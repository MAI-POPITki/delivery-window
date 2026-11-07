"""Portable planning scenarios with explicit evidence boundaries for A3."""
import pandas as pd
import streamlit as st
from pathlib import Path

from okno.corridor import CorridorConfig
from okno.regions import REGIONS
from okno.supply_scenarios import Closure, SeasonalWindow, ServiceDays, compare_supply


def render_supply():
    st.subheader('Расширение применения — сценарная лаборатория')
    st.warning('Все параметры и даты ниже — допущения. Это проверка переносимости расчёта, '
               'а не подтверждённый маршрут или экономия. ML льда не переносится на навигацию автоматически.')
    kind = st.selectbox('Применение', ['1. Другой регион: сезонный путь',
        '2. Речной завоз', '3. Снабжение удалённого объекта'], key='supply_kind')
    region = st.selectbox('Регион сценария', list(REGIONS),
                         format_func=lambda k: REGIONS[k]['name'], key='supply_region')
    st.caption('Регион задаёт контекст, а не местные тарифы или подтверждение пригодности маршрута. '
               'Существующие прогнозы регионов остаются на прежних вкладках.')
    with st.expander('Собранные источники по региону — выбрать основу для пилота'):
        registry = Path(__file__).resolve().parents[1] / 'data/external/road_events.csv'
        if registry.exists():
            events = pd.read_csv(registry).query('region == @region')
            st.dataframe(events[['road', 'hydro_year', 'kind', 'actual_date', 'source_actual']],
                         hide_index=True, width='stretch')
            st.caption('Записи существующего реестра команды; источники требуют предметной сверки. '
                       'Эти фактические даты не подставляются автоматически как заранее известный график. '
                       'Даты открытия могут означать ограниченную массу, а не полную доступность.')
        else:
            st.info('Региональный реестр отсутствует.')
    route = st.text_input('Маршрут: откуда → промежуточный пункт → получатель',
                         'База → терминал → склад получателя', key='supply_route')
    c1, c2 = st.columns(2)
    horizon = c1.number_input('Горизонт расчёта, суток', 14, 180, 60, key='supply_horizon')
    start, end = c1.slider('Период [начало, конец), дни от начала расчёта',
                           0, int(horizon), (int(horizon)//3, 2*int(horizon)//3), key='supply_window')
    if kind.startswith('1'):
        schedule = Closure(start, end)
        st.caption('Выбран период закрытия. В остальные дни основной канал доступен.')
    elif kind.startswith('2'):
        schedule = SeasonalWindow(start, end)
        st.caption('Выбрано окно навигации. Суточная мощность — агрегированная; отдельные рейсы судов не моделируются.')
    else:
        interval = c1.number_input('Интервал обслуживания объекта, суток', 1, 30, 7, key='supply_interval')
        schedule = ServiceDays(tuple(range(start, end, int(interval))))
        st.caption('Внутри выбранного периода основной канал работает в дни обслуживания. '
                   'Мощность означает объём доставки за такой день.')
    demand = c1.number_input('Потребность, т/сут', 0.1, 1000., 10., key='supply_demand')
    stock = c1.number_input('Начальный запас, т', 0., 100000., 100., key='supply_stock')
    capacity = c1.number_input('Ёмкость склада получателя, т', 1., 100000., 600., key='supply_space')
    lag = c1.number_input('Время первого участка, суток', 0, 30, 3, key='supply_lag')
    first_cap = c1.number_input('Мощность первого участка, т/сут', 1., 10000., 100., key='supply_first_cap')
    cap = c2.number_input('Основная доставка: т за доступный день', 1., 10000., 100., key='supply_cap')
    backup = c2.number_input('Резерв: т/сут вне основного окна', 0., 1000., 2., key='supply_backup')
    first_cost = c2.number_input('Первый участок, ₽/т', 0., 1000000., 5000., key='supply_first_cost')
    normal = c2.number_input('Основная доставка, ₽/т', 0., 1000000., 2000., key='supply_normal')
    alternative = c2.number_input('Резервная доставка, ₽/т', 0., 1000000., 30000., key='supply_alternative')
    holding = c2.number_input('Хранение на каждом складе, ₽/т·сут', 0., 10000., 50., key='supply_hold')
    delay = c2.number_input('Условная стоимость ожидания, ₽/т·сут', 0., 100000., 1000., key='supply_delay')
    st.caption('Два транспортных участка; второй считается доставкой в тот же день. '
               'Резерв доступен только вне основного календаря. Тонны делимы; очереди сохраняются. '
               'Стоимость включает условный штраф и оценку довоза остатка после горизонта без будущего хранения. '
               'Календарь целиком задан заранее; устойчивость к ошибкам дат здесь не проверяется.')
    if st.button('Сравнить планы', key='supply_run'):
        try:
            cfg = CorridorConfig(daily_demand_tons=demand, horizon_days=int(horizon),
                initial_yak_tons=stock, yak_warehouse_cap_tons=capacity,
                rail_lead_time_days=int(lag), rail_cap_tons=first_cap,
                crossing_cap_normal_tons=cap, crossing_cap_alt_tons=backup,
                rail_cost_per_ton=first_cost, crossing_cost_normal=normal,
                crossing_cost_alt=alternative, nb_holding_cost_per_ton_day=holding,
                yak_holding_cost_per_ton_day=holding, shortage_penalty_per_ton_day=delay)
            results = compare_supply(cfg, schedule)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return
        summary = pd.DataFrame({name: {
            'Расходы за период, ₽': s['cash_cost_rub'],
            'Оценка будущего довоза, ₽': s['terminal_commitment_rub'],
            'Всего с оценкой довоза, ₽': s['total_cost_rub'],
            'Выдано, т': s['served_tons'],
            'Очередь на конец, т': s['terminal_backlog_tons'],
            'Ожидание, т·сут': s['shortage_ton_days'],
            'Ошибка баланса, т': s['mass_balance_error_tons'],
        } for name, (s, _) in results.items()}).T
        st.dataframe(summary.round(2), width='stretch')
        st.caption('Сравнивайте стоимость вместе с ожиданием и выполнением спроса; '
                   'базовое правило не является подтверждённой практикой оператора.')
        daily = pd.concat([frame.assign(policy=name, region=region, route=route,
            calendar_assumed_known=True) for name, (_, frame) in results.items()], ignore_index=True)
        st.line_chart(daily.pivot(index='day', columns='policy', values='backlog_t'))
        st.download_button('Скачать дневной расчёт CSV', daily.to_csv(index=False).encode('utf-8-sig'),
                           'supply-scenario.csv', 'text/csv')
    st.info('Для А3: укажите конкретный российский маршрут и получателя; подтвердите проблему, '
            'календарь, мощности и цены источниками; сравните весь путь при одинаковом спросе. '
            'Для навигации дополнительно нужны рейсы, осадка и ограничения воды; '
            'для удалённого объекта — потребление и доступность резерва.')
