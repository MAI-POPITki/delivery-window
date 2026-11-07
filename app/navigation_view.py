"""Речной сценарий с существующими данными М1 и М2, без новой калибровки."""
import datetime as dt
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from okno import forecast as FC
from okno.ice import to_season_day
from okno.corridor import CorridorConfig
from okno.navigation import navigation_windows, compare_navigation, observation_as_of


def render_navigation():
    st.subheader('Лена: существующий М1 → план речной доставки')
    st.caption('Погода, обученная модель вскрытия и спутниковые наблюдения используются из текущего проекта. '
               'Другая нейросеть не обучается. Сравниваются одинаковые оптимизаторы с разными датами.')
    issue = st.date_input('Дата выпуска прогноза для речного сценария', dt.date.today(), key='nav_issue')
    year = issue.year+1 if issue.month >= 9 else issue.year
    origin = dt.date(year, 3, 1)
    end = dt.date(year, 10, 1)
    horizon = (end-origin).days
    offset = st.number_input('Сдвиг от физического вскрытия до начала перевозок, суток — допущение',
                             0, 60, 14, key='nav_offset')
    st.warning('М1 здесь прогнозирует спутниковое вскрытие, а не официальное открытие навигации. '
               'Сдвиг и конец сезона 1 октября условные. Нужны график оператора и ограничения судна. '
               'Осеннее закрытие по этой модели не прогнозируется.')
    demand = st.number_input('Потребность речного сценария, т/сут', 0.1, 1000., 10., key='nav_demand')
    normal = st.number_input('Тариф речного участка, ₽/т — допущение', 0., 1000000., 2000., key='nav_rate')
    st.caption('Остальные параметры пока одинаковые модельные: запас 100 т, склад 600 т, '
               'первый участок 3 суток / 100 т в сутки / 5000 ₽ за тонну; речной участок 100 т в сутки; '
               'резерв 2 т в сутки / 30000 ₽ за тонну; хранение 50 ₽ за тонно-сутки, ожидание 1000 ₽ за тонно-сутки.')
    if st.checkbox('Показать положение выбранной точки на карте', False, key='nav_map'):
        meta = FC.point_meta('yakutsk_crossing')
        fig = go.Figure(go.Scattermap(lat=[meta['lat']], lon=[meta['lon']], mode='markers',
            marker=dict(size=14, color='#1479b8'), text=['Точка прогноза: переправа через Лену'], hoverinfo='text'))
        fig.update_layout(map=dict(style='carto-positron', center=dict(lat=meta['lat'], lon=meta['lon']), zoom=8),
                          height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption('Одна точка наблюдения. Геометрия судового хода не подтверждена, поэтому линия маршрута не рисуется.')
    st.markdown('**Цепочка:** база → терминал Нижнего Бестяха → речной участок через Лену → склад получателя в Якутске.')
    if st.button('Рассчитать по существующему М1', key='nav_run'):
        try:
            q = FC.forecast('yakutsk_crossing', year, to_season_day(issue, year), 'close')
            windows = navigation_windows(q, origin, horizon, end, int(offset))
            cfg = CorridorConfig(horizon_days=horizon, daily_demand_tons=demand,
                initial_yak_tons=100, yak_warehouse_cap_tons=600, rail_lead_time_days=3,
                rail_cap_tons=100, crossing_cap_normal_tons=100, crossing_cap_alt_tons=2,
                rail_cost_per_ton=5000, crossing_cost_normal=normal, crossing_cost_alt=30000,
                nb_holding_cost_per_ton_day=50, yak_holding_cost_per_ton_day=50,
                shortage_penalty_per_ton_day=1000)
            results = compare_navigation(cfg, windows)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            st.error(f'Расчёт недоступен: {exc}')
            return
        st.write(f"Дата погоды, на которую выпущен М1: день {q['issue_day']} сезона {year-1}/{year}. "
                 f"Вскрытие: {q['q50_date']}, диапазон {q['q10_date']} — {q['q90_date']}.")
        names = {'calendar':'Календарь', 'q10':'М1: раннее вскрытие', 'q50':'М1: медиана', 'q90':'М1: позднее вскрытие'}
        rows=[]
        for key,(s, daily) in results.items():
            rows.append({'Источник':names[key], 'Начало перевозок (условное)':str(origin+dt.timedelta(days=windows[key].start)),
                'Расходы периода, ₽':s['cash_cost_rub'], 'Оценка довоза, ₽':s['terminal_commitment_rub'],
                'Всего, ₽':s['total_cost_rub'], 'Выдано, т':s['served_tons'],
                'Очередь на конец, т':s['terminal_backlog_tons'], 'Ожидание, т·сут':s['shortage_ton_days']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Четыре плана по четырём предполагаемым календарям. Это чувствительность плана к дате, '
                   'не доказательство превосходства ML: планы ещё не проиграны на общем фактическом будущем. '
                   'Диапазон физического события не является откалиброванным диапазоном навигации.')
        frames = pd.concat([d.assign(source=k, forecast_issue_day=q['issue_day'],
            physical_event=str(q[k+'_date']), navigation_offset_days=offset,
            navigation_end=str(end)) for k,(_,d) in results.items()],ignore_index=True)
        st.download_button('Скачать речные планы CSV', frames.to_csv(index=False).encode('utf-8-sig'), 'lena-navigation.csv','text/csv')
        path=Path(__file__).resolve().parents[1]/'data/s1/predictions.csv'
        if path.exists():
            obs=observation_as_of(pd.read_csv(path), 'yakutsk_crossing', issue)
            if obs is not None:
                st.write(f"М2, снимок {obs['date']:%d.%m.%Y}: P(лёд) = {obs.p_ice:.2f}.")
                st.caption('Последний доступный на дату решения снимок; М2 показывает состояние точки, '
                           'но не прогнозирует прекращение перевозок. В даты планов автоматически не вмешивается.')
