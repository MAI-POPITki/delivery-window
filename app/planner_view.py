"""Prospective M3 plans linked to the sidebar and the user's economic inputs."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno.ice import from_season_day
from okno.planner import Params, compare_ml_weekly_plans


def render_planner(forecast, hydro_year, issue_day, margin, point_name, *, transit_days=None,quantity_t=None,require_quantity=False,availability=None,start_day=None,start_note=None,point_id=None):
    if require_quantity: st.session_state.pop('journey_m3_summary',None)
    st.subheader('Планировщик отправок (М3)')
    st.caption('Оба графика используют путь, сезон и дату решения из левой панели и параметры ниже. '
               'Цены и мощности — сценарные значения; результат показывает ожидаемые расходы.')
    if quantity_t is not None and quantity_t>0:
        st.session_state['planner_quantity']=float(quantity_t)
        st.caption(f'Объём завоза взят из ваших заявок: {quantity_t:g} т. Для изменения объёма исправьте заявки выше.')
    if transit_days is not None:
        st.caption(f'Время рейса взято из выбранного маршрута: {transit_days*24:.1f} ч. Ограничения полной массы и возврат проверяются в рейсовом плане выше.')
    if start_note: st.caption('Начало сезона отправок: '+start_note+'.')
    c1, c2, c3 = st.columns(3)
    p = Params(Q_t=c1.number_input('Осталось вывезти на дату решения, т', 0.1, 100000.0, quantity_t if quantity_t else None if require_quantity else 12000.0, 1.0,
                                   format='%.1f', key='planner_quantity',disabled=quantity_t is not None and quantity_t>0),
        K_t_week=c1.number_input('Своя техника, т/нед', 0.1, 20000.0, None if require_quantity else 1000.0, 1.0,
                                 format='%.1f', key='planner_capacity'),
        K_extra_t_week=c1.number_input('Доп. техника максимум, т/нед', 0.0, 20000.0, 0.0 if require_quantity else 600.0, 1.0,
                                       format='%.1f', key='planner_extra_capacity'),
        c_road=c2.number_input('Зимник, ₽/т', 1000.0, 200000.0, None if require_quantity else 20000.0, 1000.0,
                               format='%.2f', key='planner_road_rate'),
        c_extra=c2.number_input('Надбавка за доп. технику, ₽/т', 0.0, 200000.0, 0.0 if require_quantity else 10000.0, 1000.0,
                                format='%.2f', key='planner_extra_rate'),
        c_exp=c3.number_input('Дорогой канал (авиа/СВП), ₽/т', 10000.0, 1000000.0, None if require_quantity else 120000.0, 5000.0,
                              format='%.2f', key='planner_backup_rate'),
        c_hold_week=c3.number_input('Хранение, ₽/т·нед', 0.0, 10000.0, None if require_quantity else 400.0, 50.0,
                                    format='%.2f', key='planner_holding_rate'),
        close_margin_days=margin,transit_days=transit_days if transit_days is not None else 7,
        start_day=start_day if start_day is not None else 150)
    if p.Q_t is None:
        st.info('Введите остаток заказа либо добавьте свои заявки в рейсовый план. Учебный объём не подставляется.')
        return p
    if any(x is None for x in (p.K_t_week,p.c_road,p.c_exp,p.c_hold_week)):
        st.info('Введите мощности и свои тарифы. Действующие ставки для произвольного зимника не выводятся из тарифов парома.')
        return p
    if not forecast or 'error' in forecast:
        st.info('Нет прогноза закрытия для этой точки.')
        return p
    from app.handling_wind_view import render_wind
    from okno.handling_wind import constrain
    days=[from_season_day(d,hydro_year).date() for d in range(int(issue_day),300)]
    arrivals=sorted({from_season_day(d+p.transit_days,hydro_year).date() for d in range(int(issue_day),300)})
    wind=render_wind('winter_m3_wind',from_season_day(issue_day,hydro_year).date(),days,arrivals,default_point=point_id)
    handling_capacity=None
    if wind is not None:
        def handling_capacity(starts,ends):
            calendar=pd.DataFrame(dict(date=[from_season_day(d,hydro_year).date() for d in starts],
                arrival=[from_season_day(d,hydro_year).date() for d in ends],capacity_t=[1e30]*len(starts)))
            return constrain(calendar,*wind).capacity_t.to_numpy()
        st.info('Ограничения обработки включены: отправки рассчитываются по суткам. Недельная мощность техники делится на 7; '
                'мощность обработки ограничивает её дополнительно. За пределами прогноза ветра отправки не планируются. '
                'Оба сравниваемых способа получают одинаковые ограничения.')
    weekly, summary = compare_ml_weekly_plans(forecast, issue_day, p,availability=availability,handling_capacity=handling_capacity)
    if require_quantity: st.session_state['journey_m3_summary']=summary.copy()
    if availability is not None:
        st.caption('Оба плана и рекомендации изменения старта исключают недели, для которых реестр не подтверждает массу или сообщает закрытие. '
                   'Неизвестные дни не считаются открытыми. Это может оставить весь груз для резервного канала.')
        st.caption('Остаток при недостатке сведений означает необходимость уточнить допуск или обеспечить резерв, а не доказанный фактический недовоз.')
    names = {'ml':'Обычный ML', 'economic_ml':'Экономический ML'}
    st.write(f'**{point_name} · дата решения {from_season_day(issue_day, hydro_year):%d.%m.%Y}**')
    st.caption(f"Прогноз выпущен {from_season_day(forecast.get('issue_day', issue_day), hydro_year):%d.%m.%Y}. "
               f"Медиана закрытия с запасом {margin} сут: {from_season_day(forecast['q50']-margin, hydro_year):%d.%m.%Y}. "
               f'Время рейса — {p.transit_days:g} суток. Начало сезонных отправок — не раньше '
               f'{from_season_day(p.start_day, hydro_year):%d.%m.%Y} и выбранной даты решения. '
               'Дата закрытия оценочная: для проезда нужны действующие разрешения и ограничения массы.')
    st.caption('Обычный ML распределяет груз равномерно до прогнозной медианы закрытия с учётом времени рейса. '
               'Экономический ML использует прежний расчёт М3: выбирает недели и свою/дополнительную технику '
               'по ожидаемой стоимости перевозки, хранения и дорогого канала. Это план на выбранную дату; '
               'при новом прогнозе его следует пересчитать.')
    economic = summary.loc['economic_ml']
    st.write(f"Рекомендация: своей техникой **{economic.planned_base_t:,.1f} т**, привлечь дополнительную на "
             f"**{economic.planned_extra_t:,.1f} т** — экономический ML.".replace(',', ' '))
    if weekly.empty:
        st.info('На выбранную дату в расчётном периоде не осталось недель для отправок по зимнику. '
                'Весь остаток отнесён к дорогому каналу.')
    max_volume = max(1., float(weekly.shipped_t.max())) if not weekly.empty else 1.
    for column, rule in zip(st.columns(2), ('ml', 'economic_ml')):
        with column:
            st.markdown(f'**{names[rule]}**')
            s = summary.loc[rule]
            delta = f"{s.expected_cost_rub-summary.loc['ml','expected_cost_rub']:+,.0f} ₽ к обычному ML".replace(',', ' ') if rule == 'economic_ml' else None
            st.metric('Ожидаемые расходы', f'{s.expected_cost_rub:,.0f} ₽'.replace(',', ' '),
                       delta=delta, delta_color='inverse', height=120)
            plan = weekly[weekly.rule.eq(rule)]
            dates = [from_season_day(day, hydro_year).date() for day in plan.dispatch_day]
            chart = go.Figure()
            chart.add_bar(x=dates, y=plan.base_t, name='Своя техника')
            chart.add_bar(x=dates, y=plan.extra_t, name='Дополнительная техника')
            chart.update_layout(barmode='stack', title=f'{names[rule]}: отправки '+('по дням' if wind is not None else 'по неделям'),
                height=400, yaxis=dict(title='т/сут' if wind is not None else 'т/нед', range=[0, max_volume*1.1]),
                xaxis=dict(title='Дата отправки', tickformat='%d.%m'),
                legend=dict(orientation='h', y=-.3))
            st.plotly_chart(chart, width='stretch', key=f'planner_chart_{rule}')
            st.caption(f'Запланировано по зимнику: {s.planned_road_t:,.1f} т. '
                       f'Ожидается дорогим каналом: {s.expected_backup_t:,.1f} т.'.replace(',', ' '))
    view = summary[['expected_cost_rub','planned_base_t','planned_extra_t','expected_delivered_t','expected_backup_t']].copy()
    view.index = view.index.map(names)
    view.index.name = 'Подход'
    view.columns = ['Ожидаемые расходы, ₽','Своя техника: план, т','Доп. техника: план, т',
                    'Ожидается по зимнику, т','Ожидается дорогим каналом, т']
    st.dataframe(view.round(2), width='stretch', key='planner_comparison', column_config={
        col:st.column_config.NumberColumn(format='localized') for col in view})
    table = weekly[weekly.rule.eq('ml')].set_index('week')
    other = weekly[weekly.rule.eq('economic_ml')].set_index('week')
    dates = [from_season_day(day, hydro_year).date() for day in table.dispatch_day]
    shipments = pd.DataFrame({'День' if wind is not None else 'Неделя':table.index, 'Дата отправки':dates,
        'Обычный ML: своя, т':table.base_t, 'Обычный ML: доп., т':table.extra_t,
        'Экономический ML: своя, т':other.base_t, 'Экономический ML: доп., т':other.extra_t,
        'Риск не успеть за рейс, %':100*table.p_closed}).reset_index(drop=True)
    if wind is not None:shipments['Мощность обработки, т/сут']=table.handling_capacity_t.to_numpy()
    st.dataframe(shipments.round(2), width='stretch', hide_index=True, key='planner_weekly')
    st.caption('Дорогой канал включает остаток, не назначенный на зимник, и ожидаемый объём рейсов, '
               'которые не успеют до закрытия. При нехватке мощности этот объём не исчезает из расходов. '
               'Расписание дорогого канала не задано и на недельных графиках не показано. '
               'По принятой модели расходы зимника и хранения начисляются на успешно доставленные тонны; '
               'затраты неудачного рейса не учтены. Расчёт М3 описывает выбранный участок, сквозной коридор А3 считается отдельно.')
    export = weekly.copy()
    export['dispatch_date'] = [from_season_day(day, hydro_year).date().isoformat() for day in export.dispatch_day]
    export['arrival_date'] = [from_season_day(day, hydro_year).date().isoformat() for day in export.arrival_day]
    if wind is not None:
        for label,calendar in zip(['loading','unloading'],wind):
            for name,value in calendar.attrs.get('wind',{}).items():export[label+'_'+name]=value
    st.download_button('Скачать оба ML-плана'+(' по дням' if wind is not None else ' по неделям'), export.to_csv(index=False).encode('utf-8-sig'),
                       'planner-ml-weekly.csv', 'text/csv', key='planner_weekly_download', on_click='ignore')
    from app.decision_view import render_decisions
    with st.expander('Что изменить, чтобы успеть: сравнить дату начала отправок и устойчивость к тарифам', expanded=False):
        render_decisions(forecast, issue_day, p, hydro_year,availability=availability,handling_capacity=handling_capacity)
    return p
