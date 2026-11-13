from okno import map_style as _flood_map_style
"""Общая рабочая область прогноза паводков и доставки."""
from datetime import date, datetime, timedelta, timezone
import math
import xml.etree.ElementTree as ET

import plotly.graph_objects as go
import streamlit as st

from okno.flood import load_points, evaluate_window, evaluate_points, available_forecast_dates
from okno.flood_routes import load_routes
from okno.map_style import AVAILABILITY_SCALE, UNKNOWN_COLOR


def human_date(value):
    return datetime.fromisoformat(str(value)).strftime('%d.%m.%Y')


def number(value):
    return f'{value:,.0f}'.replace(',', ' ')


def _open_saved_forecast(day):
    st.session_state['flood_decision'] = day
    for key in list(st.session_state):
        if key.startswith('flood_target_'):
            st.session_state.pop(key,None)
    st.session_state['flood_tab'] = 'Прогноз'


def make_map(rows, selected=None, routes=(), selected_route=None):
    fig = go.Figure()
    for highlight in (False, True):
        lat, lon, text = [], [], []
        for route in routes:
            if (route['group'] == selected_route) != highlight:
                continue
            coords = route['coordinates']
            lon.extend([p[0] for p in coords] + [None])
            lat.extend([p[1] for p in coords] + [None])
            text.extend([route['name']] * len(coords) + [''])
        if lat:
            fig.add_trace(go.Scattermap(lat=lat, lon=lon, text=text, mode='lines', hoverinfo='text',
                          line=dict(width=5 if highlight else 2, color='#2563eb' if highlight else '#64748b'),
                          name='Выбранный путь' if highlight else 'Дороги OSM'))
    for available, group in rows.groupby(rows.fraction.notna()):
        marker = dict(size=[23 if pid == selected else 15 for pid in group.pid], color=UNKNOWN_COLOR)
        text = [f'{r.river} · {r.name}<br>Нет выпуска на дату планирования' for r in group.itertuples()]
        if available:
            marker.update(color=(1-group.fraction)*100, colorscale=AVAILABILITY_SCALE, cmin=0, cmax=100,
                          showscale=True, colorbar=dict(title=dict(text='Без превышения порога, %', side='top'),
                          orientation='h', thickness=10, len=.9, x=.5, xanchor='center', y=-.06))
            text = [f'{r.river} · {r.name}<br>Высокая вода: {r.fraction:.0%} сценариев'
                    f'<br>{human_date(r.start)} — {human_date(r.end)}'
                    + (f'<br>Расход в выбранную дату: {number(r.median_m3s)} м³/с' if hasattr(r,'median_m3s') else '')
                    for r in group.itertuples()]
        labels = [r.name if len(rows) <= 2 or r.pid == selected else '' for r in group.itertuples()]
        fig.add_trace(go.Scattermap(lat=group.lat, lon=group.lon, text=labels, mode='markers+text',
                                   textposition='top right', textfont=dict(color='#172033', size=13),
                                   hovertext=text, hoverinfo='text', customdata=group[['pid']], marker=marker))
    lat_span = float(rows.lat.max()-rows.lat.min())
    lon_span = float(rows.lon.max()-rows.lon.min())
    zoom = max(2., min(10., math.log2(180/max(lat_span*2, lon_span, .1))+.4))
    center = dict(lat=float((rows.lat.min()+rows.lat.max())/2), lon=float((rows.lon.min()+rows.lon.max())/2))
    if len(rows) == 1 and routes:
        coords = [p for r in routes for p in r['coordinates']] + [[float(rows.lon.iloc[0]), float(rows.lat.iloc[0])]]
        west, east = min(p[0] for p in coords), max(p[0] for p in coords)
        south, north = min(p[1] for p in coords), max(p[1] for p in coords)
        center = dict(lat=(south+north)/2, lon=(west+east)/2)
        zoom = min(11., math.log2(180/max((north-south)*2, east-west, .1)))
    fig.update_layout(map=dict(style=_flood_map_style.FLOOD_LOCAL_MAP_STYLE, center=center, zoom=zoom), height=530,
                      margin=dict(l=0, r=0, t=0, b=65), showlegend=False)
    return fig


def render_forecast(d, target=None, travel=1):
    if d is None:
        st.info('Нет сохранённого прогноза, известного на выбранную дату. Поздние выпуски и исторические факты его не заменяют.')
        return
    st.subheader(f"Прогноз: {human_date(d['start'])} — {human_date(d['end'])}")
    target = target or datetime.fromisoformat(d['start']).date()
    selected = evaluate_window(d, target, travel)
    if selected is None:
        st.info('Этот выпуск не покрывает выбранную дату и весь период проезда.')
        return
    a,b,c = st.columns(3)
    a.metric('Расход на выбранную дату', f"{number(selected['median'])} м³/с")
    b.metric('Высокая вода во время проезда', f"{round(selected['fraction']*selected['members'])} из {selected['members']}")
    c.metric('Порог сравнения', f"{number(d['threshold'])} м³/с")
    curve = d['curve']
    fig = go.Figure()
    fig.add_scatter(x=curve.index, y=curve.q90, line=dict(width=0), showlegend=False, hoverinfo='skip')
    fig.add_scatter(x=curve.index, y=curve.q10, fill='tonexty', line=dict(width=0),
                   fillcolor='rgba(37,99,235,.15)', name='Средние 80% сценариев', hoverinfo='skip')
    fig.add_scatter(x=curve.index, y=curve.q50, name='Медиана', line=dict(color='#2563eb', width=3))
    fig.add_scatter(x=[str(target)], y=[selected['median']], mode='markers', marker=dict(size=12,color='#e11d48'),
                   name='Выбранная дата')
    if st.checkbox('Показать порог высокой воды', key='flood_show_threshold'):
        fig.add_hline(y=d['threshold'], line_dash='dash', annotation_text='Исторический порог')
    fig.update_layout(yaxis_title='Расход, м³/с', xaxis=dict(tickformat='%d.%m'), height=350,
                      legend=dict(orientation='h'), hovermode='x unified')
    st.plotly_chart(fig, width='stretch', config={'displaylogo':False})
    daily = curve.rename(columns={'q10':'Нижняя граница, м³/с','q50':'Медиана, м³/с','q90':'Верхняя граница, м³/с'}).copy()
    daily['Сценариев выше порога'] = (d['scenarios'] > d['threshold']).sum(axis=1)
    daily['Выбранная дата'] = daily.index.date == target
    daily.index = daily.index.date
    st.dataframe(daily, width='stretch')
    st.caption(f"Исторический порог: {number(d['threshold'])} м³/с. Получено {human_date(d['fetched_at'])}; "
               f"{d['members']} сценариев, {d['years']} полных лет архива. Даты — UTC.")
    st.download_button('Скачать прогноз (CSV)', curve.rename(columns={'q10':'Нижняя граница, м³/с',
                       'q50':'Медиана, м³/с','q90':'Верхняя граница, м³/с'}).to_csv(index_label='Дата UTC').encode('utf-8-sig'),
                       'flood-forecast.csv', 'text/csv')


def render_flood():
    from app.flood_workspace import render_risks, render_optimizer, render_data
    st.title('Паводки')
    st.caption('Прогноз воды, проверка пути и планирование доставки')
    today = datetime.now(timezone.utc).date()
    st.session_state.setdefault('flood_decision',today)
    decision = st.sidebar.date_input('Данные, известные на дату', value=None, min_value=date(1997,1,1),
                                   max_value=today, format='DD.MM.YYYY', key='flood_decision')
    rows, details = load_points(planning_date=decision)
    region = st.sidebar.selectbox('Регион', ['Все четыре региона', *dict.fromkeys(rows.region)], index=1, key='flood_region')
    shown = rows if region == 'Все четыре региона' else rows[rows.region == region]
    indexed = rows.set_index('pid')
    ids = shown.pid.tolist()
    if st.session_state.get('flood_point') not in ids:
        st.session_state['flood_point'] = ids[0]
    pid = st.sidebar.selectbox('Река и населённый пункт', ids, key='flood_point',
                              format_func=lambda p:f"{indexed.loc[p,'river']} · {indexed.loc[p,'name']}")
    from app.flood_live_view import render_live
    render_live(pid)
    saved_dates = available_forecast_dates(pid)
    st.sidebar.caption('Сохранены данные, полученные: '+(', '.join(day.strftime('%d.%m.%Y') for day in saved_dates[-5:]) or 'нет выпусков'))
    try:
        routes, metadata = load_routes(pid)
    except (OSError, ValueError, KeyError, ET.ParseError):
        routes, metadata = [], {}
        st.sidebar.warning('Данные дорог повреждены. Прогноз воды доступен; выбор пути требует обновления геометрии.')
    for route in routes:
        route['group'] = route['kind']+' · '+route['name']
    groups = sorted({r['group'] for r in routes}, key=lambda name:('Участок OSM' in name,
                    not name.startswith(('Мост','Переправа')), name))
    if groups:
        if st.session_state.get('flood_route') not in groups:
            st.session_state['flood_route'] = groups[0]
        selected_route = st.sidebar.selectbox('Путь / участок', groups, key='flood_route')
    else:
        selected_route = None
        st.sidebar.info('Геометрии дорог пока не загружены.')
    travel = st.sidebar.number_input('Время проезда, суток', min_value=1, max_value=7, value=1, key='flood_travel')
    d = details.get(pid)
    first = datetime.fromisoformat(d['start']).date() if d else decision + timedelta(days=1)
    last = datetime.fromisoformat(d['end']).date() if d else first
    target_key = f'flood_target_{pid}_{decision}_{first}_{last}'
    if not first <= st.session_state.get(target_key,first) <= last:
        st.session_state[target_key] = first
    st.session_state.setdefault(target_key,first)
    target = st.sidebar.date_input('Дата оценки / проезда', value=None, min_value=first, max_value=last,
                                  format='DD.MM.YYYY', key=target_key, disabled=d is None)
    st.sidebar.caption('Первая дата выбирает доступный выпуск, вторая — сутки внутри него. '
                       'Исторический ряд не заменяет отсутствующий прогноз.')
    selected = evaluate_window(d, target, travel) if d else None
    shown = evaluate_points(shown, details, target, travel)
    if d:
        st.caption(f"Выпуск получен {human_date(d['fetched_at'])}; оценка на {target:%d.%m.%Y}, проезд {travel} сут.")
    if selected:
        st.markdown(f"**На {target:%d.%m.%Y}: {number(selected['median'])} м³/с** · "
                    f"сценарии высокой воды за проезд: **{selected['fraction']:.0%}**.")
        if selected['fraction'] == 0:
            st.caption('Расход меняется по датам, но во всех сценариях остаётся ниже исторического порога. '
                       'Поэтому показатель превышения равен 0%; это не означает отсутствия любой угрозы.')
    with st.expander('На чём построен прогноз'):
        st.write('Метеопрогноз ECMWF → гидрологическая модель LISFLOOD в GloFAS → суточный расход реки → '
                 'наше сравнение с историческим порогом. Мы получаем готовый расчёт расхода через Open-Meteo, '
                 'а не обучаем собственную модель паводка.')
        st.markdown('[Метод GloFAS](https://ewds.climate.copernicus.eu/datasets/cems-glofas-forecast?tab=overview) · '
                    '[Описание API](https://open-meteo.com/en/docs/flood-api)')
        st.write('Наш порог — 80-й процентиль годовых максимумов модельного архива. Это исследовательское правило, '
                 'не официальная опасная отметка. Геометрия дороги OSM не участвует в прогнозе расхода; '
                 'связь расхода с затоплением и закрытием конкретной дороги не установлена.')
    for key in ('flood_Q','flood_road_cost','flood_backup_cost','flood_holding','flood_policy'):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    st.warning('Экспериментальная гидрология: превышение расходного порога не равно затоплению или закрытию дороги. '
               'Допуск и мощности пути подтверждаются отдельно.')
    if d is None:
        st.info(f'Для решения на {decision:%d.%m.%Y}: '+str(indexed.loc[pid,'status']))
        if saved_dates:
            st.write('Сохранённые прогнозы доступны за '+', '.join(day.strftime('%d.%m.%Y') for day in saved_dates[-5:])+'. '
                     'Для более раннего решения нужен архив действительно выпущенных тогда прогнозов. '
                     'Исторический ряд расхода нельзя выдавать за такой выпуск.')
            st.button('Открыть последний сохранённый прогноз',on_click=_open_saved_forecast,
                       args=(saved_dates[-1],),key='flood_open_available')
    aliases = {'Прогноз воды':'Прогноз','Риски и рекомендации':'Риски','Планировщик':'План','О данных':'Данные'}
    current_tab = st.session_state.get('flood_tab')
    if current_tab in aliases:
        st.session_state['flood_tab'] = aliases[current_tab]
    tabs = st.tabs(['Карта путей','Прогноз','Риски','План','Данные'],
                   key='flood_tab', on_change='rerun')
    with tabs[0]:
        if tabs[0].open is not False:
            st.subheader(selected_route or 'Карта речных точек')
            st.plotly_chart(make_map(shown, pid, routes, selected_route), width='stretch',
                            config={'scrollZoom':True,'displaylogo':False}, key='flood_map')
            st.caption('Палитра как у зимников: красный → жёлтый → зелёный. Цвет точки — доля сценариев без высокой воды; '
                       'серый — нет прогноза. Синий — выбранный путь, серые линии — дороги OSM. '
                       'Цвет реки не подтверждает проходимость дороги.')
            st.caption(f"Дорог и мостовых участков в данных: {len(routes)}. © OpenStreetMap contributors, ODbL.")
    with tabs[1]:
        if tabs[1].open is not False:
            render_forecast(d, target, travel)
    with tabs[2]:
        if tabs[2].open is not False:
            render_risks(d, selected_route, decision, travel, target)
    with tabs[3]:
        if tabs[3].open is not False:
            render_optimizer(d, selected_route, pid, decision, travel, target)
    with tabs[4]:
        if tabs[4].open is not False:
            render_data(pid, decision, d, routes, metadata)


if __name__ == '__main__':
    render_flood()
