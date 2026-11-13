"""Обзор всех реальных участков на общей дате, выбор карты и русские подписи."""
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from okno import flood_m1_quantile as M
from okno.flood_probability import exceedance
from okno.flood_day import event_definition
from okno.flood_sites import SITES as POINTS
from okno import map_style as _flood_map_style
from okno.map_style import AVAILABILITY_SCALE,UNKNOWN_COLOR


@st.cache_data(max_entries=64,ttl=60,show_spinner=False)
def overview_rows(issue,horizon):
    rows=[]
    for pid,point in POINTS.items():
        source=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
        lon,lat=source['station']['geometry']['coordinates']
        row=dict(pid=pid,name=point['name'],river=point['river'],lat=lat,lon=lon,probability=None,
                 satellite_only=pid not in M.STATIONS,
                 status='Только спутниковое наблюдение; М1 не обучена' if pid not in M.STATIONS else 'Нет прогноза на эту дату')
        try:
            series=M.load_levels(pid)
            forecast,_=M.forecast(pid,issue);forecast=forecast.iloc[:horizon]
            definition=event_definition(pid,series,pd.Timestamp(issue).year)
            threshold=definition['threshold']
            p=exceedance(pid,issue,forecast,threshold).iloc[-1]
            row.update(probability=float(p.probability),status=f"{definition['label']}: {p.probability:.1%} (≥{threshold:.0f} см)")
        except (ValueError,OSError,KeyError):pass
        rows.append(row)
    return rows


def select_point(pid):
    if pid in POINTS:
        st.session_state['m1_flood_region']='Все регионы'
        st.session_state['m1_flood_point']=pid


def open_example():
    from datetime import date
    select_point('ural_orsk')
    st.session_state['m1_issue_ural_orsk']=date(2024,4,11)
    st.session_state['m1_flood_tab']='Общий результат'


def open_satellite(pid):
    st.session_state['flood_model_source']='Спутниковые наблюдения · М2'
    st.session_state['m2_flood_region']=POINTS[pid]['region']
    st.session_state['m2_flood_point']=pid


def render_overview(issue,horizon,selected):
    with st.spinner('Рассчитываем обзор участков…'):
        rows=overview_rows(str(issue),int(horizon))
    st.subheader('1. Выберите участок')
    st.write('Выберите город кнопкой под картой или в боковой панели. Затем проверьте прогноз и спутниковую обстановку, '
             'а для плана доставки укажите условия проезда и свои тарифы.')
    st.button('Пример: паводок в Орске 11 апреля 2024',key='flood_ready_example',on_click=open_example)
    st.caption(f"{len(rows)} участков в {len({p['region'] for p in POINTS.values()})} регионах · расчёт на {pd.Timestamp(issue):%d.%m.%Y} · горизонт {horizon} сут. "
               'Карта показывает все участки проекта, а не только выбранный гидропост.')
    fig=go.Figure()
    for row in rows:
        known=row['probability'] is not None
        group=[row]
        marker=dict(size=[23 if r['pid']==selected else 16 for r in group],
            color=[1-r['probability'] for r in group] if known else '#0b3d91' if row['satellite_only'] else UNKNOWN_COLOR)
        if known:marker.update(colorscale=AVAILABILITY_SCALE,cmin=0,cmax=1)
        fig.add_trace(go.Scattermap(lon=[r['lon'] for r in group],lat=[r['lat'] for r in group],
            mode='markers+text',text=[r['name'] for r in group],
            textposition='top left' if row['pid'] in ('ural_orenburg','tobol_kurgan') else 'bottom right' if row['pid']=='ural_orsk' else 'top right',
            textfont=dict(size=14,color='#111827'),marker=marker,
            customdata=[[r['pid']] for r in group],
            hovertext=[f"{r['name']} · {r['river']}<br>{r['status']}" for r in group],hoverinfo='text'))
    fig.update_layout(map=dict(style=_flood_map_style.FLOOD_MAP_STYLE,center=dict(lon=94,lat=57),zoom=2.35),
        height=370,margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
    st.plotly_chart(fig,key='flood_overview_map',width='stretch',
        config={'displaylogo':False,'scrollZoom':True,'modeBarButtonsToRemove':['toImage','lasso2d','select2d']})
    st.caption('Зелёный → красный: вероятность события в выбранные сутки. Урал: выход на пойму '
               '(Орск — верхняя отметка 450 см); Якутск/Ишим: высокая вода q90, порог разлива не подтверждён. '
               'Синий: только спутниковый режим. Серый: нет расчёта на выбранную дату. '
               'Это не разрешение проезда. Наведите на точку для просмотра оценки. Выбор участка — кнопками ниже.')
    columns=st.columns(len(rows))
    for col,row in zip(columns,rows):
        col.button(('● ' if row['pid']==selected else '')+row['name'],key='choose_flood_'+row['pid'],
            on_click=select_point,args=(row['pid'],),width='stretch')
