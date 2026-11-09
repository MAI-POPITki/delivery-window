"""Собственный М1 по измеренным уровням: исторический прогноз и независимая проверка."""
import json
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno import flood_m1 as M
from okno.flood import POINTS
from okno.flood_routes import load_routes
from okno.map_style import AVAILABILITY_SCALE

NAMES={'catboost':'CatBoost','persistence':'Последнее измерение','calendar':'Календарная медиана','last_year':'Как в прошлом году'}


def render_m1():
    st.title('Паводки')
    st.caption('М1 · собственный прогноз измеренного уровня воды на 1–7 суток')
    regions=list(dict.fromkeys(p['region'] for p in POINTS.values()))
    region=st.sidebar.selectbox('Регион',['Все четыре региона',*regions],index=1,key='m1_flood_region')
    points=[pid for pid,p in POINTS.items() if region=='Все четыре региона' or p['region']==region]
    if st.session_state.get('m1_flood_point') not in points:
        st.session_state['m1_flood_point']=points[0]
    pid=st.sidebar.selectbox('Гидропост',points,key='m1_flood_point',
                            format_func=lambda p:f"{POINTS[p]['river']} · {POINTS[p]['name']}")
    if pid not in M.STATIONS:
        st.warning('Курган: М1 пока не обучена. В доступном архиве всего один полный год ГМВО; другие записи имеют большие разрывы. '
                   'Прогноз другой реки сюда не переносится.')
        st.markdown('Источники и покрытие: [Уровень-Спутник](https://level-satellite.ru). '
                    'Внешний расход GloFAS доступен отдельным пунктом в боковой панели.')
        return
    series=M.load_levels(pid)
    manifest=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
    valid=series.dropna()
    earliest=max(valid.index.min().date(),date(2017,1,1))
    latest=valid.index.max().date()
    key=f'm1_issue_{pid}'
    st.session_state.setdefault(key,min(latest,date(latest.year,5,1)))
    issue=st.sidebar.date_input('Дата измерения / решения',value=None,min_value=earliest,max_value=latest,
                                format='DD.MM.YYYY',key=key)
    horizon=st.sidebar.slider('Горизонт М1, суток',1,7,7,key='m1_flood_horizon')
    st.sidebar.caption(f"Наблюдения: {valid.index.min():%d.%m.%Y} — {valid.index.max():%d.%m.%Y}. "
                       f"Источник ряда: {M.STATIONS[pid][1]}. Данные за 2026 год отсутствуют.")
    st.info('Исторический расчёт по реальным измерениям. Модель использует только данные до выбранной даты. '
            'Свежего собственного прогноза на 2026 год нет: не получены текущие уровни гидропостов.')
    try:
        forecast,meta=M.forecast(pid,issue)
    except ValueError as exc:
        st.warning(str(exc));return
    forecast=forecast.iloc[:horizon]
    selected=forecast.iloc[-1]
    current=float(series.loc[pd.Timestamp(issue)])
    validation_path=M.MODELS/'validation.json'
    validation=json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    check=validation.get(f'{pid}:{horizon}',{})
    a,b,c=st.columns(3)
    a.metric('Измерено на дату решения',f'{current:.0f} см')
    b.metric(f'Прогноз на {selected.date:%d.%m.%Y}',f'{selected.predicted_cm:.0f} см',
              delta=f'{selected.predicted_cm-current:+.0f} см')
    c.metric('Оценочный диапазон 80%',f'{selected.lower_cm:.0f} … {selected.upper_cm:.0f} см')
    st.caption(f"Уровни — сантиметры над нулём этого гидропоста. Выбран по предыдущему году: {NAMES[selected.method]}. "
               f"Обучение по {meta['train_until']}; калибровка — {meta['calibration_year']} год.")
    if check.get('point_skill_success'):
        st.success('Точечный прогноз для этой точки и горизонта прошёл критерий сравнения с простыми правилами на отдельных будущих годах.')
    else:
        st.warning('Для этой точки/горизонта преимущество по всем критериям не подтверждено. Модель остаётся экспериментальной.')
    if not check.get('interval_valid'):
        coverage=check.get('spring_coverage')
        st.warning('Диапазон ещё ненадёжен в половодье'
                   +(f': в проверке апреля–июля факт попадал в него только в {coverage:.0%} случаев.' if coverage is not None else '.')
                   +' Это не гарантия 80%-й вероятности и не разрешение проезда.')
    tabs=st.tabs(['Карта','М1: прогноз','Проверка','Данные'],key='m1_flood_tab',on_change='rerun')
    with tabs[0]:
        if tabs[0].open is not False:
            roads,_=load_routes(pid)
            fig=go.Figure()
            lat,lon=[],[]
            for r in roads:
                lon.extend([p[0] for p in r['coordinates']]+[None]);lat.extend([p[1] for p in r['coordinates']]+[None])
            if lat: fig.add_trace(go.Scattermap(lat=lat,lon=lon,mode='lines',line=dict(color='#64748b',width=2),hoverinfo='skip'))
            gauge_lon,gauge_lat=manifest['station']['geometry']['coordinates']
            fig.add_trace(go.Scattermap(lat=[gauge_lat],lon=[gauge_lon],mode='markers',
                marker=dict(size=23,color=[current-selected.predicted_cm],colorscale=AVAILABILITY_SCALE,cmin=-100,cmax=100,
                            showscale=True,colorbar=dict(title='Изменение уровня, см',tickvals=[-100,0,100],
                            ticktext=['Рост ≥100','Без изменения','Снижение ≥100'],orientation='h',thickness=10,y=-.06)),
                hovertext=[f"{manifest['station']['properties']['name']}<br>Измерено: {current:.0f} см"
                           f"<br>М1 на {selected.date:%d.%m.%Y}: {selected.predicted_cm:.0f} см"],hoverinfo='text'))
            fig.update_layout(map=dict(style='carto-voyager',center=dict(lat=gauge_lat,lon=gauge_lon),zoom=10),
                              height=500,margin=dict(l=0,r=0,t=0,b=65),showlegend=False)
            st.plotly_chart(fig,width='stretch',config={'scrollZoom':True,'displaylogo':False})
            st.caption('Красный — прогноз роста уровня, зелёный — снижения. Это не вероятность безопасности дороги. '
                       'Координаты взяты из метаданных гидропоста, а не из ячейки GloFAS. © OpenStreetMap contributors.')
    with tabs[1]:
        if tabs[1].open is not False:
            known=series.loc[:pd.Timestamp(issue)].tail(45)
            fig=go.Figure()
            fig.add_scatter(x=known.index,y=known,name='Измерено до решения',line=dict(color='#64748b'))
            fig.add_scatter(x=forecast.date,y=forecast.upper_cm,line=dict(width=0),showlegend=False,hoverinfo='skip')
            fig.add_scatter(x=forecast.date,y=forecast.lower_cm,line=dict(width=0),fill='tonexty',name='Оценочный диапазон 80%')
            fig.add_scatter(x=forecast.date,y=forecast.predicted_cm,name='М1',line=dict(color='#2563eb',width=3))
            fig.add_scatter(x=forecast.date,y=forecast.persistence_cm,name='Сохранить последнее',line=dict(dash='dot'))
            if st.checkbox('Показать последующие измерения для проверки',key='m1_show_future'):
                future=series.reindex(pd.DatetimeIndex(forecast.date))
                fig.add_scatter(x=future.index,y=future,name='Последующий факт — не вход модели',line=dict(color='#16a34a',dash='dash'))
            fig.update_layout(yaxis_title='Уровень над нулём поста, см',xaxis=dict(tickformat='%d.%m'),height=400)
            st.plotly_chart(fig,width='stretch')
            view=forecast.rename(columns={'date':'Дата','lead':'Горизонт, сут','predicted_cm':'М1, см',
                  'lower_cm':'Нижняя граница, см','upper_cm':'Верхняя граница, см','persistence_cm':'Последнее измерение, см',
                  'calendar_cm':'Календарь, см','last_year_cm':'Прошлый год, см','method':'Выбранный метод'}).copy()
            view['Выбранный метод']=view['Выбранный метод'].map(NAMES)
            st.dataframe(view,hide_index=True,width='stretch')
            st.download_button('Скачать прогноз М1',view.to_csv(index=False).encode('utf-8-sig'),'flood-m1.csv','text/csv')
    with tabs[2]:
        if tabs[2].open is not False:
            report=M.ROOT/'reports/flood_m1.md'
            if report.exists(): st.markdown(report.read_text(encoding='utf-8'))
            scores=pd.read_csv(M.ROOT/'reports/flood_m1_scores.csv')
            st.dataframe(scores[scores.pid.eq(pid)],hide_index=True,width='stretch')
            st.write('MAE — средняя абсолютная ошибка в сантиметрах. Сравниваются целые будущие годы, '
                     'которые не использовались ни при обучении, ни при выборе алгоритма и калибровке интервала.')
            st.download_button('Скачать проверку по годам',scores.to_csv(index=False).encode('utf-8-sig'),'flood-m1-scores.csv','text/csv')
    with tabs[3]:
        if tabs[3].open is not False:
            st.markdown('Источник: [Уровень-Спутник](https://level-satellite.ru), '
                        'Головлев П.П., Иннокентьев А.И. (2026). Записи gmvo опубликованы как данные АИС ГМВО; '
                        'vodinfo — отдельный источник. Источники в обучающем ряду одной точки не смешиваются.')
            st.write(f"Гидропост {M.STATIONS[pid][0]}, источник {M.STATIONS[pid][1]}; "
                     f"наблюдений: {valid.size}; пропусков календарных суток: {series.isna().sum()}. "
                     'Пропуски не интерполируются. Отрицательные уровни сохраняются.')
            st.write('Признаки: текущий уровень, лаги 1/3/7/14/30 суток, скорость изменения, средние за 7/30 дней, '
                     'сезонность и горизонт. Цель — изменение уровня. Погода, спутниковые снимки и GloFAS '
                     'в эту первую собственную модель не входят. Обучение — CatBoost; алгоритм выбирается по прошлому году.')
            st.write('Диапазон — по ошибкам предыдущего года. Его фактическое покрытие указано в проверке. '
                     'Рельеф, площадь затопления, опасные отметки и допуск дорог не прогнозируются. '
                     'Архив получен позднее: исторический расчёт не равен сохранённому оперативному выпуску.')
            st.json(manifest['summary'].get('zero_marks',[]))
            st.download_button('Метаданные и источник',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'),
                               f'{pid}-source.json','application/json')
            protocol=M.ROOT/'docs/FLOOD_M1_VALIDATION_PLAN.md'
            st.download_button('Протокол проверки М1',protocol.read_bytes(),protocol.name,'text/markdown')
