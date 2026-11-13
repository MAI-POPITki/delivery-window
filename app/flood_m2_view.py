from okno import map_style as _flood_map_style
"""Отдельная спутниковая карта открытой воды, без подмены прогноза М1."""
import base64
from datetime import date
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno import flood_m2 as M
from okno.flood_sites import SITES as POINTS
from okno.flood_routes import load_routes


def set_observation_date(key, value):
    st.session_state[key]=value


def follow_latest(state,key,latest,today):
    """Follow new scenes only while the user is looking at the latest scene."""
    current=min(latest,today)
    previous=state.get(key+'_latest_seen',current)
    if key not in state or (state[key]==previous and current>previous):
        state[key]=current
    state[key+'_latest_seen']=current


def render_m2():
    st.title('Паводки · М2: вода со спутника')
    st.caption('Sentinel-2 · наблюдение открытой воды и её расширения относительно более ранних снимков')
    regions=list(dict.fromkeys(p['region'] for p in POINTS.values()))
    region=st.sidebar.selectbox('Регион М2',regions,key='m2_flood_region')
    points=[p for p in POINTS if POINTS[p]['region']==region]
    if st.session_state.get('m2_flood_point') not in points:
        st.session_state['m2_flood_point']=points[0]
    pid=st.sidebar.selectbox('Участок М2',points,key='m2_flood_point',
        format_func=lambda p:f"{POINTS[p]['river']} · {POINTS[p]['name']}")
    from app.flood_live_view import render_live
    render_live(pid)
    runs=M.available_runs(pid)
    if not runs:
        st.info('Для этого участка снимки ещё не обработаны. Отсутствие снимка не означает отсутствие воды.')
        return
    try:
        metadata=[M.load_run(p) for p in runs]
        latest=max(pd.Timestamp(m['product_generated_at']).date() for m in metadata)
        earliest=min(pd.Timestamp(m['acquired_at']).date() for m in metadata)
        key='m2_as_of_'+pid
        follow_latest(st.session_state,key,latest,date.today())
        archive=[pd.Timestamp(m['product_generated_at']).date() for m in metadata if m['acquired_at'].startswith('2024-')]
        if archive:
            st.sidebar.button('Архивный снимок 2024',key='m2_example_'+pid,
                on_click=set_observation_date,args=(key,max(archive)))
        st.sidebar.button('Последний обработанный снимок',key='m2_latest_'+pid,
            on_click=set_observation_date,args=(key,min(latest,date.today())))
        st.sidebar.caption('Доступны обработанные снимки, выбранные по условиям обработки; '
                           'она не обязательно самая новая в каталоге.')
        as_of=st.sidebar.date_input('Дата просмотра М2',value=None,min_value=earliest,
            max_value=max(date.today(),latest),format='DD.MM.YYYY',key=key)
        result=M.select_run(pid,as_of)
    except (ValueError, OSError, KeyError) as exc:
        st.error('Не удалось проверить целостность снимков М2. Данные не показаны.')
        return
    if result is None:
        st.info('К выбранной дате нет подходящего обработанного продукта. Более поздний снимок не используется.')
        return
    path,meta=result
    folder=path.parent
    acquired=pd.Timestamp(meta['acquired_at'])
    age=(as_of-acquired.date()).days
    stats=meta['statistics']
    st.info('М2 показывает наблюдение на дату съёмки, а не прогноз. '
            'Архивные продукты получены позднее; это не восстановленный оперативный выпуск. '
            'Новая вода относительно летней базы — кандидат на затопление, требующий проверки причины.')
    st.write(f"**Съёмка: {acquired:%d.%m.%Y %H:%M} UTC** · возраст к выбранной дате: {age} сут. · "
             f"видно {stats['observed_fraction']:.0%} квадрата 12×12 км.")
    st.caption(f"Возраст снимка на сегодня: {(date.today()-acquired.date()).days} сут. "
               'Между съёмками дата просмотра может меняться, а наблюдение оставаться тем же: новой съёмки ещё нет.')
    if max(age,(date.today()-acquired.date()).days)>10:
        st.warning(f'Снимок устарел для оперативной оценки: на сегодня ему {(date.today()-acquired.date()).days} суток. '
                   'Он пригоден для исторического просмотра; текущее состояние неизвестно.')
    if stats['observed_fraction']<.5:
        st.warning('Больше половины территории не наблюдается уверенно. Облака, тени, снег и пропуски — неизвестное, а не сухая территория.')
    if stats.get('scl_water_agreement_iou') is not None and stats['scl_water_agreement_iou']<.5:
        st.warning('Водный индекс сильно расходится с классом воды поставщика. Часть воды может быть пропущена; '
                   'маленькая площадь не доказывает отсутствие воды или паводка. Спорные водные пиксели не считаются сушей.')
    a,b,c=st.columns(3)
    a.metric('Наблюдаемая открытая вода',f"{stats['water_km2']:.2f} км²" if stats['observed_fraction']>0 else 'Неизвестно')
    b.metric('Новая вода относительно базы',f"{stats['new_water_km2']:.2f} км²" if stats['observed_fraction']>0 else 'Неизвестно')
    c.metric('Сопоставимая территория',f"{stats['comparable_km2']:.1f} км²")
    st.caption(f"Площади только внутри вырезки и видимой территории, не по всему городу. "
               f"У воды на {stats['water_unknown_baseline_km2']:.2f} км² нет надёжной базы для сравнения. "
               'Пиксель 20 м; глубина воды не измеряется.')
    tabs=st.tabs(['Карта воды','Снимки до и после','Как работает М2','Источники'],key='m2_flood_tab',on_change='rerun')
    with tabs[0]:
        if tabs[0].open is not False:
            lon,lat=meta['station']['geometry']['coordinates']
            fig=go.Figure()
            roads,_=load_routes(pid)
            xs,ys=[],[]
            for road in roads:
                xs.extend([p[0] for p in road['coordinates']]+[None]);ys.extend([p[1] for p in road['coordinates']]+[None])
            if xs:
                fig.add_trace(go.Scattermap(lon=xs,lat=ys,mode='lines',line=dict(color='#475569',width=2),hoverinfo='skip'))
            fig.add_trace(go.Scattermap(lon=[lon],lat=[lat],mode='markers',marker=dict(size=10,color='#111827'),
                text=[meta['station']['properties']['name']],hoverinfo='text'))
            source='data:image/png;base64,'+base64.b64encode((folder/'overlay.png').read_bytes()).decode()
            fig.update_layout(map=dict(style=_flood_map_style.FLOOD_LOCAL_MAP_STYLE,center=dict(lon=lon,lat=lat),zoom=10,
                layers=[dict(sourcetype='image',source=source,coordinates=meta['overlay_coordinates'],below='traces',opacity=.85)]),
                height=550,margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
            st.plotly_chart(fig,width='stretch',config={'scrollZoom':True,'displaylogo':False})
            st.markdown('🔴 **Новая вода** — вода на месте суши летней базы. '
                        '🔵 **Остальная вода** — базовая или без определённой базы. '
                        '🟢 **Наблюдаемая суша**. ⚪ **Неизвестно** — облака, снег, тени или нет данных.')
            st.caption('Дороги — геометрия OpenStreetMap. Пересечение с красным слоем не доказывает непроходимость: '
                       'мосты, высота полотна и глубина не учтены. © Copernicus Sentinel / OpenStreetMap contributors.')
    with tabs[1]:
        if tabs[1].open is not False:
            left,right=st.columns(2)
            left.image(str(folder/'baseline.png'),caption='Летняя база: медиана видимых участков трёх более ранних дат')
            right.image(str(folder/'rgb.png'),caption=f'Реальный снимок {acquired:%d.%m.%Y}; облака в RGB видны, но из водной маски исключены')
            dates=', '.join(pd.Timestamp(r['acquired_at']).strftime('%d.%m.%Y') for r in meta['references'])
            st.write('Даты базы: '+dates+'. Чёрное в базе означает отсутствие подходящего наблюдения, а не воду.')
            st.download_button('Скачать маску и базу (NPZ)',(folder/'mask.npz').read_bytes(),f'{pid}-m2-mask.npz')
    with tabs[2]:
        if tabs[2].open is not False:
            st.markdown('1. Читаем реальные каналы Sentinel-2: зелёный, красный, ближний и коротковолновый ИК. '
                        'Переводим DN в отражательную способность по XML данного продукта.\n'
                        '2. Убираем облака, тени, снег, сомнительные пиксели и 40-метровую полосу вокруг них.\n'
                        '3. Открытая вода: **MNDWI > 0.1**, **NDVI < 0.3**, **NIR < 0.15**. '
                        'Это экспериментальное спектральное правило, не обученная модель затоплений.\n'
                        'Если SCL отмечает воду, а правило не подтверждает её, пиксель остаётся неизвестным.\n'
                        '4. Сравниваем с тремя снимками предыдущего лета: нужно минимум два пригодных наблюдения. '
                        'Красным отмечаем воду на месте уверенной суши базы.')
            st.warning('Точность отображаемых российских сцен ещё не проверена по совпадающим ручным контурам. '
                       'Независимый тест выделения воды Sen1Floods11 доступен во вкладке проверки общей оценки. '
                       'Мутная вода, мокрый грунт, растительность, снег и тени могут давать ошибки. '
                       'М2 пока не изменяет М1 и не разрешает проезд.')
            agreement=stats.get('scl_water_agreement_iou')
            if agreement is not None:
                st.caption(f'Диагностика: IoU с водным классом SCL на видимых пикселях — {agreement:.2f}. '
                           'Это согласие двух методов на одном снимке, не независимая оценка точности.')
    with tabs[3]:
        if tabs[3].open is not False:
            st.markdown('Источник: [Copernicus Sentinel-2 L2A через Microsoft Planetary Computer]'
                        '(https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a). '
                        'Исходные вырезки DN, STAC, XML и контрольные суммы сохранены в проекте.')
            rows=[]
            for label,scene in [('Наблюдение',meta['observation']),*[('База',r) for r in meta['references']]]:
                rows.append({'Роль':label,'Съёмка UTC':scene['acquired_at'],'Генерация продукта UTC':scene['generated_at'],
                             'Каталог':scene['stac_url']})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption('Время обработки архива: '+meta['processed_at']+'. Выбираются продукты не позже даты просмотра; '
                       'поздняя переработка снимка не считается доступной раньше её генерации.')
            st.download_button('Метаданные М2 и SHA-256',json.dumps(meta,ensure_ascii=False,indent=2).encode('utf-8'),
                               f'{pid}-m2-source.json','application/json')
