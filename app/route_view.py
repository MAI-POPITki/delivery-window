"""Построение вариантов А — Б и прозрачная проверка условий доставки."""
import datetime as dt
import hashlib
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from okno.routing import search_places,fetch_routes,rank_routes


@st.cache_data(ttl=86400,max_entries=150,show_spinner=False)
def _places(query): return search_places(query)


@st.cache_data(ttl=3600,max_entries=150,show_spinner=False)
def _routes(points): return fetch_routes(points)


def render_routes():
    st.title('Маршрут и условия доставки')
    st.write('Постройте варианты по дорогам, сравните запас времени и исключите известные препятствия.')
    st.info('Рекомендация выбирается среди найденных вариантов по заданным ограничениям и сценариям. '
            'Вероятность недовоза не вычисляется: сведений обо всех закрытиях и пригодности дорог для грузовиков нет.')
    results_slot=st.container()
    points=[]
    for column,role,default in zip(st.columns(2),('А','Б'),('Иркутск','Братск')):
        with column:
            query=st.text_input('Населённый пункт '+role,default,key='route_query_'+role)
            if st.checkbox('Указать координаты '+role,key='route_manual_'+role):
                lat=st.number_input('Широта '+role,-90.,90.,52.2869 if role=='А' else 56.151,key='route_lat_'+role,format='%.6f')
                lon=st.number_input('Долгота '+role,-180.,180.,104.2807 if role=='А' else 101.614,key='route_lon_'+role,format='%.6f')
                points.append(dict(name=query,lat=lat,lon=lon))
                continue
            if st.button('Найти '+role,key='route_search_'+role):
                try: st.session_state['route_places_'+role]=(query,_places(query))
                except ValueError as e: st.error(str(e))
            saved=st.session_state.get('route_places_'+role)
            if saved and saved[0]==query and saved[1]:
                choice=st.selectbox('Выберите '+role,range(len(saved[1])),format_func=lambda i,places=saved[1]:places[i]['name'],key='route_place_'+role)
                points.append(saved[1][choice])
            elif saved and saved[0]==query: st.warning('Населённый пункт не найден.')
    st.caption('Поиск локальный: GeoNames, CC BY 4.0, https://www.geonames.org/ — 1108 населённых пунктов из cities15000. '
               'Для отсутствующего поселения укажите координаты. Координаты пути передаются FOSSGIS при построении. '
               '© участники OpenStreetMap, ODbL · https://www.openstreetmap.org/copyright')
    with st.expander('Промежуточные точки — проехать в указанном порядке'):
        via=st.data_editor(pd.DataFrame(columns=['Название','Широта','Долгота']),num_rows='dynamic',key='route_via',hide_index=True)
        st.caption('До четырёх промежуточных точек. Координаты позволяют уточнить путь без автоматического поиска адреса.')
    l,r=st.columns(2)
    date=l.date_input('Дата отправки',dt.date.today(),key='route_date')
    hour=l.time_input('Время отправки',dt.time(8),key='route_hour')
    due=r.date_input('Доставить не позднее даты',date+dt.timedelta(days=2),key='route_due')
    due_hour=r.time_input('Время окончания доставки',dt.time(20),key='route_due_hour')
    gross=l.number_input('Полная масса машины с грузом, т',0.1,200.,18.,key='route_gross')
    correction=r.number_input('Поправка времени под машину и остановки',1.,5.,1.3,0.1,key='route_correction')
    stress=st.slider('Проверить дополнительное увеличение времени пути, %',0,100,50,key='route_stress')
    st.caption('Время OSRM ориентировочное, для автомобильного профиля. Поправка и увеличение времени — '
               'ваши сценарные допущения, не статистическая вероятность. Погрузку и разгрузку включите в запас до срока.')
    with st.expander('Известные ограничения: загрузить CSV с источниками'):
        st.caption('Зоны задаются прямоугольниками. Пересечение проверяется для всей линии маршрута; '
                   'ограничения консервативно учитываются при пересечении периода отправка — срок доставки. '
                   'Гидропосты автоматически не привязываются к дорогам.')
        template=pd.DataFrame(columns=['name','min_lon','min_lat','max_lon','max_lat','kind','value','start','end','published','source'])
        st.download_button('Шаблон ограничений',template.to_csv(index=False),'route-restrictions.csv',key='route_template')
        st.caption('kind: closed — закрыто; mass — предел полной массы в тоннах; delay — задержка в часах; unknown — неизвестно. '
                   'Даты: ГГГГ-ММ-ДД ЧЧ:ММ; source — ссылка или документ. Зона и источник проверяются пользователем.')
        uploaded=st.file_uploader('Ограничения CSV',type='csv',key='route_restrictions')
    restrictions=[]
    try:
        if uploaded:
            frame=pd.read_csv(uploaded)
            if not set(template.columns)<=set(frame): raise ValueError('В CSV отсутствуют столбцы шаблона.')
            if len(frame)>200 or frame.isna().any().any(): raise ValueError('До 200 ограничений, без пустых полей.')
            restrictions=frame.to_dict('records')
        if len(points)==2:
            points=[points[0]]+[dict(name=str(x['Название']),lat=float(x['Широта']),lon=float(x['Долгота'])) for _,x in via.iterrows()]+[points[1]]
        departure=dt.datetime.combine(date,hour); deadline=dt.datetime.combine(due,due_hour)
        if deadline<=departure: raise ValueError('Срок доставки должен быть позже отправки.')
    except (ValueError,TypeError,KeyError) as e:
        st.error(str(e)); return
    signature=json.dumps(dict(points=points,departure=str(departure),deadline=str(deadline),gross=gross,
                              correction=correction,stress=stress,restrictions=restrictions),sort_keys=True,ensure_ascii=False)
    if st.button('Построить и сравнить маршруты',type='primary',key='route_build'):
        if len(points)<2: st.warning('Найдите и выберите оба населённых пункта.'); return
        with st.spinner('Ищу дорожные варианты туда и обратно…'):
            try:
                routes=_routes(points)
                reverse=_routes(list(reversed(points)))[0]
                ranked=rank_routes(routes,departure=departure,deadline=deadline,gross_t=gross,
                    restrictions=restrictions,factors=(1.,1+stress/200,1+stress/100),time_factor=correction)
                st.session_state['route_result']=(signature,ranked,reverse)
            except (ValueError,TypeError) as e: st.error('Проверьте даты без часовых поясов и данные ограничений: '+str(e)); return
    saved=st.session_state.get('route_result')
    if not saved or saved[0]!=signature:
        if saved: st.info('Условия изменены: пересчитайте маршрут. Старый результат скрыт.')
        return
    ranked,reverse=saved[1:]
    with results_slot:
        st.subheader('Сравнение найденных вариантов')
        st.warning('Все варианты имеют непроверенные участки: загруженный список ограничений не является полным реестром. '
                   'Отсутствие отметок не подтверждает открытие зимников, работу парома или допуск грузовика.')
        table=pd.DataFrame([{'Вариант':x['route']['id'],'Км':round(x['route']['distance_km'],1),
            'Базовое время, ч':round(x['assessment']['base_h'],1),'При замедлении, ч':round(x['assessment']['stress_h'],1),
            'Запас до срока, ч':round(x['assessment']['slack_h'],1),
            'Не уложился в сценариях':f"{x['assessment']['late_scenarios']}/{x['assessment']['scenarios']}",
            'Блокирующие ограничения':len(x['assessment']['blocked']),'Непроверенные отмеченные зоны':len(x['assessment']['unknown']),
            'Паромные шаги OSM':x['route']['ferries']} for x in ranked])
        st.dataframe(table,hide_index=True,width='stretch')
        eligible=[x for x in ranked if not x['assessment']['blocked']]
        if not eligible:
            st.error('Все найденные варианты пересекают заданное закрытие или превышают предел массы. Передача в планировщик заблокирована.')
        else:
            st.write(f"Предпочтительный по заданным условиям: **вариант {eligible[0]['route']['id']}**. "
                     'Сначала меньше отмеченных неизвестных зон, затем меньше опозданий в сценариях и больше запас времени.')
            if eligible[0]['assessment']['late_scenarios']:
                st.warning('Даже предпочтительный вариант не укладывается в срок в части заданных сценариев. Увеличьте запас времени или измените путь.')
        if len(ranked)==1:
            st.caption('Сервис нашёл только один вариант. Чтобы исследовать другой путь, задайте промежуточные точки.')
        selected=st.selectbox('Вариант для просмотра',range(len(ranked)),format_func=lambda i:f"Вариант {ranked[i]['route']['id']}",key='route_selected')
        item=ranked[selected]; route=item['route']; assessment=item['assessment']
        fig=go.Figure()
        for x in ranked:
            coords=x['route']['coordinates']; active=x['route']['id']==route['id']
            fig.add_trace(go.Scattermap(lon=[p[0] for p in coords],lat=[p[1] for p in coords],mode='lines',
                line=dict(width=5 if active else 2,color='#28768e' if active else '#a9b6bf'),name=f"Вариант {x['route']['id']}"))
        fig.add_trace(go.Scattermap(lon=[p['lon'] for p in points],lat=[p['lat'] for p in points],mode='markers',
            marker=dict(size=12,color='#243d53'),text=[p['name'] for p in points],name='Ваши точки'))
        fig.update_layout(map=dict(style='carto-positron',center=dict(lat=sum(p['lat'] for p in points)/len(points),
            lon=sum(p['lon'] for p in points)/len(points)),zoom=5),height=450,margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig,width='stretch',key='route_map')
        if assessment['blocked']: st.error('Препятствия: '+', '.join(assessment['blocked']))
        if assessment['unknown']: st.warning('; '.join(assessment['unknown']))
        if assessment['matched']: st.dataframe(pd.DataFrame(assessment['matched']),hide_index=True)
        st.caption('Обратный путь посчитан отдельным запросом и может отличаться. Его ограничения и время возврата проверяются отдельно оператором.')
        st.download_button('Скачать маршрут и расчёт условий',json.dumps(dict(points=points,outward=route,backward=reverse,
            assessment=assessment,inputs=json.loads(signature)),ensure_ascii=False,indent=2),'route.json','application/json',key='route_export')
        if st.button('Передать расстояния и время в завоз / обратный груз',key='route_apply',disabled=bool(assessment['blocked'])):
            st.session_state.update(bh_distance=route['distance_km'],bh_travel=route['travel_h']*correction+assessment['delay_h'],
                bh_return_distance=reverse['distance_km'],bh_return_travel=reverse['travel_h']*correction,
                bh_a=points[0]['name'],bh_b=points[-1]['name'],
                bh_route_import_note='Дороги OSM / OSRM, получено '+route['fetched_at']+'. Окно и допуск массы нужно подтвердить отдельно.')
            st.query_params['module']='backhaul'; st.rerun()
        st.caption('OSRM / FOSSGIS · © участники OpenStreetMap · https://www.openstreetmap.org/copyright · '
                   'Исправить карту: https://www.openstreetmap.org/ · Данные дороги текущие, исторический граф не восстанавливается.')
