"""Условный план доставки по собственному прогнозу М1; без вымышленных тарифов."""
import hashlib
import pandas as pd
import streamlit as st
from okno.flood_m3 import plan_variants


def render_m3(pid,decision,forecast):
    for name in list(st.session_state):
        if name.startswith('own_m3_') and not name.startswith('own_m3_calendar_') and name!='own_m3_run':
            st.session_state[name]=st.session_state[name]
    if st.session_state.get('_own_m3_post') not in (None,pid):
        st.session_state['own_m3_threshold']=None
        st.session_state['own_m3_evidence']=''
    st.session_state['_own_m3_post']=pid
    st.session_state.setdefault('own_m3_route','')
    st.session_state.setdefault('own_m3_evidence','')
    st.session_state.setdefault('own_m3_travel',1)
    st.subheader('3. Рассчитайте доставку')
    st.write('Минимизирует перевозку и хранение по подтверждённым датам и мощностям. '
             'Основной вариант требует, чтобы верхняя граница М1 оставалась ниже вашего порога весь рейс.')
    st.warning('Это условная политика предосторожности, а не вероятность закрытия дороги. '
               '80%-й диапазон уровня не даёт 80%-й гарантии проезда. М2 не закрывает дорогу автоматически.')
    route=st.text_input('Маршрут / участок',value=None,key='own_m3_route') or ''
    a,b=st.columns(2)
    threshold=a.number_input('Порог уровня этого гидропоста, см',value=None,key='own_m3_threshold')
    evidence=b.text_input('Основание порога и связь с участком',value=None,key='own_m3_evidence') or ''
    travel=a.number_input('Длительность проезда, суток',min_value=1,max_value=7,value=None,key='own_m3_travel')
    quantity=b.number_input('Объём груза, т',min_value=0.,value=None,key='own_m3_quantity')
    road=a.number_input('Перевозка по участку, ₽/т',min_value=0.,value=None,key='own_m3_road')
    backup=b.number_input('Оценка довоза остатка, ₽/т',min_value=0.,value=None,key='own_m3_backup')
    holding=a.number_input('Хранение, ₽/т·сут',min_value=0.,value=None,key='own_m3_holding')
    st.caption('Тарифы и порог не подставляются. Порог — см над нулём выбранного поста, не глубина на дороге. '
               'Остаток для другого способа не считается доставленным.')
    context=hashlib.sha256(f'{pid}|{decision}|{route}|{len(forecast)}'.encode()).hexdigest()[:12]
    records=pd.DataFrame(dict(date=pd.to_datetime(forecast.date).dt.date,confirmed=False,
                              capacity_t=[None]*len(forecast),source='',published=[None]*len(forecast)))
    saved=st.session_state.setdefault('_own_m3_confirmations',{})
    records=saved.get(context,records)
    edited=st.data_editor(records,key='own_m3_calendar_'+context,hide_index=True,disabled=['date'],
        column_config={'date':st.column_config.DateColumn('Дата',format='DD.MM.YYYY'),
            'confirmed':st.column_config.CheckboxColumn('Допуск подтверждён'),
            'capacity_t':st.column_config.NumberColumn('Мощность отправки, т',min_value=0.),
            'source':st.column_config.TextColumn('Основание допуска'),
            'published':st.column_config.DateColumn('Сведения получены',format='DD.MM.YYYY')})
    saved[context]=edited.copy()
    from app.handling_wind_view import render_wind,CAPACITY_LABELS
    wind=render_wind('own_m3_wind',decision,list(edited.date),
                     list(pd.to_datetime(edited.date).add(pd.Timedelta(days=int(travel)-1)).dt.date))
    if not st.button('Рассчитать план доставки',key='own_m3_run',type='primary'):return
    if not route.strip() or not evidence.strip() or any(x is None for x in [threshold,quantity,road,backup,holding]):
        st.info('Введите маршрут, порог с основанием, объём и тарифы. Неизвестные условия не заменяются примерами.');return
    try:
        variants=plan_variants(edited,forecast,decision,int(travel),threshold,quantity,road,backup,holding,wind=wind)
    except ValueError as exc:
        st.error(str(exc));return
    plan,summary=variants['upper']
    st.dataframe(pd.DataFrame([{'Вариант':label,'Расходы, ₽':variants[name][1]['cost'],
        'По участку, т':variants[name][1]['road_t'],'Остаток, т':variants[name][1]['backup_t']}
        for name,label in [('lower','Нижняя граница'),('median','Медиана'),('upper','Верхняя граница · основной')]]),hide_index=True)
    st.caption('Три варианта — чувствительность к прогнозу, не три независимых вероятностных сценария. '
               'Тарифы и подтверждения одинаковы во всех вариантах.')
    st.bar_chart(plan.set_index('date')[['optimized_t','uniform_t']].rename(
        columns={'optimized_t':'М3, т','uniform_t':'Равномерный план, т'}))
    st.write(f"Основной план: {summary['cost']:.0f} ₽; равномерный на тех же разрешённых мощностях: "
             f"{summary['baseline_cost']:.0f} ₽. Фактическая экономия на реальных рейсах не подтверждена.")
    output=plan.drop(columns=['fraction']).rename(columns={'date':'Отправка','arrival':'Окончание проезда',
        'capacity_t':'Мощность, т','confirmed':'Допуск подтверждён','policy_allowed':'Проходит политику',
        'max_level_cm':'Максимум верхней границы, см','optimized_t':'М3, т','uniform_t':'Равномерно, т'})
    output=output.rename(columns=CAPACITY_LABELS)
    st.dataframe(output[[c for c in output if not c.startswith(('wind_load_','wind_unload_'))]],hide_index=True)
    st.download_button('Скачать план М3',output.to_csv(index=False).encode('utf-8-sig'),'flood-m3.csv','text/csv')
