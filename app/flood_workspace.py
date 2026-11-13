"""Проверка даты рейса, ввод подтверждений и оптимизация паводкового плана."""
from datetime import timedelta
import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno.flood import ROOT, SOURCE, DATA, read_response
from okno.flood_planning import passage_risks, confirmed_calendar, optimize_dispatch


def risk_table(d, travel):
    table = passage_risks(d, travel)
    return table.rename(columns={'date':'Дата отправки','arrival':'Окончание проезда',
                  'fraction':'Сценарии высокой воды','complete':'Полный прогноз на рейс'})


def render_risks(d, route, decision, travel, target=None):
    st.subheader('Риски и рекомендации')
    st.write(f"**{route or 'Путь не выбран'}** · решение на {decision:%d.%m.%Y} · проезд {travel} сут.")
    st.warning('Текущий официальный допуск для этого участка в базе не подтверждён. '
               'Геометрия дороги и прогноз речного расхода не дают разрешения на проезд.')
    if d is None:
        st.info('На эту дату нет сохранённого гидрологического прогноза. Уточните обстановку и действующие ограничения у владельца дороги.')
        return
    risks = passage_risks(d, travel)
    passage = target or st.date_input('Планируемая дата проезда', value=risks.date.iloc[0], min_value=risks.date.iloc[0],
                            max_value=risks.date.iloc[-1], format='DD.MM.YYYY', key='flood_passage_'+str(decision))
    st.write(f'Оценка проезда с {passage:%d.%m.%Y}. Дата меняется в боковой панели.')
    current = risks.loc[risks.date == passage].iloc[0]
    if pd.isna(current.fraction):
        st.info('Прогноз не покрывает весь рейс. Уменьшите горизонт проезда или дождитесь нового выпуска.')
    else:
        st.metric('Сценарии высокой воды во время проезда', f'{current.fraction:.0%}')
        st.caption('Это доля гидрологических сценариев, а не вероятность закрытия выбранной дороги или аварии.')
        earlier = risks[(risks.date < passage) & risks.complete & (risks.fraction < current.fraction)]
        if len(earlier):
            best = earlier.sort_values(['fraction','date']).iloc[0]
            st.write(f"**Рассмотрите отправку {best.date:%d.%m.%Y}:** гидрологический индикатор "
                     f"{current.fraction:.0%} → {best.fraction:.0%}. Возможность проезда и переноса груза нужно подтвердить.")
        else:
            st.write('В доступном прогнозе нет более ранней даты с меньшей долей сценариев высокой воды.')
    st.write('Перед отправкой подтвердите у владельца пути допуск на всю длительность рейса, '
             'допустимую массу машины и наличие объезда. При росте воды пересчитайте план по новому выпуску.')
    table = risk_table(d, travel)
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'Сценарии высокой воды':st.column_config.NumberColumn(format='percent')})
    st.download_button('Скачать сравнение дат', table.to_csv(index=False).encode('utf-8-sig'),
                        'flood-date-risks.csv', 'text/csv')


def render_optimizer(d, route, pid, decision, travel, start_date=None):
    st.subheader('Планировщик доставки')
    st.write('Распределяет объём по датам с минимальной суммой перевозки, хранения и оценки довоза остатка. '
             'Сравнение — с равномерной загрузкой тех же доступных мощностей.')
    st.caption('Мощности, тарифы и подтверждения вводит оператор. Значения не подставляются из учебного примера. '
               'Остаток не считается доставленным: расписание другого способа перевозки здесь не задано.')
    if not route:
        st.info('Сначала выберите путь в боковой панели.')
        return
    context = hashlib.sha256(f'{pid}|{route}|{decision}|{travel}'.encode()).hexdigest()[:12]
    a,b = st.columns(2)
    quantity = a.number_input('Осталось доставить, т', min_value=0., value=None, key='flood_Q')
    road_cost = b.number_input('Перевозка по пути, ₽/т', min_value=0., value=None, key='flood_road_cost')
    backup = a.number_input('Оценка довоза остатка, ₽/т', min_value=0., value=None, key='flood_backup_cost')
    holding = b.number_input('Хранение, ₽/т в сутки', min_value=0., value=None, key='flood_holding')
    st.session_state.setdefault('flood_policy',100)
    limit = st.slider('Предельная доля сценариев высокой воды для отправки, %',0,100,value=None,key='flood_policy')
    st.caption('Это выбранная вами политика предосторожности. Превышение порога не объявляет дорогу закрытой.')
    allow_without = False
    if d is None:
        allow_without = st.checkbox('Рассчитать только по подтверждениям оператора, без прогноза воды', key='flood_without_forecast')
    days = pd.date_range(decision+timedelta(days=1), periods=7).date
    records = pd.DataFrame(dict(date=days, confirmed=False, capacity_t=[None]*7, source='', published=[None]*7))
    saved = st.session_state.setdefault('_flood_confirmations', {})
    records = saved.get(context, records)
    st.markdown('**Подтверждённые условия проезда**')
    st.caption('Для каждого разрешённого дня укажите мощность отправки, основание и дату получения сведений. '
               'Незаполненные дни исключаются. Для многодневного рейса допуск нужен на каждый день проезда.')
    edited = st.data_editor(records, hide_index=True, width='stretch', num_rows='fixed',
               key='flood_calendar_'+context, disabled=['date'], column_config={
               'date':st.column_config.DateColumn('Дата',format='DD.MM.YYYY'),
               'confirmed':st.column_config.CheckboxColumn('Проезд подтверждён'),
               'capacity_t':st.column_config.NumberColumn('Мощность отправки, т',min_value=0.),
               'source':st.column_config.TextColumn('Основание / источник'),
               'published':st.column_config.DateColumn('Сведения получены',format='DD.MM.YYYY')})
    saved[context] = edited.copy()
    from app.handling_wind_view import render_wind,CAPACITY_LABELS
    wind=render_wind('external_m3_wind',decision,list(days),
                     [(pd.Timestamp(day)+pd.Timedelta(days=travel-1)).date() for day in days],default_point=pid)
    if not st.button('Рассчитать план', key='flood_optimize', type='primary'):
        return
    if any(v is None for v in (quantity,road_cost,backup,holding)) or not quantity:
        st.info('Введите объём и свои тарифы — учебные числа автоматически не используются.')
        return
    try:
        calendar = confirmed_calendar(edited, passage_risks(d,travel) if d is not None else None,
                                      decision, travel, limit/100, allow_without)
        if start_date is not None:
            calendar.loc[calendar.date < start_date,'policy_allowed'] = False
        if wind is not None:
            from okno.handling_wind import constrain
            calendar=constrain(calendar,*wind)
        if not calendar.confirmed.any():
            st.info('Нет дней с подтверждённым допуском на весь рейс. Заполните основание и даты; неизвестный путь не считается открытым.')
            return
        if not calendar.policy_allowed.any():
            st.info('Нет дат, удовлетворяющих выбранной политике и полноте прогноза. Измените условия или дождитесь нового выпуска.')
            return
        plan, summary = optimize_dispatch(calendar,quantity,road_cost,backup,holding)
    except ValueError as exc:
        st.error(str(exc))
        return
    a,b,c=st.columns(3)
    a.metric('По выбранному пути', f"{summary['road_t']:,.1f} т".replace(',',' '))
    b.metric('Остаток для другого способа', f"{summary['backup_t']:,.1f} т".replace(',',' '))
    c.metric('Расчётные расходы', f"{summary['cost']:,.0f} ₽".replace(',',' '))
    st.write(f"Равномерный план: {summary['baseline_cost']:,.0f} ₽. "
             f"Разница расчётных расходов: {summary['cost']-summary['baseline_cost']:+,.0f} ₽.".replace(',',' '))
    fig = go.Figure()
    fig.add_bar(x=plan.date,y=plan.uniform_t,name='Равномерный план')
    fig.add_bar(x=plan.date,y=plan.optimized_t,name='Минимум расходов')
    fig.update_layout(barmode='group',yaxis_title='Отправка, т',xaxis=dict(tickformat='%d.%m'),height=350)
    st.plotly_chart(fig,width='stretch')
    output = plan.rename(columns={'date':'Дата отправки','arrival':'Окончание проезда','capacity_t':'Мощность, т',
               'confirmed':'Допуск подтверждён','policy_allowed':'Проходит политику','fraction':'Сценарии высокой воды',
               'optimized_t':'Оптимальный план, т','uniform_t':'Равномерный план, т'})
    output=output.rename(columns=CAPACITY_LABELS)
    st.dataframe(output[[c for c in output if not c.startswith(('wind_load_','wind_unload_'))]],hide_index=True,width='stretch')
    st.download_button('Скачать план',output.to_csv(index=False).encode('utf-8-sig'),'flood-plan.csv','text/csv')
    st.caption('Это условный расчёт по введённым данным. Он не подтверждает фактический допуск, '
               'свободную технику или доставку остатка другим способом; выигрыш прогноза на реальных рейсах не доказан.')


def render_data(pid, decision, d, routes, metadata):
    st.subheader('О данных')
    st.markdown(f'**Гидрология:** [GloFAS v4 через Open-Meteo]({SOURCE}), модельный расход, м³/с; не замеры уровня гидропоста.')
    if d is not None:
        st.write(f"На решение {decision:%d.%m.%Y} выбран выпуск, полученный {d['fetched_at']}. "
                 f"Полных сценариев: {d['members']}; полных исторических лет: {d['years']}.")
    else:
        st.info('Для выбранной даты сохранённого оперативного выпуска нет. Исторический ряд не подставляется вместо прогноза.')
    st.write('Исторический запрос: 1991–2020, в сохранённых ответах полны 1997–2020. '
             'Порог — q80 годовых максимумов. Доля сценариев считается за весь рейс. '
             'Записи позже даты решения исключаются; старые и неполные выпуски не используются.')
    if metadata:
        st.markdown(f"**Дороги:** [официальный API OpenStreetMap]({metadata['source_url']}), "
                    f"получено {metadata['fetched_at']}; участков: {len(routes)}. © OpenStreetMap contributors, ODbL.")
        st.caption('Это текущая геометрия дорог, не архив разрешений движения. Связь дорожного затопления с расходом реки не калибрована.')
        st.dataframe(pd.DataFrame([{'Участок':r['name'],'Тип':r['kind'],'Источник':r['source']} for r in routes]),
                     hide_index=True,width='stretch',column_config={'Источник':st.column_config.LinkColumn()})
    st.markdown('**Что ещё не доказано:** опасные уровни и границы затопления не рассчитываются. '
                'В диагностике 2024 года порог не был превышен у Орска, несмотря на известное наводнение. '
                'Официальные ограничения, осевые нагрузки и влияние заторов требуют отдельных данных.')
    for label,path in [('Как построен прогноз',ROOT/'docs/FLOOD_METHOD.md'),
                       ('План проверок',ROOT/'docs/FLOOD_VALIDATION_PLAN.md'),
                       ('Результаты проверки данных',ROOT/'reports/flood_validation.md'),
                       ('Проверки программы',ROOT/'reports/flood_implementation_checks.md')]:
        if path.exists():
            st.download_button(label,path.read_bytes(),path.name,'text/markdown')
