"""Даты закрытия/открытия как главный результат, с явными условиями расчёта."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from okno.flood_road_window import analog_paths, summarize
from okno.flood_m1 import ROOT


@st.cache_data(show_spinner=False)
def _paths(series, issue):
    return analog_paths(series, issue, nearest=False)


def render_road_window(pid, issue, series, short_forecast):
    st.subheader('Когда дорога может закрыться и открыться')
    st.write('Предварительный горизонт — 60 суток. Для конкретной дороги нужны уровни '
             'гидропоста, при которых проезд становится невозможным и снова возможным.')
    mode=st.radio('Основание расчёта дат',['Дорога с известным порогом','Пример по высокой воде · не дорожный прогноз'],
                  key='road_window_mode_'+pid,horizontal=True)
    example=mode.startswith('Пример')
    travel=1
    initial_closed=None
    if example:
        historical=series[(series.index.year<issue.year)&series.index.month.isin([4,5,6,7])].dropna()
        close_cm=reopen_cm=float(historical.quantile(.9))
        stable,recovery=3,0
        route='Сценарий высокой воды'
        evidence=f'q90 измерений апреля–июля до {issue.year} года'
        st.warning(f'Порог {close_cm:.0f} см — {evidence}. Это пример работы прогноза дат, '
                   'а не установленный порог закрытия какой-либо дороги.')
    else:
        with st.expander('Условия дороги для прогноза дат',expanded=True):
            route=st.text_input('Какая дорога / мост',key='road_window_route_'+pid)
            a,b=st.columns(2)
            close_cm=a.number_input('Закрытие при уровне ≥, см',value=None,key='road_window_close_'+pid)
            reopen_cm=b.number_input('Возобновление при уровне <, см',value=None,key='road_window_reopen_'+pid)
            evidence=st.text_input('Источник порогов и привязка к этому гидропосту',key='road_window_evidence_'+pid)
            state=st.selectbox('Состояние дороги на дату расчёта',['Неизвестно · оценить по уровню','Открыта','Закрыта'],key='road_window_state_'+pid)
            initial_closed={'Открыта':False,'Закрыта':True}.get(state)
            stable=a.number_input('Сколько суток уровень должен быть ниже',1,14,3,key='road_window_stable_'+pid)
            recovery=b.number_input('Дополнительно на восстановление, суток',0,30,0,key='road_window_recovery_'+pid)
            travel=a.number_input('Сколько суток занимает весь рейс',1,30,1,key='road_window_travel_'+pid)
            st.caption('3 суток устойчивого снижения и 0 суток восстановления — начальные настройки сценария, '
                       'не измеренные характеристики дороги. Укажите свои условия; пороги не подставляются.')
        if not route.strip() or not evidence.strip() or close_cm is None or reopen_cm is None:
            st.info('Укажите дорогу, два порога и их источник — даты появятся без ввода тарифов и заказа. '
                    'Чтобы сразу посмотреть механизм, выберите пример по высокой воде выше.')
            return
    try:
        paths,sources=_paths(series,pd.Timestamp(issue))
        result=summarize(paths,float(series.loc[pd.Timestamp(issue)]),close_cm,reopen_cm,stable,recovery,travel,initial_closed)
    except ValueError as exc:
        st.warning(str(exc));return
    st.markdown(f'**{route} · расчёт на {issue:%d.%m.%Y} · до {result["through"]:%d.%m.%Y}**')
    a,b=st.columns(2)
    for col,event,title in [(a,result['close'],'Начало высокой воды' if example else 'Ожидаемое закрытие'),
                            (b,result['reopen'],'Спад ниже порога' if example else 'Ожидаемое возобновление')]:
        col.markdown('**'+title+'**')
        if col is a and result['initial_closed']:
            col.write('Период ограничения уже начался по указанному состоянию / порогу. Дата начала не прогнозируется.')
            continue
        if event:
            if event['n']>result['scenarios']/2:
                col.write(f"Ориентир: **{event['median']:%d.%m.%Y}**; разброс **{event['early']:%d.%m}–{event['late']:%d.%m.%Y}**.")
            else:
                col.write(f"Единой даты нет. В отдельных сценариях: **{event['early']:%d.%m}–{event['late']:%d.%m.%Y}**.")
            col.caption(f"Событие найдено в {event['n']} из {result['scenarios']} аналогов. Даты рассчитаны только по этим сценариям.")
        else:
            col.write(f"Дата не определена до {result['through']:%d.%m.%Y}.")
    if result['already_above']:
        a.warning('На дату расчёта уровень уже выше порога. Это не подтверждение фактического закрытия дороги.')
    if result['censored']:
        b.warning(f"В {result['censored']} сценариях спад/выдержка не завершились за 60 суток; открытие может быть позже.")
    if not example:
        if result['last_departure']:
            st.info(f"Ориентир для последней отправки: **{result['last_departure']:%d.%m.%Y}**, "
                    f"чтобы весь рейс ({travel} сут.) закончился до самой ранней даты риска среди аналогов.")
        elif result['departure_expired']:
            st.warning('Запаса времени до самого раннего риска среди аналогов уже нет.')
        else:
            st.info('Закрытие не найдено в аналогах за 60 суток. Это не подтверждает доступность дороги.')
        st.caption('Это условная оценка при указанных порогах, а не приказ о закрытии/открытии. '
                   'Повреждения, локальные заторы и решения владельца дороги могут сдвинуть даты.')
    with st.expander('На чём основаны даты, график и сценарии'):
        st.write('Берём все полные прошлые сезоны этого гидропоста на той же календарной дате, минимум 5 разных лет. '
                 'Продолжения — реальные измерения тех сезонов с поправкой к текущему уровню, затухающей за 14 суток. '
                 'Календарный вариант выбран после сравнения с подбором похожих лет: тот оказался хуже. '
                 'Это предварительный сезонный ориентир до 60 дней; CatBoost М1 остаётся прогнозом на 1–7 дней.')
        st.write('Закрытие — первое достижение порога. Возобновление — после устойчивого нахождения ниже второго '
                 'порога и срока восстановления. Показан первый период ограничения; повторные подъёмы возможны. '
                 'Разброс — минимум и максимум дат среди аналогов с событием; это не 80% или 90% гарантии.')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=paths.index,y=paths.min(axis=1),line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=paths.index,y=paths.max(axis=1),fill='tonexty',fillcolor='rgba(148,163,184,.25)',
                                line=dict(width=0),name='Разброс аналогов'))
        fig.add_trace(go.Scatter(x=paths.index,y=paths.median(axis=1),name='Медиана аналогов',line=dict(color='#475569')))
        fig.add_trace(go.Scatter(x=short_forecast.date,y=short_forecast.predicted_cm,name='М1 · 1–7 суток',line=dict(color='#1a9850')))
        fig.add_hline(y=close_cm,line_color='#d73027',annotation_text='Порог закрытия / высокой воды')
        fig.add_hline(y=reopen_cm,line_color='#1a9850',line_dash='dot',annotation_text='Порог возобновления')
        fig.update_layout(height=330,yaxis_title='См над нулём выбранного поста',xaxis=dict(tickformat='%d.%m'))
        st.plotly_chart(fig,width='stretch',key='road_window_chart_'+pid)
        out=sources.merge(result['events'],left_on=sources.year.astype(str),right_on='scenario').drop(columns=['key_0'],errors='ignore')
        st.dataframe(out.rename(columns={'year':'Год-аналог','anchor':'Дата-аналог','score_cm':'Расхождение, см','scenario':'Сценарий',
            'source_end':'Последнее использованное наблюдение','close':'Начало риска','reopen':'Возобновление',
            'already_above':'Порог уже превышен','initial_closed':'Начальный период риска','censored':'Открытие за горизонтом'}),hide_index=True)
        st.caption('Ни один использованный сезон не относится к году расчёта или будущему. '
                   'Погода, будущие сбросы водохранилищ и состояние полотна здесь не прогнозируются. '
                   'Точность дат нужно оценивать отдельно от точности уровня М1.')
        st.download_button('Скачать сценарии дат',out.to_csv(index=False).encode('utf-8-sig'),f'{pid}-road-dates.csv','text/csv')
        report=ROOT/'reports/flood_road_dates.md'
        if report.exists():st.markdown(report.read_text(encoding='utf-8'))
