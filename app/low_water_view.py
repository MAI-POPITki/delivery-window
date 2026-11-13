"""One-region low-water beta, explicit archive replay and navigation limits."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno import low_water as L


def render_low_water():
    st.title('Маловодье · бета')
    st.write('Республика Саха (Якутия) · Лена у Якутска · гидропост 3045')
    levels = L.M.load_levels(L.PID)
    metrics_path = L.M.ROOT / 'reports/low_water_metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else None
    st.info(f'Архивная проверка, не прогноз на сегодня. Последний замер модельного ряда: '
            f'{levels.last_valid_index():%d.%m.%Y}. Горизонт — 1–7 суток летом.')
    st.caption('Оцениваем низкий уровень на выбранную дату. Он не равен глубине фарватера '
               'и сам по себе не определяет допустимую осадку судна.')
    years = [y for y in range(2019, levels.last_valid_index().year+1)
             if (L.M.MODELS/f'{L.PID}_{y}.json').exists()]
    if not years:
        st.warning('Нет подготовленного архивного пакета для беты.')
        return
    demo = metrics['demo'] if metrics else dict(issue=f'{years[-1]}-08-20', lead=3)
    with st.sidebar:
        st.subheader('Низкая вода · Лена')
        year = st.selectbox('Архивный сезон', years, index=len(years)-1, key='low_water_year')
        default = pd.Timestamp(demo['issue']).replace(year=int(year)).date()
        issue = st.date_input('Дата решения', default, min_value=pd.Timestamp(year,6,1).date(),
            max_value=pd.Timestamp(year,9,29).date(), format='DD.MM.YYYY', key=f'low_water_issue_{year}')
        max_lead = min(7, (pd.Timestamp(year,9,30)-pd.Timestamp(issue)).days)
        lead = st.selectbox('Горизонт, суток', list(range(1,max_lead+1)),
            index=min(int(demo['lead']),max_lead)-1, key=f'low_water_lead_{year}_{max_lead}')
        custom = st.checkbox('Использовать отметку оператора', key='low_water_custom')
        limit = st.number_input('Отметка уровня, см относительно нуля поста', value=0.,
            step=1., key='low_water_limit') if custom else None
    try:
        forecast, meta = L.forecast(issue, limit)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return
    row = forecast[forecast.lead.eq(lead)].iloc[0]
    a,b,c = st.columns(3)
    a.metric(f'Уровень на {row.target_date:%d.%m.%Y}', f'{row.predicted_cm:.1f} см')
    b.metric('Вероятность уровня не выше отметки', f'{row.probability:.0%}')
    c.metric('Выбранная отметка', f"{meta['threshold_cm']:.1f} см")
    st.write(f'Диапазон уровня: **{row.lower_cm:.1f}…{row.upper_cm:.1f} см**. '
             f"Исходный замер: **{meta['latest_input_cm']:.0f} см** на {issue:%d.%m.%Y}.")
    if custom:
        st.caption('Отметка введена оператором. Привязку к участку, источник и соответствие нулю поста '
                   'нужно подтвердить отдельно. Метрики ниже относятся к статистическому порогу.')
    else:
        st.caption('Отметка q20: нижние 20% летних уровней ранних учебных лет. Это исследовательский '
                   'сигнал необычно низкой воды, а не официальный порог ограничения судоходства.')
    if row.probability >= .5:
        st.warning('Сигнал для проверки условий отправки: уточните у оператора глубины на маршруте, '
                   'осадку и ограничения загрузки. Автоматическое сокращение загрузки не рассчитывается.')
    else:
        st.write('По выбранной отметке сигнал ниже 50%. Это не подтверждает достаточную глубину на маршруте.')
    plot, check, sources = st.tabs(['Уровень и даты', 'Проверка качества', 'Источники и метод'])
    with plot:
        fig = go.Figure()
        history = levels.loc[pd.Timestamp(issue)-pd.Timedelta(days=20):pd.Timestamp(issue)]
        fig.add_scatter(x=history.index,y=history.values,name='Замеры до решения',line=dict(color='#54718e'))
        fig.add_scatter(x=forecast.target_date,y=forecast.upper_cm,line=dict(width=0),showlegend=False)
        fig.add_scatter(x=forecast.target_date,y=forecast.lower_cm,fill='tonexty',
            fillcolor='rgba(40,140,210,.16)',line=dict(width=0),name='Номинальный интервал 80%')
        fig.add_scatter(x=forecast.target_date,y=forecast.predicted_cm,name='Прогноз уровня',line=dict(color='#2487c8'))
        fig.add_hline(y=meta['threshold_cm'],line_dash='dash',annotation_text='Выбранная отметка')
        if st.checkbox('Показать последующий факт · только для проверки', key='low_water_show_fact'):
            actual = levels.reindex(pd.DatetimeIndex(forecast.target_date))
            fig.add_scatter(x=actual.index,y=actual.values,name='Последующий факт',line=dict(color='#e39243'))
            st.caption('Последующий факт раскрыт после расчёта и не используется в признаках прогноза.')
        fig.update_layout(height=400,yaxis_title='Уровень, см относительно нуля поста',xaxis_title='Дата',
            legend=dict(orientation='h'))
        st.plotly_chart(fig,width='stretch')
        display = forecast[['target_date','lead','predicted_cm','lower_cm','upper_cm','probability','method']].rename(columns={
            'target_date':'Дата','lead':'Горизонт, сут','predicted_cm':'Уровень, см','lower_cm':'Нижняя, см',
            'upper_cm':'Верхняя, см','probability':'Вероятность ≤ отметки','method':'Метод'})
        st.dataframe(display,hide_index=True,width='stretch',column_config={
            'Вероятность ≤ отметки':st.column_config.NumberColumn(format='percent')})
        export = forecast.copy();export['threshold_cm']=meta['threshold_cm'];export['gauge_code']='3045'
        export['calibration_year']=meta['calibration_year'];export['mode']='archive_low_water_beta'
        st.download_button('Скачать прогноз и основание', export.to_csv(index=False).encode('utf-8-sig'),
            'low-water-lena.csv','text/csv')
    with check:
        if metrics is None:
            st.info('Отчёт проверки пока отсутствует.')
        else:
            score=metrics['scores'][str(lead)]
            st.write(f"**Лена у Якутска, июнь–сентябрь 2019–2023, {lead} сут.** "
                     f"{score['n']} прогнозов, {score['events']} дней ниже или на отметке q20.")
            st.dataframe(pd.DataFrame([
                {'Показатель':'MAE уровня, см','Бета':score['mae_cm'],'Сохранение уровня':score['persistence_mae_cm'],'Календарь':score['calendar_mae_cm']},
                {'Показатель':'Brier вероятности · меньше лучше','Бета':score['brier'],'Сохранение уровня':score['persistence_brier'],'Календарь':score['calendar_brier']}
            ]),hide_index=True,width='stretch')
            st.write(f"При предупреждении от 50%: обнаружено **{score['tp']}** дней, пропущено **{score['fn']}**, "
                     f"ложных предупреждений **{score['fp']}**. Фактическое покрытие номинального "
                     f"80%-го интервала: **{score['coverage80']:.1%}**.")
            st.caption('Соседние дни зависимы; это пять сезонов, а не сотни независимых событий. '
                       'Оценочный год исключён из обучения и калибровки. Архив ранее изучался в проекте. '
                       'На отдельных сезонах вероятность хуже базового метода; новая независимая проверка впереди.')
        report=L.M.ROOT/'reports/low_water_beta.md'
        if report.exists():st.download_button('Полный отчёт по сезонам',report.read_bytes(),report.name,'text/markdown')
    with sources:
        st.markdown('[Архив наземных замеров «Уровень-Спутник»](https://level-satellite.ru/) · '
                    'АИС ГМВО, пост 3045. Отрицательные уровни допустимы относительно нуля поста.')
        st.write(f"Версия: {meta['version']}. Обучение CatBoost до {meta['train_until']}; "
                 f"выбор метода и ошибки калибровки — лето {meta['calibration_year']} года.")
        st.write('Для каждого горизонта на предыдущем лете выбирается лучший из CatBoost, сохранения уровня, '
                 'календаря и трендов за 7/14 суток. Остатки задают диапазон и вероятность H ≤ отметки. '
                 'Погода и свежие спутниковые снимки в этот прогноз не входят.')
        st.write('Для оперативного пилота нужны свежие 31 ежедневное измерение с тем же нулём поста '
                 'и пакет обучения/калибровки для текущего года. Для решений по судну — связь уровня '
                 'с глубиной конкретного участка и ограничения оператора.')
