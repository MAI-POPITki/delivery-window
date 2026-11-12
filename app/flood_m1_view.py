from okno import map_style as _flood_map_style
"""Собственный М1 по измеренным уровням: исторический прогноз и независимая проверка."""
import json
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno import flood_m1 as M
from okno import flood_m1_quantile as Q
from okno.flood_sites import SITES as POINTS
from okno.flood_routes import load_routes
from okno.map_style import AVAILABILITY_SCALE

NAMES={'catboost':'CatBoost','persistence':'Последнее измерение','calendar':'Календарная медиана','last_year':'Как в прошлом году'}


def render_m1():
    # Сохраняем ввод даже на проходе, где вкладка М3 не отрисовывается.
    for name in list(st.session_state):
        if name.startswith('own_m3_') and not name.startswith('own_m3_calendar_') and name!='own_m3_run':
            st.session_state[name]=st.session_state[name]
    st.title('Паводки')
    st.caption('Выберите участок и дату — получите вероятность разлива или высокой воды на эти сутки')
    regions=list(dict.fromkeys(p['region'] for p in POINTS.values()))
    if st.session_state.get('m1_flood_region')=='Все четыре региона':
        st.session_state['m1_flood_region']='Все регионы'
    region=st.sidebar.selectbox('Регион',['Все регионы',*regions],index=0,key='m1_flood_region')
    points=[pid for pid,p in POINTS.items() if region=='Все регионы' or p['region']==region]
    if st.session_state.get('m1_flood_point') not in points:
        st.session_state['m1_flood_point']=points[0]
    pid=st.sidebar.selectbox('Участок реки',points,key='m1_flood_point',
                            format_func=lambda p:f"{POINTS[p]['river']} · {POINTS[p]['name']}")
    from app.flood_live_view import render_live
    render_live(pid)
    if pid not in M.STATIONS:
        from app.flood_overview import render_overview,open_satellite
        render_overview(st.session_state.get('flood_overview_date',date(2023,5,1)),7,pid)
        if pid=='tobol_kurgan':
            st.warning('Курган: М1 пока не обучена. В доступном архиве всего один полный год ГМВО; другие записи имеют большие разрывы.')
        else:
            st.info(f"{POINTS[pid]['name']}: подключён спутниковый мониторинг. М1 уровня ещё не обучена; план М3 по чужому прогнозу не строится.")
        st.markdown('Источники и покрытие: [Уровень-Спутник](https://level-satellite.ru). '
                    'Для выбранного участка доступны только явно перечисленные возможности.')
        st.button('Открыть спутниковую карту',on_click=open_satellite,args=(pid,),key='kurgan_satellite')
        return
    with st.sidebar.expander('Настройки модели'):
        version=st.radio('Версия М1',['v2 · квантили уровня','v1 · точечная модель'],key='m1_flood_version')
    engine=Q if version.startswith('v2') else M
    series=M.load_levels(pid)
    manifest=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
    valid=series.dropna()
    earliest=max(valid.index.min().date(),date(2019 if engine is Q else 2017,1,1))
    latest=valid.index.max().date()
    key=f'm1_issue_{pid}'
    st.session_state.setdefault(key,min(latest,date(latest.year,5,1)))
    if st.session_state[key]<earliest:
        st.session_state[key]=earliest
    issue=st.sidebar.date_input('Дата расчёта (архив)',value=None,min_value=earliest,max_value=latest,
                                format='DD.MM.YYYY',key=key)
    target_key='flood_target_date_'+pid
    lower=pd.Timestamp(issue)+pd.Timedelta(days=1)
    upper=pd.Timestamp(issue)+pd.Timedelta(days=7)
    if target_key not in st.session_state or not lower.date()<=st.session_state[target_key]<=upper.date():
        st.session_state[target_key]=upper.date()
    target=st.date_input('На какую дату нужен прогноз?',min_value=lower.date(),max_value=upper.date(),
                         value=None,format='DD.MM.YYYY',key=target_key)
    horizon=(target-issue).days
    st.session_state['m1_flood_horizon']=horizon
    st.caption('Вероятностный прогноз доступен на 1–7 суток после даты исходных измерений. '
               'Дату исходных измерений можно изменить слева для архивной проверки.')
    st.sidebar.caption(f"Наблюдения: {valid.index.min():%d.%m.%Y} — {valid.index.max():%d.%m.%Y}. "
                       f"Источник ряда: {M.STATIONS[pid][1]}. Данные за 2026 год отсутствуют.")
    st.info('Исторический расчёт по реальным измерениям. Модель использует только данные до выбранной даты. '
            'Свежего собственного прогноза на 2026 год нет: не получены текущие уровни гидропостов.')
    try:
        forecast,meta=engine.forecast(pid,issue)
    except ValueError as exc:
        st.warning(str(exc));return
    forecast=forecast.iloc[:horizon]
    selected=forecast.iloc[-1]
    from app.flood_day_view import render_day
    day_result=render_day(pid,issue,target)
    st.session_state['flood_overview_date']=issue
    from app.flood_overview import render_overview
    render_overview(issue,horizon,pid)
    with st.expander('Дополнительно: приблизительные даты ограничения дороги',expanded=False):
        from app.flood_road_window_view import render_road_window
        render_road_window(pid,issue,series,forecast)
    st.subheader('2. Оцените обстановку на выбранном участке')
    current=float(series.loc[pd.Timestamp(issue)])
    validation_path=engine.MODELS/'validation.json'
    validation=json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    check=validation.get(f'{pid}:{horizon}',{})
    a,b,c=st.columns(3)
    a.metric('Измерено на дату решения',f'{current:.0f} см')
    b.metric(f'Прогноз на {selected.date:%d.%m.%Y}',f'{selected.predicted_cm:.0f} см',
              delta=f'{selected.predicted_cm-current:+.0f} см')
    c.metric('Оценочный диапазон 80%',f'{selected.lower_cm:.0f} … {selected.upper_cm:.0f} см')
    st.caption('Уровень указан относительно нулевой отметки гидропоста. Это не глубина воды на дороге.')
    if engine is Q:
        with st.expander('Дополнительно: другие пороги высокой воды',expanded=False):
            kind=st.radio('Какое событие оцениваем',['Высокий уровень · q80','Очень высокий уровень · q90','Своя отметка'],index=1,key='own_probability_kind_'+pid)
            q=.8 if 'q80' in kind else .9
            historical=series[(series.index.year<issue.year-1)&series.index.month.isin([4,5,6,7])].dropna()
            threshold=(st.number_input('Проверяемый уровень, см над нулём поста',value=None,key='own_probability_threshold_'+pid)
                       if kind=='Своя отметка' else float(historical.quantile(q)))
            if kind!='Своя отметка':
                st.write(f'Порог события: **{threshold:.0f} см**, квантиль измерений апреля–июля учебных лет по {issue.year-2}. '
                         'Это относительная высокая вода, не официальная опасная отметка.')
            st.caption('Вы задаёте событие: уровень ≥ этой отметки. Это не вероятность затопления или закрытия дороги. '
                       'Опасная отметка автоматически не подставляется.')
            if threshold is not None:
                from okno.flood_probability import exceedance
                try:
                    probabilities=exceedance(pid,issue,forecast,threshold)
                    st.dataframe(probabilities.assign(probability=probabilities.probability*100).rename(
                        columns={'date':'Дата','lead':'Суток вперёд','probability':'Оценка превышения, %',
                                 'n':'Примеров ошибок','calibration_year':'Год ошибок'}),hide_index=True)
                    st.warning('Экспериментальная эмпирическая оценка по ошибкам предыдущего года. '
                               'Её точность зависит от режима реки; новая надстройка проверена повторно на уже изученном архиве. '
                               'Проценты не являются гарантией проезда.')
                    evidence_path=M.ROOT/'models/flood_probability/validation.json'
                    if evidence_path.exists():
                        evidence=json.loads(evidence_path.read_text()).get(f'{pid}:{horizon}:{q}',{})
                        if kind=='Своя отметка' or issue.month not in (4,5,6,7):
                            st.caption('Доказательная таблица относится к весенним q80/q90; для этого события отдельный критерий не проверен.')
                        elif evidence.get('skill'):
                            st.info('Для этого поста, горизонта и типа события Brier score лучше обоих простых правил '
                                    'с положительным 90%-интервалом выигрыша на повторной архивной оценке.')
                        else:
                            st.warning('Для этого события преимущество вероятностей над обоими простыми правилами не подтверждено.')
                    calibration_path=M.ROOT/'reports/flood_probability_reliability.csv'
                    if kind!='Своя отметка' and calibration_path.exists():
                        with st.expander('Насколько проценты совпали с фактами'):
                            reliability=pd.read_csv(calibration_path)
                            subset=reliability[(reliability.pid==pid)&(reliability.lead==horizon)&(reliability.threshold_quantile==q)]
                            if not subset.empty:
                                st.dataframe(subset[['predicted','observed','n']].assign(
                                    predicted=subset.predicted*100,observed=subset.observed*100).rename(
                                    columns={'predicted':'Средний прогноз, %','observed':'Фактические превышения, %','n':'Примеров'}),hide_index=True)
                                st.caption('Близкие проценты означают хорошую калибровку. Малые корзины ненадёжны; '
                                           'соседние дни зависимы. Это повторная архивная проверка, не гарантия для следующего паводка.')
                except (ValueError,FileNotFoundError,KeyError) as exc:
                    st.info('Для этой версии нет проверенных сохранённых ошибок; вероятность не показана.')
            report=M.ROOT/'reports/flood_probability.md'
            if report.exists():
                st.download_button('Проверка вероятностей: Brier score и соперники',report.read_bytes(),report.name)
    with st.expander('Надёжность прогноза и ограничения'):
        if check.get('point_skill_success'):
            if engine is Q:
                st.info('На повторной архивной оценке точечный прогноз прошёл критерии сравнения с простыми правилами. '
                        'Эти годы уже изучались при создании v1; независимое подтверждение v2 ещё требуется.')
            else:
                st.success('Точечный прогноз для этой точки и горизонта прошёл критерий сравнения с простыми правилами на отдельных будущих годах.')
        else:
            st.warning('Для этой точки/горизонта преимущество по всем критериям не подтверждено. Модель остаётся экспериментальной.')
        if not check.get('interval_valid'):
            coverage=check.get('spring_coverage')
            st.warning('Диапазон ещё ненадёжен в половодье'
                       +(f': в проверке апреля–июля факт попадал в него только в {coverage:.0%} случаев.' if coverage is not None else '.')
                       +' Это не гарантия 80%-й вероятности и не разрешение проезда.')
        elif engine is Q:
            st.caption(f"На повторной проверке весной покрытие диапазона — {check['spring_coverage']:.0%}. "
                       'Оценка по прошлым годам не гарантирует покрытие следующего паводка.')
        if engine is Q and check.get('worst_spring_coverage',1)<.75:
            st.warning(f"В отдельном паводке диапазон был ненадёжен: весна {check['worst_spring_year']} года — "
                       f"покрытие {check['worst_spring_coverage']:.0%}. Среднее по годам скрывает этот провал.")
    with st.expander('Как получен этот прогноз'):
        st.caption(f"Уровни — сантиметры над нулём этого гидропоста. Выбран по предыдущему году: {NAMES[selected.method]}. "
                   f"Обучение по {meta['train_until']}; калибровка — {meta['calibration_year']} год.")
        known=series.loc[:pd.Timestamp(issue)]
        features=engine.feature_frame(known,horizon).loc[pd.Timestamp(issue)]
        st.write(f"1. На {issue:%d.%m.%Y} измерено {current:.0f} см. "
                 f"За сутки уровень изменился на {features.d1:+.0f} см, за 7 суток — на {features.d7:+.0f} см.")
        st.write('2. Модель получает текущий уровень, прошлые измерения за 30 суток, изменения, средние, '
                 'сезон и число суток до целевой даты. Последующие измерения в расчёт не входят.')
        st.write(f"3. На горизонте {horizon} суток выбран метод «{NAMES[selected.method]}» по ошибке {meta['calibration_year']} года. "
                 f"Изменение от текущего уровня до прогноза: {selected.predicted_cm-current:+.0f} см.")
        comparison=pd.DataFrame([{'Метод':NAMES[name],'Ошибка на прошлом году, см':value}
                                for name,value in meta['calibration_mae'][str(horizon)].items()])
        st.dataframe(comparison,hide_index=True,width='stretch')
        if engine is Q:
            season='spring' if pd.Timestamp(issue).month in (4,5,6,7) else 'other'
            correction=meta['corrections'][str(horizon)][season]
            st.write(f"4. CatBoost оценивает квантили 10/50/90%. Диапазон выбранного метода расширен по ошибкам "
                     f"прошлого года на {correction['cm']:.1f} см с каждой стороны "
                     f"({correction['n']} примеров; {'весь год' if correction['fallback_all_year'] else 'апрель–июль' if season=='spring' else 'остальные месяцы'}). "
                     'Если выбрано простое правило, диапазон строится по его собственным ошибкам.')
        else:
            st.write('4. Диапазон — прогноз ± 80-й процентиль абсолютных ошибок выбранного метода в прошлом году.')
        st.write('Уровень относительно нуля поста не равен глубине воды на дороге. '
                 'Без подтверждённых опасных отметок и рельефа этот расчёт не определяет затопление или возможность проезда.')
    tabs=st.tabs(['Общий результат','Карта','М1: прогноз','Проверка','Данные','М3: план доставки'],key='m1_flood_tab',on_change='rerun')
    with tabs[0]:
        if tabs[0].open is not False:
            from app.flood_assessment_view import render_assessment
            render_assessment(pid,issue,forecast,day_result['threshold'] if day_result else threshold if engine is Q else None,engine is Q)
    with tabs[5]:
        if tabs[5].open is not False:
            from app.flood_m3_view import render_m3
            render_m3(pid,issue,forecast)
    with tabs[1]:
        if tabs[1].open is not False:
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
            fig.update_layout(map=dict(style=_flood_map_style.FLOOD_LOCAL_MAP_STYLE,center=dict(lat=gauge_lat,lon=gauge_lon),zoom=10),
                              height=500,margin=dict(l=0,r=0,t=0,b=65),showlegend=False)
            st.plotly_chart(fig,width='stretch',config={'scrollZoom':True,'displaylogo':False})
            st.caption('Красный — прогноз роста уровня, зелёный — снижения. Это не вероятность безопасности дороги. '
                       'Координаты взяты из метаданных гидропоста, а не из ячейки GloFAS. © OpenStreetMap contributors.')
    with tabs[2]:
        if tabs[2].open is not False:
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
    with tabs[3]:
        if tabs[3].open is not False:
            benchmark=M.ROOT/'reports/flood_m2_benchmark.md'
            if benchmark.exists():
                with st.expander('М2: независимая ручная разметка — полный тест и перенос на другое событие'):
                    st.markdown(benchmark.read_text(encoding='utf-8'))
            report=M.ROOT/('reports/flood_m1_quantile.md' if engine is Q else 'reports/flood_m1.md')
            if report.exists(): st.markdown(report.read_text(encoding='utf-8'))
            scores=pd.read_csv(M.ROOT/('reports/flood_m1_quantile_scores.csv' if engine is Q else 'reports/flood_m1_scores.csv'))
            st.dataframe(scores[scores.pid.eq(pid)],hide_index=True,width='stretch')
            st.write('MAE — средняя абсолютная ошибка в сантиметрах. Сравниваются целые будущие годы, '
                     'которые не использовались ни при обучении, ни при выборе алгоритма и калибровке интервала.')
            st.download_button('Скачать проверку по годам',scores.to_csv(index=False).encode('utf-8-sig'),'flood-m1-scores.csv','text/csv')
    with tabs[4]:
        if tabs[4].open is not False:
            st.markdown('Источник: [Уровень-Спутник](https://level-satellite.ru), '
                        'Головлев П.П., Иннокентьев А.И. (2026). Записи gmvo опубликованы как данные АИС ГМВО; '
                        'vodinfo — отдельный источник. Источники в обучающем ряду одной точки не смешиваются.')
            st.write(f"Гидропост {M.STATIONS[pid][0]}, источник {M.STATIONS[pid][1]}; "
                     f"наблюдений: {valid.size}; пропусков календарных суток: {series.isna().sum()}. "
                     'Пропуски не интерполируются. Отрицательные уровни сохраняются.')
            st.write('Признаки: текущий уровень, лаги 1/3/7/14/30 суток, скорость изменения, средние за 7/30 дней, '
                     'сезонность и горизонт. Цель — изменение уровня. Погода, спутниковые снимки и GloFAS '
                     'в собственный М1 уровней не входят. Обучение — CatBoost; алгоритм выбирается по прошлому году.')
            st.write(('v2: CatBoost MultiQuantile оценивает 10/50/90%; сезонная CQR-калибровка по предыдущему году. '
                      if engine is Q else 'v1: симметричный диапазон по ошибкам предыдущего года. ')
                     +'Его фактическое покрытие указано в проверке. '
                     'Рельеф, площадь затопления, опасные отметки и допуск дорог не прогнозируются. '
                     'Архив получен позднее: исторический расчёт не равен сохранённому оперативному выпуску.')
            st.json(manifest['summary'].get('zero_marks',[]))
            st.download_button('Метаданные и источник',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'),
                               f'{pid}-source.json','application/json')
            protocol=M.ROOT/('docs/FLOOD_M1_V2_PLAN.md' if engine is Q else 'docs/FLOOD_M1_VALIDATION_PLAN.md')
            st.download_button('Протокол проверки М1',protocol.read_bytes(),protocol.name,'text/markdown')
