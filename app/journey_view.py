"""Анализ рисков и проверяемые предложения для выбранной карточки пути."""
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from okno.journey import road_rows, registry_coverage, evidence
from okno.risk_advice import scenarios
from okno.regions import label
from okno.ice import from_season_day, to_season_day
from okno.planner import sample_quantiles

NAMES={'unknown':'Нет опубликованных сведений на эту дату','closed':'Путь закрыт',
       'overweight':'Масса превышена','limited':'Предел массы опубликован',
       'mass_unknown':'Предел массы не указан'}


def render_journey(pid,hy,issue_date,issue_day,margin,points,forecast):
    st.subheader('Риски и рекомендации')
    st.write(f'**{label(pid)}** · сведения, известные на {issue_date:%d.%m.%Y} · сезон {hy-1}/{hy}')
    st.caption('Выберите дату поездки и укажите массу машины с грузом. '
               'Опубликованные ограничения и прогноз закрытия помогут оценить возможность проезда.')
    events=road_rows(pid,hy)
    roads=sorted(events.road.unique())
    road=st.selectbox('Дорога из реестра',roads,index=0 if len(roads)==1 else None,
                      placeholder='Выберите конкретную дорогу',key='risk_road_'+pid+str(hy)) if roads else None
    if not roads:
        st.info('В реестре нет дорог этой точки за выбранный сезон. Допуск и предел массы неизвестны; прогноз не заменяет их.')
        _seasons=sorted(registry_coverage(point_id=pid).hydro_year.unique())
        _here=registry_coverage(hydro_year=hy)
        _hint=[]
        if _seasons:
            _hint.append('По этой точке сведения есть за сезоны: '+', '.join(f'{y-1}/{str(y)[2:]}' for y in _seasons)+
                         ' — выберите сезон в левой панели.')
        if len(_here):
            _hint.append('В сезоне '+f'{hy-1}/{str(hy)[2:]}'+' сведения есть по точкам: '+
                         ', '.join(sorted({label(p) for p in _here.point_id.unique()}))+'.')
        if _hint:
            st.caption(' '.join(_hint)+' Реестр пополняет команда по официальным сообщениям (data/external/road_events.csv, '
                       'график текущего сезона — road_schedule_2027.csv).')
    a,b=st.columns(2)
    passage=a.date_input('Планируемая дата проезда',value=issue_date+dt.timedelta(days=7),min_value=issue_date,
                         key='risk_passage_'+str(issue_date))
    tare=a.number_input('Масса пустой машины с оборудованием, т',0.,200.,12.,.1,key='risk_tare')
    cargo=b.number_input('Масса груза, т',0.,200.,6.,.1,key='risk_cargo')
    gross=tare+cargo
    st.write(f'Полная масса машины: **{gross:g} т**. Измените значения под свою машину.')
    q=forecast(pid,hy,issue_day,'close')
    rows=scenarios(events,road,q,issue_date,passage,hy,margin,gross)
    current=rows[0]
    _kinds={'open':'открытие','open10':'открытие до 10 т','limit20':'предел 20 т','limit30':'предел 30 т','close':'закрытие'}
    if not road and len(events):
        st.info('У этой точки несколько дорог, ограничения у каждой свои. Выберите дорогу в списке «Дорога из реестра». '
                'Ниже — что известно по каждой на выбранную дату проезда.')
        _tab=[]
        for _r in sorted(events.road.unique()):
            _ev=evidence(events,_r,issue_date,passage,gross)
            _rec=_ev['record']
            _tab.append({'Дорога':_r,'Состояние на дату проезда':NAMES[_ev['status']],
                         'Последняя запись':(f"{_kinds.get(_rec['kind'],_rec['kind'])} с {_rec['date']:%d.%m.%Y} ({_rec['basis'].lower()})" if _rec else '—'),
                         'Предел массы, т':'—' if _ev['limit_t'] is None else f"{_ev['limit_t']:g}"})
        st.dataframe(pd.DataFrame(_tab),hide_index=True,width='stretch')
    elif road and current['status']=='unknown':
        _all=evidence(events,road,issue_date,pd.Timestamp('2100-01-01'),gross)['records']
        _later=sorted([r for r in _all if r['date']>pd.Timestamp(passage)],key=lambda r:r['date'])
        if _later:
            st.info('На выбранную дату проезда сведений ещё нет, но опубликовано дальше: '+
                    '; '.join(f"{_kinds.get(r['kind'],r['kind'])} с {r['date']:%d.%m.%Y} ({r['basis'].lower()})" for r in _later[:4])+
                    '. Выберите дату проезда не раньше первой из них.')
    st.subheader('Что известно о рисках')
    st.write('**Опубликованные ограничения: '+(NAMES[current['status']] if road or not len(events) else 'выберите дорогу')+'**')
    record=current['record']
    if record:
        st.markdown(f"{record['basis']} · действует с {record['date']:%d.%m.%Y} · опубликовано {record['publication']:%d.%m.%Y} · [Источник]({record['source']})")
        st.caption(record['comment'])
    q_open=forecast(pid,hy,issue_day,'open')
    if q_open and 'error' not in q_open and all(np.isfinite(q_open.get(k,np.nan)) for k in ('q10','q50','q90')):
        _s=sample_quantiles(q_open['q10'],q_open['q50'],q_open['q90'],4000,np.random.default_rng(0))
        _day=float(to_season_day(pd.Timestamp(passage),hy))
        p_not_open=float((_s>_day).mean())
        if p_not_open>0.005 or _day<q_open['q90']:
            st.metric('М1: риск, что к дате проезда лёд ещё не держит 20 т',f'{p_not_open:.1%}')
            st.caption(f"Прогноз открытия (лёд держит 20 т): медиана {q_open['q50_date']:%d.%m.%Y}, "
                       f"диапазон {q_open['q10_date']:%d.%m}–{q_open['q90_date']:%d.%m}. Физическая готовность льда, "
                       'а не приказ об открытии: ехать можно только после официального открытия.')
    risk=current['risk']
    if risk is not None:
        st.metric('М1: риск весеннего закрытия к дате проезда',f'{risk:.1%}')
        st.caption(f"Выпуск модели: {from_season_day(q['issue_day'],hy):%d.%m.%Y}; запас до физического закрытия: {margin} сут. "
                   'Это модельная оценка для порога 20 т, не вероятность аварии или недовоза всей поставки. '
                   'Она не учитывает отдельным процентом административное открытие и другие причины недоступности.')
    else:
        st.info('Процент закрытия не рассчитывается: нет подходящего прогноза либо масса выше 20 т. Экстраполяция модели на более тяжёлую машину не применяется.')
    if q and 'error' not in q:
        st.caption(f"Карточка М1: медиана физического закрытия {q['q50_date']:%d.%m.%Y}, "
                   f"диапазон q10–q90 {q['q10_date']:%d.%m}–{q['q90_date']:%d.%m}.")
    st.subheader('Что можно изменить')
    limit=current['limit_t']
    if current['status']=='closed':
        st.error('По известной записи путь закрыт на выбранную дату. Уменьшение массы не снимает закрытие.')
    elif limit is not None:
        if tare>limit:
            st.warning(f'Пустая машина тяжелее предела {limit:g} т. Нужна более лёгкая машина: уменьшения груза недостаточно.')
        elif gross>limit:
            st.write(f'**Уменьшить груз минимум на {gross-limit:g} т:** при этой массе машины под груз остаётся {limit-tare:g} т. '
                     'Это устранит превышение указанного предела, но не гарантирует проезд. Число рейсов и стоимость здесь не рассчитываются.')
        else: st.write(f'По записанному пределу {limit:g} т превышения массы нет. Допустимость поездки нужно подтвердить у владельца дороги.')
    else:
        st.write('**Уточнить ограничение у владельца дороги.** Пока предела массы нет, рекомендовать конкретную загрузку нельзя.')
    candidates=[x for x in rows[1:] if x['status']=='limited' and x['risk'] is not None and risk is not None and x['risk']<risk]
    if candidates:
        best=min(candidates,key=lambda x:(x['risk'], -x['date'].toordinal()))
        st.write(f"**Проехать раньше — {best['date']:%d.%m.%Y}:** модельный риск закрытия {risk:.1%} → {best['risk']:.1%} "
                 f"(снижение на {100*(risk-best['risk']):.1f} процентного пункта). На эту дату есть запись допустимой массы. "
                 'Возможность отправить груз раньше проверяет перевозчик; это не измеренная экономия.')
    else:
        st.write('**Более ранняя дата:** в проверенном интервале нет варианта с одновременно меньшим модельным риском и известным пределом, допускающим эту массу. Улучшение не заявляется.')
    st.caption('Сравнение охватывает до семи дней перед проездом, не раньше даты решения. Все варианты используют один выпуск прогноза и только уже опубликованные записи.')
    table=pd.DataFrame([{'Дата проезда':x['date'],'Состояние по реестру':NAMES[x['status']],
                         'Предел массы, т':x['limit_t'],'М1: риск закрытия, %':None if x['risk'] is None else round(100*x['risk'],1),
                         'Источник':x['record']['source'] if x['record'] else ''} for x in rows])
    st.dataframe(table,hide_index=True,width='stretch')
    st.download_button('Скачать сравнение дат',table.to_csv(index=False).encode('utf-8-sig'),'risk-scenarios.csv','text/csv')
    st.caption('Расчёт всей цепочки хаб → ж/д → Нижний Бестях → переправа → Якутск и режим «сервис» — на вкладке «Сквозной коридор (А3)».')
