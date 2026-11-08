"""Единое рабочее место: карточка данных → маршрут → рейсы → М3 → сравнение рисков."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from okno.journey import road_rows,evidence,seasonal_segment,weekly_access
from okno.regions import label
from okno.ice import from_season_day

ROOT=Path(__file__).resolve().parents[1]


def _pilot():
    st.session_state.update(filter_region='yakutia',filter_point='yakutsk_crossing')


def _archive_pilot():
    import datetime as dt
    st.session_state.update(filter_region='yakutia',filter_point='yakutsk_crossing',filter_season=2026,
                            filter_date=dt.date(2026,4,8))


def render_journey(pid,hy,issue_date,issue_day,margin,points,forecast):
    st.subheader('Маршрут и план завоза')
    st.caption(f'Общие условия: {label(pid)} · решение {issue_date:%d.%m.%Y} · сезон {hy-1}/{hy}. '
               'Карточка пути, реестр, маршрут и план ниже используют эту дату. Изменение точки или даты сбрасывает выбранный маршрут.')
    events=road_rows(pid,hy)
    segment=seasonal_segment(pid)
    roads=sorted(events.road.unique())
    if segment and segment['road'] not in roads: roads.append(segment['road'])
    default_road=roads.index(segment['road']) if segment and segment['road'] in roads else 0 if len(roads)==1 else None
    road=st.selectbox('Дорога для привязки карточки',roads,index=default_road,
                      placeholder='Выберите точную дорогу: один пост может относиться к нескольким направлениям',key='journey_road_'+pid+str(hy)) if roads else None
    gross=st.number_input('Полная масса машины для сравнения маршрутов, т',0.1,200.,18.,.1,key='journey_gross')
    qclose=forecast(pid,hy,issue_day,'close');qopen=forecast(pid,hy,issue_day,'open')
    context=dict(pid=pid,hy=hy,issue_date=issue_date,issue_day=issue_day,margin=margin,events=events,
                 road=road,gross_t=gross,close_forecast=qclose,open_forecast=qopen,
                 segment=segment if segment and segment['road']==road else None)
    context['version']=hashlib.sha256((ROOT/'data/external/road_events.csv').read_bytes()+
        json.dumps({k:(qclose or {}).get(k) for k in ('issue_day','q10','q50','q90')},default=str,sort_keys=True).encode()).hexdigest()
    token=hashlib.sha256(json.dumps(dict(pid=pid,road=road,date=str(issue_date),margin=margin,gross=gross,version=context['version']),sort_keys=True).encode()).hexdigest()
    if st.session_state.get('journey_context_token')!=token:
        for k in ('journey_selected','journey_orders_summary','route_result','bh_result'):
            st.session_state.pop(k,None)
        st.session_state['journey_context_token']=token
    if st.session_state.get('journey_decision_seen')!=str(issue_date):
        st.session_state['route_date']=issue_date
        st.session_state['route_due']=issue_date+pd.Timedelta(days=2)
        st.session_state['journey_decision_seen']=str(issue_date)
    with st.expander('Данные из карточки пути и источники',expanded=True):
        rows=[]
        for kind,q in (('Открытие',qopen),('Физическое закрытие',qclose)):
            if q and 'error' not in q:
                rows.append({'Событие':kind,'М1, q10':from_season_day(q['q10'],hy).date(),
                    'М1, медиана':from_season_day(q['q50'],hy).date(),'М1, q90':from_season_day(q['q90'],hy).date(),
                    'Выпуск':from_season_day(q['issue_day'],hy).date()})
        if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        else: st.info('Для выбранной даты прогноза карточки нет. Вероятность не подставляется.')
        st.caption(f'Источник: те же М1 и погодные ряды, что во вкладке «Карточка пути». '
                   f'Переход от физического таяния к закрытию использует выбранный запас {margin} сут. '
                   'Это допущение модели, не административное разрешение. Ретроспективный показ не является независимым бэктестом.')
        if road:
            facts=evidence(events,road,issue_date,issue_date,gross)
            names={'closed':'Закрыто по записи','overweight':'Превышен предел массы','limited':'Есть запись ограничения массы',
                   'mass_unknown':'Открытие записано, структурированной массы нет','unknown':'На эту дату опубликованных сведений нет'}
            st.write('**Реестр дороги:** '+names[facts['status']])
            if facts['record']:
                r=facts['record'];st.markdown(f"{r['basis']} · {r['date']:%d.%m.%Y} · [Источник]({r['source']})")
                st.caption(r['comment'])
            known=pd.DataFrame([dict(Событие=x['kind'],Дата=x['date'].date(),Опубликовано=x['publication'].date(),
                                     Основание=x['basis'],Источник=x['source']) for x in facts['records']])
            if not known.empty: st.dataframe(known.drop_duplicates(),hide_index=True,width='stretch')
        if pid=='yakutsk_crossing':
            path=ROOT/'data/s1/predictions.csv'
            if path.exists():
                shots=pd.read_csv(path,parse_dates=['date'])
                shots=shots[(shots.point==pid)&(shots.date<=pd.Timestamp(issue_date))].sort_values('date').tail(1)
                if not shots.empty:
                    st.write(f"Последний снимок М2 до решения: {shots.iloc[0]['date']:%d.%m.%Y}; P(лёд)={shots.iloc[0]['p_ice']:.2f}.")
                    st.caption('Спутник подтверждает состояние поверхности; вероятность льда не равна разрешению проезда или риску недовоза.')
    if not context['segment']:
        st.caption('Для выбранной точки ещё нет явной связи геометрии дороги с карточкой. '
                   'Для пилота такая связь добавлена у переправы Якутск — Нижний Бестях.')
        st.button('Выбрать переправу Якутск — Нижний Бестях',on_click=_pilot,key='journey_pilot')
    st.button('Открыть архивный пилот: 8 апреля 2026, переправа Якутск — Нижний Бестях',
              on_click=_archive_pilot,key='journey_archive',help='Реальные записи реестра, известные на эту дату. Это исторический разбор, не текущий допуск.')
    from app.route_view import render_routes
    render_routes(context)
    selected=st.session_state.get('journey_selected')
    valid=selected and selected.get('point_id')==pid and selected.get('road')==road and selected.get('issue_date')==str(issue_date)
    if valid:
        context['selected_route']=selected
        if not selected['project'].get('covered'):
            st.info('Выбран дорожный вариант без подтверждённой связи с сезонным участком. '
                    'Реестр выбранной карточки не переносится на него. Для рейсов с ограничениями выберите привязанный вариант.')
            context['selected_route']=None
    from app.backhaul_view import render_backhaul
    render_backhaul(context)
    st.subheader('3. Недельный план и риск недовоза на выбранном сезонном участке')
    with st.expander('Историческая проверка всей цепочки (А3)',expanded=False):
        st.caption('Отдельный ретроспективный расчёт хаб → ж/д → переправа → склад. '
                   'В нём свой подтверждённый исторический сезон и спрос; это не текущий маршрут пользователя.')
        if st.checkbox('Открыть проверку сквозного коридора',key='journey_corridor'):
            from app.corridor_view import render_corridor
            render_corridor()
    order_summary=st.session_state.get('journey_orders_summary') or {}
    quantity=order_summary.get('out_t') or None
    linked=bool(context.get('selected_route'))
    handling=order_summary.get('max_out_handling_h',0)
    travel=(st.session_state.get('bh_travel')+handling)/24 if linked and st.session_state.get('bh_travel') else None
    if linked and handling:
        st.caption(f'Время М3 включает путь и максимальную длительность обработки среди ваших прямых заявок: {handling:g} ч. '
                   'Точный график каждой машины, загрузки и возврата проверяется в рейсовом плане.')
    if not linked:
        st.caption('Пока маршрут не выбран, М3 считает только выбранный сезонный участок; время рейса задаётся отдельно.')
        travel=st.number_input('Время рейса для М3 по данным перевозчика, ч',0.1,None,value=None,key='journey_m3_hours')
        travel=travel/24 if travel is not None else None
    if travel is None:
        st.info('Выберите связанный маршрут выше или укажите время рейса. Вымышленная длительность не используется.')
        return
    from app.planner_view import render_planner
    applicable=max(gross,order_summary.get('max_gross_t',0))<=20
    if not applicable:
        st.warning('В заявках или сравнении указана машина тяжелее 20 т. М1 обучена на физическом пороге 20 т: '
                   'её вероятность нельзя переносить на эту машину. Рейсы по известному дорожному ограничению проверяются выше, модельный недовоз М3 не оценивается.')
    availability=weekly_access(events,road,issue_date,hy,gross) if road else None
    earliest=(pd.Timestamp(selected['departure'])-pd.Timestamp(hy-1,9,1)).total_seconds()/86400 if linked else None
    params=render_planner(qclose if applicable else None,hy,issue_day,margin,label(pid),transit_days=travel,quantity_t=quantity,require_quantity=True,availability=availability,start_day=earliest)
    st.subheader('4. Риск недовоза и сравнение расписаний для этого заказа')
    summary=st.session_state.get('journey_m3_summary')
    if summary is None:
        st.info('Сначала заполните параметры М3. Рейтинг появится по тому же заказу, маршруту и прогнозу — без отдельного условного объёма.')
    else:
        risk=summary[['expected_backup_t','expected_cost_rub','planned_road_t']].copy()
        risk.index=risk.index.map({'ml':'Равномерный план по М1','economic_ml':'Экономический план по М1'})
        risk['Ожидаемая доля для резерва, %']=100*risk.expected_backup_t/params.Q_t
        risk=risk.rename(columns={'expected_backup_t':'Ожидаемый остаток для резерва, т',
            'expected_cost_rub':'Ожидаемые расходы, ₽','planned_road_t':'Запланировано по сезонному пути, т'})
        st.dataframe(risk.sort_values('Ожидаемый остаток для резерва, т'),width='stretch')
        st.caption('Рейтинг использует расчёт М3 выше, тот же объём и допуск из реестра. Ожидаемая доля груза для резерва '
                   'не равна вероятности недовоза всего заказа. Срок резервной доставки не моделируется. '
                   'Риск закрытия сезонного участка для вариантов маршрута показан в первой таблице.')
