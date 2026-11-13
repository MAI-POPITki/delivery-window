"""Отдельное рабочее место для совместного планирования завоза и вывоза."""
from dataclasses import asdict
import datetime as dt
import hashlib
import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from okno.backhaul import Route, compare, fixed_plan_sensitivity, validate_inputs
from okno.backhaul_io import example_frames, parse_frames, schedule_frame, report_payload
from okno.journey import route_calendar


def _csv(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


def render_backhaul(context=None):
    context=context or {}
    integrated=bool(context)
    st.subheader('2. Грузы, машины и обратная загрузка')
    st.write('Проверьте, можно ли выполнить завоз и вывоз одними машинами, сократить пустой пробег и расходы.')
    st.caption('Один маршрут А — Б, все машины базируются в А. Сравниваем одни и те же заявки, сроки и парк. '
               'Каждый рейс включает возвращение в А. Расчёт не требует нейросети.')
    if not integrated: st.warning('Начальные значения — учебный пример. Реальных заявок на обратный груз в проекте пока нет. '
               'Для своего расчёта замените таблицы и параметры; программа не проверяет достоверность введённых источников.')
    calculate_slot=st.container()
    result_slot=st.container()
    if not integrated and st.button('Построить маршрут по дорогам',key='bh_open_route'):
        st.query_params['module']='routes'; st.rerun()
    if st.session_state.get('bh_route_import_note'):
        st.info(st.session_state['bh_route_import_note'])
    if integrated and not context.get('selected_route'):
        st.info('Сначала постройте и выберите маршрут в разделе выше. Рейсы будут использовать его расстояние и время.'); return
    c1,c2,c3=st.columns(3)
    a=c1.text_input('Точка А — база', 'Терминал А',key='bh_a',disabled=integrated)
    b=c2.text_input('Точка Б — получатель завоза', 'Терминал Б',key='bh_b',disabled=integrated)
    base=context['issue_date'] if integrated else c3.date_input('Дата решения, время 00:00',dt.date(2026,3,1),key='bh_date')
    if integrated: c3.caption(f'Общая дата решения: {base:%d.%m.%Y}')
    st.caption('Все даты и часы — в одном выбранном местном времени. Дата и путь из боковой панели М1 здесь не используются.')
    with st.expander('Маршрут, транспортное окно и источник параметров',expanded=True):
        left,right=st.columns(2)
        distance=left.number_input('Расстояние А → Б, км',0.01,None,300.,key='bh_distance',disabled=integrated)
        travel=left.number_input('Время движения А → Б, ч',0.01,None,8.,0.5,key='bh_travel',disabled=integrated)
        back_distance=left.number_input('Расстояние Б → А, км',0.01,None,300.,key='bh_return_distance',disabled=integrated)
        back_travel=left.number_input('Время движения Б → А, ч',0.01,None,8.,0.5,key='bh_return_travel',disabled=integrated)
        turnaround=left.number_input('Подготовка машины к следующему рейсу, ч',0.,72.,1.,0.5,key='bh_turn')
        step=left.selectbox('Шаг начала рейсов, ч',[24,12,6],key='bh_step')
        opened=base if integrated else right.date_input('Движение разрешено с 00:00',base,min_value=base,key='bh_open')
        closed=base+dt.timedelta(days=60) if integrated else right.date_input('Движение закрывается в 00:00',base+dt.timedelta(days=4),min_value=base,key='bh_close')
        calendar=None
        if integrated:
            calendar,sources=route_calendar(context['events'],context['road'],context['issue_date'],base,60,context['gross_t'])
            limit=max((x for x in calendar if x is not None),default=context['gross_t'])
            right.write('Календарь и полная масса — по записям реестра для выбранной дороги. Неизвестные дни исключаются из рейсов.')
            right.caption('Реестр имеет дневную точность: при неизвестном часе день закрытия исключается целиком.')
            for link in sources: right.markdown(f'[Источник календаря]({link})')
        else: limit=right.number_input('Допустимая полная масса, т',0.1,200.,20.,0.1,key='bh_limit')
        source=st.text_input('Откуда взяты расстояние, время, окно и ограничение массы?',
                             'Календарь road_events.csv и выбранный маршрут OSM; источники приведены выше' if integrated else 'Учебные значения; разрешение на движение не подтверждено',key='bh_source')
        st.caption('Окно должно быть известно на дату решения. По нему проверяется весь путь туда и обратно. '
                   'Прогноз льда не заменяет разрешение проезда. Время движения задавайте с учётом отдыха и остановок. '
                   'Расстояния можно ввести вручную или передать из построителя маршрутов.')
    demo_orders,demo_vehicles=example_frames(base)
    if integrated:
        demo_orders=st.session_state.get('journey_order_frame',demo_orders.iloc[:0]).copy()
        demo_vehicles=st.session_state.get('journey_vehicle_frame',demo_vehicles.iloc[:0]).copy()
    st.subheader('Заявки и доступные машины')
    st.caption('Заявка — одна неделимая партия. Тип кузова в заявке и у машины должен совпадать. '
               'Готовность — начало возможной погрузки, срок — окончание разгрузки. До 12 заявок и 4 машин.')
    l,r=st.columns(2)
    order_file=l.file_uploader('Загрузить заявки CSV',type='csv',key='bh_order_file')
    vehicle_file=r.file_uploader('Загрузить машины CSV',type='csv',key='bh_vehicle_file')
    l.download_button('Шаблон заявок',_csv(demo_orders),'backhaul-orders.csv',key='bh_order_template',on_click='ignore')
    r.download_button('Шаблон машин',_csv(demo_vehicles),'backhaul-vehicles.csv',key='bh_vehicle_template',on_click='ignore')
    try:
        order_input=pd.read_csv(io.BytesIO(order_file.getvalue())) if order_file else demo_orders
        vehicle_input=pd.read_csv(io.BytesIO(vehicle_file.getvalue())) if vehicle_file else demo_vehicles
    except (ValueError,UnicodeError) as error:
        st.error(f'Не удалось прочитать CSV: {error}')
        return
    # Новый файл/дата создаёт новый редактор; изменения в старой таблице не подменяют импорт.
    order_key=hashlib.sha256(_csv(order_input)).hexdigest()[:12]
    vehicle_key=hashlib.sha256(_csv(vehicle_input)).hexdigest()[:12]
    order_frame=st.data_editor(order_input,num_rows='dynamic',hide_index=True,width='stretch',key='bh_orders_'+order_key)
    st.caption('Ставка движения — собственные расходы на километр, а не цена продажи перевозки. '
               'Стоимость часа покрывает занятость, включая ожидание и обработку. Не учитывайте одну затрату дважды.')
    vehicle_frame=st.data_editor(vehicle_input,num_rows='dynamic',hide_index=True,width='stretch',key='bh_vehicles_'+vehicle_key)
    if integrated:
        st.session_state['journey_order_frame']=order_frame.copy()
        st.session_state['journey_vehicle_frame']=vehicle_frame.copy()
    st.caption('При смене груза оператор должен подтвердить совместимость и нужную обработку кузова. '
               'Модель проверяет массу и указанный тип кузова; габариты, осевые нагрузки и специальные требования отдельно не рассчитывает.')
    metadata={'point_a':a,'point_b':b,'route_source':source,
              'order_sources':order_frame.get('Источник',pd.Series(dtype=str)).astype(str).tolist(),
              'vehicle_sources':vehicle_frame.get('Источник',pd.Series(dtype=str)).astype(str).tolist(),
              'evidence':'Пользовательские заявки, календарь реестра проекта' if integrated else 'Пользовательский сценарий; фактическая экономия не подтверждена'}
    if integrated:
        metadata.update(point_id=context['pid'],road=context['road'],decision_date=str(context['issue_date']),
                        registry_version=context.get('version'),calendar_sources=sources,
                        forecast_is_not_permission=True)
    if integrated and (order_frame.empty or vehicle_frame.empty):
        st.session_state.pop('journey_orders_summary',None)
        st.info('Добавьте свои заявки и машины в таблицы или загрузите CSV. Учебные заказы не подставляются.'); return
    try:
        if not source.strip() or not a.strip() or not b.strip() or a.strip()==b.strip():
            raise ValueError('Укажите разные точки А и Б и источник параметров.')
        orders,vehicles=parse_frames(order_frame,vehicle_frame,base)
        earliest=max(0.,(pd.Timestamp(context['selected_route']['departure'])-pd.Timestamp(base)).total_seconds()/3600) if integrated else (opened-base).days*24
        route=Route(distance,travel,earliest,(closed-base).days*24,limit,float(step),turnaround,back_distance,back_travel,
                    tuple(calendar) if calendar is not None else None)
        validate_inputs(orders,vehicles,route)
        if integrated:
            max_gross=max((v.tare_t+o.tons for v in vehicles for o in orders if v.body==o.body and o.tons<=v.capacity_t),default=context['gross_t'])
            st.session_state['journey_orders_summary']=dict(out_t=sum(o.tons for o in orders if o.direction=='out'),
                back_t=sum(o.tons for o in orders if o.direction=='back'),orders=len(orders),vehicles=len(vehicles),max_gross_t=max_gross,
                max_out_handling_h=max((o.loading_h+o.unloading_h for o in orders if o.direction=='out'),default=0))
            if max_gross>context['gross_t']+.01:
                st.warning(f'По заявкам возможна полная масса до {max_gross:g} т. Сравнение маршрутов выше выполнено для {context["gross_t"]:g} т — обновите общую массу и пересчитайте маршрут. В рейсовом плане каждая фактическая масса проверяется отдельно.')
    except (ValueError,TypeError) as error:
        if integrated: st.session_state.pop('journey_orders_summary',None)
        st.error(str(error))
        return
    signature=json.dumps(dict(route=asdict(route),orders=[asdict(o) for o in orders],
                              vehicles=[asdict(v) for v in vehicles],date=str(base),metadata=metadata),ensure_ascii=False,sort_keys=True)
    with calculate_slot:
        pressed=st.button('Сравнить отдельные рейсы и обратную загрузку',type='primary',key='bh_calculate')
    if pressed:
        with st.spinner('Подбираю рейсы и проверяю занятость машин…'):
            try:
                plans=compare(orders,vehicles,route)
                st.session_state['bh_result']=(signature,plans)
            except ValueError as error:
                st.error(str(error))
                return
    saved=st.session_state.get('bh_result')
    if not saved or saved[0]!=signature:
        st.info('Нажмите кнопку сравнения.' if not saved else 'Параметры изменены. Пересчитайте результат — старые цифры скрыты.')
        return
    plans=saved[1]
    names={'separate':'Отдельные рейсы','combined':'С возможностью обратной загрузки'}
    with result_slot:
        st.subheader('Результат для одинаковых заказов')
        names={'separate':'Отдельные рейсы','combined':'С возможностью обратной загрузки'}
        for key,plan in plans.items():
            if not plan.feasible:
                st.warning(names[key]+': '+plan.reason)
            elif plan.status!='optimal':
                st.warning(names[key]+': '+plan.reason)
        if all(p.feasible for p in plans.values()):
            before,after=(plans[k].summary(orders) for k in ('separate','combined'))
            l,m,r=st.columns(3)
            l.metric('Расходы с совмещением',f"{after['cost_rub']:,.0f} ₽",f"{after['cost_rub']-before['cost_rub']:+,.0f} ₽",delta_color='inverse')
            m.metric('Общий пробег',f"{after['total_km']:,.0f} км",f"{after['total_km']-before['total_km']:+,.0f} км",delta_color='inverse')
            r.metric('Пустой пробег',f"{after['empty_km']:,.0f} км",f"{after['empty_km']-before['empty_km']:+,.0f} км",delta_color='inverse')
            st.caption('Изменения относительно отдельных рейсов. Сравнение расходов, не прибыли. '
                       'Контрольная схема оптимизирована под те же ограничения; она не объявляется реальной практикой перевозчика.')
            if after['paired']==0:
                st.info('Совмещение не выбрано: при заданных условиях отдельные рейсы не дороже либо совместимых пар нет.')
            elif before['cost_rub']>after['cost_rub']+.01:
                st.success(f"В расчёте совмещено оборотов: {after['paired']}. Выполнены все заявки ({len(orders)}), машины возвращены на базу.")
            rows=[]
            for field,label in [('trips','Обороты А — Б — А'),('tons','Перевезено, т'),('orders','Выполнено заявок'),
                                ('occupied_h','Занятость машин, ч'),('wait_h','Ожидание в Б, ч'),
                                ('movement_rub','Движение, ₽'),('time_rub','Занятость, ₽'),('fixed_rub','Фиксированные расходы, ₽')]:
                rows.append({'Показатель':label,'Отдельные рейсы':before[field],'С совмещением':after[field]})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            with st.expander('Зависимость вывода от стоимости часа'):
                st.caption('Расходы тех же двух расписаний при другой стоимости часа. Это не повторная оптимизация. '
                           'Отрицательная экономия означает, что выбранное совмещение стало дороже.')
                st.dataframe(fixed_plan_sensitivity(plans,orders),hide_index=True,width='stretch')
        elif plans['combined'].feasible:
            st.info('Совмещённый план выполняет все заказы. Стоимостную экономию относительно невыполнимой схемы не рассчитываем.')
    for key,plan in plans.items():
        if not plan.feasible:
            continue
        st.markdown('**'+names[key]+'**')
        frame=schedule_frame(plan,base)
        timeline=frame.copy()
        timeline['Начало']=pd.to_datetime(timeline['Начало работы в А'])
        timeline['Конец']=pd.to_datetime(timeline['Машина снова свободна'])
        timeline['Рейс']=timeline['Завоз']+' / '+timeline['Вывоз']
        fig=px.timeline(timeline,x_start='Начало',x_end='Конец',y='Машина',color='Рейс',
                        hover_data=['Выезд из А','Выезд из Б','Пустой пробег, км','Ожидание в Б, ч'])
        fig.update_layout(height=260,xaxis_title='Местное время',showlegend=False)
        st.plotly_chart(fig,width='stretch',key='bh_chart_'+key)
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button('Скачать расписание: '+names[key],_csv(frame),f'backhaul-{key}.csv',key='bh_export_'+key,on_click='ignore')
    st.download_button('Скачать расчёт с исходными данными и источниками',
        json.dumps(report_payload(plans,orders,vehicles,route,base,metadata),ensure_ascii=False,indent=2),
        'backhaul-calculation.json','application/json',key='bh_export_json',on_click='ignore')
    st.caption('Оптимальность относится к указанной сетке начала рейсов и рассматриваемым оборотам. '
               'Каждая партия перевозится целиком; дополнительных заездов нет. Достоверность заказов и окна требует проверки оператором.')
