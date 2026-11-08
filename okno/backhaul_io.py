"""Таблицы заявок и машин: импорт, проверка дат и экспорт расписания."""
from dataclasses import asdict
import pandas as pd
from .backhaul import Order, Vehicle

ORDER_COLUMNS = ['Номер', 'Направление', 'Масса, т', 'Кузов', 'Готовность', 'Доставить до',
                 'Погрузка, ч', 'Разгрузка, ч', 'Источник']
VEHICLE_COLUMNS = ['Машина', 'Кузов', 'Грузоподъёмность, т', 'Собственная масса, т',
                   'С грузом, ₽/км', 'Пустой, ₽/км', 'Занятость, ₽/ч', 'На оборот, ₽', 'Источник']


def example_frames(base_date):
    """Учебные заявки, не сведения о перевозчике или конкретном направлении."""
    base=pd.Timestamp(base_date)
    stamp=lambda h:(base+pd.Timedelta(hours=h)).strftime('%Y-%m-%d %H:%M')
    orders=pd.DataFrame([
        ['Завоз-1','А → Б',8.,'тент',stamp(0),stamp(24),1.,1.,'Учебный пример'],
        ['Вывоз-1','Б → А',6.,'тент',stamp(12),stamp(72),1.,1.,'Учебный пример'],
    ],columns=ORDER_COLUMNS)
    vehicles=pd.DataFrame([
        ['Машина-1','тент',10.,8.,55.,40.,500.,1000.,'Учебный пример']
    ],columns=VEHICLE_COLUMNS)
    return orders,vehicles


def parse_frames(order_frame, vehicle_frame, base_date):
    base=pd.Timestamp(base_date)
    def text(row,key):
        value=row[key]
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f'Не заполнено поле «{key}».')
        return str(value).strip()
    def hours(value):
        stamp=pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is not None:
            raise ValueError('Укажите дату и время без часового пояса, в одном местном времени.')
        return (stamp-base).total_seconds()/3600
    try:
        for frame,columns in ((order_frame,ORDER_COLUMNS),(vehicle_frame,VEHICLE_COLUMNS)):
            if not set(columns)<=set(frame.columns):
                raise ValueError('Нужны все столбцы из шаблона CSV: '+', '.join(columns))
            for _,row in frame.iterrows():
                text(row,'Источник')
        orders=[]
        for _,r in order_frame.iterrows():
            direction={'А → Б':'out','Б → А':'back'}.get(text(r,'Направление'))
            if direction is None:
                raise ValueError('Направление: только «А → Б» или «Б → А».')
            orders.append(Order(text(r,'Номер'),direction,float(r['Масса, т']),text(r,'Кузов'),
                hours(r['Готовность']),hours(r['Доставить до']),float(r['Погрузка, ч']),float(r['Разгрузка, ч'])))
        vehicles=[Vehicle(text(r,'Машина'),text(r,'Кузов'),float(r['Грузоподъёмность, т']),
            float(r['Собственная масса, т']),float(r['С грузом, ₽/км']),float(r['Пустой, ₽/км']),
            float(r['Занятость, ₽/ч']),float(r['На оборот, ₽'])) for _,r in vehicle_frame.iterrows()]
        return orders,vehicles
    except (TypeError,KeyError,OverflowError) as error:
        raise ValueError('Проверьте формат таблиц, чисел и дат.') from error


def schedule_frame(plan,base_date):
    rows=[]
    def stamp(h):
        return (pd.Timestamp(base_date)+pd.Timedelta(hours=h)).strftime('%Y-%m-%d %H:%M')
    for t in plan.trips:
        rows.append({'Машина':t.vehicle,'Завоз':t.outward or 'Порожний ход',
            'Вывоз':t.backward or 'Порожний ход','Начало работы в А':stamp(t.start_h),
            'Выезд из А':stamp(t.outward_depart_h),'Прибытие в Б':stamp(t.outward_arrive_h),
            'Завоз разгружен':stamp(t.outward_done_h) if t.outward_done_h is not None else '',
            'Начало обратной погрузки':stamp(t.backward_load_h) if t.backward else '',
            'Выезд из Б':stamp(t.backward_depart_h),'Возврат в А':stamp(t.backward_arrive_h),
            'Вывоз разгружен':stamp(t.backward_done_h) if t.backward_done_h is not None else '',
            'Машина снова свободна':stamp(t.finish_h),'Ожидание в Б, ч':t.wait_h,
            'С грузом, км':t.loaded_km,'Пустой пробег, км':t.empty_km,
            'Движение, ₽':t.movement_rub,'Занятость, ₽':t.time_rub,'На оборот, ₽':t.fixed_rub,
            'Расходы, ₽':t.cost_rub})
    return pd.DataFrame(rows)


def report_payload(plans,orders,vehicles,route,base_date,metadata):
    return dict(method='backhaul-v1',base_date=str(base_date),metadata=metadata,
                route=asdict(route),orders=[asdict(o) for o in orders],vehicles=[asdict(v) for v in vehicles],
                results={key:dict(status=p.status,reason=p.reason,summary=p.summary(orders),
                                 trips=[asdict(t) for t in p.trips]) for key,p in plans.items()})
