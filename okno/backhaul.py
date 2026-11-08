"""Совмещение завоза и вывоза на одном маршруте А — Б с возвращением машин в А."""
from dataclasses import dataclass
from heapq import heappush, heappop
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

EPS = 1e-7


@dataclass(frozen=True)
class Order:
    id: str
    direction: str
    tons: float
    body: str
    ready_h: float
    due_h: float
    loading_h: float = 1.
    unloading_h: float = 1.


@dataclass(frozen=True)
class Vehicle:
    id: str
    body: str
    capacity_t: float
    tare_t: float
    loaded_rub_km: float
    empty_rub_km: float
    occupied_rub_h: float
    fixed_rub: float = 0.


@dataclass(frozen=True)
class Route:
    distance_km: float
    travel_h: float
    open_h: float
    close_h: float
    gross_limit_t: float
    step_h: float = 24.
    turnaround_h: float = 1.


@dataclass(frozen=True)
class Trip:
    vehicle: str
    outward: str | None
    backward: str | None
    start_h: float
    outward_depart_h: float
    outward_arrive_h: float
    backward_load_h: float
    backward_depart_h: float
    backward_arrive_h: float
    finish_h: float
    outward_done_h: float | None
    backward_done_h: float | None
    wait_h: float
    loaded_km: float
    empty_km: float
    movement_rub: float
    time_rub: float
    fixed_rub: float

    @property
    def cost_rub(self):
        return self.movement_rub + self.time_rub + self.fixed_rub

    @property
    def orders(self):
        return tuple(x for x in (self.outward, self.backward) if x is not None)


@dataclass
class Plan:
    status: str
    trips: list[Trip]
    reason: str = ''
    candidate_count: int = 0

    @property
    def feasible(self):
        return self.status in ('optimal', 'feasible')

    def summary(self, orders):
        if not self.feasible:
            return None
        return dict(trips=len(self.trips), paired=sum(len(t.orders) == 2 for t in self.trips),
                    orders=len(orders), tons=sum(o.tons for o in orders),
                    total_km=sum(t.loaded_km+t.empty_km for t in self.trips),
                    empty_km=sum(t.empty_km for t in self.trips),
                    occupied_h=sum(t.finish_h-t.start_h for t in self.trips),
                    wait_h=sum(t.wait_h for t in self.trips),
                    movement_rub=sum(t.movement_rub for t in self.trips),
                    time_rub=sum(t.time_rub for t in self.trips),
                    fixed_rub=sum(t.fixed_rub for t in self.trips),
                    cost_rub=sum(t.cost_rub for t in self.trips))


def _numbers(values, *, positive=False):
    return all(isinstance(x, (int, float)) and math.isfinite(x)
               and (x > 0 if positive else x >= 0) for x in values)


def validate_inputs(orders, vehicles, route):
    if not 1 <= len(orders) <= 12 or not 1 <= len(vehicles) <= 4:
        raise ValueError('Нужно от 1 до 12 заявок и от 1 до 4 машин.')
    for items in (orders, vehicles):
        if any(not isinstance(x.id, str) or not x.id.strip() for x in items):
            raise ValueError('У каждой заявки и машины должен быть непустой номер.')
        if len({x.id for x in items}) != len(items):
            raise ValueError('Номера внутри таблицы не должны повторяться.')
        if any(not isinstance(x.body, str) or not x.body.strip() for x in items):
            raise ValueError('Укажите тип кузова.')
    if not _numbers((route.distance_km, route.travel_h, route.gross_limit_t, route.step_h), positive=True):
        raise ValueError('Расстояние, время пути, предел массы и шаг должны быть положительными.')
    if not _numbers((route.open_h, route.close_h, route.turnaround_h)) or route.close_h <= route.open_h:
        raise ValueError('Закрытие должно быть позже открытия; времена не могут быть отрицательными.')
    if route.close_h > 24*60 or route.step_h < 1:
        raise ValueError('Горизонт ограничен 60 сутками, шаг — не менее часа.')
    for o in orders:
        if o.direction not in ('out', 'back') or not _numbers((o.tons,), positive=True):
            raise ValueError('Направление должно быть out/back, масса — положительной.')
        if not math.isfinite(o.ready_h) or not _numbers((o.due_h, o.loading_h, o.unloading_h)) or o.due_h <= max(0,o.ready_h):
            raise ValueError('Проверьте готовность, срок и длительность обработки заявки.')
    for v in vehicles:
        if not _numbers((v.capacity_t, v.tare_t), positive=True) or not _numbers(
            (v.loaded_rub_km, v.empty_rub_km, v.occupied_rub_h, v.fixed_rub)
        ):
            raise ValueError('Массы машины должны быть положительными, ставки — неотрицательными.')


def make_trip(v, route, start, outward=None, backward=None):
    """Вся ось времени задана в часах от даты решения; ожидание в Б учитывается."""
    if outward is None and backward is None:
        return None
    for order in (outward, backward):
        if order and (order.body != v.body or order.tons > v.capacity_t + EPS):
            return None
    if start < 0 or (outward and start < outward.ready_h - EPS):
        return None
    dep_out = start + (outward.loading_h if outward else 0)
    arr_out = dep_out + route.travel_h
    done_out = arr_out + (outward.unloading_h if outward else 0)
    load_back = max(done_out, backward.ready_h) if backward else done_out
    dep_back = load_back + (backward.loading_h if backward else 0)
    arr_back = dep_back + route.travel_h
    done_back = arr_back + (backward.unloading_h if backward else 0)
    # Для обеих сторон проверяется полная масса и весь интервал движения.
    for depart, arrive, order in ((dep_out, arr_out, outward), (dep_back, arr_back, backward)):
        if depart < route.open_h-EPS or arrive > route.close_h+EPS:
            return None
        if v.tare_t + (order.tons if order else 0) > route.gross_limit_t + EPS:
            return None
    if outward and done_out > outward.due_h+EPS:
        return None
    if backward and done_back > backward.due_h+EPS:
        return None
    finish = done_back + route.turnaround_h
    loaded = route.distance_km * (int(outward is not None)+int(backward is not None))
    empty = 2*route.distance_km-loaded
    return Trip(v.id, outward.id if outward else None, backward.id if backward else None,
                start, dep_out, arr_out, load_back, dep_back, arr_back, finish,
                done_out if outward else None, done_back if backward else None,
                load_back-done_out, loaded, empty,
                loaded*v.loaded_rub_km+empty*v.empty_rub_km,
                (finish-start)*v.occupied_rub_h, v.fixed_rub)


def candidates(orders, vehicles, route, paired):
    validate_inputs(orders, vehicles, route)
    outward = [o for o in orders if o.direction == 'out']
    backward = [o for o in orders if o.direction == 'back']
    combinations = [(o, None) for o in outward]+[(None, o) for o in backward]
    if paired:
        combinations += [(a, b) for a in outward for b in backward]
    result = []
    for v in vehicles:
        for start in np.arange(0, route.close_h, route.step_h):
            for a, b in combinations:
                trip = make_trip(v, route, float(start), a, b)
                if trip:
                    result.append(trip)
                    if len(result) > 12000:
                        raise ValueError('Слишком много вариантов. Увеличьте шаг отправок или сократите горизонт/число заявок.')
    return result


def audit_plan(trips, orders, vehicles, route):
    """Проверка ограничений и баланса результата отдельно от матрицы оптимизатора."""
    od, vd = {o.id:o for o in orders}, {v.id:v for v in vehicles}
    seen = []
    for t in trips:
        if t.vehicle not in vd or any(k not in od for k in t.orders):
            raise ValueError('В плане неизвестная машина или заявка.')
        v = vd[t.vehicle]
        out, back = od.get(t.outward), od.get(t.backward)
        if not t.orders or (out and out.direction != 'out') or (back and back.direction != 'back'):
            raise ValueError('Неверное направление заявки.')
        for o in (out, back):
            if o and (o.body != v.body or o.tons > v.capacity_t+EPS or o.tons+v.tare_t > route.gross_limit_t+EPS):
                raise ValueError('Нарушены ограничения кузова или массы.')
        if t.start_h < -EPS or abs(t.start_h/route.step_h-round(t.start_h/route.step_h)) > EPS:
            raise ValueError('Начало вне сетки отправок.')
        for dep, arr in ((t.outward_depart_h,t.outward_arrive_h),(t.backward_depart_h,t.backward_arrive_h)):
            if dep < route.open_h-EPS or arr > route.close_h+EPS or abs(arr-dep-route.travel_h)>EPS:
                raise ValueError('Нарушено окно движения или время пути.')
        if v.tare_t > route.gross_limit_t+EPS:
            raise ValueError('Пустая машина превышает ограничение.')
        out_done = t.outward_arrive_h+(out.unloading_h if out else 0)
        if abs(t.outward_depart_h-t.start_h-(out.loading_h if out else 0)) > EPS:
            raise ValueError('Не учтена погрузка в А.')
        if out and (t.start_h < out.ready_h-EPS or out_done > out.due_h+EPS or t.outward_done_h != out_done):
            raise ValueError('Нарушен срок прямой заявки.')
        if t.backward_load_h < out_done-EPS or (back and t.backward_load_h < back.ready_h-EPS):
            raise ValueError('Обратная погрузка начинается слишком рано.')
        if abs(t.backward_depart_h-t.backward_load_h-(back.loading_h if back else 0)) > EPS:
            raise ValueError('Не учтена обратная погрузка.')
        back_done=t.backward_arrive_h+(back.unloading_h if back else 0)
        if back and (back_done > back.due_h+EPS or t.backward_done_h != back_done):
            raise ValueError('Нарушен срок обратной заявки.')
        if abs(t.finish_h-back_done-route.turnaround_h)>EPS or abs(t.wait_h-(t.backward_load_h-out_done))>EPS:
            raise ValueError('Неверное время занятости или ожидания.')
        loaded=route.distance_km*len(t.orders)
        empty=2*route.distance_km-loaded
        if any(abs(a-b)>1e-5 for a,b in ((t.loaded_km,loaded),(t.empty_km,empty),
            (t.movement_rub,loaded*v.loaded_rub_km+empty*v.empty_rub_km),
            (t.time_rub,(t.finish_h-t.start_h)*v.occupied_rub_h),(t.fixed_rub,v.fixed_rub))):
            raise ValueError('Неверный пробег или расчёт расходов.')
        seen.extend(t.orders)
    if sorted(seen) != sorted(od):
        raise ValueError('Не все заявки выполнены ровно один раз.')
    for v in vehicles:
        chain=sorted((t for t in trips if t.vehicle==v.id),key=lambda t:t.start_h)
        if any(a.finish_h>b.start_h+EPS for a,b in zip(chain,chain[1:])):
            raise ValueError('Рейсы одной машины пересекаются.')


def solve(orders, vehicles, route, *, paired=True, time_limit=20.):
    trips = candidates(orders, vehicles, route, paired)
    missing=[o.id for o in orders if not any(o.id in t.orders for t in trips)]
    if missing:
        return Plan('infeasible', [], 'Нет допустимого рейса для заявок: '+', '.join(missing)+
                    '. Проверьте кузов, массы, сроки, окно и шаг отправок.', len(trips))
    rr, cc, vv, lo, hi = [], [], [], [], []
    def constraint(indices, lower, upper):
        row=len(lo)
        rr.extend([row]*len(indices)); cc.extend(indices); vv.extend([1.]*len(indices))
        lo.append(lower); hi.append(upper)
        if len(vv)>2_000_000:
            raise ValueError('Задача слишком большая. Увеличьте шаг отправок.')
    for o in orders:
        constraint([i for i,t in enumerate(trips) if o.id in t.orders],1,1)
    # В каждый момент старта машина может участвовать не более чем в одном обороте.
    for v in vehicles:
        grouped={}
        for i,t in enumerate(trips):
            if t.vehicle==v.id:
                grouped.setdefault(t.start_h,[]).append(i)
        active, ends=set(), []
        for start,indices in sorted(grouped.items()):
            while ends and ends[0][0] <= start+EPS:
                _,i=heappop(ends); active.discard(i)
            for i in indices:
                active.add(i); heappush(ends,(trips[i].finish_h,i))
            if len(active)>1:
                constraint(sorted(active),0,1)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lo),len(trips))).tocsc()
    result=milp(np.array([t.cost_rub for t in trips]), integrality=np.ones(len(trips)),
                bounds=Bounds(0,1), constraints=LinearConstraint(matrix,lo,hi),
                options={'time_limit':time_limit,'mip_rel_gap':0.})
    if result.status==2:
        return Plan('infeasible', [], 'При заданном парке все заказы в срок не помещаются. Проверьте сроки и занятость машин.',len(trips))
    if result.x is None:
        return Plan('unknown', [], 'За отведённое время план не найден; это не доказательство невыполнимости.',len(trips))
    chosen=[t for t,x in zip(trips,result.x) if x>.5]
    try:
        audit_plan(chosen,orders,vehicles,route)
    except ValueError as error:
        return Plan('unknown', [], 'Результат решателя не прошёл проверку: '+str(error),len(trips))
    return Plan('optimal' if result.status==0 else 'feasible',
                sorted(chosen,key=lambda t:(t.start_h,t.vehicle)),
                '' if result.status==0 else 'Найден выполнимый план, оптимальность не доказана.',len(trips))


def compare(orders, vehicles, route):
    return {'separate':solve(orders,vehicles,route,paired=False),
            'combined':solve(orders,vehicles,route,paired=True)}


def fixed_plan_sensitivity(plans, orders):
    """Меняем стоимость часа для тех же графиков; не приписываем им новый оптимум."""
    if not all(p.feasible for p in plans.values()):
        return pd.DataFrame()
    a,b=(plans[k].summary(orders) for k in ('separate','combined'))
    return pd.DataFrame([{'Множитель стоимости часа':f,
        'Отдельные рейсы, ₽':a['movement_rub']+a['fixed_rub']+a['time_rub']*f,
        'С совмещением, ₽':b['movement_rub']+b['fixed_rub']+b['time_rub']*f,
        'Экономия, ₽':a['movement_rub']+a['fixed_rub']-b['movement_rub']-b['fixed_rub']+(a['time_rub']-b['time_rub'])*f}
        for f in (0.,.5,1.,2.)])
