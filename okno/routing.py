"""Дорожные маршруты OSM и проверка явных ограничений без вымышленных вероятностей."""
import datetime as dt
import math
import csv
from pathlib import Path
import threading
import time

import requests

HEADERS={'User-Agent':'delivery-window-MVP/1.0 (educational route planning)'}
_lock=threading.Lock()
_last=0.


def _get(url,params):
    global _last
    # Ограничение публичных сервисов: не более одного запроса в секунду на процесс.
    with _lock:
        time.sleep(max(0.,1.1-(time.monotonic()-_last)))
        try:
            response=requests.get(url,params=params,headers=HEADERS,timeout=35)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException,ValueError) as error:
            raise ValueError('Картографический сервис не ответил. Повторите позже; расстояние не заменено прямой линией.') from error
        finally:
            _last=time.monotonic()


def search_places(query):
    if not 2<=len(query.strip())<=150:
        raise ValueError('Введите название населённого пункта (2–150 символов).')
    q=query.strip().casefold().replace('ё','е')
    with (Path(__file__).resolve().parents[1]/'data/osm/settlements_ru.csv').open(encoding='utf-8') as stream:
        rows=[x for x in csv.DictReader(stream) if q in x['search'].casefold().replace('ё','е') or q in x['name'].casefold().replace('ё','е')]
    rows.sort(key=lambda x:(x['name'].casefold()!=q,-int(x['population'])))
    return [dict(name=f"{x['name']} ({float(x['lat']):.3f}, {float(x['lon']):.3f})",lat=float(x['lat']),lon=float(x['lon'])) for x in rows[:10]]


def fetch_routes(points):
    if not 2<=len(points)<=6:
        raise ValueError('Нужно от 2 до 6 точек по порядку движения.')
    for p in points:
        if not all(math.isfinite(p[k]) for k in ('lat','lon')) or not -90<=p['lat']<=90 or not -180<=p['lon']<=180:
            raise ValueError('Проверьте координаты точек.')
    coordinates=';'.join(f"{p['lon']:.6f},{p['lat']:.6f}" for p in points)
    url='https://routing.openstreetmap.de/routed-car/route/v1/driving/'+coordinates
    data=_get(url,dict(alternatives='true',steps='true',geometries='geojson',overview='full',
                       radiuses=';'.join(['1000']*len(points))))
    if data.get('code')!='Ok' or not data.get('routes'):
        raise ValueError('Сервис не нашёл дорожный маршрут. Проверьте точки или добавьте промежуточные. Проезд по зимнику может отсутствовать в графе.')
    result=[]
    for i,r in enumerate(data['routes'][:3]):
        if r['distance']<=0 or r['duration']<=0:
            continue
        steps=[s for leg in r['legs'] for s in leg.get('steps',[])]
        result.append(dict(id=i+1,distance_km=r['distance']/1000,travel_h=r['duration']/3600,
            coordinates=r['geometry']['coordinates'],ferries=sum(s.get('mode')=='ferry' for s in steps),
            roads=sorted({s.get('name','') for s in steps if s.get('name')}),
            fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),source=url,
            data_version=data.get('data_version','не указана сервисом')))
    if not result:
        raise ValueError('Маршрут имеет нулевую длину. Выберите разные точки.')
    return result


def _intersects(a,b,box):
    """Пересечение отрезка с прямоугольником: учитывает пересечение между вершинами."""
    xmin,ymin,xmax,ymax=box
    dx,dy=b[0]-a[0],b[1]-a[1]
    lo,hi=0.,1.
    for p,q in ((-dx,a[0]-xmin),(dx,xmax-a[0]),(-dy,a[1]-ymin),(dy,ymax-a[1])):
        if abs(p)<1e-12:
            if q<0: return False
        elif p<0: lo=max(lo,q/p)
        else: hi=min(hi,q/p)
        if lo>hi: return False
    return True


def assess(route, departure, deadline, gross_t, restrictions, factors=(1.,1.25,1.5),time_factor=1.):
    """Сценарии времени — условия пользователя, не вероятностный прогноз недовоза."""
    departure=dt.datetime.fromisoformat(str(departure))
    deadline=dt.datetime.fromisoformat(str(deadline))
    if deadline<=departure or gross_t<=0 or not math.isfinite(gross_t) or not math.isfinite(time_factor) or time_factor<1:
        raise ValueError('Проверьте срок, полную массу и поправку времени (не менее 1).')
    if not factors or any(not math.isfinite(f) or f<1 for f in factors):
        raise ValueError('Множители времени должны быть не меньше 1.')
    blocked=[]; unknown=[]; delay=0.; matched=[]
    if route.get('ferries',0):
        unknown.append('Паром на маршруте: расписание и допуск машины не проверены')
    # Для временных ограничений консервативно рассматриваем весь интервал до крайнего срока.
    for r in restrictions:
        box=[float(r[k]) for k in ('min_lon','min_lat','max_lon','max_lat')]
        if not all(math.isfinite(x) for x in box) or not (-180<=box[0]<=box[2]<=180 and -90<=box[1]<=box[3]<=90):
            raise ValueError('Проверьте границы зоны ограничения.')
        start,end,published=(dt.datetime.fromisoformat(str(r[k])) for k in ('start','end','published'))
        if end<=start or not str(r['source']).strip() or r['kind'] not in ('closed','mass','delay','unknown'):
            raise ValueError('У ограничения нужны период, дата публикации, источник и тип closed/mass/delay/unknown.')
        value=float(r.get('value',0))
        if not math.isfinite(value) or value<0:
            raise ValueError('Значение ограничения не может быть отрицательным.')
        if not any(_intersects(a,b,box) for a,b in zip(route['coordinates'],route['coordinates'][1:])):
            continue
        if end<=departure or start>=deadline: continue
        name=str(r.get('name','Ограничение')); matched.append(dict(r))
        known_at=min(departure,dt.datetime.now()) if departure.tzinfo is None else departure
        if published>known_at:
            unknown.append(name+': опубликовано после отправки'); continue
        if r['kind']=='closed' or (r['kind']=='mass' and gross_t>value): blocked.append(name)
        elif r['kind']=='unknown': unknown.append(name)
        elif r['kind']=='delay': delay+=value
    available_h=(deadline-departure).total_seconds()/3600
    hours=route['travel_h']*time_factor
    durations=[hours*f+delay for f in factors]
    failures=sum(h>available_h for h in durations)
    return dict(blocked=blocked,unknown=unknown,matched=matched,delay_h=delay,
        scenarios=len(factors),late_scenarios=failures,slack_h=available_h-max(durations),
        base_h=hours+delay,stress_h=max(durations),
        rank=(bool(blocked),len(unknown),failures,-(available_h-max(durations)),route['distance_km']))


def rank_routes(routes,**kwargs):
    return sorted([dict(route=r,assessment=assess(r,**kwargs)) for r in routes],key=lambda x:x['assessment']['rank'])
