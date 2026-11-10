"""Связь маршрута, реестра дорог и прогнозов карточки. Факты не заменяются моделью."""
import datetime as dt
import math
import re
from pathlib import Path
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from .external_data import extract_point_id
from .ice import to_season_day
from .planner import sample_quantiles

ROOT=Path(__file__).resolve().parents[1]


_BARE_POINT=re.compile(r"\b(gauge_\d+|yakutsk_crossing|lena_\d+)\b")


def _point_of(comment):
    """point_id=<id>; в графике 2026/27 точка записана без тега («ближайший гидропост: gauge_3414»)."""
    tagged=extract_point_id(comment)
    if tagged:
        return tagged
    m=_BARE_POINT.search(comment) if isinstance(comment,str) else None
    return m.group(1) if m else None


@lru_cache(maxsize=4)
def _registry(version):
    """Реестр фактов и опубликованных графиков: road_events.csv + график текущего сезона (road_schedule_*.csv)."""
    parts=[pd.read_csv(ROOT/'data/external/road_events.csv').fillna('')]
    for path in sorted((ROOT/'data/external').glob('road_schedule_*.csv')):
        parts.append(pd.read_csv(path).fillna(''))
    frame=pd.concat(parts,ignore_index=True)
    frame['point_id']=frame.comment.map(_point_of)
    return frame


def _registry_version():
    files=[ROOT/'data/external/road_events.csv',*sorted((ROOT/'data/external').glob('road_schedule_*.csv'))]
    return tuple(f.stat().st_mtime_ns for f in files if f.exists())


def road_rows(point_id,hydro_year):
    frame=_registry(_registry_version())
    return frame[(frame.point_id==point_id)&(frame.hydro_year==hydro_year)].drop(columns='point_id').copy()


def planned_opening(point_id,hydro_year,decision):
    """Самое раннее открытие дорог точки в сезоне, известное на дату решения: (день сезона, пояснение) или None.

    Факт используется, если он уже опубликован; иначе опубликованный график. Ничего не додумывается."""
    decision=pd.Timestamp(decision).normalize()
    frame=road_rows(point_id,hydro_year)
    best=None
    for _,row in frame[frame.kind.isin(['open','open10'])].iterrows():
        for basis,effective,published in (('факт','actual_date','actual_published'),('график','announced_date','announced_published')):
            when=pd.to_datetime(row[effective],errors='coerce');known=pd.to_datetime(row[published],errors='coerce')
            if pd.isna(when) or pd.isna(known) or known>decision: continue
            if basis=='факт' and when>decision: continue
            if best is None or when<best[0]: best=(when,basis,row['road'])
            break
    if best is None: return None
    return int(to_season_day(best[0],hydro_year)),f"{best[1]} открытия {best[0]:%d.%m.%Y} — {best[2]} (реестр)"


def registry_coverage(point_id=None,hydro_year=None):
    """Какие точки и сезоны вообще есть в реестре — для подсказки «где данные есть»."""
    frame=_registry(_registry_version())
    frame=frame[frame.point_id.notna()]
    if point_id is not None:
        frame=frame[frame.point_id==point_id]
    if hydro_year is not None:
        frame=frame[frame.hydro_year==hydro_year]
    return frame


def evidence(frame,road,decision,passage,gross_t):
    """На дату проезда учитываются только сведения, опубликованные к решению."""
    decision=pd.Timestamp(decision).normalize();passage=pd.Timestamp(passage).normalize()
    records=[]
    for _,row in frame[frame.road.eq(road)].iterrows():
        # Если факт события уже опубликован к дате решения, прежний график этого же события не используется.
        _fact=pd.to_datetime(row['actual_date'],errors='coerce');_fact_pub=pd.to_datetime(row['actual_published'],errors='coerce')
        fact_known=not pd.isna(_fact) and not pd.isna(_fact_pub) and _fact_pub<=decision and _fact<=decision and bool(row['source_actual'])
        for prefix,effective,published,source in (
            ('Факт','actual_date','actual_published','source_actual'),
            ('Объявленный график','announced_date','announced_published','source_announced')):
            when=pd.to_datetime(row[effective],errors='coerce');known=pd.to_datetime(row[published],errors='coerce')
            if pd.isna(when) or pd.isna(known) or known>decision or not row[source]: continue
            if fact_known and prefix=='Объявленный график': continue
            # Датированное сообщение о будущем событии — график, а не наблюдение будущего.
            kind=prefix if when<=decision else 'Объявленный график'
            records.append(dict(kind=row.kind,date=when,publication=known,source=row[source],
                                basis=kind,comment=row.comment))
    applicable=[r for r in records if r['date']<=passage]
    if not applicable:
        return dict(status='unknown',limit_t=None,record=None,records=records)
    last_date=max(r['date'] for r in applicable)
    latest=[r for r in applicable if r['date']==last_date]
    facts=[r for r in latest if r['basis']=='Факт']
    latest=facts or latest
    if len({r['kind'] for r in latest})!=1:
        return dict(status='unknown',limit_t=None,record=None,records=records)
    row=max(latest,key=lambda r:r['publication'])
    limit={'limit20':20.,'limit30':30.,'open10':10.}.get(row['kind'])  # open10 — открытие до 10 т полной массы
    status='closed' if row['kind']=='close' else 'overweight' if limit and gross_t>limit else 'limited' if limit else 'mass_unknown'
    return dict(status=status,limit_t=limit,record=row,records=records)


def model_risk(q,passage,decision,hydro_year,margin,gross_t):
    """Условная вероятность физического закрытия с заданным запасом; не допуск проезда."""
    if not q or 'error' in q or gross_t>20:
        return None
    issue=float(q.get('issue_day',math.inf))
    if issue>to_season_day(decision,hydro_year): return None
    quantiles=[q.get(k,float('nan')) for k in ('q10','q50','q90')]
    if not all(math.isfinite(x) for x in quantiles) or quantiles!=sorted(quantiles): return None
    samples=sample_quantiles(*quantiles,4000,np.random.default_rng(0))-margin
    stamp=pd.Timestamp(passage)
    day=float(to_season_day(stamp,hydro_year))+(stamp-stamp.normalize()).total_seconds()/86400
    return float((samples<day).mean())


def assess_project(route,context,departure,gross_t):
    """Привязка только по явно выбранному сезонному участку, а не ближайшему посту."""
    binding=route.get('project_binding')
    if not binding or binding['point_id']!=context['pid'] or binding['road']!=context.get('road'):
        return dict(covered=False,status='unlinked',risk=None,record=None)
    passage=pd.Timestamp(departure)+pd.Timedelta(hours=route['travel_h'])
    facts=evidence(context['events'],binding['road'],context['issue_date'],passage,gross_t)
    risk=model_risk(context.get('close_forecast'),passage,context['issue_date'],context['hy'],context['margin'],gross_t)
    return dict(covered=True,status=facts['status'],risk=risk,record=facts['record'],passage=str(passage),
                forecast_issue=(context.get('close_forecast') or {}).get('issue_day'),limit_t=facts['limit_t'])


def distance_km(coordinates):
    def hav(a,b):
        la,lb=math.radians(a[1]),math.radians(b[1]);dlat=lb-la;dlon=math.radians(b[0]-a[0])
        return 6371*2*math.asin(min(1.,math.sqrt(math.sin(dlat/2)**2+math.cos(la)*math.cos(lb)*math.sin(dlon/2)**2)))
    return sum(hav(a,b) for a,b in zip(coordinates,coordinates[1:]))


def seasonal_segment(point_id):
    path=ROOT/'data/osm/project_segments.json'
    return next((s for s in json.loads(path.read_text(encoding='utf-8')) if s['point_id']==point_id),None)


def seasonal_route(points,segment,speed_kmh,fetch):
    """Дорожные подъезды + сохранённая трасса зимней переправы. Скорость задаёт оператор."""
    if not math.isfinite(speed_kmh) or speed_kmh<=0: raise ValueError('Укажите положительную скорость на сезонном участке.')
    if len(points)!=2: raise ValueError('Для сезонного варианта пока нужны А и Б без промежуточных точек.')
    coordinates=segment['coordinates']
    # Выбираем направление участка по суммарной длине подходов, а не меняем геометрию.
    pos=lambda p:[p['lon'],p['lat']]
    if distance_km([pos(points[0]),coordinates[-1]])+distance_km([coordinates[0],pos(points[1])])<distance_km([pos(points[0]),coordinates[0]])+distance_km([coordinates[-1],pos(points[1])]):
        coordinates=list(reversed(coordinates))
    access=[]
    for a,b in ((points[0],dict(lon=coordinates[0][0],lat=coordinates[0][1])),
                (dict(lon=coordinates[-1][0],lat=coordinates[-1][1]),points[1])):
        if distance_km([pos(a),pos(b)])<.03:
            access.append(dict(distance_km=0.,travel_h=0.,coordinates=[pos(a),pos(b)],source='Конец участка',ferries=0))
        else: access.append(fetch([a,b])[0])
    winter_km=distance_km(coordinates)
    gaps=distance_km([access[0]['coordinates'][-1],coordinates[0]])+distance_km([coordinates[-1],access[1]['coordinates'][0]])
    if gaps>.25:
        raise ValueError('Подъезды не состыкованы с трассой переправы (разрыв больше 250 м). Сезонный вариант не добавлен.')
    return dict(id=100,distance_km=sum(r['distance_km'] for r in access)+winter_km+gaps,
                travel_h=sum(r['travel_h'] for r in access)+(winter_km+gaps)/speed_kmh,
                coordinates=access[0]['coordinates']+coordinates+access[1]['coordinates'],
                ferries=sum(r.get('ferries',0) for r in access),roads=[segment['road']],
                project_binding={'point_id':segment['point_id'],'road':segment['road']},
                source=segment['source'],fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                data_version='Геометрия проекта + текущие подъезды OSM',
                speed_input_kmh=speed_kmh,geometry_note=segment['note'],access_gap_km=gaps)


def route_calendar(frame,road,decision,base,horizon,gross_t):
    """В парк передаётся календарь из записей: неизвестные дни не открываются автоматически."""
    values=[];sources=[]
    for day in range(horizon):
        result=evidence(frame,road,decision,pd.Timestamp(base)+pd.Timedelta(days=day),gross_t)
        values.append(result['limit_t'] if result['status'] in ('limited','overweight') else None)
        if result['record']: sources.append(result['record']['source'])
    return values,sorted(set(sources))


def weekly_access(frame,road,decision,hydro_year,gross_t):
    """Допуск на всю длительность недельной отправки; статус не смешивается с вероятностью М1."""
    base=pd.Timestamp(hydro_year-1,9,1)
    @lru_cache(maxsize=500)
    def allowed(day):
        return evidence(frame,road,decision,base+pd.Timedelta(days=day),gross_t)['status']=='limited'
    def mask(departures,arrivals):
        result=[]
        for departure,arrival in zip(departures,arrivals):
            days=range(int(math.floor(departure)),max(int(math.floor(departure))+1,int(math.ceil(arrival-1e-9))))
            result.append(all(allowed(day) for day in days))
        return np.asarray(result,dtype=bool)
    return mask
