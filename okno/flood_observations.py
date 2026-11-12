"""Fresh post observations, separate from immutable flood M1 training archives."""
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from . import official_hydro as H
from .flood_sites import SITES
from .flood_store import atomic_json, read_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/flood/live_levels'
LEVEL_API='https://level-satellite.ru/api/series'
METEO_API='https://meteo-dv.ru/hydro_dfo/cgi/get_hydro_graph.php'


def station(pid):
    if pid not in SITES:raise ValueError('Неизвестный участок')
    meta=json.loads((ROOT/f'data/flood/levels/{pid}.manifest.json').read_text(encoding='utf-8'))
    return meta['station']


def record(pid,when,value,source,url,digest,now,precision='instant'):
    value=H.number(value,'water_level_cm')
    if value is None:return None
    observed=H.stamp(when)
    if observed>now:return None
    identity=f'{pid}|{source}|{observed.isoformat()}|{value}'
    return dict(id=hashlib.sha256(identity.encode()).hexdigest()[:24],pid=pid,
                station_code=str(station(pid)['properties']['post_code']),value_cm=value,
                observed_at=observed.isoformat(),observation_precision=precision,
                available_at=now.isoformat(),first_seen_at=now.isoformat(),
                source=source,source_url=url,raw_sha256=digest,units='cm above gauge zero')


def parse_levels(pid,payload,url,digest,now):
    if not isinstance(payload,dict) or not isinstance(payload.get('level'),list):
        raise ValueError('Изменилась схема измерений уровня')
    rows=[]
    # SWOT is a different measurement product and zero: it is never substituted.
    for row in payload['level']:
        if not isinstance(row,dict) or not {'d','v','source'}<=row.keys():
            raise ValueError('Некорректная запись уровня')
        if row['source'] not in ('gmvo','vodinfo'):continue
        value=record(pid,row['d'],row['v'],'level_satellite:'+row['source'],url,digest,now,'day')
        if value:rows.append(value)
    return rows


def parse_meteo(pid,payload,url,digest,now):
    if pid!='lena_yakutsk':raise ValueError('Нет проверенной привязки Метео ДВ')
    point=H.registry()['yakutsk_crossing']
    if str(point['station_code'])!=str(station(pid)['properties']['post_code']):
        raise ValueError('Разошлись индексы гидропоста')
    from .hydro_ingest import parse_graph
    original=parse_graph(payload,point,now,url,digest)
    return [r for row in original if (r:=record(pid,row['observed_at'],row['value'],
            'meteo_dv',url,digest,now)) is not None]


def save_raw(directory,raw):
    digest=hashlib.sha256(raw).hexdigest()
    folder=directory/'raw';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{digest}.gz'
    if not path.exists():path.write_bytes(gzip.compress(raw,mtime=0))
    return digest


def merge(old,new):
    rows={r['id']:r for r in old}
    for value in new:
        if value['id'] in rows:
            prior=rows[value['id']]
            value={**value,'available_at':prior['available_at'],'first_seen_at':prior['first_seen_at']}
        rows[value['id']]=value
    return sorted(rows.values(),key=lambda r:(r['observed_at'],r['available_at'],r['id']))


def refresh(pid,*,directory=DATA,now=None):
    live_clock=now is None
    now=now or datetime.now(timezone.utc)
    directory=Path(directory)
    path=directory/f'{pid}.json'
    previous=read_json(path)
    code=str(station(pid)['properties']['post_code'])
    jobs=[('level_satellite',LEVEL_API,'POST',dict(post_code=code,
           date_from=(now-timedelta(days=45)).date().isoformat(),date_to=now.date().isoformat()),parse_levels)]
    if pid=='lena_yakutsk':
        jobs.insert(0,('meteo_dv',METEO_API,'GET',dict(station_code=code,
                    date_loc=now.date().isoformat()+' 00:00',time_offset=0),parse_meteo))
    new=[];statuses=[]
    for source,url,method,params,parser in jobs:
        try:
            kwargs={'json':params} if method=='POST' else {'params':params}
            response=requests.request(method,url,timeout=(8,35),**kwargs)
            response.raise_for_status()
            received=datetime.now(timezone.utc) if live_clock else now
            digest=save_raw(directory,response.content)
            rows=parser(pid,response.json(),response.url,digest,received)
            new.extend(rows)
            statuses.append(dict(source=source,state='ok' if rows else 'no_data',records=len(rows),
                                 raw_sha256=digest,url=response.url,request=params))
        except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
            statuses.append(dict(source=source,state='error',error=type(exc).__name__))
    if all(row['state']=='error' for row in statuses):
        raise RuntimeError('Источники измерений не ответили')
    rows=merge(previous.get('records',[]),new)
    completed=datetime.now(timezone.utc) if live_clock else now
    snapshot=dict(schema_version=1,pid=pid,station=station(pid),checked_at=completed.isoformat(),
                  last_success_at=completed.isoformat() if new else previous.get('last_success_at'),
                  records=rows,sources=statuses)
    atomic_json(path,snapshot)
    return dict(state='ok' if new else 'no_data',records=len(rows),
                observed_at=max((r['observed_at'] for r in rows),default=None),
                version=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())


def observations(pid,*,directory=DATA,now=None):
    now=now or datetime.now(timezone.utc)
    rows=read_json(Path(directory)/f'{pid}.json').get('records',[])
    if not rows and pid=='lena_yakutsk':
        rows=[record(pid,r['observed_at'],r['value'],r['source'],r['source_url'],r['raw_sha256'],
                     H.stamp(r['available_at'])) for r in H.observations('yakutsk_crossing',as_of=now)
              if r['metric']=='water_level_cm' and r['quality']=='measured']
    return [r for r in rows if r and H.stamp(r['observed_at'])<=now and H.stamp(r['available_at'])<=now]


def backfill_meteo(pid='lena_yakutsk',*,days=31,directory=DATA,now=None):
    """Read prior weekly guest graphs; caller must hold the regular levels lock.

    This adds observed levels only. It never changes model archives or starts ML.
    """
    if pid!='lena_yakutsk':raise ValueError('Нет проверенной привязки Метео ДВ')
    if not isinstance(days,int) or not 7<=days<=93:raise ValueError('Ожидается от 7 до 93 суток')
    live_clock=now is None
    now=now or datetime.now(timezone.utc)
    directory=Path(directory)
    path=directory/f'{pid}.json'
    previous=read_json(path)
    rows=[];sources=[]
    code=str(station(pid)['properties']['post_code'])
    for offset in range(0,days,7):
        date=(now-timedelta(days=offset)).date().isoformat()
        params=dict(station_code=code,date_loc=date+' 00:00',time_offset=0)
        try:
            response=requests.get(METEO_API,params=params,timeout=(8,35))
            response.raise_for_status()
            received=datetime.now(timezone.utc) if live_clock else now
            digest=save_raw(directory,response.content)
            parsed=parse_meteo(pid,response.json(),response.url,digest,received)
            rows.extend(parsed)
            sources.append(dict(source='meteo_dv_backfill',state='ok' if parsed else 'no_data',
                                records=len(parsed),url=response.url,raw_sha256=digest,request=params,
                                fetched_at=received.isoformat()))
        except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
            sources.append(dict(source='meteo_dv_backfill',state='error',request=params,error=type(exc).__name__))
            if isinstance(exc,requests.HTTPError) and exc.response is not None and exc.response.status_code==429:
                break
    if not rows:raise RuntimeError('Датированных замеров при догрузке не получено; прежний ряд сохранён')
    merged=merge(previous.get('records',[]),rows)
    completed=datetime.now(timezone.utc) if live_clock else now
    # Keep every prior request provenance; deduplicate repeated successful captures.
    history=previous.get('backfill_sources',[])
    captures={json.dumps(s,sort_keys=True):s for s in [*history,*sources]}
    atomic_json(path,{**previous,'schema_version':1,'pid':pid,'station':station(pid),
                     'records':merged,'backfill_sources':list(captures.values()),
                     'last_backfill_at':completed.isoformat()})
    state=summary(pid,directory=directory,now=completed)
    return dict(records=len(merged),days=state['days'],continuous=state['continuous'],
                latest=state['latest']['observed_at'],failed_windows=sum(s['state']=='error' for s in sources))


def summary(pid,*,directory=DATA,now=None):
    now=now or datetime.now(timezone.utc)
    rows=observations(pid,directory=directory,now=now)
    if not rows:return dict(latest=None,days=0,continuous=False,model_ready=False)
    latest=max(rows,key=lambda r:(r['observed_at'],r['available_at']))
    same=[r for r in rows if r['source']==latest['source']]
    days={H.stamp(r['observed_at']).date() for r in same}
    end=H.stamp(latest['observed_at']).date()
    expected={end-timedelta(days=k) for k in range(31)}
    model=ROOT/f'models/flood_m1_quantile/{pid}_{now.year}.json'
    return dict(latest=latest,days=len(days&expected),continuous=expected<=days,
                age_days=(now-H.stamp(latest['observed_at'])).total_seconds()/86400,
                model_ready=model.exists(),units='cm above gauge zero')
