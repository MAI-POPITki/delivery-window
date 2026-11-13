"""Hourly wind forecasts -> conservative daily handling capacity, not road permits."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .flood_store import atomic_json

API='https://api.open-meteo.com/v1/forecast'
DATA=Path(__file__).resolve().parents[1]/'data/handling_wind'


def fetch(lat,lon,*,directory=DATA):
    if not np.isfinite([lat,lon]).all() or not -90<=lat<=90 or not -180<=lon<=180:
        raise ValueError('Некорректные координаты терминала')
    response=requests.get(API,params=dict(latitude=lat,longitude=lon,
        hourly='wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation',wind_speed_unit='ms',timezone='GMT',
        forecast_days=7),timeout=(8,40))
    response.raise_for_status()
    payload=response.json()
    now=datetime.now(timezone.utc)
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'))
    envelope=dict(payload=payload,sha256=hashlib.sha256(raw.encode()).hexdigest(),
                  fetched_at=now.isoformat(),url=response.url,latitude=lat,longitude=lon,
                  source=API,kind='weather_forecast')
    parse(envelope,now)
    path=Path(directory)/f'{now:%Y%m%dT%H%M%S%f}-{envelope["sha256"][:12]}.json'
    atomic_json(path,envelope)
    return envelope


def parse(envelope,as_of):
    if not isinstance(envelope,dict) or not isinstance(envelope.get('payload'),dict):
        raise ValueError('Некорректный файл выпуска ветра')
    as_of=pd.Timestamp(as_of)
    if as_of.tzinfo is None:raise ValueError('Дата решения должна иметь часовой пояс')
    fetched=pd.Timestamp(envelope['fetched_at'])
    if fetched.tzinfo is None or fetched>as_of or as_of-fetched>pd.Timedelta(hours=24):
        raise ValueError('Выпуск ветра получен позже решения или старше 24 часов')
    p=envelope['payload']
    raw=json.dumps(p,sort_keys=True,separators=(',',':'))
    if hashlib.sha256(raw.encode()).hexdigest()!=envelope['sha256']:
        raise ValueError('Не совпала контрольная сумма прогноза ветра')
    if envelope.get('source')!=API or envelope.get('kind')!='weather_forecast':
        raise ValueError('Нужен сохранённый прогноз погоды, а не реанализ')
    if p.get('utc_offset_seconds')!=0 or p.get('timezone')!='GMT':
        raise ValueError('Ожидались исходные часы UTC')
    names=['wind_speed_10m','wind_gusts_10m']
    if any(p['hourly_units'].get(k)!='m/s' for k in names):
        raise ValueError('Ожидался ветер в м/с')
    for key,units in [('temperature_2m','°C'),('precipitation','mm')]:
        if key in p['hourly']:
            if p['hourly_units'].get(key)!=units:raise ValueError('Некорректные единицы температуры или осадков')
            names.append(key)
    frame=pd.DataFrame({k:p['hourly'][k] for k in names},index=pd.to_datetime(p['hourly']['time'],utc=True))
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Некорректные часы прогноза ветра')
    if not (frame.index==frame.index.floor('h')).all():raise ValueError('Ожидались целые часы')
    for key in names:frame[key]=pd.to_numeric(frame[key],errors='raise')
    if np.isinf(frame).any().any() or (frame[[k for k in names if k!='temperature_2m']]<0).any().any():
        raise ValueError('Некорректная скорость ветра или осадки')
    return frame


def capacity(envelope,days,*,as_of,tz,speed_limit,gust_limit,duration_h,batch_t,start_hour=0,end_hour=24,
             min_temperature=None,max_precipitation=None,max_operation_hours=24):
    """Nonoverlapping whole-hour batches within a local daily shift.

    Missing wind in any scheduled hour makes that day unknown (zero capacity).
    Values at either limit are excluded. Slots never cross the shift/day boundary.
    """
    if not np.isfinite([speed_limit,gust_limit,batch_t]).all() or min(speed_limit,gust_limit,batch_t)<=0:
        raise ValueError('Задайте положительные пределы ветра и объём партии')
    if isinstance(duration_h,bool) or int(duration_h)!=duration_h or not 1<=duration_h<=24:
        raise ValueError('Длительность операции — целое число часов от 1 до 24')
    if not 0<=start_hour<end_hour<=24 or int(start_hour)!=start_hour or int(end_hour)!=end_hour:
        raise ValueError('Некорректная рабочая смена')
    if not np.isfinite(max_operation_hours) or not 0<=max_operation_hours<=24:
        raise ValueError('Лимит часов операций должен быть от 0 до 24')
    if min_temperature is not None and not np.isfinite(min_temperature):raise ValueError('Некорректная температура')
    if max_precipitation is not None and (not np.isfinite(max_precipitation) or max_precipitation<0):
        raise ValueError('Некорректный предел осадков')
    zone=ZoneInfo(tz)
    frame=parse(envelope,as_of)
    rows=[]
    for day in days:
        midnight=pd.Timestamp(day).normalize().tz_localize(zone)
        finish=(pd.Timestamp(day).normalize()+pd.Timedelta(days=1)).tz_localize(zone)
        hours=pd.date_range(midnight,finish,freq='h',inclusive='left')
        hours=hours[(hours.hour>=start_hour)&(hours.hour<end_hour)]
        # Never allocate past hours in a current-day plan.
        hours=hours[hours.tz_convert('UTC')>=pd.Timestamp(as_of).ceil('h')]
        needed=['wind_speed_10m','wind_gusts_10m']
        if min_temperature is not None:needed.append('temperature_2m')
        if max_precipitation is not None:needed.append('precipitation')
        utc=hours.tz_convert('UTC')
        subset=frame.reindex(index=utc,columns=needed).copy()
        end_values=frame.reindex(index=utc+pd.Timedelta(hours=1),columns=needed)
        # Open-Meteo gusts and precipitation describe the preceding hour.
        # For [08,09) use the 09:00 interval value, not the 08:00 value.
        subset['wind_speed_10m']=np.maximum(subset.wind_speed_10m.to_numpy(),end_values.wind_speed_10m.to_numpy())
        subset['wind_gusts_10m']=end_values.wind_gusts_10m.to_numpy()
        if min_temperature is not None:
            subset['temperature_2m']=np.minimum(subset.temperature_2m.to_numpy(),end_values.temperature_2m.to_numpy())
        if max_precipitation is not None:subset['precipitation']=end_values.precipitation.to_numpy()
        known=bool(len(hours)) and subset.notna().all().all()
        allowed=((subset.wind_speed_10m<speed_limit)&(subset.wind_gusts_10m<gust_limit)).to_numpy(copy=True)
        if min_temperature is not None:allowed&=(subset.temperature_2m>=min_temperature).to_numpy()
        if max_precipitation is not None:allowed&=(subset.precipitation<=max_precipitation).to_numpy()
        run=batches=0
        for okay in allowed if known else []:
            run=run+1 if okay else 0
            if run==duration_h:batches+=1;run=0
        batches=min(batches,int(max_operation_hours//duration_h))
        rows.append(dict(date=pd.Timestamp(day).date(),known=bool(known),
                         working_hours=len(hours),safe_hours=int(allowed.sum()) if known else 0,
                         batches=batches,capacity_t=float(batches*batch_t)))
    result=pd.DataFrame(rows,columns=['date','known','working_hours','safe_hours','batches','capacity_t'])
    result.attrs['wind']=dict(fetched_at=envelope['fetched_at'],sha256=envelope['sha256'],
                              speed_limit=speed_limit,gust_limit=gust_limit,duration_h=duration_h,
                              batch_t=batch_t,tz=tz,start_hour=start_hour,end_hour=end_hour,
                              min_temperature=min_temperature,max_precipitation=max_precipitation,
                              max_operation_hours=max_operation_hours)
    return result


def constrain(calendar,loading,unloading):
    result=calendar.copy()
    load=loading.set_index('date');unload=unloading.set_index('date')
    a=load.reindex(result.date).reset_index(drop=True)
    b=unload.reindex(result.arrival if 'arrival' in result else result.date).reset_index(drop=True)
    known=a.known.fillna(False).astype(bool)&b.known.fillna(False).astype(bool)
    result['capacity_before_wind_t']=result.capacity_t
    result['wind_known']=known.to_numpy()
    result['loading_wind_t']=a.capacity_t.fillna(0).to_numpy()
    result['unloading_wind_t']=b.capacity_t.fillna(0).to_numpy()
    result['capacity_t']=np.minimum.reduce([pd.to_numeric(result.capacity_t,errors='coerce').fillna(0).to_numpy(),
                                            result.loading_wind_t.to_numpy(),result.unloading_wind_t.to_numpy()])
    result.loc[~result.wind_known,'capacity_t']=0.
    for label,table in [('load',loading),('unload',unloading)]:
        for field,value in table.attrs.get('wind',{}).items():
            result['wind_'+label+'_'+field]=value
    return result
