"""Incremental ERA5-Land snow depth, plus weather for new PANGAEA gauges.

Daily snow_depth_mean is in metres. It describes modelled ground snow, not a
measurement on the river. ERA5 alone has no snow-depth field; use era5_seamless.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import requests
from okno.paths import DATA,PROCESSED,WEATHER
from okno.northeast import STATIONS
from okno.regions import all_gauges

API='https://archive-api.open-meteo.com/v1/archive'


def normalise_daily(payload,start,end):
    """Reject incomplete/mislabelled responses before persisting daily data."""
    frame=pd.DataFrame(payload['daily']).rename(columns={'time':'date'})
    dates=pd.to_datetime(frame.date,errors='raise')
    expected=pd.date_range(start,end)
    if not pd.DatetimeIndex(dates).equals(expected):
        raise ValueError('Snow-depth dates are incomplete, duplicated or unordered')
    if payload['daily_units'].get('snow_depth_mean')!='m':
        raise ValueError('Unexpected snow depth units')
    if frame.snow_depth_mean.notna().mean()<.95:
        raise ValueError('Snow depth coverage below 95%')
    if (frame.snow_depth_mean.dropna()<0).any():
        raise ValueError('Negative snow depth')
    return frame


def write_manifest():
    """Keep constant provenance once per collection, not on every daily row."""
    directory=DATA/'snow_depth'
    gauges=pd.read_csv(PROCESSED/'gauges.csv').set_index('gauge_code')
    files={}
    for path in sorted(directory.glob('*.csv')):
        frame=pd.read_csv(path,usecols=['date','snow_depth_cm'])
        if path.stem=='yakutsk_crossing':
            lat,lon=62.,129.82
        else:
            g=gauges.loc[int(path.stem.split('_')[1])];lat,lon=float(g.lat),float(g.lon)
        # Compact legacy files carrying the same constants on every row.
        if list(pd.read_csv(path,nrows=0).columns)!=['date','snow_depth_cm']:
            frame.to_csv(path,index=False,float_format='%.3f')
        files[path.name]=dict(latitude=lat,longitude=lon,first_date=frame.date.min(),
            last_date=frame.date.max(),rows=len(frame),missing_depth=int(frame.snow_depth_cm.isna().sum()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    metadata=dict(api=API,documentation='https://open-meteo.com/en/docs/historical-weather-api',
        models='era5_seamless',variable='snow_depth_mean',original_units='m',stored_units='cm',
        timezone='UTC',nominal_release_lag_days=5,
        description='Modelled snow depth on land from ERA5-Land; not a measurement on river ice.',
        files=files)
    (directory/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


def update_point(row,start='1979-01-01',end=None):
    end=end or (dt.date.today()-dt.timedelta(days=7)).isoformat()
    pid=f'gauge_{row.gauge_code}' if row.gauge_code else 'yakutsk_crossing'
    out=DATA/'snow_depth'/f'{pid}.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    weather=WEATHER/'gauges'/f'{pid}.csv'
    old=pd.read_csv(out,usecols=['date','snow_depth_cm']) if out.exists() else None
    if old is not None:
        start=max(start,(pd.Timestamp(old.date.max())+pd.Timedelta(days=1)).date().isoformat())
    need_weather=row.gauge_code in STATIONS and not weather.exists()
    if need_weather:
        # >15 years of preceding climate for the 2013–2018 holdout, without
        # repeatedly requesting decades unused by these newer-station tests.
        start='1995-01-01'
    if start>end:
        return f'{pid}: current'
    variables=['snow_depth_mean']
    if need_weather:
        variables += ['temperature_2m_mean','snowfall_sum','precipitation_sum']
    params=dict(latitude=row.lat,longitude=row.lon,start_date=start,end_date=end,
                daily=','.join(variables),models='era5_seamless',timezone='UTC')
    for attempt in range(8):
        try:
            r=requests.get(API,params=params,timeout=100)
            if r.status_code==429 or r.status_code>=500:
                reason=r.json().get('reason','') if r.headers.get('Content-Type','').startswith('application/json') else ''
                print(f'{pid}: HTTP {r.status_code} {reason}, retry {attempt+1}',flush=True)
                remaining=max(1,int(r.headers.get('Retry-After',min(45,10*(attempt+1)))))
                # Respect the provider's hourly quota. Individual sleeps stay
                # short so a background update can be stopped normally.
                while remaining:
                    pause=min(45,remaining);time.sleep(pause);remaining-=pause
                continue
            r.raise_for_status();payload=r.json()
            frame=normalise_daily(payload,start,end)
            if need_weather:
                wf=frame.drop(columns='snow_depth_mean').copy();wf.insert(0,'point_id',pid)
                wf['source']='Open-Meteo ERA5-Seamless (ERA5-Land temperature)'
                weather.parent.mkdir(parents=True,exist_ok=True);wf.to_csv(weather,index=False)
            snow=pd.DataFrame(dict(date=frame.date,snow_depth_cm=frame.snow_depth_mean*100))
            if old is not None:
                snow=pd.concat([old,snow],ignore_index=True).drop_duplicates('date',keep='last').sort_values('date')
            snow.to_csv(out,index=False,float_format='%.3f')
            return f'{pid}: {len(snow)} snow days, weather={need_weather}'
        except (requests.RequestException,KeyError,ValueError) as exc:
            if attempt==7:
                raise RuntimeError(f'{pid}: {exc}') from exc
            print(f'{pid}: {exc}, retry',flush=True)
            time.sleep(5)
    raise RuntimeError(f'{pid}: download failed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recent',action='store_true',help='Incremental update of existing files; full series only when missing')
    parser.add_argument('--codes',nargs='*',type=int)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    gauges=pd.read_csv(PROCESSED/'gauges.csv')
    gauges=gauges[gauges.gauge_code.isin(args.codes or all_gauges())]
    rows=list(gauges.itertuples())
    if not args.codes:
        from types import SimpleNamespace
        rows.append(SimpleNamespace(gauge_code=0,lat=62.,lon=129.82))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(update_point,rows):
            print(result,flush=True)
    write_manifest()


if __name__=='__main__':
    main()
