"""Causal snow-depth features; ground reanalysis and river measurements stay separate."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from .paths import DATA,MODELS,PROCESSED
from .northeast import STATIONS

FEATURES=['snow_ground_cm','snow_ground_mean14_cm','snow_ground_change14_cm',
          'snow_on_ice_cm','snow_on_ice_age_days','snow_on_ice_observed']
METADATA=MODELS/'m1_snow.json'
LAG_DAYS=5
MAX_OBSERVATION_AGE=30


def _mtime(path):
    return path.stat().st_mtime_ns if path.exists() else 0


@lru_cache(maxsize=100)
def _ground(point_id,version):
    path=DATA/'snow_depth'/f'{point_id}.csv'
    if not path.exists():
        return pd.DataFrame({'date':pd.Series(dtype='datetime64[ns]'),'snow_depth_cm':pd.Series(dtype=float)})
    return pd.read_csv(path,usecols=['date','snow_depth_cm'],parse_dates=['date']).sort_values('date')


@lru_cache(maxsize=2)
def _measurements(version):
    path=PROCESSED/'hydro_snow_on_ice.csv'
    if not path.exists():
        return pd.DataFrame({'gauge_code':pd.Series(dtype=int),'hydro_year':pd.Series(dtype=int),
                             'date':pd.Series(dtype='datetime64[ns]'),'snow_on_ice_cm':pd.Series(dtype=float)})
    return pd.read_csv(path,parse_dates=['date'])


@lru_cache(maxsize=5000)
def _season(point_id,year,ground_version,measurement_version):
    start=pd.Timestamp(year-1,9,1);end=pd.Timestamp(year,9,1)
    g=_ground(point_id,ground_version)
    g=g[(g.date>=start)&(g.date<end)].dropna(subset=['snow_depth_cm'])
    obs=_measurements(measurement_version)
    if point_id.startswith('gauge_'):
        obs=obs[(obs.gauge_code==int(point_id.split('_')[1]))&(obs.hydro_year==year)].sort_values('date')
    else:
        obs=obs.iloc[:0]
    return ((g.date-start).dt.days.to_numpy(),g.snow_depth_cm.to_numpy(dtype=float),
            (obs.date-start).dt.days.to_numpy(),obs.snow_on_ice_cm.to_numpy(dtype=float))


def features(point_id,year,issue_day):
    """Backward-only observations; no interpolation from future snow measurements.

    Reanalysis has a nominal five-day release lag. Measurement dates are used
    in a research reconstruction: their historical publication times are unknown.
    An old season's snow is never used for a new season.
    """
    gv=_mtime(DATA/'snow_depth'/f'{point_id}.csv')
    ov=_mtime(PROCESSED/'hydro_snow_on_ice.csv')
    gd,gs,od,os=_season(point_id,int(year),gv,ov)
    n=np.searchsorted(gd,issue_day-LAG_DAYS,side='right')
    result=dict.fromkeys(FEATURES,np.nan)
    result['snow_on_ice_observed']=0.
    if n and issue_day-gd[n-1]<=LAG_DAYS+7:
        first=np.searchsorted(gd,gd[n-1]-13,side='left')
        result.update(snow_ground_cm=float(gs[n-1]),snow_ground_mean14_cm=float(np.mean(gs[first:n])),
                      snow_ground_change14_cm=float(gs[n-1]-gs[first]))
    k=np.searchsorted(od,issue_day,side='right')
    if k and issue_day-od[k-1]<=MAX_OBSERVATION_AGE:
        result.update(snow_on_ice_cm=float(os[k-1]),snow_on_ice_age_days=float(issue_day-od[k-1]),
                      snow_on_ice_observed=1.)
    # New official snow measurements must have an explicit date at THIS post
    # and must already have been available to the project on the issue date.
    # Regional snow depth and undated ice-card values are never substituted.
    from .official_hydro import latest_measurement, stamp
    cutoff=(pd.Timestamp(year-1,9,1)+pd.Timedelta(days=int(issue_day))).date()
    latest=latest_measurement(point_id,'snow_on_ice_cm',cutoff,max_age=MAX_OBSERVATION_AGE)
    if latest:
        observation_date=stamp(latest['observed_at']).date()
        age=(cutoff-observation_date).days
        if observation_date>=pd.Timestamp(year-1,9,1).date() and (
                not result['snow_on_ice_observed'] or age<=result['snow_on_ice_age_days']):
            result.update(snow_on_ice_cm=latest['value'],snow_on_ice_age_days=float(age),snow_on_ice_observed=1.)
    return result


@lru_cache(maxsize=2)
def _metadata(version):
    return json.loads(METADATA.read_text()) if METADATA.exists() else {}


@lru_cache(maxsize=8)
def _model(path,version):
    from catboost import CatBoostRegressor
    model=CatBoostRegressor();model.load_model(path)
    return model


def predict(point_id,year,issue_day,kind,base_features,*,use_snow=True):
    if not (point_id.startswith('gauge_') or point_id=='yakutsk_crossing'):
        return dict(message='Для спутниковых точек притоков Лены снежная модель не проверена; сохранена отдельная модель вскрытия.')
    info=_metadata(_mtime(METADATA))
    if not info:
        return dict(message='Модель толщины снега ещё не подготовлена.')
    bounds=info.get('issue_ranges',{}).get(kind,[0,365])
    if not bounds[0]<=issue_day<=bounds[1]:
        return dict(message='Дата решения вне периода обучения снежной модели для этого события; используется обычная М1.')
    candidates=[m for m in info['models'].get(kind,[]) if m['train_max_year']<year]
    if not candidates:
        return dict(message='Для этого года нет снежной модели, обученной только на прошлых сезонах.')
    chosen=max(candidates,key=lambda m:m['train_max_year'])
    f=features(point_id,year,issue_day)
    if use_snow and not np.isfinite(f['snow_ground_cm']) and not np.isfinite(f['snow_on_ice_cm']):
        return dict(message='На дату решения нет свежих данных о толщине снега; используется обычная М1.')
    groups=info.get('validation',{}).get(kind,{}).get('groups',{})
    group='satellite' if point_id=='yakutsk_crossing' else 'new_gauges' if int(point_id.split('_')[1]) in STATIONS else 'existing_gauges'
    quality=groups.get(group,{})
    if not f.get('snow_on_ice_observed',0):
        quality=quality.get('without_river',quality)
    if use_snow and point_id not in quality.get('validated_points',[]):
        return dict(message='На этом посту снежная поправка не прошла независимую проверку; используется обновлённая М1.')
    if use_snow and year>quality.get('last_year',year-1) and quality.get('improvement_pct',0)<=0:
        return dict(message='В проверке прошлых сезонов снег не снизил ошибку этого события; сохранён обновлённый прогноз без снежной поправки.')
    model_path=MODELS/chosen['file' if use_snow else 'baseline_file']
    if not model_path.exists():
        return dict(message='Файл снежной модели недоступен; используется обычная М1.')
    model=_model(str(model_path),_mtime(model_path))
    row=pd.DataFrame([{**base_features,**f}])
    columns=info['features'][kind] if use_snow else info['baseline_features'][kind]
    q=np.sort(np.asarray(model.predict(row[columns]))[0])+base_features['cal_target']
    correction=chosen.get('correction_days' if use_snow else 'baseline_correction_days',0.)
    q[0]-=correction;q[2]+=correction
    message=('Толщина снега учтена отдельной экспериментальной моделью. '
             'Снег на суше — ERA5-Land; замер на льду используется только при наличии свежей записи.'
             if use_snow else 'Обновлённая экспериментальная М1 без признаков толщины снега.')
    return dict(quantiles=np.maximum(q,issue_day),features=f,train_max_year=chosen['train_max_year'],
        model_file=model_path.name,
        validation=quality,message=message)
