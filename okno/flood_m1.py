"""Собственный М1: прогноз уровня по прошлым измерениям конкретного гидропоста."""
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/flood/levels'
MODELS=ROOT/'models/flood_m1'
STATIONS={'lena_yakutsk':('3045','gmvo'),'ishim_ishim':('11412','gmvo'),
          'ural_orenburg':('19063','vodinfo'),'ural_orsk':('19054','vodinfo')}
FEATURES=['h0','h1','h3','h7','h14','h30','d1','d3','d7','d14','mean7','mean30','sin','cos','lead']
METHODS=['catboost','persistence','calendar','last_year']


def load_levels(pid):
    if pid not in STATIONS:
        raise ValueError('Для этой точки недостаточно однородных наблюдений; М1 не обучена')
    path=DATA/f'{pid}.json.gz'
    return _levels(pid,path.stat().st_mtime_ns,(DATA/f'{pid}.manifest.json').stat().st_mtime_ns).copy()


@lru_cache(maxsize=8)
def _levels(pid,mtime,meta_mtime):
    raw=gzip.decompress((DATA/f'{pid}.json.gz').read_bytes())
    meta=json.loads((DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest()!=meta['sha256'] or meta['station']['properties']['post_code']!=STATIONS[pid][0]:
        raise ValueError('Нарушена целостность или привязка наблюдений')
    d=pd.DataFrame(json.loads(raw)['level'])
    d=d[d.source==STATIONS[pid][1]].copy()
    d['d']=pd.to_datetime(d.d,errors='raise')
    if d.d.duplicated().any():
        raise ValueError('Повторяющиеся даты наблюдений')
    values=pd.to_numeric(d.v,errors='raise')
    if not np.isfinite(values).all():
        raise ValueError('Некорректные уровни')
    s=pd.Series(values.to_numpy(),index=d.d,name='level_cm').sort_index()
    return s.reindex(pd.date_range(s.index.min(),s.index.max(),freq='D'))


def feature_frame(series,lead):
    s=series.reindex(pd.date_range(series.index.min(),series.index.max(),freq='D'))
    f=pd.DataFrame({'h0':s})
    for lag in (1,3,7,14,30):
        f[f'h{lag}']=s.shift(lag)
    for lag in (1,3,7,14):
        f[f'd{lag}']=s-s.shift(lag)
    f['mean7']=s.rolling(7,min_periods=7).mean()
    f['mean30']=s.rolling(30,min_periods=30).mean()
    phase=2*np.pi*(s.index.day_of_year-1)/365.25
    f['sin'],f['cos'],f['lead']=np.sin(phase),np.cos(phase),lead
    return f[FEATURES]


def dataset(series):
    rows=[]
    for lead in range(1,8):
        f=feature_frame(series,lead)
        f['issue']=f.index
        f['target_date']=f.index+pd.Timedelta(days=lead)
        f['target']=series.reindex(f.target_date).to_numpy()
        f['delta']=f.target-f.h0
        rows.append(f)
    return pd.concat(rows,ignore_index=True).dropna(subset=FEATURES+['target'])


def baselines(series,rows,year):
    past=series[series.index.year<year].dropna()
    calendar=past.groupby(past.index.strftime('%m-%d')).median()
    dates=pd.DatetimeIndex(rows.target_date)
    seasonal=calendar.reindex(dates.strftime('%m-%d')).to_numpy()
    seasonal=np.where(np.isnan(seasonal),past.median(),seasonal)
    previous=series.reindex(dates-pd.DateOffset(years=1)).to_numpy()
    return pd.DataFrame(dict(persistence=rows.h0.to_numpy(),calendar=seasonal,last_year=previous),index=rows.index)


def fit_year(series,year):
    table=dataset(series)
    train=table[table.target_date<pd.Timestamp(year-1,1,1)]
    full=series[series.index.year<year-1].groupby(series[series.index.year<year-1].index.year).count()
    required=pd.Series([.9*len(pd.date_range(f'{year}-01-01',f'{year}-12-31')) for year in full.index],index=full.index)
    if (full>=required).sum()<8:
        raise ValueError('Для обучения нужны минимум 8 полных прошлых лет')
    cal=table[(table.issue.dt.year==year-1)&(table.target_date<pd.Timestamp(year,1,1))].copy()
    if len(cal)<1000:
        raise ValueError('Недостаточно измерений предыдущего года для калибровки')
    model=CatBoostRegressor(iterations=200,depth=5,learning_rate=.06,l2_leaf_reg=10,
                           loss_function='RMSE',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    model.fit(train[FEATURES],train.delta)
    preds=baselines(series,cal,year-1)
    preds['catboost']=model.predict(cal[FEATURES])+cal.h0.to_numpy()
    methods={};widths={};scores={}
    for lead in range(1,8):
        mask=cal.lead==lead
        actual=cal.loc[mask,'target']
        maes={method:float((preds.loc[mask,method]-actual).abs().mean()) for method in METHODS}
        method=min(maes,key=maes.get)
        error=(preds.loc[mask,method]-actual).abs().dropna()
        methods[str(lead)]=method
        widths[str(lead)]=float(np.quantile(error,.8,method='higher'))
        scores[str(lead)]=maes
    meta=dict(year=year,train_until=str(train.target_date.max().date()),calibration_year=year-1,
              methods=methods,widths_cm=widths,calibration_mae=scores,
              features=FEATURES,interval='80% residual quantile on previous year; coverage not guaranteed')
    return model,meta


def predict_rows(series,rows,year,model,meta):
    p=baselines(series,rows,year)
    p['catboost']=model.predict(rows[FEATURES])+rows.h0.to_numpy()
    p['m1']=[p.loc[i,meta['methods'][str(int(rows.loc[i,'lead']))]] for i in rows.index]
    widths=rows.lead.map(lambda lead:meta['widths_cm'][str(int(lead))]).to_numpy()
    p['lower']=p.m1-widths
    p['upper']=p.m1+widths
    return p


@lru_cache(maxsize=16)
def _bundle(pid,year,version):
    model=CatBoostRegressor()
    model.load_model(str(MODELS/f'{pid}_{year}.cbm'))
    meta=json.loads((MODELS/f'{pid}_{year}.json').read_text(encoding='utf-8'))
    return model,meta


def forecast(pid,issue_date):
    series=load_levels(pid)
    issue=pd.Timestamp(issue_date).normalize()
    if issue not in series.index or pd.isna(series.loc[issue]):
        raise ValueError(f'На {issue:%d.%m.%Y} нет измерения уровня; последнее: {series.last_valid_index():%d.%m.%Y}')
    path=MODELS/f'{pid}_{issue.year}.json'
    if not path.exists():
        raise ValueError('Для выбранного года нет проверенной версии модели')
    model,meta=_bundle(pid,issue.year,path.stat().st_mtime_ns)
    source=json.loads((DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
    if meta.get('data_sha256')!=source['sha256']:
        raise ValueError('Наблюдения изменились после обучения: модель нужно переобучить')
    if pd.Timestamp(meta['train_until'])>=issue or meta['calibration_year']>=issue.year:
        raise ValueError('Обучение или калибровка используют будущее')
    # Физически отрезаем будущее прежде, чем строить признаки и базовые прогнозы.
    known=series.loc[:issue]
    rows=[]
    for lead in range(1,8):
        f=feature_frame(known,lead).loc[[issue]].copy()
        f['target_date']=issue+pd.Timedelta(days=lead)
        rows.append(f)
    rows=pd.concat(rows,ignore_index=True)
    if rows[FEATURES].isna().any().any():
        raise ValueError('Перед датой решения есть пропуски: нужны 30 ежедневных наблюдений')
    p=predict_rows(known,rows,issue.year,model,meta)
    output=pd.DataFrame(dict(date=rows.target_date,lead=rows.lead,predicted_cm=p.m1,
                            lower_cm=p.lower,upper_cm=p.upper,persistence_cm=p.persistence,
                            calendar_cm=p.calendar,last_year_cm=p.last_year,
                            method=rows.lead.map(lambda h:meta['methods'][str(int(h))])))
    return output,meta
