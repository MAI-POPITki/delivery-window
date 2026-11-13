"""Эмпирическое превышение заданного уровня, без выдуманного распределения квантилей."""
from functools import lru_cache
import hashlib
import json
import numpy as np
import pandas as pd
from . import flood_m1_quantile as M
from .flood_m2 import sha256

DATA=M.ROOT/'models/flood_probability'


def metadata_hash(path):
    canonical=json.dumps(json.loads(path.read_text(encoding='utf-8')),sort_keys=True,separators=(',',':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def probability(samples, forecast, scale, threshold):
    samples=np.asarray(samples,dtype=float)
    if not len(samples) or not np.isfinite(samples).all() or not np.isfinite([forecast,scale,threshold]).all() or scale<=0:
        raise ValueError('Для вероятности нужны конечные ошибки, прогноз, порог и положительный масштаб')
    return float((np.count_nonzero(samples >= (threshold-forecast)/scale)+1)/(len(samples)+2))


@lru_cache(maxsize=16)
def _load(pid,year,stamp):
    path=DATA/f'{pid}_{year}.csv.gz'
    meta=json.loads(path.with_suffix('').with_suffix('.json').read_text(encoding='utf-8'))
    if (sha256(path)!=meta['sha256'] or sha256(M.MODELS/f'{pid}_{year}.cbm')!=meta['model_sha256']
            or metadata_hash(M.MODELS/f'{pid}_{year}.json')!=meta['metadata_sha256']):
        raise ValueError('Изменились ошибки или модель вероятностей')
    source=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
    if source['sha256']!=meta['data_sha256']:
        raise ValueError('Изменились наблюдения')
    table=pd.read_csv(path,parse_dates=['issue'])
    if not table.issue.dt.year.eq(year-1).all():raise ValueError('Калибровка не из предыдущего года')
    return table


def calibration(pid,year):
    path=DATA/f'{pid}_{year}.csv.gz'
    return _load(pid,year,(path.stat().st_mtime_ns,(M.MODELS/f'{pid}_{year}.cbm').stat().st_mtime_ns,
        (M.DATA/f'{pid}.manifest.json').stat().st_mtime_ns,(M.MODELS/f'{pid}_{year}.json').stat().st_mtime_ns))


def exceedance(pid,issue_date,forecast,threshold):
    issue=pd.Timestamp(issue_date)
    if not np.isfinite(threshold):raise ValueError('Некорректный порог')
    table=calibration(pid,issue.year)
    wet=issue.month in (4,5,6,7)
    rows=[]
    for f in forecast.itertuples():
        if pd.Timestamp(f.date)!=issue+pd.Timedelta(days=int(f.lead)):
            raise ValueError('Прогноз не соответствует дате решения')
        subset=table[table.lead.eq(f.lead)]
        seasonal=subset[subset.issue.dt.month.isin([4,5,6,7]).eq(wet)]
        selected=seasonal if len(seasonal)>=60 else subset
        scale=max((f.upper_cm-f.lower_cm)/2,1.)
        p=probability(selected.normalized_error,f.predicted_cm,scale,threshold)
        rows.append(dict(date=f.date,lead=f.lead,probability=p,n=len(selected),calibration_year=issue.year-1))
    return pd.DataFrame(rows)
