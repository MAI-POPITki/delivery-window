"""Low-water beta for Lena/Yakutsk: summer levels, rolling previous-year calibration.

The statistical q20 is not a navigation limit. Existing CatBoost weights are reused;
method selection, residuals and threshold use years strictly before the issue year.
"""
from functools import lru_cache
import json

import numpy as np
import pandas as pd

from . import flood_m1_quantile as M
from .flood_probability import probability

PID = 'lena_yakutsk'
MONTHS = (6, 7, 8, 9)
VERSION = 'low-water-beta-v1'
METHODS = ('persistence', 'calendar', 'trend7', 'trend14', 'catboost')


def season(dates):
    return pd.DatetimeIndex(dates).month.isin(MONTHS)


def threshold(series, year):
    """Freeze a summer q20 before the calibration year (not the evaluation year)."""
    past = series[(series.index.year < year - 1) & season(series.index)].dropna()
    if len(past) < 8 * 100:
        raise ValueError('Для летнего порога нужны минимум восемь прошлых сезонов')
    return float(past.quantile(.2))


def candidates(series, rows, year, model):
    # Calendar uses the same pre-calibration archive for calibration and evaluation.
    out = M.baselines(series, rows, year - 1)[['persistence', 'calendar']]
    for lag in (7, 14):
        out[f'trend{lag}'] = rows.h0 + (rows.h0 - rows[f'h{lag}']) / lag * rows.lead
    out['catboost'] = M.quantiles(model, rows)[:, 1]
    return out[list(METHODS)]


def fit_calibration(series, year, model):
    # Truncate BEFORE feature/target construction: evaluation measurements cannot enter.
    past = series.loc[:pd.Timestamp(year, 1, 1) - pd.Timedelta(days=1)]
    rows = M.dataset(past)
    rows = rows[(rows.issue.dt.year == year - 1) & season(rows.issue) & season(rows.target_date)]
    pred = candidates(past, rows, year, model)
    result = {}
    for lead in range(1, 8):
        r = rows[rows.lead.eq(lead)]
        if len(r) < 90:
            raise ValueError('Недостаточно ежедневных летних замеров предыдущего года')
        mae = pred.loc[r.index].sub(r.target, axis=0).abs().mean()
        method = mae.idxmin()  # Fixed candidate list, selected only on previous year.
        result[lead] = dict(method=method, n=len(r), calibration_mae=mae.to_dict(),
            errors=(r.target - pred.loc[r.index, method]).to_numpy(),
            persistence_errors=(r.target - r.h0).to_numpy(),
            calendar_errors=(r.target - pred.loc[r.index, 'calendar']).to_numpy(),
            targets=r.target.to_numpy())
    return result


def _paths(year):
    return (M.DATA / f'{PID}.json.gz', M.DATA / f'{PID}.manifest.json',
            M.MODELS / f'{PID}_{year}.cbm', M.MODELS / f'{PID}_{year}.json')


@lru_cache(maxsize=8)
def _bundle(year, stamps):
    paths = _paths(year)
    series = M.load_levels(PID)  # Validates raw SHA-256 and gauge code.
    meta = json.loads(paths[3].read_text(encoding='utf-8'))
    source = json.loads(paths[1].read_text(encoding='utf-8'))
    if (meta.get('data_sha256') != source['sha256'] or meta.get('point_id') != PID
            or meta.get('gauge_code') != '3045' or meta.get('observation_source') != 'gmvo'):
        raise ValueError('Изменились архив или привязка модели к посту')
    if pd.Timestamp(meta['train_until']) >= pd.Timestamp(year - 1, 1, 1) or meta['calibration_year'] != year - 1:
        raise ValueError('Модель должна быть обучена строго до года калибровки')
    model, _ = M._bundle(PID, year, paths[2].stat().st_mtime_ns, paths[3].stat().st_mtime_ns)
    return model, meta, fit_calibration(series, year, model), threshold(series, year)


def bundle(year):
    paths = _paths(year)
    if not all(p.exists() for p in paths):
        raise ValueError('Для этого года нет проверенного пакета модели; текущие данные не заменяются архивом')
    return _bundle(year, tuple(p.stat().st_mtime_ns for p in paths))


def below_probability(errors, point, limit):
    """P(H <= limit), including ties, with add-one smoothing; not 1-P(H >= limit)."""
    return probability(-np.asarray(errors, dtype=float), -point, 1., -limit)


def predict_rows(series, rows, year, model, calibration, limit):
    if not np.isfinite(limit):
        raise ValueError('Порог должен быть конечным числом в сантиметрах')
    pred = candidates(series, rows, year, model)
    out = rows[['issue', 'target_date', 'lead']].copy()
    result = []
    for idx, row in rows.iterrows():
        c = calibration[int(row.lead)]
        point = float(pred.loc[idx, c['method']])
        lo, hi = np.quantile(c['errors'], [.1, .9])
        result.append(dict(predicted_cm=point, lower_cm=point+lo, upper_cm=point+hi,
            probability=below_probability(c['errors'], point, limit),
            persistence_cm=float(row.h0), calendar_cm=float(pred.loc[idx, 'calendar']),
            persistence_probability=below_probability(c['persistence_errors'], float(row.h0), limit),
            calendar_probability=below_probability(c['calendar_errors'], float(pred.loc[idx, 'calendar']), limit),
            climatology_probability=(np.count_nonzero(c['targets'] <= limit)+1)/(len(c['targets'])+2),
            method=c['method'], calibration_n=c['n']))
    values = pd.DataFrame(result, index=rows.index)
    return pd.concat([out, values], axis=1)


def forecast(issue_date, limit=None):
    issue = pd.Timestamp(issue_date).normalize()
    if issue.month not in MONTHS or (issue.month == 9 and issue.day == 30):
        raise ValueError('Бета проверена только для июня–сентября, с целью не позже 30 сентября')
    series = M.load_levels(PID)
    if issue not in series.index or pd.isna(series.loc[issue]):
        raise ValueError('Нет измерения на дату решения: свежий прогноз нельзя построить по архивной отметке')
    model, meta, cal, q20 = bundle(issue.year)
    if limit is not None and not np.isfinite(limit):
        raise ValueError('Некорректный порог уровня')
    known = series.loc[:issue]
    rows = []
    for lead in range(1, 8):
        target = issue + pd.Timedelta(days=lead)
        if target.month not in MONTHS:
            break
        r = M.feature_frame(known, lead).loc[[issue]].copy()
        r['issue'], r['target_date'] = issue, target
        rows.append(r)
    rows = pd.concat(rows, ignore_index=True)
    if not np.isfinite(rows[M.FEATURES].to_numpy()).all():
        raise ValueError('Нужны последние 31 ежедневное измерение без пропусков')
    limit = q20 if limit is None else float(limit)
    out = predict_rows(known, rows, issue.year, model, cal, limit)
    return out, dict(version=VERSION, pid=PID, gauge_code='3045', units='см относительно нуля поста',
        observation_source='gmvo', issue=str(issue.date()), last_measurement=str(series.last_valid_index().date()),
        train_until=meta['train_until'], calibration_year=issue.year-1, threshold_cm=limit,
        statistical_threshold_cm=q20, threshold_kind='statistical_q20' if limit == q20 else 'operator',
        latest_input_cm=float(known.loc[issue]))
