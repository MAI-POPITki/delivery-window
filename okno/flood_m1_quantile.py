"""М1 v2: квантили уровня и сезонная калибровка строго по предыдущему году."""
from functools import lru_cache
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from . import flood_m1 as V1

ROOT, DATA, STATIONS, FEATURES = V1.ROOT, V1.DATA, V1.STATIONS, V1.FEATURES
MODELS = ROOT / 'models/flood_m1_quantile'
load_levels, feature_frame, dataset, baselines = V1.load_levels, V1.feature_frame, V1.dataset, V1.baselines
METHODS = V1.METHODS


def spring(dates):
    return pd.DatetimeIndex(dates).month.isin([4, 5, 6, 7])


def correction(values):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Нет ошибок для калибровки интервала')
    rank = min(len(values), int(np.ceil((len(values) + 1) * .8)))
    return float(max(0., values[rank - 1]))


def quantiles(model, rows):
    return np.sort(np.asarray(model.predict(rows[FEATURES])), axis=1) + rows.h0.to_numpy()[:, None]


def fit_year(series, year):
    table = dataset(series)
    train = table[table.target_date < pd.Timestamp(year - 1, 1, 1)]
    past = series[series.index.year < year - 1]
    counts = past.groupby(past.index.year).count()
    full = sum(n >= .9 * (366 if pd.Timestamp(y, 12, 31).is_leap_year else 365) for y, n in counts.items())
    if full < 8:
        raise ValueError('Для обучения нужны минимум 8 полных прошлых лет')
    cal = table[(table.issue.dt.year == year - 1) & (table.target_date < pd.Timestamp(year, 1, 1))].copy()
    if len(cal) < 1000:
        raise ValueError('Недостаточно измерений предыдущего года для калибровки')
    model = CatBoostRegressor(iterations=200, depth=5, learning_rate=.06, l2_leaf_reg=10,
                             loss_function='MultiQuantile:alpha=0.1,0.5,0.9', random_seed=42,
                             thread_count=2, verbose=False, allow_writing_files=False)
    model.fit(train[FEATURES], train.delta)
    q = quantiles(model, cal)
    preds = baselines(series, cal, year - 1)
    preds['catboost'] = q[:, 1]
    methods, scores, adjustments = {}, {}, {}
    wet = spring(cal.issue)
    for lead in range(1, 8):
        mask = cal.lead.eq(lead).to_numpy()
        truth = cal.target.to_numpy()
        maes = {method: float(np.nanmean(np.abs(preds[method].to_numpy()[mask] - truth[mask]))) for method in METHODS}
        method = min(maes, key=maes.get)
        methods[str(lead)], scores[str(lead)] = method, maes
        error = (np.maximum(q[:, 0] - truth, truth - q[:, 2]) if method == 'catboost'
                 else np.abs(preds[method].to_numpy() - truth))
        adjustments[str(lead)] = {}
        for season in ('spring', 'other'):
            selected = mask & (wet if season == 'spring' else ~wet)
            fallback = int(np.isfinite(error[selected]).sum()) < 60
            selected = mask if fallback else selected
            adjustments[str(lead)][season] = dict(cm=correction(error[selected]),
                n=int(np.isfinite(error[selected]).sum()), fallback_all_year=fallback)
    return model, dict(year=year, train_until=str(train.target_date.max().date()),
        calibration_year=year - 1, methods=methods, calibration_mae=scores,
        corrections=adjustments, features=FEATURES, version='quantile-v2',
        interval='q10/q90 + previous-year seasonal CQR; coverage not guaranteed')


def predict_rows(series, rows, year, model, meta):
    p = baselines(series, rows, year)
    q = quantiles(model, rows)
    p['catboost'] = q[:, 1]
    methods = [meta['methods'][str(int(h))] for h in rows.lead]
    p['m1'] = [p.iloc[i][method] for i, method in enumerate(methods)]
    use_q = np.array(methods) == 'catboost'
    wet = spring(rows.issue if 'issue' in rows else rows.target_date - pd.to_timedelta(rows.lead, unit='D'))
    corr = np.array([meta['corrections'][str(int(h))]['spring' if w else 'other']['cm']
                     for h, w in zip(rows.lead, wet)])
    p['lower'] = np.where(use_q, q[:, 0], p.m1) - corr
    p['upper'] = np.where(use_q, q[:, 2], p.m1) + corr
    return p


@lru_cache(maxsize=16)
def _bundle(pid, year, model_stamp, meta_stamp):
    model = CatBoostRegressor()
    model.load_model(str(MODELS / f'{pid}_{year}.cbm'))
    return model, json.loads((MODELS / f'{pid}_{year}.json').read_text(encoding='utf-8'))


def forecast(pid, issue_date):
    series = load_levels(pid)
    issue = pd.Timestamp(issue_date).normalize()
    if issue not in series.index or pd.isna(series.loc[issue]):
        raise ValueError(f'На {issue:%d.%m.%Y} нет измерения уровня; последнее: {series.last_valid_index():%d.%m.%Y}')
    path = MODELS / f'{pid}_{issue.year}.json'
    model_path = path.with_suffix('.cbm')
    if not path.exists() or not model_path.exists():
        raise ValueError('Для выбранного года нет версии М1 v2; доступны годы с 2019')
    model, meta = _bundle(pid, issue.year, model_path.stat().st_mtime_ns, path.stat().st_mtime_ns)
    source = json.loads((DATA / f'{pid}.manifest.json').read_text(encoding='utf-8'))
    if (meta.get('data_sha256') != source['sha256'] or meta.get('point_id') != pid
            or meta.get('gauge_code') != STATIONS[pid][0] or meta.get('observation_source') != STATIONS[pid][1]):
        raise ValueError('Наблюдения или привязка модели изменились: требуется переобучение')
    if pd.Timestamp(meta['train_until']) >= issue or meta['calibration_year'] >= issue.year:
        raise ValueError('Обучение или калибровка используют будущее')
    known = series.loc[:issue]
    rows = []
    for lead in range(1, 8):
        f = feature_frame(known, lead).loc[[issue]].copy()
        f['issue'], f['target_date'] = issue, issue + pd.Timedelta(days=lead)
        rows.append(f)
    rows = pd.concat(rows, ignore_index=True)
    if rows[FEATURES].isna().any().any():
        raise ValueError('Перед датой решения есть пропуски: нужны последние 30 ежедневных измерений и уровень 30 суток назад')
    p = predict_rows(known, rows, issue.year, model, meta)
    out = pd.DataFrame(dict(date=rows.target_date, lead=rows.lead, predicted_cm=p.m1,
        lower_cm=p.lower, upper_cm=p.upper, persistence_cm=p.persistence, calendar_cm=p.calendar,
        last_year_cm=p.last_year, method=rows.lead.map(lambda h: meta['methods'][str(int(h))])))
    return out, meta
