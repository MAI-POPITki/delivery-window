"""
М2 — состояние льда по радарным снимкам Sentinel-1: P(лёд) для вырезки 3×3 км над руслом.

1. Маска русла для каждой точки: пиксели, которые на летних снимках с водой стабильно тёмные
   (вода отражает радар от спутника — тёмная; лёд шероховатый — светлый).
2. Признаки — только по пикселям русла: перцентили и разброс VV и VH (дБ), разница VV−VH,
   доля «светлых» пикселей, текстура. Дата в признаки не входит: модель смотрит на снимок,
   а не угадывает по сезону.
3. Классификатор CatBoost «вода / лёд». Проверка — по годам (сезон целиком вне обучения).

Встраивание в М1 — «отсечение сценариев»: если на свежем снимке уверенно лёд, вскрытие ещё
впереди; если уверенно вода — оно уже прошло.
"""

import numpy as np
import pandas as pd

from .paths import DATA, MODELS

S1 = DATA / "s1"
MASKS = S1 / "river_masks.npz"
MODEL = MODELS / "m2_ice.cbm"
WATER_DB = -18.0       # порог «тёмного» пикселя на воде
MIN_SHARE = 0.6        # пиксель тёмный хотя бы на 60% летних снимков с водой
FEATURES = ["vv_p10", "vv_p25", "vv_p50", "vv_p75", "vv_p90", "vv_std",
            "vh_p10", "vh_p50", "vh_p90", "vh_std", "diff_p50", "bright_share", "vv_grad", "river_px"]


def load_chip(path):
    z = np.load(DATA.parent / path)
    return z["vv_db"].astype("float32"), z["vh_db"].astype("float32")


def build_masks(index: pd.DataFrame) -> dict:
    """Маска русла по летним (июнь–август) снимкам с меткой water."""
    masks = {}
    for pt, g in index.groupby("point"):
        g = g[(g.label == "water") & pd.to_datetime(g.date).dt.month.isin([6, 7, 8])]
        if len(g) < 3:
            g = index[(index.point == pt) & (index.label == "water")]
        stack = [load_chip(f)[0] for f in g.file]
        dark = np.mean([np.nan_to_num(v, nan=0) < WATER_DB for v in stack], axis=0)
        m = dark >= MIN_SHARE
        masks[pt] = m
    np.savez_compressed(MASKS, **masks)
    return masks


def load_masks() -> dict:
    z = np.load(MASKS)
    return {k: z[k] for k in z.files}


def chip_features(vv, vh, mask) -> dict:
    v, h = vv[mask], vh[mask]
    v, h = v[np.isfinite(v)], h[np.isfinite(h)]
    if len(v) < 50:
        return {k: np.nan for k in FEATURES}
    p = np.percentile
    gy, gx = np.gradient(np.where(np.isfinite(vv), vv, np.nanmedian(vv)))
    grad = np.hypot(gx, gy)[mask]
    return {"vv_p10": p(v, 10), "vv_p25": p(v, 25), "vv_p50": p(v, 50), "vv_p75": p(v, 75), "vv_p90": p(v, 90),
            "vv_std": v.std(), "vh_p10": p(h, 10), "vh_p50": p(h, 50), "vh_p90": p(h, 90), "vh_std": h.std(),
            "diff_p50": p(v, 50) - p(h, 50), "bright_share": float((v > -15).mean()),
            "vv_grad": float(np.nanmean(grad)), "river_px": int(mask.sum())}


def features_table(index: pd.DataFrame, masks: dict) -> pd.DataFrame:
    rows = []
    for r in index.itertuples():
        if r.point not in masks:
            continue
        try:
            vv, vh = load_chip(r.file)
        except FileNotFoundError:
            continue
        rows.append({"item_id": r.item_id, "point": r.point, **chip_features(vv, vh, masks[r.point])})
    return index.merge(pd.DataFrame(rows), on=["item_id", "point"])


def make_model():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, random_seed=42, verbose=False,
                              allow_writing_files=False)


def load_model():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model(str(MODEL))
    return m


def apply_observation(q10, q50, q90, obs_day, p_ice, n=4000, conf=0.8, seed=0):
    """Отсечение сценариев вскрытия по снимку в день obs_day (день сезона).
    p_ice ≥ conf — лёд стоит → вскрытие позже obs_day; p_ice ≤ 1−conf — вода → вскрытие уже было.
    Возвращает новые (q10, q50, q90) или исходные, если снимок неуверенный."""
    from .planner import sample_quantiles
    s = sample_quantiles(q10, q50, q90, n, np.random.default_rng(seed))
    if p_ice >= conf:
        s = s[s > obs_day]
    elif p_ice <= 1 - conf:
        s = s[s <= obs_day]
    else:
        return q10, q50, q90
    if len(s) < n * 0.02:  # снимок противоречит почти всем сценариям — не доверяем ни тому, ни другому
        return q10, q50, q90
    return tuple(float(x) for x in np.percentile(s, [10, 50, 90]))
