"""
М1 — вероятностный прогноз даты события (открытие / закрытие сезонного пути).

CatBoost с квантильной функцией потерь (10/50/90%) предсказывает ОТКЛОНЕНИЕ даты
от календаря (средней даты прошлых лет); физическая оценка (Стефан / градусо-дни) — признак.
Одна общая модель на все точки: мало лет на одном посту дополняются данными других.

Проверка: leave-one-season-out — сезон выкидывается целиком (у всех точек сразу).
"""

import numpy as np
import pandas as pd

from .features import FEATURES

QUANTILES = (0.1, 0.5, 0.9)
CB_PARAMS = dict(iterations=600, depth=4, learning_rate=0.03, l2_leaf_reg=5,
                 random_seed=42, verbose=False, allow_writing_files=False)


def _model():
    from catboost import CatBoostRegressor
    loss = "MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES)
    return CatBoostRegressor(loss_function=loss, **CB_PARAMS)


def fit(df: pd.DataFrame, kind: str):
    m = _model()
    m.fit(df[FEATURES[kind]], df["target_day"] - df["cal_target"])
    return m


def predict(model, df: pd.DataFrame, kind: str) -> pd.DataFrame:
    p = np.asarray(model.predict(df[FEATURES[kind]]))
    p = np.sort(p, axis=1)  # квантили не должны пересекаться
    out = pd.DataFrame(p + df["cal_target"].to_numpy()[:, None],
                       columns=[f"q{int(q * 100)}" for q in QUANTILES], index=df.index)
    return out


def backtest(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Leave-one-season-out: для каждого сезона обучаемся на остальных.
    Затем конформная калибровка диапазона (CQR): поправка для сезона y считается только
    по ошибкам других сезонов, чтобы покрытие 10–90% было честным."""
    parts = []
    for y in sorted(df["hydro_year"].unique()):
        test = df[df.hydro_year == y]
        train = df[df.hydro_year != y]
        m = fit(train, kind)
        parts.append(pd.concat([test, predict(m, test, kind)], axis=1))
    bt = pd.concat(parts).sort_index()
    bt["q10_raw"], bt["q90_raw"] = bt["q10"], bt["q90"]
    score = np.maximum(bt["q10"] - bt["target_day"], bt["target_day"] - bt["q90"])
    for y in bt["hydro_year"].unique():
        mask = bt["hydro_year"] == y
        corr = conformal_correction(score[~mask])
        bt.loc[mask, "q10"] -= corr
        bt.loc[mask, "q90"] += corr
    return bt


TARGET_COVERAGE = 0.8


def conformal_correction(scores: pd.Series) -> float:
    s = np.sort(np.asarray(scores.dropna()))
    n = len(s)
    k = min(n - 1, int(np.ceil((n + 1) * TARGET_COVERAGE)) - 1)
    return float(s[k]) if n else 0.0


LEAD_BUCKETS = {"7": (4, 10), "14": (11, 17), "28": (25, 31)}


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {"гидропосты (замеры)": bt[bt.is_satellite == 0],
              "переправа Якутск (спутник)": bt[bt.is_satellite == 1]}
    for gname, g in groups.items():
        if g.empty:
            continue
        for lead, (lo, hi) in {**LEAD_BUCKETS, "все": (0, 400)}.items():
            s = g[g.lead_days.between(lo, hi)]
            if s.empty:
                continue
            e_cal = (s.cal_target - s.target_day).abs().mean()
            e_phys = (s.phys_pred - s.target_day).abs().mean()
            e_m1 = (s.q50 - s.target_day).abs().mean()
            cover = ((s.target_day >= s.q10) & (s.target_day <= s.q90)).mean()
            rows.append({"kind": s.kind.iloc[0], "группа": gname, "за_сколько_дней": lead, "n": len(s),
                         "сезонов": s.hydro_year.nunique(),
                         "MAE_календарь": round(e_cal, 1), "MAE_физика": round(e_phys, 1),
                         "MAE_М1": round(e_m1, 1),
                         "выигрыш_к_календарю_%": round(100 * (1 - e_m1 / e_cal), 1),
                         "покрытие_10_90": round(cover, 2),
                         "ширина_10_90_сут": round((s.q90 - s.q10).mean(), 1)})
    return pd.DataFrame(rows)
