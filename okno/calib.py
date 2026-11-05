"""
Калибровка закона Стефана по реальным замерам толщины льда на гидропостах.

Для каждого замера в фазе нарастания (до 31 марта) считаем FDD от фактической даты
ледостава (по посту) до даты замера по погоде ERA5 в точке поста, затем подбираем
alpha в h = alpha * sqrt(FDD) методом наименьших квадратов (через ноль).
"""

import numpy as np
import pandas as pd

from .ice import load_weather
from .paths import PROCESSED


def stefan_pairs(gauge_code: int, seasons: pd.DataFrame, thick: pd.DataFrame) -> pd.DataFrame:
    try:
        w = load_weather(f"gauge_{gauge_code}")
    except FileNotFoundError:
        return pd.DataFrame()
    w = w.set_index("date").sort_index()
    frost = np.maximum(0.0, -w["t"])
    cum = frost.cumsum()
    s = seasons[(seasons.gauge_code == gauge_code) & seasons.valid & seasons.freezeup_date.notna()]
    th = thick[(thick.gauge_code == gauge_code) & thick.ice_cm.notna()].copy()
    th["date"] = pd.to_datetime(th["date"])
    rows = []
    for r in s.itertuples():
        fz = pd.Timestamp(r.freezeup_date)
        if fz < w.index.min() or fz > w.index.max():
            continue
        m = th[(th.hydro_year == r.hydro_year) & (th.date > fz) & (th.date.dt.month.isin([10, 11, 12, 1, 2, 3]))]
        m = m[m.date <= w.index.max()]
        if m.empty:
            continue
        base = cum.loc[:fz].iloc[-1]
        fdd = cum.reindex(m["date"]).to_numpy() - base
        rows.append(pd.DataFrame({
            "gauge_code": gauge_code, "hydro_year": r.hydro_year, "date": m["date"].to_numpy(),
            "fdd": fdd, "ice_cm": m["ice_cm"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_alpha(pairs: pd.DataFrame) -> dict:
    x = np.sqrt(np.maximum(pairs["fdd"].to_numpy(), 0))
    y = pairs["ice_cm"].to_numpy()
    alpha = float((x @ y) / (x @ x))
    pred = alpha * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"alpha": alpha, "mae_cm": float(np.abs(y - pred).mean()),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "n": int(len(y))}


def calibrate_all(gauge_codes=None):
    seasons = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    thick = pd.read_csv(PROCESSED / "hydro_ice_thickness.csv")
    gauges = pd.read_csv(PROCESSED / "gauges.csv")
    codes = gauge_codes or gauges.loc[gauges.yakutia, "gauge_code"].tolist()
    all_pairs, per = [], []
    for g in codes:
        p = stefan_pairs(g, seasons, thick)
        if p.empty:
            continue
        all_pairs.append(p)
        per.append({"gauge_code": g, **fit_alpha(p)})
    if not all_pairs:
        return None, pd.DataFrame(), pd.DataFrame()
    pairs = pd.concat(all_pairs, ignore_index=True)
    pooled = fit_alpha(pairs)
    return pooled, pd.DataFrame(per), pairs
