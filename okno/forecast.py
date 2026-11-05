"""
Прогноз на произвольную дату без знания факта — для дашборда, бота и разбора сезонов.

forecast(point_id, hydro_year, issue_day, kind) → квантили даты события, календарь, физика.
Использует только погоду до issue_day и историю дат за прошлые сезоны.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import features as F
from .ice import from_season_day, load_weather
from .paths import MODELS, PROCESSED

POINTS = {
    "yakutsk_crossing": {"name": "Переправа Якутск — Нижний Бестях", "lat": 62.0, "lon": 129.82,
                         "is_satellite": 1, "gauge": None},
}


@lru_cache(maxsize=None)
def _gauges():
    return pd.read_csv(PROCESSED / "gauges.csv")


@lru_cache(maxsize=None)
def history(point_id: str) -> pd.DataFrame:
    """Фактические даты прошлых сезонов: index = hydro_year; open_day, close_day, freezeup_day."""
    if point_id == "yakutsk_crossing":
        s = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv")
        return pd.DataFrame({"open_day": np.nan, "close_day": s["sat_breakup_day"].to_numpy(),
                             "freezeup_day": s["sat_freezeup_day"].to_numpy()}, index=s["hydro_year"].to_numpy())
    code = int(point_id.split("_")[1])
    s = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    s = s[(s.gauge_code == code) & s.valid]
    return pd.DataFrame({"open_day": s[f"open_{F.OPEN_LOAD_T}t_day"].to_numpy(),
                         "close_day": s["melt_start_day"].to_numpy(),
                         "freezeup_day": s["freezeup_day"].to_numpy()}, index=s["hydro_year"].to_numpy())


def point_meta(point_id: str) -> dict:
    if point_id in POINTS:
        return POINTS[point_id]
    code = int(point_id.split("_")[1])
    g = _gauges().set_index("gauge_code").loc[code]
    return {"name": g["name"], "lat": float(g["lat"]), "lon": float(g["lon"]), "is_satellite": 0, "gauge": code}


@lru_cache(maxsize=None)
def _model(kind):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor()
    m.load_model(str(MODELS / f"m1_{kind}.cbm"))
    return m


@lru_cache(maxsize=None)
def _conformal(kind):
    import json
    p = MODELS / f"m1_{kind}_conformal.json"
    return json.loads(p.read_text())["correction_days"] if p.exists() else 0.0


@lru_cache(maxsize=None)
def _matrices(point_id):
    w = load_weather(point_id)
    T_all = w.pivot_table(index="hydro_year", columns="season_day", values="t").reindex(columns=range(365))
    S_all = w.pivot_table(index="hydro_year", columns="season_day", values="snow_cm").reindex(columns=range(365))
    full = T_all.notna().sum(axis=1) > 330
    return T_all, S_all, T_all[full]


@lru_cache(maxsize=None)
def _pairs(gauge):
    from .calib import stefan_pairs
    return stefan_pairs(gauge, pd.read_csv(PROCESSED / "hydro_ice_seasons.csv"),
                        pd.read_csv(PROCESSED / "hydro_ice_thickness.csv"))


@lru_cache(maxsize=None)
def _alpha(gauge, hydro_year):
    """alpha поста без текущего сезона; для точек без замеров толщины — общий по постам."""
    from .calib import fit_alpha
    if not gauge:
        return F.load_alpha()
    p = _pairs(gauge)
    return fit_alpha(p[p.hydro_year != hydro_year])["alpha"] if not p.empty else F.load_alpha()


def last_weather_day(point_id: str, hydro_year: int):
    T_all, _, _ = _matrices(point_id)
    if hydro_year not in T_all.index:
        return None
    v = T_all.loc[hydro_year].dropna()
    return int(v.index.max()) if len(v) else None


def calendar_day(point_id, hydro_year, kind):
    h = history(point_id)[f"{kind}_day"]
    cal = F.past_mean(h, hydro_year, F.CAL_WINDOW, F.CAL_MIN)
    if np.isnan(cal):  # нет свежих лет (посты — до 2012): берём последние доступные 10 сезонов
        past = h[h.index < hydro_year].dropna().tail(F.CAL_WINDOW)
        cal = float(past.mean()) if len(past) >= F.CAL_MIN else np.nan
    return cal


def forecast(point_id: str, hydro_year: int, issue_day: int, kind: str) -> dict | None:
    T_all, S_all, T_full = _matrices(point_id)
    if hydro_year not in T_all.index:
        return None
    last = last_weather_day(point_id, hydro_year)
    if last is None or last < issue_day:
        issue_day = last if last is not None else issue_day
    clim = F.past_clim(T_full, hydro_year)
    cal = calendar_day(point_id, hydro_year, kind)
    if clim is None or np.isnan(cal):
        return None
    hist = history(point_id)
    meta = point_meta(point_id)
    fz_obs = float(hist.loc[hydro_year, "freezeup_day"]) if hydro_year in hist.index else np.nan
    alpha = _alpha(meta.get("gauge"), hydro_year)
    theta = np.nan
    if kind == "close":
        th = F.thaw_thresholds(T_full, pd.concat([hist["close_day"], pd.Series({hydro_year: np.nan})]))
        theta = th.get(hydro_year, np.nan)
        if np.isnan(theta):
            vals = [v for y, v in th.items() if y < hydro_year and not np.isnan(v)]
            theta = vals[-1] if vals else np.nan
    f = F.season_features(T_all, S_all, hydro_year, int(issue_day), clim, alpha, fz_obs, theta)
    phys = f.pop("phys_open") if kind == "open" else f.pop("phys_close")
    f.pop("phys_close" if kind == "open" else "phys_open")
    phys = phys if not np.isnan(phys) else cal
    row = {"issue_day": issue_day, "lat": meta["lat"], "is_satellite": meta["is_satellite"],
           "cal_target": cal, "phys_pred": phys, "phys_minus_cal": phys - cal, **f}
    X = pd.DataFrame([row])[F.FEATURES[kind]]
    q = np.sort(np.asarray(_model(kind).predict(X))[0]) + cal
    corr = _conformal(kind)
    q[0], q[2] = q[0] - corr, q[2] + corr
    out = {"point_id": point_id, "hydro_year": hydro_year, "issue_day": int(issue_day), "kind": kind,
           "q10": q[0], "q50": q[1], "q90": q[2], "calendar": cal, "physics": phys,
           "actual": float(hist.loc[hydro_year, f"{kind}_day"]) if hydro_year in hist.index else np.nan,
           "h_stefan_now_cm": f["h_stefan_now"]}
    for k in ("q10", "q50", "q90", "calendar", "physics", "actual"):
        out[f"{k}_date"] = from_season_day(out[k], hydro_year).date() if not np.isnan(out[k]) else None
    return out
