"""
Датасет для М1: одна строка = точка × сезон × дата выпуска прогноза.

Честность во времени:
- погодные признаки — только наблюдения до даты выпуска (issue_day) включительно;
- будущая погода заменяется климатологией за ПРОШЛЫЕ сезоны (до 15 лет назад);
- «календарь» = средняя дата события за прошлые сезоны (до 10 лет назад) — так планируют сейчас;
- коэффициент Стефана alpha для поста подобран по замерам других сезонов (без текущего).

Цели (день сезона, 0 = 1 сентября):
- open  — измеренная толщина льда достигла нормативной для 20 т (гидропосты);
- close — начало разрушения ледостава: «начало таяния» на гидропосте,
          для переправы Якутск — спутниковое вскрытие (source_type = спутник).
"""

import numpy as np
import pandas as pd

from .calib import fit_alpha, stefan_pairs
from .ice import load_weather, required_thickness_cm
from .paths import PROCESSED, REPORTS

ALPHA_DEFAULT = 1.8
OPEN_LOAD_T = 20
SPRING_START = 181  # 1 марта (день сезона): с него копим оттепельные градусо-дни (TDD)
CAL_WINDOW, CAL_MIN = 10, 3
CLIM_WINDOW, CLIM_MIN = 15, 5

OPEN_ISSUE_DAYS = list(range(30, 121, 7))     # 1 окт … конец дек, раз в неделю
CLOSE_ISSUE_DAYS = list(range(150, 271, 7))   # 29 янв … конец мая, раз в неделю

FEATURES_COMMON = ["issue_day", "lat", "is_satellite", "cal_target", "phys_pred", "phys_minus_cal",
                   "fdd_to_date", "fdd_anom", "t_last14", "t_last30", "t_anom30", "snow_to_date",
                   "freezeup_known", "freezeup_used", "h_stefan_now"]
FEATURES = {"open": FEATURES_COMMON,
            "close": FEATURES_COMMON + ["tdd_to_date", "tdd_anom", "h_stefan_max", "thaw_days_14", "t_anom_spring"]}


def season_matrix(point_id: str, var: str = "t") -> pd.DataFrame:
    w = load_weather(point_id)
    m = w.pivot_table(index="hydro_year", columns="season_day", values=var)
    m = m.reindex(columns=range(365))
    full = m.notna().sum(axis=1) > 330
    return m[full]


def past_mean(series: pd.Series, year: int, window: int, min_n: int):
    past = series[(series.index < year) & (series.index >= year - window)].dropna()
    return float(past.mean()) if len(past) >= min_n else np.nan


def past_clim(T: pd.DataFrame, year: int):
    past = T[(T.index < year) & (T.index >= year - CLIM_WINDOW)]
    if len(past) < CLIM_MIN:
        return None
    c = past.mean(axis=0).to_numpy()
    return np.where(np.isnan(c), np.nanmean(c), c)


def _first_cross(values, thr, start):
    idx = np.where(values[start:] >= thr)[0]
    return float(start + idx[0]) if len(idx) else np.nan


def _freezeup_est(t, window=5):
    neg = (t < 0).astype(int)
    run = np.convolve(neg, np.ones(window, dtype=int), mode="full")[: len(t)]
    fdd = np.cumsum(np.maximum(0, -t))
    idx = np.where((run >= window) & (fdd >= 30))[0]
    return float(idx[0]) if len(idx) else np.nan


def season_features(T, S, y, d, clim, alpha, fz_obs, theta=np.nan, t_ahead=None):
    """Признаки и физический прогноз для сезона y на дату выпуска d (день сезона)."""
    t_obs = T.loc[y].to_numpy()
    t_proj = clim.copy()
    t_proj[: d + 1] = np.where(np.isnan(t_obs[: d + 1]), clim[: d + 1], t_obs[: d + 1])
    if t_ahead is not None:  # живой режим: прогноз погоды на ближайшие дни вместо климатологии
        fut = (np.arange(365) > d) & np.isfinite(t_ahead)
        t_proj = np.where(fut, t_ahead, t_proj)
    snow = np.nan_to_num(S.loc[y].to_numpy()) if S is not None and y in S.index else np.zeros(365)
    fz_known = fz_obs if (not np.isnan(fz_obs) and fz_obs <= d) else np.nan
    fz = fz_known if not np.isnan(fz_known) else _freezeup_est(t_proj)
    fz_i = int(fz) if not np.isnan(fz) else 60
    days = np.arange(365)
    h = alpha * np.sqrt(np.cumsum(np.where(days >= fz_i, np.maximum(0, -t_proj), 0)))
    tdd = np.cumsum(np.where(days >= SPRING_START, np.maximum(0, t_proj), 0))
    tdd_clim = np.cumsum(np.where(days >= SPRING_START, np.maximum(0, clim), 0))
    fdd_obs = np.cumsum(np.maximum(0, -t_proj))
    fdd_clim = np.cumsum(np.maximum(0, -clim))
    sl = slice(max(0, d - 29), d + 1)
    return {
        "fdd_to_date": fdd_obs[d], "fdd_anom": fdd_obs[d] - fdd_clim[d],
        "t_last14": float(np.mean(t_proj[max(0, d - 13): d + 1])),
        "t_last30": float(np.mean(t_proj[sl])),
        "t_anom30": float(np.mean(t_proj[sl] - clim[sl])),
        "snow_to_date": float(snow[: d + 1].sum()),
        "freezeup_known": fz_known, "freezeup_used": fz,
        "h_stefan_now": float(h[d]), "h_stefan_max": float(h[SPRING_START]),
        "tdd_to_date": float(tdd[d]), "tdd_anom": float(tdd[d] - tdd_clim[d]),
        "thaw_days_14": int((t_proj[max(0, d - 13): d + 1] > 0).sum()),
        "t_anom_spring": float(np.mean(t_proj[SPRING_START: d + 1] - clim[SPRING_START: d + 1]))
        if d >= SPRING_START else 0.0,
        "phys_open": _first_cross(h, required_thickness_cm(OPEN_LOAD_T), fz_i),
        "phys_close": _first_cross(tdd, theta, max(d, SPRING_START)) if not np.isnan(theta) else np.nan,
    }


def thaw_thresholds(T: pd.DataFrame, targets: pd.Series) -> dict:
    """θ для сезона y — медиана TDD к дате события в прошлых сезонах."""
    vals = {}
    for y, td in targets.dropna().items():
        if y in T.index:
            t = np.nan_to_num(T.loc[y].to_numpy())
            tdd = np.cumsum(np.where(np.arange(365) >= SPRING_START, np.maximum(0, t), 0))
            vals[y] = tdd[int(td)]
    s = pd.Series(vals)
    return {y: past_mean(s, y, CAL_WINDOW, CAL_MIN) for y in targets.index}


def alpha_by_year(gauge_code, seasons, thick, years):
    """alpha поста, подобранный по замерам всех сезонов, кроме данного."""
    pairs = stefan_pairs(gauge_code, seasons, thick)
    if pairs.empty:
        return {y: load_alpha() for y in years}
    return {y: fit_alpha(pairs[pairs.hydro_year != y])["alpha"] for y in years}


def load_alpha() -> float:
    p = REPORTS / "stefan_calibration.csv"
    if p.exists():
        c = pd.read_csv(p)
        return float(np.average(c["alpha"], weights=c["n"]))
    return ALPHA_DEFAULT


def point_rows(point_id, meta, targets: pd.DataFrame, alphas: dict, kind: str):
    """targets: index = hydro_year, колонки target_day, freezeup_day."""
    T = season_matrix(point_id, "t")
    S = season_matrix(point_id, "snow_cm")
    theta = thaw_thresholds(T, targets["target_day"]) if kind == "close" else {}
    rows = []
    issue_days = OPEN_ISSUE_DAYS if kind == "open" else CLOSE_ISSUE_DAYS
    for y in targets.index:
        if y not in T.index:
            continue
        clim = past_clim(T, y)
        cal = past_mean(targets["target_day"], y, CAL_WINDOW, CAL_MIN)
        target = targets.loc[y, "target_day"]
        if clim is None or np.isnan(cal) or np.isnan(target):
            continue
        for d in issue_days:
            if d >= target:
                continue
            f = season_features(T, S, y, d, clim, alphas[y] if y in alphas else load_alpha(),
                                targets.loc[y, "freezeup_day"], theta.get(y, np.nan))
            phys = f.pop("phys_open") if kind == "open" else f.pop("phys_close")
            f.pop("phys_close" if kind == "open" else "phys_open")
            phys = phys if not np.isnan(phys) else cal
            rows.append({"point_id": point_id, "hydro_year": y, "issue_day": d, "kind": kind,
                         "lat": meta["lat"], "lon": meta["lon"], "is_satellite": int(meta.get("is_satellite", 0)),
                         "target_day": target, "cal_target": cal, "lead_days": target - d,
                         "phys_pred": phys, "phys_minus_cal": phys - cal, **f})
    return rows


def build_m1_dataset(kind: str, codes=None, include_sat: bool = True,
                     local_alpha: bool = True, lena: bool = False) -> pd.DataFrame:
    """codes — гидропосты (по умолчанию все якутские); local_alpha=False — коэффициент Стефана
    общий по Якутии, как для нового региона без замеров толщины."""
    seasons = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    thick = pd.read_csv(PROCESSED / "hydro_ice_thickness.csv")
    gauges = pd.read_csv(PROCESSED / "gauges.csv")
    col = f"open_{OPEN_LOAD_T}t_day" if kind == "open" else "melt_start_day"
    sel = gauges[gauges.yakutia] if codes is None else gauges[gauges.gauge_code.isin(codes)]
    rows = []
    for g in sel.itertuples():
        pid = f"gauge_{g.gauge_code}"
        try:
            load_weather(pid)
        except FileNotFoundError:
            continue
        s = seasons[(seasons.gauge_code == g.gauge_code) & seasons.valid]
        t = pd.DataFrame({"target_day": s[col].to_numpy(), "freezeup_day": s["freezeup_day"].to_numpy()},
                         index=s["hydro_year"].to_numpy())
        if t["target_day"].notna().sum() < 10:
            continue
        alphas = alpha_by_year(g.gauge_code, seasons, thick, t.index) if kind == "open" and local_alpha else {}
        rows += point_rows(pid, {"lat": g.lat, "lon": g.lon}, t, alphas, kind)
    if kind == "close" and lena:  # точки по всему руслу Лены и притокам (scripts/lena_points.py)
        rows += lena_rows(kind)
    if kind == "close" and include_sat:
        sat = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv")
        t = pd.DataFrame({"target_day": sat["sat_breakup_day"].to_numpy(),
                          "freezeup_day": sat["sat_freezeup_day"].to_numpy()}, index=sat["hydro_year"].to_numpy())
        rows += point_rows("yakutsk_crossing", {"lat": 62.0, "lon": 129.82, "is_satellite": 1}, t, {}, kind)
    return pd.DataFrame(rows)


def lena_rows(kind="close"):
    """Спутниковые сезоны вскрытия в точках бассейна Лены (Qiu et al. 2026), где есть погода."""
    from .paths import DATA
    pts_f, sat_f = DATA / "lena_points.csv", PROCESSED / "lena_sat_seasons.csv"
    if not (pts_f.exists() and sat_f.exists()):
        return []
    pts, sat = pd.read_csv(pts_f), pd.read_csv(sat_f)
    rows = []
    for p in pts.itertuples():
        try:
            load_weather(p.point_id)
        except FileNotFoundError:
            continue
        s = sat[sat.point_id == p.point_id]
        t = pd.DataFrame({"target_day": s["sat_breakup_day"].to_numpy(), "freezeup_day": s["sat_freezeup_day"].to_numpy()},
                         index=s["hydro_year"].to_numpy())
        if t["target_day"].notna().sum() < 10:
            continue
        rows += point_rows(p.point_id, {"lat": p.lat, "lon": p.lon, "is_satellite": 1}, t, {}, kind)
    return rows
