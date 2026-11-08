"""
Прогноз на произвольную дату без знания факта — для дашборда, бота и разбора сезонов.

forecast(point_id, hydro_year, issue_day, kind) → квантили даты события, календарь, физика.
Использует только погоду до issue_day и историю дат за прошлые сезоны.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import features as F
from . import weather_forecast as WF
from . import snow as SN
from .ice import from_season_day, load_weather
from .paths import MODELS, PROCESSED, RECENT, WEATHER, WEATHER_FORECASTS

POINTS = {
    "yakutsk_crossing": {"name": "Переправа Якутск — Нижний Бестях", "lat": 62.0, "lon": 129.82,
                         "is_satellite": 1, "gauge": None},
}


@lru_cache(maxsize=None)
def _gauges():
    return pd.read_csv(PROCESSED / "gauges.csv")


@lru_cache(maxsize=None)
@lru_cache(maxsize=None)
def _lena_points():
    from .paths import DATA
    f = DATA / "lena_points.csv"
    return pd.read_csv(f).set_index("point_id") if f.exists() else pd.DataFrame(columns=["lat", "lon"])


def is_basin(point_id: str) -> bool:
    """Точка бассейна Лены (притоки) — свой прогноз закрытия (models/m1_close_lena.cbm), открытия нет."""
    return point_id.startswith("lena_")


def history(point_id: str) -> pd.DataFrame:
    """Фактические даты прошлых сезонов: index = hydro_year; open_day, close_day, freezeup_day."""
    if is_basin(point_id):
        s = pd.read_csv(PROCESSED / "lena_sat_seasons.csv")
        s = s[s.point_id == point_id]
        return pd.DataFrame({"open_day": np.nan, "close_day": s["sat_breakup_day"].to_numpy(),
                             "freezeup_day": s["sat_freezeup_day"].to_numpy()}, index=s["hydro_year"].to_numpy())
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
    if is_basin(point_id):
        r = _lena_points().loc[point_id]
        return {"name": point_id, "lat": float(r["lat"]), "lon": float(r["lon"]), "is_satellite": 1, "gauge": None}
    code = int(point_id.split("_")[1])
    g = _gauges().set_index("gauge_code").loc[code]
    return {"name": g["name"], "lat": float(g["lat"]), "lon": float(g["lon"]), "is_satellite": 0, "gauge": code}


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _weather_path(point_id):
    for p in (WEATHER / f"{point_id}.csv", WEATHER / "gauges" / f"{point_id}.csv", WEATHER / "lena" / f"{point_id}.csv"):
        if p.exists():
            return p
    return WEATHER / "gauges" / f"{point_id}.csv"


def data_version() -> float:
    """Метка свежести данных: меняется, когда ежедневное обновление дописывает погоду
    или переобучаются модели. Кэши ниже привязаны к ней, поэтому дашборд подхватывает
    новые данные без перезапуска."""
    files = (list(WEATHER.glob("*.csv")) + list((WEATHER / "gauges").glob("*.csv")) + list(MODELS.glob("*"))
             + list(RECENT.glob("*.csv")) + list(WEATHER_FORECASTS.glob("*.csv"))
             + list((PROCESSED.parent/'snow_depth').glob('*.csv'))
             + [PROCESSED/'hydro_snow_on_ice.csv',PROCESSED/'gauges.csv',PROCESSED/'hydro_ice_seasons.csv'])
    return max((_mtime(f) for f in files), default=0.0)


def _model(kind):
    return _model_v(kind, _mtime(MODELS / f"m1_{kind}.cbm"))


@lru_cache(maxsize=None)
def _model_v(kind, _ver):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor()
    m.load_model(str(MODELS / f"m1_{kind}.cbm"))
    return m


def _conformal(kind):
    return _conformal_v(kind, _mtime(MODELS / f"m1_{kind}_conformal.json"))


@lru_cache(maxsize=None)
def _conformal_v(kind, _ver):
    import json
    p = MODELS / f"m1_{kind}_conformal.json"
    return json.loads(p.read_text())["correction_days"] if p.exists() else 0.0


def _matrices(point_id):
    """(T_all, S_all, T_full): наблюдения (архив ERA5 + оперативные последние дни) и полные сезоны."""
    return _matrices_v(point_id, _mtime(_weather_path(point_id)), _mtime(RECENT / f"{point_id}.csv"))[:3]


def _ahead(point_id):
    """Прогноз погоды на ближайшие 16 дней: таблица сезон × день (только прогнозные дни)."""
    return _matrices_v(point_id, _mtime(_weather_path(point_id)), _mtime(RECENT / f"{point_id}.csv"))[3]


@lru_cache(maxsize=32)
def _matrices_v(point_id, _ver, _ver_recent):
    w = load_weather(point_id, live=True)
    obs, fut = w[~w["is_forecast"]], w[w["is_forecast"]]

    def mat(d, col):
        return d.pivot_table(index="hydro_year", columns="season_day", values=col).reindex(columns=range(365))

    T_all, S_all = mat(obs, "t"), mat(obs, "snow_cm")
    full = T_all.notna().sum(axis=1) > 330
    T_ahead = mat(fut, "t") if len(fut) else pd.DataFrame(columns=range(365))
    return T_all, S_all, T_all[full], T_ahead


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


def forecast(point_id: str, hydro_year: int, issue_day: int, kind: str, *, use_weather=False,use_snow=False) -> dict | None:
    if is_basin(point_id) and kind == "open":
        return None  # в бассейне Лены нет замеров толщины — прогнозируем только вскрытие
    model_key = "close_lena" if is_basin(point_id) and kind == "close" else kind
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
    # живой режим (прогноз выпускается на последний день наблюдений): ближайшие 16 дней берём из
    # прогноза погоды, а не из климатологии. В обучении прогнозов не было — это улучшает физическую часть.
    T_ahead = _ahead(point_id)
    ahead = None
    if last is not None and issue_day >= last and hydro_year in T_ahead.index:
        ahead = T_ahead.loc[hydro_year].to_numpy(dtype=float)
    f = F.season_features(T_all, S_all, hydro_year, int(issue_day), clim, alpha, fz_obs, theta, t_ahead=ahead)
    phys = f.pop("phys_open") if kind == "open" else f.pop("phys_close")
    f.pop("phys_close" if kind == "open" else "phys_open")
    phys = phys if not np.isnan(phys) else cal
    row = {"issue_day": issue_day, "lat": meta["lat"], "is_satellite": meta["is_satellite"],
           "cal_target": cal, "phys_pred": phys, "phys_minus_cal": phys - cal, **f}
    X = pd.DataFrame([row])[F.FEATURES[kind]]
    q = np.sort(np.asarray(_model(model_key).predict(X))[0]) + cal
    corr = _conformal(model_key)
    q[0], q[2] = q[0] - corr, q[2] + corr
    q = np.maximum(q, issue_day)  # событие ещё не наступило — раньше даты прогноза его быть не может
    prediction_model=f'm1_{model_key}.cbm'
    # Updated models with/without snow share the same training data and fixed
    # physical coefficient. The checkbox therefore isolates snow, not a change
    # in the training set or calibration method.
    sf=F.season_features(T_all,S_all,hydro_year,int(issue_day),clim,F.ALPHA_DEFAULT,fz_obs,theta)
    sp=sf.pop('phys_open' if kind=='open' else 'phys_close')
    sf.pop('phys_close' if kind=='open' else 'phys_open')
    sp=sp if np.isfinite(sp) else cal
    sr={'issue_day':issue_day,'lat':meta['lat'],'is_satellite':meta['is_satellite'],
        'cal_target':cal,'phys_pred':sp,'phys_minus_cal':sp-cal,**sf}
    paired=SN.predict(point_id,hydro_year,int(issue_day),kind,sr,use_snow=False)
    if 'quantiles' in paired:
        q=paired['quantiles'];phys=sp;f['h_stefan_now']=sf['h_stefan_now']
        prediction_model=paired['model_file']
    weather_message, weather_applied, weather_delta = "", False, None
    days_used = int(((np.arange(365) > issue_day) & np.isfinite(ahead)).sum()) if ahead is not None else 0
    if kind == "close" and use_weather:
        # Эксперимент обучен с климатологической физикой и отдельными признаками GFS.
        # Не подмешиваем сюда оперативный best_match, которого не было в обучении.
        wf = F.season_features(T_all, S_all, hydro_year, int(issue_day), clim, alpha, fz_obs, theta)
        wf_phys = wf.pop("phys_close")
        wf.pop("phys_open")
        wf_phys = wf_phys if np.isfinite(wf_phys) else cal
        wf_row = {"issue_day": issue_day, "lat": meta["lat"], "is_satellite": meta["is_satellite"],
                  "cal_target": cal, "phys_pred": wf_phys, "phys_minus_cal": wf_phys - cal, **wf}
        try:
            result = WF.predict_close(point_id, hydro_year, int(issue_day), wf_row, clim)
        except (OSError, ValueError, KeyError):
            result = {"message": "Архив или погодная модель недоступны; используется обычная М1."}
        weather_message = result["message"]
        if "quantiles" in result:
            weather_delta = float(result["quantiles"][1] - q[1])
            q = result["quantiles"]
            phys = wf_phys
            days_used, weather_applied = result["days"], True
            prediction_model=WF.MODEL_PATH.name
    snow_applied,snow_message,snow_delta,snow_inputs,snow_validation=False,'',None,{},{}
    if use_snow:
        if weather_applied:
            snow_message='Применён погодный эксперимент GFS; отдельная снежная поправка с ним не складывается.'
        else:
            result=SN.predict(point_id,hydro_year,int(issue_day),kind,sr)
            snow_message=result['message']
            if 'quantiles' in result:
                snow_delta=float(result['quantiles'][1]-q[1]);q=result['quantiles'];phys=sp
                snow_applied=True;snow_inputs=result['features'];snow_validation=result['validation']
                prediction_model=result['model_file']
    out = {"point_id": point_id, "hydro_year": hydro_year, "issue_day": int(issue_day), "kind": kind,
           "q10": q[0], "q50": q[1], "q90": q[2], "calendar": cal, "physics": phys,
           "actual": float(hist.loc[hydro_year, f"{kind}_day"]) if hydro_year in hist.index else np.nan,
           "h_stefan_now_cm": f["h_stefan_now"],
           "forecast_days_used": days_used, "weather_model_applied": weather_applied,
           "weather_model_message": weather_message, "weather_delta_days": weather_delta,
           'snow_model_applied':snow_applied,'snow_model_message':snow_message,'snow_delta_days':snow_delta,
           'snow_features':snow_inputs,'snow_validation':snow_validation,
           'updated_model_applied':'quantiles' in paired,'prediction_model':prediction_model}
    for k in ("q10", "q50", "q90", "calendar", "physics", "actual"):
        out[f"{k}_date"] = from_season_day(out[k], hydro_year).date() if not np.isnan(out[k]) else None
    return out
