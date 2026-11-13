"""
Физика льда: градусо-дни и закон Стефана.

h(t) = alpha * sqrt(FDD(t)),  FDD(t) = сумма max(0, -T) от даты ледостава до t (°C·сут).
alpha подбирается по реальным замерам толщины на гидропостах (см/(°C·сут)^0.5).
Литературный диапазон для речного льда под снегом: ~1.4–2.7.

Сезон (гидрологический год hy): с 1 сентября года hy-1 по 31 августа года hy.
День сезона: 0 = 1 сентября.
"""

import datetime as dt

import numpy as np
import pandas as pd

from .paths import WEATHER

SEASON_START_MONTH = 9


def season_start(hy: int) -> pd.Timestamp:
    return pd.Timestamp(hy - 1, SEASON_START_MONTH, 1)


def to_season_day(date, hy: int) -> float:
    if date is None or (isinstance(date, float) and np.isnan(date)) or pd.isna(date):
        return np.nan
    return float((pd.Timestamp(date) - season_start(hy)).days)


def from_season_day(day: float, hy: int) -> pd.Timestamp:
    return season_start(hy) + pd.Timedelta(days=float(day))


def hydro_year(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year + (d.dt.month >= SEASON_START_MONTH).astype(int)


def load_weather(point_id: str, live: bool = False) -> pd.DataFrame:
    """Суточная погода ERA5 для точки; ищет в data/weather и data/weather/gauges.
    live=True — дописывает после конца архива оперативные данные Open-Meteo (последние дни)
    и прогноз на 16 дней (колонка is_forecast). Для обучения не используется."""
    from .paths import RECENT
    for p in (WEATHER / f"{point_id}.csv", WEATHER / "gauges" / f"{point_id}.csv", WEATHER / "lena" / f"{point_id}.csv"):
        if p.exists():
            w = pd.read_csv(p, parse_dates=["date"])
            w["is_forecast"] = False
            rp = RECENT / f"{point_id}.csv"
            if live and rp.exists():
                r = pd.read_csv(rp, parse_dates=["date"])
                r = r[r["date"] > w["date"].max()]
                w = pd.concat([w, r[[c for c in w.columns if c in r.columns]]], ignore_index=True)
                w["is_forecast"] = w["is_forecast"].astype(bool)
            w = w.rename(columns={"temperature_2m_mean": "t", "snowfall_sum": "snow_cm",
                                  "precipitation_sum": "precip_mm"})
            w["hydro_year"] = hydro_year(w["date"])
            # векторно (раньше — season_start на каждую строку, ~1 млн вызовов при открытии дашборда)
            start = pd.to_datetime(pd.DataFrame({"year": w["hydro_year"] - 1, "month": SEASON_START_MONTH, "day": 1}))
            w["season_day"] = (w["date"] - start).dt.days
            return w
    raise FileNotFoundError(point_id)


def stefan_thickness(t_mean: np.ndarray, alpha: float, start_idx: int = 0) -> np.ndarray:
    """Толщина льда по дням от start_idx (дня ледостава); до ледостава — 0."""
    fdd = np.cumsum(np.maximum(0.0, -np.asarray(t_mean, dtype=float)))
    fdd = fdd - (fdd[start_idx - 1] if start_idx > 0 else 0.0)
    h = alpha * np.sqrt(np.maximum(fdd, 0.0))
    h[:start_idx] = 0.0
    return h


def required_thickness_cm(load_t: float, k: float = 11.0) -> float:
    """
    ДОПУЩЕНИЕ: нормативная толщина пресноводного льда для колёсной нагрузки P (т),
    h = k * sqrt(P), k ≈ 11 (типовая инженерная оценка: 10 т → 35 см, 20 т → 49 см, 30 т → 60 см).
    Сверить с ОДН 218.010-98 «Инструкция по проектированию, строительству и эксплуатации
    ледовых переправ»; чувствительность проверяется ±30%.
    """
    return k * float(np.sqrt(load_t))


def season_weather(w: pd.DataFrame, hy: int) -> pd.DataFrame:
    s = w[w["hydro_year"] == hy].sort_values("date").reset_index(drop=True)
    return s


def freezeup_from_weather(s: pd.DataFrame, window: int = 5) -> int | None:
    """
    Оценка дня ледостава только по погоде (там, где нет поста):
    первый день сезона, после которого `window` дней подряд средняя T < 0
    и накоплено ≥ 30 °C·сут мороза. Грубая оценка, используется как признак/запасной вариант.
    """
    t = s["t"].to_numpy()
    neg = (t < 0).astype(int)
    run = np.convolve(neg, np.ones(window, dtype=int), mode="full")[: len(t)]
    fdd = np.cumsum(np.maximum(0.0, -t))
    idx = np.where((run >= window) & (fdd >= 30))[0]
    return int(s["season_day"].iloc[idx[0]]) if len(idx) else None
