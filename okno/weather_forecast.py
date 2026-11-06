"""Прогноз закрытия с архивом GFS; наблюдения будущих дней не используются.

Previous Runs хранит фиксированные горизонты, а не полный отдельный запуск.
Для дня D+k берём offset k+1: все исходные прогнозы относятся к D-1.
Это оставляет запас на публикацию модели и даёт шесть полных будущих суток.
"""

from functools import lru_cache
import json

import numpy as np
import pandas as pd

from . import features as F
from .ice import from_season_day
from .paths import MODELS, WEATHER_FORECASTS

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_seamless"
HORIZON = 6
ARCHIVE_START = pd.Timestamp("2021-04-01")
EXTRA_FEATURES = ["weather_days", "weather_mean_6d", "weather_anomaly_6d",
                  "weather_tdd_6d", "weather_thaw_days_6d"]
MODEL_FEATURES = F.FEATURES["close"] + EXTRA_FEATURES
MODEL_PATH = MODELS / "m1_close_weather.cbm"
METADATA_PATH = MODELS / "m1_close_weather.json"
COLUMNS = ["point_id", "issue_date", "valid_date", "temperature_2m_mean", "model",
           "source_date", "offset_days", "available_at", "source"]


def validate_snapshots(frame):
    """Отсечь поздно выпущенные прогнозы, неверные горизонты и другой тип данных."""
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError("Неполная схема архива метеопрогнозов")
    frame = frame[COLUMNS].copy()
    for column in ("issue_date", "valid_date", "source_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    frame["temperature_2m_mean"] = pd.to_numeric(frame["temperature_2m_mean"], errors="raise")
    lead = (frame.valid_date - frame.issue_date).dt.days
    cutoff = (frame.issue_date + pd.Timedelta(days=1)).dt.tz_localize("UTC")
    earliest_availability = (frame.source_date + pd.Timedelta(days=1, hours=6)).dt.tz_localize("UTC")
    valid = (lead.between(1, HORIZON) & frame.offset_days.eq(lead + 1)
             & frame.source_date.eq(frame.issue_date - pd.Timedelta(days=1))
             & frame.available_at.ge(earliest_availability) & frame.available_at.lt(cutoff)
             & frame.model.eq(MODEL) & frame.source.eq(API_URL)
             & np.isfinite(frame.temperature_2m_mean) & frame.temperature_2m_mean.between(-90, 60))
    if not valid.all() or frame.isna().any().any():
        raise ValueError("Архив содержит недоступные на дату решения или некорректные прогнозы")
    if frame.duplicated(["point_id", "issue_date", "valid_date"]).any():
        raise ValueError("Повторные даты в архиве метеопрогнозов")
    return frame.sort_values(["point_id", "issue_date", "valid_date"]).reset_index(drop=True)


def from_hourly(payload, point_id, issue_dates):
    """Только 24 прогнозных часа образуют сутки; пропуски не заменяем фактом."""
    if payload.get("timezone") not in ("GMT", "UTC") or payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("Архив должен быть в UTC")
    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"], errors="raise")
    if hourly.time.duplicated().any():
        raise ValueError("Повторные часы в ответе GFS")
    hourly["day"] = hourly.time.dt.normalize()
    rows = []
    for issue in sorted(set(pd.Timestamp(date).normalize() for date in issue_dates)):
        for lead in range(1, HORIZON + 1):
            valid_date = issue + pd.Timedelta(days=lead)
            column = f"temperature_2m_previous_day{lead + 1}"
            values = hourly.loc[hourly.day.eq(valid_date), column].dropna()
            if len(values) != 24:
                continue
            rows.append({
                "point_id": point_id, "issue_date": issue, "valid_date": valid_date,
                "temperature_2m_mean": float(values.mean()), "model": MODEL,
                "source_date": issue - pd.Timedelta(days=1), "offset_days": lead + 1,
                "available_at": (issue + pd.Timedelta(hours=6)).tz_localize("UTC"),
                "source": API_URL,
            })
    return validate_snapshots(pd.DataFrame(rows, columns=COLUMNS))


@lru_cache(maxsize=32)
def _read_archive(path, version):
    return validate_snapshots(pd.read_csv(path))


def archive(point_id):
    path = WEATHER_FORECASTS / f"{point_id}.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return _read_archive(path, path.stat().st_mtime_ns)


def weather_features(snapshots, point_id, hydro_year, issue_day, clim, *, available_before=None):
    missing = {name: np.nan for name in EXTRA_FEATURES}
    missing["weather_days"] = 0
    if snapshots.empty:
        return missing
    issue = from_season_day(issue_day, hydro_year).normalize()
    rows = snapshots[snapshots.point_id.eq(point_id) & snapshots.issue_date.eq(issue)].sort_values("valid_date")
    if available_before is not None:
        rows = rows[rows.available_at.le(pd.Timestamp(available_before))]
    if len(rows) != HORIZON:
        return missing
    temperatures = rows.temperature_2m_mean.to_numpy(dtype=float)
    days = (rows.valid_date - from_season_day(0, hydro_year)).dt.days.to_numpy()
    if days.min() < 0 or days.max() >= len(clim):
        return missing
    return {
        "weather_days": HORIZON, "weather_mean_6d": float(temperatures.mean()),
        "weather_anomaly_6d": float((temperatures - clim[days]).mean()),
        "weather_tdd_6d": float(np.maximum(temperatures, 0).sum()),
        "weather_thaw_days_6d": int((temperatures > 0).sum()),
    }


def training_rows(base):
    """Те же базовые признаки и прогнозные признаки; без архива — явные пропуски."""
    out = base.copy()
    for name in EXTRA_FEATURES:
        out[name] = 0 if name == "weather_days" else np.nan
    for point_id, group in out.groupby("point_id"):
        snapshots = archive(point_id)
        if snapshots.empty:
            continue
        matrix = F.season_matrix(point_id)
        for year, rows in group.groupby("hydro_year"):
            clim = F.past_clim(matrix, year)
            if clim is None:
                continue
            for index, row in rows.iterrows():
                values = weather_features(snapshots, point_id, year, int(row.issue_day), clim)
                for name, value in values.items():
                    out.loc[index, name] = value
    return out


@lru_cache(maxsize=4)
def _load_model(path, version):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def predict_close(point_id, hydro_year, issue_day, base_row, clim):
    """Эксперимент применяется только после обучающих сезонов и с полным архивом."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return {"message": "Погодная модель ещё не обучена; используется обычная М1."}
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("features") != MODEL_FEATURES:
        return {"message": "Погодная модель требует переобучения; используется обычная М1."}
    if point_id not in metadata["point_ids"]:
        return {"message": "Для этой точки нет обучающего архива метеопрогнозов; используется обычная М1."}
    if hydro_year <= metadata["trained_through_season"]:
        return {"message": "Эта дата входит в обучающие сезоны; используется обычная М1. Проверка эксперимента — в отчёте."}
    values = weather_features(archive(point_id), point_id, hydro_year, issue_day, clim,
                              available_before=pd.Timestamp.now(tz="UTC"))
    if values["weather_days"] != HORIZON:
        return {"message": "На дату решения нет полного шестидневного прогноза GFS; используется обычная М1."}
    from catboost import CatBoostError
    try:
        model = _load_model(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
        quantiles = np.sort(np.asarray(model.predict(pd.DataFrame([{**base_row, **values}])[MODEL_FEATURES]))[0])
    except CatBoostError:
        return {"message": "Погодная модель недоступна; используется обычная М1."}
    quantiles += base_row["cal_target"]
    correction = metadata["correction_days"]
    quantiles[0] -= correction
    quantiles[2] += correction
    return {"quantiles": np.maximum(quantiles, issue_day), "days": HORIZON,
            "message": "Экспериментальная М1 с архивом GFS: учтены ближайшие 6 дней. "
                       "Выигрыш за 7/14 дней не подтверждён; диапазон пока ненадёжен."}
