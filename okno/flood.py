"""Высокая вода: проверяемые внешние расчёты GloFAS, без подмены замерами."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flood"
API = "https://flood-api.open-meteo.com/v1/flood"
SOURCE = "https://open-meteo.com/en/docs/flood-api"
# Координаты речных ячеек, не центров городов и не официальных гидропостов.
POINTS = {
    "lena_yakutsk": dict(region="Якутия", river="Лена", name="Якутск", lat=62.025, lon=129.82501),
    "ishim_ishim": dict(region="Тюменская область", river="Ишим", name="Ишим", lat=56.125, lon=69.57501),
    "tobol_kurgan": dict(region="Курганская область", river="Тобол", name="Курган", lat=55.425003, lon=65.375015),
    "ural_orenburg": dict(region="Оренбургская область", river="Урал", name="Оренбург", lat=51.725006, lon=55.125015),
    "ural_orsk": dict(region="Оренбургская область", river="Урал", name="Орск", lat=51.225006, lon=58.575012),
}


def save_response(path, payload, url, fetched_at):
    """Атомарный конверт с неизменённым JSON-ответом и контрольной суммой."""
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    envelope = dict(payload=payload, sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    url=url, fetched_at=fetched_at, source=SOURCE,
                    model=("GloFAS v4 seamless (models=seamless_v4)" if "models=seamless_v4" in url
                           else "GloFAS v4 seamless (API default at retrieval)"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(envelope, ensure_ascii=True), encoding="utf-8")
    temp.replace(path)


def read_response(path):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    raw = json.dumps(envelope["payload"], ensure_ascii=True, separators=(",", ":"))
    if hashlib.sha256(raw.encode()).hexdigest() != envelope["sha256"]:
        raise ValueError("Не совпала SHA-256 исходного ответа")
    if not envelope["url"].startswith(API + "?") or envelope["source"] != SOURCE:
        raise ValueError("Неизвестный источник")
    return envelope


def daily_frame(payload, point):
    if payload.get("timezone") != "GMT" or payload.get("utc_offset_seconds") != 0:
        raise ValueError("Ожидались исходные UTC-даты GloFAS")
    coords = (payload["latitude"], payload["longitude"])
    if not all(np.isfinite(value) for value in coords):
        raise ValueError("Некорректные координаты речной ячейки")
    if abs(coords[0] - point["lat"]) > .03 or abs(coords[1] - point["lon"]) > .03:
        raise ValueError("Поставщик изменил речную ячейку")
    fields = {k: v for k, v in payload["daily"].items() if k.startswith("river_discharge")}
    if not fields or any(payload["daily_units"].get(k) != "m³/s" for k in fields):
        raise ValueError("Не подтверждены единицы расхода м³/с")
    df = pd.DataFrame(fields, index=pd.to_datetime(payload["daily"]["time"], errors="raise"))
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Повторяющиеся или неупорядоченные даты")
    if df.index.hasnans or df.index.tz is not None or not (df.index == df.index.normalize()).all():
        raise ValueError("Ожидались календарные даты без времени")
    if ((df < 0) | np.isinf(df)).any().any():
        raise ValueError("Некорректный расход")
    return df


def threshold(history, before_year=None):
    """q80 годовых максимумов; только полные календарные годы до before_year."""
    series = history["river_discharge"]
    if before_year is not None:
        series = series[series.index.year < before_year]
    maxima = []
    for year, group in series.groupby(series.index.year):
        days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if group.notna().sum() >= .95 * days:
            maxima.append(float(group.max()))
    if len(maxima) < 20:
        raise ValueError("Менее 20 полных лет для порога")
    value = float(np.quantile(maxima, .8))
    if value <= 0:
        raise ValueError("Нулевой порог: проверьте речную ячейку")
    return value, len(maxima)


def summarize(history_envelope, forecast_envelope, point, now=None, horizon=7, *, reference=None, planning_date=None):
    now = now or datetime.now(timezone.utc)
    fetched = datetime.fromisoformat(forecast_envelope["fetched_at"])
    if fetched.tzinfo is None or not timedelta(0) <= now - fetched <= timedelta(hours=48):
        raise ValueError("Прогноз устарел (>48 ч) или время загрузки некорректно")
    forecast = daily_frame(forecast_envelope["payload"], point)
    decision_date = planning_date or now.astimezone(timezone.utc).date()
    start = decision_date + timedelta(days=1)
    dates = pd.date_range(start, periods=horizon)
    if reference is None:
        hist = daily_frame(history_envelope["payload"], point)
        q, years = threshold(hist, before_year=start.year)
    else:
        q, years = reference
    members = [c for c in forecast if c.startswith("river_discharge_member")]
    future = forecast.reindex(dates)[members].dropna(axis=1)
    if future.shape[1] < 30:
        raise ValueError("Недостаточно полных сценариев на выбранные 7 суток")
    peaks = future.max(axis=0)
    fraction = float((peaks > q).mean())
    median = future.median(axis=1)
    return dict(fraction=fraction, threshold=q, years=years, members=len(peaks),
                start=str(dates[0].date()), end=str(dates[-1].date()),
                peak_date=str(median.idxmax().date()), peak=float(median.max()),
                fetched_at=forecast_envelope["fetched_at"],
                curve=pd.DataFrame({"q10": future.quantile(.1, axis=1), "q50": median,
                                    "q90": future.quantile(.9, axis=1)}), scenarios=future)


@lru_cache(maxsize=32)
def _history_reference(path, pid, mtime_ns, size, before_year):
    """Статичный архив проверяется и обрабатывается один раз на версию файла."""
    frame = daily_frame(read_response(path)["payload"], POINTS[pid])
    return threshold(frame, before_year=before_year)


def forecast_as_of(pid, decision_time, data_dir=DATA):
    """Только действительно сохранённые ответы, полученные не позже решения."""
    paths = [data_dir / "latest" / f"{pid}.json", *(data_dir / "runs").glob(f"*/{pid}.json")]
    candidates = []
    for path in paths:
        if path.exists():
            envelope = read_response(path)
            fetched = datetime.fromisoformat(envelope["fetched_at"])
            if fetched.tzinfo is None:
                raise ValueError("В выпуске не указан часовой пояс")
            if fetched <= decision_time:
                candidates.append((fetched, envelope))
    if not candidates:
        raise ValueError("На выбранную дату нет сохранённого выпуска прогноза")
    return max(candidates, key=lambda item: item[0])[1]


def load_points(now=None, data_dir=DATA, planning_date=None):
    now = now or datetime.now(timezone.utc)
    planning_date = planning_date or now.date()
    if planning_date > now.date():
        raise ValueError("Дата решения не может быть в будущем")
    decision_time = now if planning_date == now.date() else datetime.combine(
        planning_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(microseconds=1)
    rows, details = [], {}
    for pid, point in POINTS.items():
        row = dict(pid=pid, **point, fraction=None, status="нет данных")
        try:
            path = data_dir / "history" / f"{pid}.json"
            stat = path.stat()
            cutoff = (planning_date + timedelta(days=1)).year
            reference = _history_reference(path, pid, stat.st_mtime_ns, stat.st_size, cutoff)
            result = summarize(None, forecast_as_of(pid, decision_time, data_dir),
                               point, decision_time, reference=reference, planning_date=planning_date)
            details[pid] = result
            row.update({k: v for k, v in result.items() if k not in ("curve", "scenarios")})
            row["status"] = "расчёт доступен"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["status"] = "Данные ещё не загружены" if isinstance(exc, FileNotFoundError) else str(exc)
        rows.append(row)
    return pd.DataFrame(rows), details
