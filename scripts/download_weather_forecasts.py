"""Загрузка настоящих архивных прогнозов GFS для эксперимента закрытия.

python scripts/download_weather_forecasts.py --start-year 2021 --end-year 2024
python scripts/download_weather_forecasts.py --live
"""

import argparse
import datetime as dt
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from okno.forecast import point_meta
from okno.ice import from_season_day
from okno.paths import PROCESSED, WEATHER_FORECASTS
from okno.weather_forecast import API_URL, ARCHIVE_START, HORIZON, MODEL, from_hourly, validate_snapshots


def download(point_id, issues, session):
    issues = [pd.Timestamp(date).normalize() for date in issues if pd.Timestamp(date) >= ARCHIVE_START]
    if not issues:
        return pd.DataFrame()
    meta = point_meta(point_id)
    params = {"latitude": meta["lat"], "longitude": meta["lon"], "models": MODEL, "timezone": "GMT", "temperature_unit": "celsius",
              "hourly": ",".join(f"temperature_2m_previous_day{i}" for i in range(2, 8)),
              "start_date": (min(issues) + pd.Timedelta(days=1)).date().isoformat(),
              "end_date": (max(issues) + pd.Timedelta(days=HORIZON)).date().isoformat()}
    for attempt in range(4):
        response = session.get(API_URL, params=params, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
        response.raise_for_status()
        return from_hourly(response.json(), point_id, issues)
    raise RuntimeError("Не удалось загрузить архив GFS")


def save(point_id, frames):
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"{point_id}: архив не вернул полных прогнозных суток")
    WEATHER_FORECASTS.mkdir(parents=True, exist_ok=True)
    path = WEATHER_FORECASTS / f"{point_id}.csv"
    if path.exists():
        frames.insert(0, validate_snapshots(pd.read_csv(path)))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["point_id", "issue_date", "valid_date"], keep="last")
    frame = validate_snapshots(frame)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    temporary.replace(path)
    print(f"{point_id}: {frame.issue_date.nunique()} дат решения, {len(frame)} прогнозных суток", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--points", nargs="+", default=["yakutsk_crossing"])
    parser.add_argument("--live", action="store_true", help="Сохранить прогнозы на сегодня и вчера, сохраняя прошлые даты")
    args = parser.parse_args()
    base = pd.read_csv(PROCESSED / "m1_dataset_close.csv")
    with requests.Session() as session:
        for point_id in args.points:
            frames = []
            if args.live:
                today = dt.datetime.now(dt.timezone.utc).date()
                frames.append(download(point_id, [today - dt.timedelta(days=1), today], session))
            else:
                for year in range(args.start_year, args.end_year + 1):
                    rows = base[base.point_id.eq(point_id) & base.hydro_year.eq(year)]
                    issues = [from_season_day(int(day), year) for day in rows.issue_day]
                    if issues:
                        frames.append(download(point_id, issues, session))
                        print(f"{point_id}: сезон {year} загружен", flush=True)
            save(point_id, frames)


if __name__ == "__main__":
    main()
