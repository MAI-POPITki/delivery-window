"""python scripts/download_flood.py [--history] — реальные ответы Open-Meteo/GloFAS."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno.flood import API, DATA, POINTS, daily_frame, save_response, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--case-year", type=int, help="Диагностический исторический год, не архив выпущенных прогнозов")
    args = parser.parse_args()
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])))
    failures = []
    for pid, point in POINTS.items():
        try:
            params = dict(latitude=point["lat"], longitude=point["lon"], daily="river_discharge",
                          timezone="GMT", models="seamless_v4")
            jobs = []
            if args.history or not (DATA / "history" / f"{pid}.json").exists():
                jobs.append(("history", dict(start_date="1991-01-01", end_date="2020-12-31")))
            jobs.append(("latest", dict(forecast_days=8, ensemble="true")))
            if args.case_year:
                jobs.append((f"cases/{args.case_year}", dict(start_date=f"{args.case_year}-01-01",
                             end_date=f"{args.case_year}-12-31")))
            for kind, extra in jobs:
                response = session.get(API, params={**params, **extra}, timeout=90)
                response.raise_for_status()
                payload = response.json()
                frame = daily_frame(payload, point)
                if kind == "history":
                    threshold(frame, before_year=2021)
                elif kind == "latest" and (len([c for c in frame if c.startswith("river_discharge_member")
                          and frame[c].notna().all()]) < 30 or len(frame) != 8):
                    raise ValueError("Incomplete ensemble")
                fetched = datetime.now(timezone.utc).isoformat()
                save_response(DATA / kind / f"{pid}.json", payload, response.url, fetched)
                if kind == "latest":
                    archive = DATA / "runs" / fetched[:10] / f"{pid}.json"
                    if not archive.exists():
                        save_response(archive, payload, response.url, fetched)
                print(f"{pid}: {kind} OK, {len(frame)} days")
                time.sleep(.25)
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append(f"{pid}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
