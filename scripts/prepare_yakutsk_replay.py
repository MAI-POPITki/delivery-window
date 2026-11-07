"""Build auditable inputs for the 2026 Yakutsk crossing scenario, without retraining.

    python scripts/prepare_yakutsk_replay.py
    python scripts/prepare_yakutsk_replay.py --weather-lag-days 0 --output-dir /tmp/replay-lag0

Weather is historical ERA5 reanalysis, not an archive of operational observations.
The default five-day publication lag is an explicit approximation. No future
weather or 2026 event labels enter M1. Existing M2 probabilities are independently
checked against the stored features; raw SAR chips are not bundled in the repo.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okno import forecast as FC, m2  # noqa: E402
from okno.ice import to_season_day  # noqa: E402
from okno.yakutsk_replay import fuse_observation  # noqa: E402

POINT = "yakutsk_crossing"
YEAR = 2026
PREFIX = "yakutsk_spring_2026"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_observations() -> pd.DataFrame:
    """One row per real scene; decisions are made at midnight UTC.

    Availability assumes processing 24 hours after acquisition END. For scenes
    acquired at 21:49 UTC on 18 March, the first usable daily decision is 20 March.
    This is an assumption; historical first-publication timestamps are unavailable.
    """
    pred = pd.read_csv(ROOT / "data/s1/predictions.csv")
    pred = pred[(pred.point == POINT) & pred.date.between("2026-01-01", "2026-04-15")].copy()
    index = pd.read_csv(ROOT / "data/s1/index.csv")
    index = index[index.point == POINT][["item_id", "lat", "lon", "file"]]
    pred = pred.merge(index, on="item_id", validate="one_to_one").sort_values("date")
    checked = m2.load_model().predict_proba(pred[m2.FEATURES])[:, 1]
    if not np.allclose(checked, pred.p_ice, atol=0.001, rtol=0):
        raise ValueError("Stored M2 probabilities disagree with the current model and stored features")
    rows = []
    for row, probability in zip(pred.itertuples(), checked):
        stamps = re.findall(r"\d{8}T\d{6}", row.item_id)
        end = pd.to_datetime(stamps[1], format="%Y%m%dT%H%M%S", utc=True)
        available = end + pd.Timedelta(days=1)
        first_decision = available.ceil("D")
        preview = ROOT / "docs/replay_yakutsk_2026" / f"{stamps[0][:8]}_scene_preview.png"
        rows.append({
            "date": row.date,
            "acquired_at_utc": end.isoformat(),
            "available_at_utc": available.isoformat(),
            "available_date": first_decision.date().isoformat(),
            "season_day": int(to_season_day(row.date, YEAR)),
            "p_ice": float(row.p_ice),
            "p_ice_recomputed": float(probability),
            "item_id": row.item_id,
            "lat": row.lat, "lon": row.lon,
            "raw_chip_available": (ROOT / row.file).exists(),
            "m2_probability_source": row.p_ice_source,
            "scene_preview_file": str(preview.relative_to(ROOT)) if preview.exists() else "",
        })
    return pd.DataFrame(rows)


def build_forecasts(observations: pd.DataFrame, weather_lag_days: int = 5,
                    start="2026-03-16", end="2026-04-15") -> pd.DataFrame:
    """Freeze all weather at the cutoff before calling the application's M1.

    issue_date/issue_day denote the decision. weather_issue_day is the last day
    fed into M1; its quantiles retain the application's own lower bound at that
    cutoff (they are not silently shifted forward to the decision date).
    """
    if weather_lag_days < 0:
        raise ValueError("weather_lag_days must be nonnegative")
    T, S, full = FC._matrices(POINT)
    past = full[full.index < YEAR].copy()
    rows = []
    for day in pd.date_range(start, end):
        issue_day = int(to_season_day(day, YEAR))
        cutoff = day - pd.Timedelta(days=weather_lag_days)
        weather_day = int(to_season_day(cutoff, YEAR))
        t, s = T.loc[T.index <= YEAR].copy(), S.loc[S.index <= YEAR].copy()
        t.loc[YEAR, t.columns > weather_day] = np.nan
        s.loc[YEAR, s.columns > weather_day] = np.nan
        with ExitStack() as stack:
            stack.enter_context(patch.object(FC, "_matrices", return_value=(t, s, past)))
            stack.enter_context(patch.object(FC, "_ahead", return_value=pd.DataFrame(columns=range(365))))
            q = FC.forecast(POINT, YEAR, weather_day, "close", use_weather=False)
        if q is None or q["issue_day"] != weather_day:
            raise ValueError(f"Missing weather/forecast at {cutoff.date()}")
        if not np.isnan(q["actual"]):
            raise ValueError("A 2026 physical event target unexpectedly entered the replay")
        usable = observations[
            (pd.to_datetime(observations.available_date) <= day)
            & (pd.to_datetime(observations.date) <= day)
            & (pd.to_datetime(observations.date) >= day - pd.Timedelta(days=14))
        ]
        quantiles = tuple(float(q[k]) for k in ["q10", "q50", "q90"])
        corrected = quantiles
        obs = usable.sort_values("date").iloc[-1] if len(usable) else None
        if obs is not None:
            corrected, _ = fuse_observation(quantiles, float(obs.season_day), float(obs.p_ice))
        row = {"issue_date": day.date().isoformat(), "decision_date": day.date().isoformat(),
               "issue_day": issue_day, "weather_cutoff_date": cutoff.date().isoformat(),
               "weather_issue_day": weather_day, "weather_lag_days": weather_lag_days,
               **dict(zip(["q10", "q50", "q90"], quantiles)),
               **dict(zip(["m2_q10", "m2_q50", "m2_q90"], map(float, corrected))),
               "calendar": float(q["calendar"]), "obs_used": obs is not None,
               "obs_date": obs.date if obs is not None else "",
               "obs_available_date": obs.available_date if obs is not None else "",
               "obs_age_days": int((day - pd.Timestamp(obs.date)).days) if obs is not None else np.nan,
               "p_ice": float(obs.p_ice) if obs is not None else np.nan,
               "item_id": obs.item_id if obs is not None else ""}
        rows.append(row)
    out = pd.DataFrame(rows)
    assert (out[["q10", "q50", "q90"]].diff(axis=1).iloc[:, 1:] >= 0).all().all()
    return out


def write_inputs(output_dir: Path, weather_lag_days=5):
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = build_observations()
    forecasts = build_forecasts(observations, weather_lag_days)
    observations.to_csv(output_dir / f"{PREFIX}_observations.csv", index=False)
    forecasts.to_csv(output_dir / f"{PREFIX}_forecasts.csv", index=False, float_format="%.8f")
    targets = pd.read_csv(ROOT / "data/processed/m1_dataset_close.csv", usecols=["hydro_year"])
    probabilities = pd.read_csv(ROOT / "data/s1/predictions.csv")
    labelled = probabilities[probabilities.label.isin(["water", "ice"])]
    if targets.hydro_year.max() >= YEAR or labelled.hydro_year.max() >= YEAR:
        raise ValueError("Model training targets overlap the replay season")
    source_paths = ["models/m1_close.cbm", "models/m1_close_conformal.json", "models/m2_ice.cbm",
                    "reports/stefan_calibration.csv", "data/processed/m1_dataset_close.csv",
                    "data/processed/yakutsk_sat_seasons.csv", "data/weather/yakutsk_crossing.csv",
                    "data/s1/predictions.csv", "data/s1/index.csv"]
    provenance = {
        "point_id": POINT, "hydro_year": YEAR,
        "decision_period": [forecasts.issue_date.min(), forecasts.issue_date.max()],
        "decision_time": "00:00 UTC; daily decision dates are not local Yakutsk dates",
        "weather_lag_days_assumption": weather_lag_days,
        "weather_source": "ERA5 reanalysis through the cutoff; future observations physically masked",
        "operational_weather_archive": False,
        "weather_limitation": "Ex-post reanalysis. Five-day publication lag is nominal, not verified per record; historical weather revisions are not reconstructed.",
        "model_target": "physical river-ice breakup, NOT legal load restrictions or truck-bearing ice strength",
        "m1_last_training_target_hydro_year": int(targets.hydro_year.max()),
        "m2_last_labelled_target_hydro_year": int(labelled.hydro_year.max()),
        "model_limitation": "Current repository model artifacts are held fixed; they were not deployed historically in March 2026.",
        "satellite_source": "Existing repository M2 results independently recomputed from stored radar features",
        "satellite_publication_lag_assumption_hours": 24,
        "satellite_max_age_days": 14,
        "satellite_first_publication_verified": False,
        "raw_sar_download": "Metadata and signing succeeded; Azure asset read failed: proxy tunnel HTTP 403. No alternate network route used.",
        "scene_previews": observations[observations.scene_preview_file.ne("")][["date", "scene_preview_file"]].to_dict("records"),
        "preview_limitation": "Rendered full-scene overview, not raw 3 km classification chips and not an ice-thickness measurement",
        "m2_fusion": "Existing M2 truncation with matched-sample quantile reconstruction error subtracted; no truncation gives exactly zero effect",
        "factual_load_restrictions_used_for_evaluation_only": [
            {"date": "2026-03-27", "gross_mass_t": 30},
            {"date": "2026-04-08", "gross_mass_t": 20},
            {"date": "2026-04-15", "event": "closed in the evening"}],
        "source_sha256": {str(p): checksum(ROOT / p) for p in source_paths},
        "catalog": "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc",
        "weather_documentation": "https://open-meteo.com/en/docs/historical-weather-api",
    }
    (output_dir / f"{PREFIX}_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return observations, forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-lag-days", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/scenarios")
    args = parser.parse_args()
    obs, fc = write_inputs(args.output_dir, args.weather_lag_days)
    print(f"Prepared {len(fc)} daily forecasts and {len(obs)} observations: {args.output_dir}")
    print(f"Fresh M2 observations available for {int(fc.obs_used.sum())} decision dates")


if __name__ == "__main__":
    main()
