"""Read/compare M2 observations without backdating new optical information."""
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA
from .gaofen import stamp


def cutoff(value=None, now=None):
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    if value is None:
        return pd.Timestamp(now)
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time.max, tzinfo=timezone.utc)
    return pd.Timestamp(min(stamp(value), now))


def select_latest(records, *, as_of=None, now=None, max_age=None):
    if records.empty:
        return records
    end = cutoff(as_of, now)
    data = records.copy()
    data["observed_at"] = pd.to_datetime(data.observed_at, utc=True)
    data["available_at"] = pd.to_datetime(data.available_at, utc=True)
    data = data[(data.observed_at <= end) & (data.available_at <= end) & data.p_ice.notna()]
    if max_age is not None:
        data = data[data.observed_at >= end-pd.Timedelta(days=max_age)]
    chosen = []
    for _, group in data.groupby("point"):
        group = group.sort_values(["observed_at", "source"])
        last = group.iloc[-1].copy()
        last["conflict"] = False
        close = group[(group.source != last.source) & (group.observed_at >= last.observed_at-pd.Timedelta(hours=48))]
        confident = last.p_ice <= 0.2 or last.p_ice >= 0.8
        if confident and any((p >= 0.8 and last.p_ice <= 0.2) or (p <= 0.2 and last.p_ice >= 0.8) for p in close.p_ice):
            last["conflict"] = True
            last["source"] += " (расхождение со вторым спутником)"
            last["p_ice"] = np.nan
        last["date"] = last.observed_at.tz_convert(None).normalize()
        chosen.append(last)
    return pd.DataFrame(chosen).reset_index(drop=True) if chosen else data.iloc[:0]


def observations(*, as_of=None, now=None, max_age=None, point=None):
    records = []
    path = DATA/"s1"/"predictions.csv"
    if path.exists():
        s1 = pd.read_csv(path)
        s1["observed_at"] = pd.to_datetime(s1.date, utc=True)
        # Legacy Sentinel-1 rows do not record publication time. This preserves
        # their existing behaviour; optical rows require actual project receipt.
        s1["available_at"] = s1.observed_at
        s1["source"] = "Sentinel-1"
        records.append(s1[["point", "item_id", "observed_at", "available_at", "p_ice", "source"]])
    from . import landsat, m2_landsat
    path = landsat.DIRECTORY/"predictions.csv"
    meta = m2_landsat.metadata()
    if path.exists() and meta.get("enabled"):
        optical = pd.read_csv(path)
        optical = optical[optical.quality.eq("usable") & optical.model_version.eq(meta["model_sha256"][:12])]
        records.append(optical[["point", "item_id", "observed_at", "available_at", "p_ice", "source"]])
    data = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if point is not None and len(data):
        data = data[data.point == point]
    return select_latest(data, as_of=as_of, now=now, max_age=max_age)
