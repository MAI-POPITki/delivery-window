"""Strict conversion of official post observations into M1 training candidates.

Water levels, regional ranges and publication dates cannot become ice-event
targets. A single ice measurement only supplies a bound, not a crossing date.
"""
from datetime import datetime, timezone
import re

import pandas as pd

from . import official_hydro as H
from .ice import required_thickness_cm

MONTHS = {name: i+1 for i, name in enumerate(("январ", "феврал", "март", "апрел", "мая", "июн",
                                            "июл", "август", "сентябр", "октябр", "ноябр", "декабр"))}


def extract_dated_ice(report, fetched_at, *, points=None):
    """Only explicit single-post absolute thickness and an explicit survey date.

    Example verified in the 29 December 2022 Yakutia bulletin:
    'На р. Яна у г. Верхоянск толщина льда составляет 84 см'. Neither the
    adjacent river-wide ranges nor 'above normal by N cm' match this pattern.
    """
    if report.get("source_url") != "https://ykuthydromet.ru/reki/":
        return []
    text = re.sub(r"\s+", " ", report["text"])
    surveyed = re.search(r"по данным измерений\s+от\s+(\d{1,2})\s+([а-яё]+)", text, re.I)
    if not surveyed or not report.get("published_at"):
        return []
    month = next((n for m, n in MONTHS.items() if surveyed[2].lower().startswith(m)), None)
    if month is None:
        return []
    published = H.stamp(report["published_at"])
    year = published.year-(month > published.month)
    try:
        observed = datetime(year, month, int(surveyed[1]), tzinfo=timezone.utc)
    except ValueError:
        return []
    lag = (published.date()-observed.date()).days
    if not 0 <= lag <= 31:
        return []
    points = H.registry() if points is None else points
    out = []
    for point in points.values():
        if point["region"] != "yakutia" or not point.get("station_code"):
            continue
        river, site = re.escape(point["river"]), re.escape(point["site"])
        pattern = (rf"На\s+р\.\s*{river}\s+у\s+(?:г|п|с)\.\s*{site}\s+"
                   rf"толщина\s+льда\s+составляет\s+(\d+(?:[.,]\d+)?)\s*см\b")
        match = re.search(pattern, text, re.I)
        if match is None and point["point_id"] == "yakutsk_crossing":
            # The city-post statement follows the Lena sentence. Preserve its
            # reference-post scope: it is not a measurement on the ice road.
            match = re.search(r"У\s+г\.\s*Якутск\s+толщина\s+льда\s+составляет\s+(\d+(?:[.,]\d+)?)\s*см\b", text, re.I)
            if match and not re.search(r"На\s+р\.\s*Лена\b", text[max(0,match.start()-160):match.start()], re.I):
                match = None
        if match is None:
            continue
        row = H.observation(point, "ice_cm", match[1], observed, fetched_at,
                            report["source_url"], report["raw_sha256"], source="yakutia_bulletin",
                            note="Точный замер и дата прямо указаны в тексте бюллетеня.")
        if row:
            row.update(published_at=published.isoformat(), evidence=match[0], observation_precision="day")
            out.append(row)
    return out


def prepare_targets(snapshot, *, historical=None):
    """Check if NEW measured seasons support the existing opening target.

    Use the same >=3 measurements and bracketed threshold convention as the
    existing training dataset builder. Always retain bracket endpoints and
    distinguish this inferred physical date from official road opening.
    """
    from scripts.build_dataset import open_date_from_measurements
    historical = (pd.read_csv(H.PROCESSED/"hydro_ice_thickness.csv")
                  if historical is None else historical)
    exact = [r for r in snapshot.get("observations", [])
             if r.get("metric") == "ice_cm" and r.get("quality") == "measured"
             and r.get("observed_at") and r.get("scope") == "post"]
    groups = {}
    for row in exact:
        observed = H.stamp(row["observed_at"])
        year = observed.year+(observed.month >= 9)
        groups.setdefault((row["point_id"], year), []).append(row)
    candidates, targets = [], []
    threshold = required_thickness_cm(20)
    for (pid, year), rows in sorted(groups.items()):
        code = int(pid.split("_")[1])
        old = historical[historical.gauge_code.eq(code)&historical.hydro_year.eq(year)]
        new = pd.DataFrame({"date": [H.stamp(r["observed_at"]).date().isoformat() for r in rows],
                            "ice_cm": [r["value"] for r in rows]})
        series = pd.concat([old[["date","ice_cm"]], new], ignore_index=True).dropna(subset=["ice_cm"])
        series = series.drop_duplicates().sort_values("date")
        # Conflicting values on one date are not averaged or silently selected.
        conflict = series.date.duplicated(keep=False).any()
        crossing = None if conflict else open_date_from_measurements(series, threshold)
        if conflict:
            reason = "Противоречивые значения на одну дату"
        elif len(series) < 3:
            reason = "Меньше трёх датированных замеров в сезоне"
        elif crossing is None:
            reason = "Нет наблюдаемого перехода через порог 20 т"
        else:
            reason = "Порог заключён между замерами; физическая дата оценена интерполяцией"
        candidates.append(dict(point_id=pid, hydro_year=year, measured_dates=series.date.nunique(),
                               min_ice_cm=float(series.ice_cm.min()), max_ice_cm=float(series.ice_cm.max()),
                               eligible=crossing is not None, reason=reason))
        if crossing is not None:
            before = series[pd.to_datetime(series.date) <= crossing]
            after = series[pd.to_datetime(series.date) >= crossing]
            targets.append(dict(point_id=pid, gauge_code=code, hydro_year=year, kind="open",
                                target_date=crossing.date().isoformat(),
                                target_definition="interpolated_20t_thickness_crossing",
                                left_measurement=before.date.iloc[-1] if len(before) else None,
                                right_measurement=after.date.iloc[0] if len(after) else None))
    return candidates, targets
