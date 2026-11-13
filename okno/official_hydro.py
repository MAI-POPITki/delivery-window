"""Official observations, kept separate from historical training targets.

The store distinguishes measurements at a named post, undated values shown by
the source, and descriptions of a river reach. None is a transport permit.
"""
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from .paths import DATA, PROCESSED
from .regions import GAUGE_NAMES, REGIONS, basin_points, points_of, region_of

DIRECTORY = DATA / "official_hydro"
SNAPSHOT = DIRECTORY / "snapshot.json"
METRICS = {"water_level_cm": "Уровень воды, см над нулём поста",
           "ice_cm": "Толщина льда, см", "snow_on_ice_cm": "Снег на льду, см",
           "water_temperature_c": "Температура воды, °C"}
MAX_AGE = {"water_level_cm": 7, "ice_cm": 10, "snow_on_ice_cm": 30,
           "water_temperature_c": 7}
UTC = timezone.utc


def stamp(value):
    """Parse a source timestamp; the Meteo-DV UI interprets these as UTC."""
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.max)
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def normalise(value):
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b(?:пгт|гмс|г/п|мкр|г|п|с|р)\s*\.", " ", text)
    return re.sub(r"[^а-яa-z0-9]", "", text)


def registry():
    """Every supported point; source IDs never come from nearest-neighbour matching."""
    gauges = {}
    path = PROCESSED / "gauges.csv"
    if path.exists():
        import csv
        with path.open(encoding="utf-8", newline="") as stream:
            gauges = {int(r["gauge_code"]): r for r in csv.DictReader(stream)}
    result = {}
    basin = basin_points()
    for key, region in REGIONS.items():
        for pid in points_of(key):
            if pid.startswith("gauge_"):
                code = int(pid.split("_")[1])
                name = GAUGE_NAMES[code].split(" (")[0]
                river, site = name.split(" — ", 1)
                g = gauges.get(code, {})
                result[pid] = dict(point_id=pid, region=key, station_code=code,
                                   river=river, site=site, scope="post",
                                   lat=float(g["lat"]) if g.get("lat") else None,
                                   lon=float(g["lon"]) if g.get("lon") else None)
            elif pid == "yakutsk_crossing":
                # 3045 measures the Lena at Yakutsk, not the engineered ice road.
                result[pid] = dict(point_id=pid, region=key, station_code=3045,
                                   river="Лена", site="Якутск", scope="reference_post",
                                   lat=62.0, lon=129.82)
            else:
                result[pid] = dict(point_id=pid, region=key, station_code=None,
                                   river="Лена", site="", scope="satellite",
                                   lat=basin.get(pid, {}).get("lat"),
                                   lon=basin.get(pid, {}).get("lon"))
    return result


def match_station(item, points=None):
    """Require both original station index and river/site identity."""
    points = registry() if points is None else points
    try:
        code = int(item["code"])
    except (KeyError, ValueError, TypeError):
        return None
    for point in points.values():
        if point["station_code"] != code:
            continue
        river = normalise(item.get("geoobject", item.get("geo_object", "")))
        site = normalise(item.get("name", item.get("station_name", "")))
        if river == normalise(point["river"]) and normalise(point["site"]) in site:
            return point
    return None


def distance_km(point, item):
    try:
        la, lo, a, b = map(math.radians, (point["lat"], point["lon"],
                                        float(item["latitude"]), float(item["longitude"])))
        h = math.sin((a-la)/2)**2 + math.cos(la)*math.cos(a)*math.sin((b-lo)/2)**2
        return round(6371*2*math.asin(min(1, math.sqrt(h))), 1)
    except (KeyError, TypeError, ValueError):
        return None


def number(value, metric):
    """The portal's 999.999 is missing, while zero and negative levels are valid."""
    if value is None or isinstance(value, bool) or str(value).strip() in ("", "-", "—"):
        return None
    try:
        n = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n) or math.isclose(n, 999.999, abs_tol=0.00001):
        return None
    bounds = {"water_level_cm": (-3000, 10000), "ice_cm": (0, 500),
              "snow_on_ice_cm": (0, 1000), "water_temperature_c": (-5, 50)}
    lo, hi = bounds[metric]
    return n if lo <= n <= hi else None


def observation(point, metric, value, observed_at, fetched_at, source_url, raw_sha256,
                *, source="meteo_dv", quality="measured", note=""):
    n = number(value, metric)
    if n is None:
        return None
    fetched = stamp(fetched_at)
    observed = stamp(observed_at) if observed_at else None
    if observed and observed > fetched:
        return None
    # Value is in the identity: a later correction must not become known in an
    # earlier replay. Merging preserves the first-seen time of each revision.
    identity = observation_id(source, point['point_id'], metric, observed.isoformat() if observed else None, n, quality)
    return dict(id=identity,
                point_id=point["point_id"], region=point["region"],
                station_code=point["station_code"], station_name=point["site"],
                river=point["river"], metric=metric, value=n,
                observed_at=observed.isoformat() if observed else None,
                first_seen_at=fetched.isoformat(), available_at=fetched.isoformat(),
                fetched_at=fetched.isoformat(),
                scope=point["scope"], quality=quality, source=source,
                source_url=source_url, raw_sha256=raw_sha256, note=note)


def observation_id(source, point_id, metric, observed_at, value, quality):
    observed = stamp(observed_at).isoformat() if observed_at else None
    identity = f"{source}|{point_id}|{metric}|{observed}|{float(value)}|{quality}"
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _version(path):
    try:
        s = path.stat()
        return s.st_mtime_ns, s.st_size
    except OSError:
        return 0, 0


@lru_cache(maxsize=8)
def _load(path, version):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return {}
        return value
    except (OSError, ValueError, TypeError):
        return {}


def load_snapshot(path=None):
    path = SNAPSHOT if path is None else Path(path)
    return _load(str(path), _version(path))


def observations(point_id=None, *, as_of=None, snapshot=None):
    snapshot = load_snapshot() if snapshot is None else snapshot
    rows = snapshot.get("observations", [])
    if point_id:
        rows = [r for r in rows if r["point_id"] == point_id]
    if as_of is not None:
        cutoff = stamp(as_of)
        rows = [r for r in rows if r.get("observed_at")
                and stamp(r["observed_at"]) <= cutoff
                and stamp(r["available_at"]) <= cutoff]
    return sorted(rows, key=lambda r: (r.get("observed_at") or "", r["available_at"]), reverse=True)


def latest_measurement(point_id, metric, as_of, *, snapshot=None, max_age=None):
    """Only dated, fresh, already available measurements at this exact post."""
    cutoff = stamp(as_of)
    age = MAX_AGE[metric] if max_age is None else max_age
    for row in observations(point_id, as_of=cutoff, snapshot=snapshot):
        if row["metric"] != metric or row["quality"] != "measured" or row["scope"] != "post":
            continue
        if cutoff-stamp(row["observed_at"]) <= timedelta(days=age):
            return row
    return None


def bulletins_for(point_id=None, *, as_of=None, snapshot=None):
    snapshot = load_snapshot() if snapshot is None else snapshot
    rows = snapshot.get("bulletins", [])
    if point_id:
        key = region_of(point_id)
        rows = [r for r in rows if r["region"] == key
                and (not r.get("point_ids") or point_id in r["point_ids"])]
    if as_of:
        rows = [r for r in rows if r.get("published_at") and stamp(r["published_at"]) <= stamp(as_of)]
    return sorted(rows, key=lambda r: r.get("published_at") or "", reverse=True)


def coverage(*, snapshot=None, now=None):
    snapshot = load_snapshot() if snapshot is None else snapshot
    now = datetime.now(UTC) if now is None else stamp(now)
    points = registry()
    rows = []
    for key, region in REGIONS.items():
        own = {p for p, v in points.items() if v["region"] == key}
        measured = [r for r in snapshot.get("observations", [])
                    if r["point_id"] in own and r.get("observed_at") and r["quality"] == "measured"]
        fresh = {r["point_id"] for r in measured
                 if timedelta(0) <= now-stamp(r["observed_at"]) <= timedelta(days=MAX_AGE[r["metric"]])}
        reports = [r for r in snapshot.get("bulletins", []) if r["region"] == key]
        rows.append(dict(region=key, name=region["name"], points=len(own),
                         measured_points=len({r["point_id"] for r in measured}),
                         fresh_points=len(fresh), observations=len(measured), bulletins=len(reports),
                         last_observation=max((r["observed_at"] for r in measured), default=None),
                         last_bulletin=max((r.get("published_at") or "" for r in reports), default=None)))
    return rows
