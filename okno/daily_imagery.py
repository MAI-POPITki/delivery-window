"""Dated MODIS/VIIRS browse mosaics from NASA GIBS, for visual review only.

PNG colour composites are not calibrated bands and never enter M1/M2 inference.
A daily global product does not imply cloud-free coverage of every local window.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import requests

from .gaofen import _atomic_write, stamp
from .paths import DATA
from .s1_points import river_points

DIRECTORY = DATA / "daily_imagery"
WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
CAPABILITIES = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml"
SOURCE = "NASA Global Imagery Browse Services (GIBS)"
INTERVAL = timedelta(hours=6)
SIZE = 512
MAX_IMAGE_BYTES = 2_000_000
HISTORY_DAYS = 31
SOURCES = {
    "modis_terra": {"name": "MODIS · Terra", "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
                    "resolution": "250–500 м, цветной обзор"},
    "modis_aqua": {"name": "MODIS · Aqua", "layer": "MODIS_Aqua_CorrectedReflectance_TrueColor",
                   "resolution": "250–500 м, цветной обзор"},
    "viirs_noaa21": {"name": "VIIRS · NOAA-21", "layer": "VIIRS_NOAA21_CorrectedReflectance_TrueColor",
                     "resolution": "около 750 м, цветной обзор"},
    "viirs_noaa20": {"name": "VIIRS · NOAA-20", "layer": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
                     "resolution": "около 750 м, цветной обзор"},
}


def points():
    return river_points(half_size_m=40000)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and value.get("schema_version") == 1 else {}
    except (OSError, ValueError):
        return {}


def load_snapshot(directory=None):
    return read_json(Path(directory or DIRECTORY)/"snapshot.json")


def parse_catalog(content):
    if len(content) > 10_000_000:
        raise ValueError("NASA capabilities exceed size limit")
    ns = {"w": "http://www.opengis.net/wmts/1.0", "o": "http://www.opengis.net/ows/1.1"}
    root = ET.fromstring(content)
    wanted = {value["layer"]: key for key, value in SOURCES.items()}
    result = {}
    for layer in root.findall(".//w:Contents/w:Layer", ns):
        identifier = layer.findtext("o:Identifier", namespaces=ns)
        if identifier not in wanted:
            continue
        dimensions = [d for d in layer.findall("w:Dimension", ns)
                      if (d.findtext("o:Identifier", namespaces=ns) or "").lower() == "time"]
        ranges = [v.text for d in dimensions for v in d.findall("w:Value", ns) if v.text]
        if ranges:
            result[wanted[identifier]] = {"layer": identifier, "dates": ranges,
                                          "title": layer.findtext("o:Title", namespaces=ns)}
    if set(result) != set(SOURCES):
        raise ValueError("NASA catalogue does not contain all configured daily layers")
    return result


def published(catalog, source, day):
    """Require the requested exact day, including gaps; never use a nearest date."""
    day = date.fromisoformat(str(day))
    for value in catalog.get(source, {}).get("dates", []):
        parts = value.split("/")
        if len(parts) == 1 and day == date.fromisoformat(parts[0]):
            return True
        if len(parts) == 3 and parts[2] == "P1D":
            if date.fromisoformat(parts[0]) <= day <= date.fromisoformat(parts[1]):
                return True
    return False


def preview_path(point_id, source, day, directory=None):
    if point_id not in points() or source not in SOURCES:
        raise ValueError("Unknown imagery point or satellite")
    day = date.fromisoformat(str(day)).isoformat()
    return Path(directory or DIRECTORY)/"previews"/point_id/source/f"{day}.png"


def image_url(point_id, source, day):
    preview_path(point_id, source, day)  # strict identifiers and ISO date
    point = points()[point_id]
    params = {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
              "LAYERS": SOURCES[source]["layer"], "STYLES": "", "SRS": "EPSG:4326",
              "BBOX": ",".join(f"{v:.9f}" for v in point["bbox"]),
              "WIDTH": SIZE, "HEIGHT": SIZE, "FORMAT": "image/png", "TRANSPARENT": "TRUE",
              "TIME": date.fromisoformat(str(day)).isoformat()}
    return WMS+"?"+urlencode(params)


def inspect_image(content):
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("NASA image exceeds size limit")
    with Image.open(io.BytesIO(content)) as im:
        if im.format != "PNG" or im.size != (SIZE, SIZE):
            raise ValueError("Unexpected NASA image format or dimensions")
        pixels = np.asarray(im.convert("RGBA"))
    # GIBS can return opaque black fill, not just transparent fill. This is
    # an image-fill check, NOT a cloud mask or a scientific validity estimate.
    nonempty = (pixels[:, :, 3] > 0) & (pixels[:, :, :3].max(axis=2) > 0)
    fraction = float(nonempty.mean())
    return {"nonempty_fraction": round(fraction, 4),
            "status": "no_data" if fraction == 0 else "partial" if fraction < .98 else "image",
            "cloud_quality": "not_assessed"}


def fetch_image(point_id, source, day, *, session, directory=None, now=None):
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    day = date.fromisoformat(str(day))
    if day > now.date():
        raise ValueError("A daily image cannot have a future date")
    url = image_url(point_id, source, day)
    with session.get(url, timeout=(5, 30), stream=True) as response:
        response.raise_for_status()
        if response.headers.get("Content-Type", "").split(";")[0].strip().lower() != "image/png":
            raise ValueError("NASA returned a service error instead of an image")
        chunks, total = [], 0
        for chunk in response.iter_content(65536):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("NASA image exceeds size limit")
            chunks.append(chunk)
    content = b"".join(chunks)
    quality = inspect_image(content)
    path = preview_path(point_id, source, day, directory)
    _atomic_write(path, content)
    return {"point": point_id, "source": source, "date": day.isoformat(),
            "layer": SOURCES[source]["layer"], "bbox": points()[point_id]["bbox"],
            "fetched_at": now.isoformat(), "available_at": now.isoformat(), "url": url,
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), **quality,
            "usage": "visual_review_only", "ml_ready": False}


def due(directory=None, now=None):
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    snapshot = load_snapshot(directory)
    try:
        if now-stamp(snapshot["checked_at"]) >= INTERVAL or not snapshot.get("records"):
            return True
        return any(not preview_path(r["point"], r["source"], r["date"], directory).exists()
                   for r in snapshot["records"] if r.get("sha256"))
    except (KeyError, ValueError, TypeError):
        return True


def update(*, days=7, directory=None, now=None):
    from .hydro_ingest import update_lock
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= HISTORY_DAYS:
        raise ValueError("Choose 1..31 days")
    directory = Path(directory or DIRECTORY)
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    with update_lock(directory) as locked:
        if not locked:
            return load_snapshot(directory)
        previous = load_snapshot(directory)
        rows = {(r["point"], r["source"], r["date"]): r for r in previous.get("records", [])}
        errors = []
        catalog_path = directory/"catalog.json"
        catalog = read_json(catalog_path).get("layers", {})
        try:
            with requests.Session() as session:
                response = session.get(CAPABILITIES, timeout=(5, 45))
                response.raise_for_status()
                catalog = parse_catalog(response.content)
            _atomic_write(catalog_path, (json.dumps({"schema_version": 1, "checked_at": now.isoformat(),
                                                    "url": CAPABILITIES, "layers": catalog}, ensure_ascii=False, indent=2)+"\n").encode())
        except Exception as exc:
            errors.append({"source": "catalog", "error": type(exc).__name__})

        tasks = []
        for key in points():
            for source in SOURCES:
                for offset in range(days):
                    day = now.date()-timedelta(days=offset)
                    identity = (key, source, day.isoformat())
                    old = rows.get(identity)
                    if not published(catalog, source, day):
                        if old is None:
                            rows[identity] = {"point": key, "source": source, "date": day.isoformat(),
                                              "status": "not_published" if source in catalog else "source_unavailable",
                                              "usage": "visual_review_only", "ml_ready": False}
                        continue
                    path = preview_path(key, source, day, directory)
                    # Daily mosaics are revised during and after the day. Refresh
                    # the last three dates; retain older valid local files.
                    if old and path.exists() and offset >= 3:
                        continue
                    tasks.append(identity)

        def fetch(task):
            with requests.Session() as session:
                return fetch_image(*task, session=session, directory=directory, now=now)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch, task): task for task in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    row = future.result()
                    old = rows.get(key, {})
                    row["first_received_at"] = old.get("first_received_at", row["fetched_at"])
                    if old.get("sha256") == row["sha256"]:
                        row["available_at"] = old.get("available_at", row["available_at"])
                    rows[key] = row
                except Exception as exc:
                    errors.append({"point": key[0], "source": key[1], "date": key[2], "error": type(exc).__name__})
                    if key not in rows or not rows[key].get("sha256"):
                        rows[key] = {"point": key[0], "source": key[1], "date": key[2], "status": "source_unavailable",
                                     "usage": "visual_review_only", "ml_ready": False}

        cutoff = (now.date()-timedelta(days=HISTORY_DAYS-1)).isoformat()
        retained = {key: row for key, row in rows.items() if cutoff <= key[2] <= now.date().isoformat()}
        for key in rows.keys()-retained.keys():
            preview_path(*key, directory).unlink(missing_ok=True)
        snapshot = {"schema_version": 1, "checked_at": now.isoformat(), "source": SOURCE,
                    "window_km": 80, "errors": errors,
                    "records": sorted(retained.values(), key=lambda r: (r["point"], r["date"], r["source"]))}
        _atomic_write(directory/"snapshot.json", (json.dumps(snapshot, ensure_ascii=False, indent=2)+"\n").encode())
        return snapshot
