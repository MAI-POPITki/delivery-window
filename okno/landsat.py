"""USGS Landsat 8/9 C2 L2 spectral windows via the public MPC processing API.

The API returns unscaled uint16 bands with nearest-neighbour QA reprojection.
Only a 6×6 km window is requested; full scenes are not downloaded.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image
import requests

from .paths import DATA
from .s1_points import river_points
from .gaofen import _atomic_write, stamp

DIRECTORY = DATA / "landsat"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
API = "https://planetarycomputer.microsoft.com/api/data/v1"
COLLECTION = "landsat-c2-l2"
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel", "qa_radsat")
SIZE = 200
SCALE, OFFSET = 0.0000275, -0.2
SOURCE = "USGS Landsat Collection 2 Level-2 / Microsoft Planetary Computer"
INTERVAL = timedelta(hours=6)
ITEM_PATTERN = r"LC0[89]_L2S[PR]_\d{6}_\d{8}_02_T1"


def points():
    return river_points(half_size_m=3000)


def _request(session, method, url, **kwargs):
    for attempt in range(3):
        response = session.request(method, url, timeout=(5, 45), **kwargs)
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            response.raise_for_status()
            return response
        try:
            delay = min(30, max(1, float(response.headers.get("Retry-After", 2))))
        except ValueError:
            delay = 2
        response.close()
        time.sleep(delay)
    raise RuntimeError("Landsat request failed")


def search(point, start, end, *, session):
    query = {"collections": [COLLECTION], "intersects": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
             "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
             "fields": {"exclude": ["assets"]},
             "query": {"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": 80},
                       "landsat:collection_category": {"eq": "T1"}}}
    result = {}
    url, method, body = STAC+"/search", "POST", query
    for _ in range(30):
        response = _request(session, method, url, **({"json": body} if method == "POST" else {}))
        data = response.json()
        for item in data.get("features", []):
            p = item["properties"]
            if p.get("platform") not in ("landsat-8", "landsat-9") or not re.fullmatch(ITEM_PATTERN, item["id"]):
                continue
            result[item["id"]] = {"item_id": item["id"], "date": p["datetime"][:10], "observed_at": p["datetime"],
                                   "platform": p["platform"], "scene_cloud_cover": p.get("eo:cloud_cover"),
                                   "published_at": p.get("created") or p.get("updated") or p["datetime"],
                                   "catalog_url": STAC+f"/collections/{COLLECTION}/items/{item['id']}"}
        nxt = next((link for link in data.get("links", []) if link.get("rel") == "next"), None)
        if not nxt:
            return sorted(result.values(), key=lambda r: (r["observed_at"], r["item_id"]), reverse=True)
        url, method = nxt["href"], nxt.get("method", "GET").upper()
        if not url.startswith(STAC+"/"):
            raise ValueError("Unexpected STAC pagination destination")
        body = nxt.get("body", query)
    raise ValueError("Landsat catalogue pagination limit reached")


def chip_path(point_id, item_id, directory=None):
    if point_id not in points() or not re.fullmatch(ITEM_PATTERN, item_id):
        raise ValueError("Unknown Landsat point or item")
    return Path(directory or DIRECTORY) / "chips" / point_id / f"{item_id}.npz"


def load_chip(path):
    with np.load(path, allow_pickle=False) as data:
        stack = data["bands"]
    if stack.shape != (8, SIZE, SIZE) or stack.dtype != np.uint16:
        raise ValueError("Unexpected Landsat spectral grid")
    return stack


def download(point_id, item, *, session, directory=None, now=None):
    directory = Path(directory or DIRECTORY)
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    path = chip_path(point_id, item["item_id"], directory)
    if path.exists():
        try:
            load_chip(path)
            return path
        except (OSError, ValueError, KeyError):
            pass
    bbox = points()[point_id]["bbox"]
    url = API+"/item/bbox/"+",".join(f"{v:.9f}" for v in bbox)+f"/{SIZE}x{SIZE}.npy"
    params = [("collection", COLLECTION), ("item", item["item_id"]), ("return_mask", "false"),
              ("unscale", "false"), ("resampling", "nearest"), ("reproject", "nearest")]
    params += [("assets", band) for band in BANDS]
    response = _request(session, "GET", url, params=params)
    if len(response.content) > 3_000_000:
        raise ValueError("Landsat window exceeds size limit")
    stack = np.load(io.BytesIO(response.content), allow_pickle=False)
    if stack.shape != (8, SIZE, SIZE) or stack.dtype != np.uint16:
        raise ValueError("Landsat service returned unexpected bands/dtype")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, bands=stack, bbox=np.array(bbox), crs="EPSG:4326", fetched_at=now.isoformat())
    _atomic_write(path, buffer.getvalue())
    return path


def reflectance(stack):
    return stack[:6].astype(np.float32)*SCALE+OFFSET


def clear_pixels(stack):
    # Bits 0..4: fill, dilated cloud, cirrus, cloud, shadow. Bit 5 (snow) is RETAINED.
    qa = stack[6]
    saturated = stack[7] & (0b1111110 | (1 << 11))  # selected OLI bands, terrain occlusion
    return ((qa & 31) == 0) & (saturated == 0) & np.all(stack[:6] > 0, axis=0)


def rgb(stack, mask=None):
    colors = np.moveaxis(reflectance(stack)[[2, 1, 0]], 0, -1)
    colors = (np.clip(colors/0.4, 0, 1)**(1/2.2)*255).astype(np.uint8)
    colors[np.any(stack[:6] == 0, axis=0)] = 0
    if mask is not None:
        from scipy.ndimage import binary_erosion
        colors[mask & ~binary_erosion(mask)] = [255, 75, 75]
    return colors


def make_preview(point_id, item_id, directory=None, mask=None):
    directory = Path(directory or DIRECTORY)
    path = directory / "previews" / point_id / f"{item_id}.png"
    stack = load_chip(chip_path(point_id, item_id, directory))
    buffer = io.BytesIO()
    Image.fromarray(rgb(stack, mask)).save(buffer, format="PNG")
    _atomic_write(path, buffer.getvalue())
    return path


def load_index(directory=None):
    path = Path(directory or DIRECTORY) / "index.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_index(index, directory=None):
    index = index.drop_duplicates(["point", "item_id"], keep="last").sort_values(["point", "observed_at"])
    _atomic_write(Path(directory or DIRECTORY)/"index.csv", index.to_csv(index=False, lineterminator="\n").encode())


def index_row(point_id, item, path, *, now=None):
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    with np.load(path, allow_pickle=False) as saved:
        fetched = str(saved["fetched_at"])
    return {**item, "point": point_id, "available_at": max(stamp(fetched), stamp(item["published_at"])).isoformat(),
            "fetched_at": fetched, "source": SOURCE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file": str(path.relative_to(DATA.parent)) if path.is_relative_to(DATA.parent) else str(path)}


def load_snapshot(directory=None):
    try:
        return json.loads((Path(directory or DIRECTORY)/"snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def due(directory=None, now=None):
    directory = Path(directory or DIRECTORY)
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    snapshot = load_snapshot(directory)
    try:
        if now-stamp(snapshot["checked_at"]) >= INTERVAL:
            return True
        return any(not chip_path(k, row["item_id"], directory).exists() for k, row in snapshot["points"].items() if row.get("item_id"))
    except (KeyError, TypeError, ValueError):
        return True


def update(*, directory=None, days=90, now=None):
    from .hydro_ingest import update_lock
    from . import m2_landsat as model
    directory = Path(directory or DIRECTORY)
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    with update_lock(directory) as locked:
        if not locked:
            return load_snapshot(directory)
        previous = load_snapshot(directory)
        state = dict(previous.get("points", {}))
        index = load_index(directory)
        additions = []
        with requests.Session() as session:
            for key, point in points().items():
                try:
                    items = search(point, (now.date()-timedelta(days=days)).isoformat(), now.date().isoformat(), session=session)
                    status = "no_scenes"
                    for item in items[:6]:
                        path = download(key, item, session=session, directory=directory, now=now)
                        row = index_row(key, item, path, now=now)
                        additions.append(row)
                        features, quality = model.features(load_chip(path), model.load_masks().get(key))
                        state[key] = {**item, **quality, "status": "ready" if features is not None else "cloudy",
                                      "checked_at": now.isoformat()}
                        status = state[key]["status"]
                        if features is not None:
                            make_preview(key, item["item_id"], directory, model.load_masks().get(key))
                            break
                    if not items:
                        state[key] = {**state.get(key, {}), "status": status, "checked_at": now.isoformat()}
                except Exception as exc:
                    state[key] = {**state.get(key, {}), "status": "source_unavailable", "error": type(exc).__name__, "checked_at": now.isoformat()}
        if additions:
            index = pd.concat([index, pd.DataFrame(additions)], ignore_index=True)
            save_index(index, directory)
            model.predict_index(directory=directory, now=now)
        snapshot = {"schema_version": 1, "checked_at": now.isoformat(), "source": SOURCE, "points": state}
        _atomic_write(directory/"snapshot.json", (json.dumps(snapshot, ensure_ascii=False, indent=2)+"\n").encode())
        return snapshot
