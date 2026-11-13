"""Public GF-3 catalogue and browse PNGs from NSOAS, for visual review only.

Browse images have no calibrated radar bands. They must never enter the
Sentinel-1 VV/VH classifier or become automatic ice/event observations.
"""
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import requests
from PIL import Image

from .paths import DATA
from .s1_points import LENA_POINTS

BASE = "https://osdds.nsoas.org.cn"
DIRECTORY = DATA / "gaofen"
INTERVAL = timedelta(hours=6)
MAX_IMAGE_BYTES = 12 * 1024 * 1024
TIMEOUT = (5, 25)


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def load_snapshot(directory=None):
    try:
        value = json.loads((Path(directory or DIRECTORY) / "snapshot.json").read_text(encoding="utf-8"))
        return value if value.get("schema_version") == 1 else {}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def points(index_path=None):
    """Use the same river coordinates as Sentinel-1; landmarks are fallback only."""
    result = {key: {"name": name, "lat": lat, "lon": lon, "coordinate_source": "landmark"}
              for key, (name, lat, lon) in LENA_POINTS.items()}
    try:
        with Path(index_path or DATA / "s1" / "index.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("point") not in result:
                    continue
                try:
                    lat, lon = float(row["lat"]), float(row["lon"])
                    if not (math.isfinite(lat) and math.isfinite(lon) and -90 < lat < 90 and -180 <= lon <= 180):
                        continue
                except (ValueError, KeyError, TypeError):
                    continue
                result[row["point"]].update(lat=lat, lon=lon, coordinate_source="sentinel_1_index")
    except OSError:
        pass
    for point in result.values():
        dy = 1500 / 111320
        dx = dy / math.cos(math.radians(point["lat"]))
        point["bbox"] = [point["lon"]-dx, point["lat"]-dy, point["lon"]+dx, point["lat"]+dy]
    return result


def _inside(x, y, ring):
    inside = False
    for (ax, ay), (bx, by) in zip(ring, ring[1:]):
        cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
        if abs(cross) < 1e-10 and min(ax, bx)-1e-10 <= x <= max(ax, bx)+1e-10 and min(ay, by)-1e-10 <= y <= max(ay, by)+1e-10:
            return True
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside = not inside
    return inside


def covers_aoi(geometry, bbox):
    """Require a valid convex scene footprint covering the entire 3×3 km AOI."""
    if geometry.get("type") == "Feature":
        geometry = geometry.get("geometry") or {}
    rings = geometry.get("coordinates", [])
    if geometry.get("type") != "Polygon" or len(rings) != 1:
        return False
    ring = rings[0]
    if not 4 <= len(ring) <= 30 or ring[0] != ring[-1]:
        return False
    if any(len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
           or not (-180 <= p[0] <= 180 and -90 <= p[1] <= 90) for p in ring):
        return False
    if max(p[0] for p in ring)-min(p[0] for p in ring) >= 180:
        return False
    signs = set()
    vertices = ring[:-1]
    for i, (ax, ay) in enumerate(vertices):
        bx, by = vertices[(i+1) % len(vertices)]
        cx, cy = vertices[(i+2) % len(vertices)]
        cross = (bx-ax)*(cy-by)-(by-ay)*(cx-bx)
        if abs(cross) > 1e-10:
            signs.add(cross > 0)
    if len(signs) != 1:
        return False
    w, s, e, n = bbox
    return all(_inside(x, y, ring) for x, y in ((w, s), (w, n), (e, s), (e, n)))


def browse_url(path):
    # A provider NAS path is a path on this public gateway, never a new host.
    if not re.fullmatch(r"//metadata_\d+/DDS_IMAGES/browse/GF3/[A-Z0-9_]+/L1A/\d{8}/[A-Za-z0-9_-]+_browse\.png", str(path)):
        raise ValueError("Unsupported NSOAS browse path")
    return BASE + "/loadImg/browse/" + path


def normalize_scene(row, point, now):
    try:
        if row["satelliteName"] != "GF3" or row["productLevelName"] != "L1A":
            return None
        name = row["productName"]
        if not name.startswith("GF3_"):
            return None
        acquired = date.fromisoformat(row["imagingStartTime"][:10])
        if acquired > now.date()+timedelta(days=1):
            return None
        geometry = json.loads(row["geomSearchGeoJson"])
        if not covers_aoi(geometry, point["bbox"]):
            return None
        return {"id": str(row["searchId"]), "product_name": name,
                "acquired_at_source": row["imagingStartTime"], "acquired_date": acquired.isoformat(),
                "time_zone": "unspecified_by_catalogue", "sensor": row["sensorName"],
                "processing_level": "L1A", "polarization_code": name.split("_")[-2],
                "footprint": geometry.get("geometry", geometry), "browse_url": browse_url(row["browseFileLocation"]),
                "usage": "visual_review_only", "ml_ready": False}
    except (KeyError, ValueError, TypeError, IndexError, AttributeError):
        return None


def search_point(point, *, now, days=90, session):
    w, s, e, n = point["bbox"]
    query = {"currentPage": 1, "pageSize": 100,
             "sortedList": [{"sorted": "data_size", "type": "asc"}],
             "filterObject": {"satelliteId": ["530"], "productLevelStr": ["L1A"], "selectType": 3,
                              "imagingTime": [(now.date()-timedelta(days=days)).isoformat(), now.date().isoformat()],
                              "cloudCoverQuote": [0, 100], "isPublish": True, "geometryReturnTypeList": [1, 2],
                              "geometryQueryType": "1", "geometryQuery": f"POLYGON(({w} {n},{e} {n},{e} {s},{w} {s},{w} {n}))"}}
    scenes = {}
    for page in range(1, 4):
        query["currentPage"] = page
        response = session.post(BASE+"/api/metaData/search", json=query, timeout=TIMEOUT, allow_redirects=False)
        response.raise_for_status()
        data = response.json()
        if response.status_code != 200 or data.get("code") != 0:
            raise ValueError("NSOAS catalogue returned an error")
        payload = data["data"]
        count = int(payload["recordCount"])
        if count < 0 or count > 300:
            raise ValueError("NSOAS catalogue exceeds bounded pagination")
        rows = payload["pageData"]
        if len(rows) != min(100, max(0, count-(page-1)*100)):
            raise ValueError("NSOAS catalogue page is incomplete")
        for row in rows:
            scene = normalize_scene(row, point, now)
            if scene and date.fromisoformat(scene["acquired_date"]) >= now.date()-timedelta(days=days):
                scenes[scene["id"]] = scene
        if page*100 >= count:
            return sorted(scenes.values(), key=lambda r: (r["acquired_at_source"], r["id"]), reverse=True), count
    raise ValueError("NSOAS catalogue pagination incomplete")


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="gaofen-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def preview_path(scene, directory=None):
    key = hashlib.sha256((scene["id"]+scene["browse_url"]).encode()).hexdigest()[:24]
    return Path(directory or DIRECTORY) / "previews" / f"{key}.png"


def _image_details(content):
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("NSOAS preview exceeds size limit")
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
        if image.format != "PNG" or min(width, height) < 128 or width*height > 25_000_000:
            raise ValueError("Invalid NSOAS preview dimensions")
        image.verify()
    return width, height, hashlib.sha256(content).hexdigest()


def download_preview(scene, *, directory, session, now):
    path = preview_path(scene, directory)
    prefix = BASE+"/loadImg/browse/"
    if not scene["browse_url"].startswith(prefix) or browse_url(scene["browse_url"][len(prefix):]) != scene["browse_url"]:
        raise ValueError("Untrusted browse URL")
    content = None
    if path.exists() and path.stat().st_size <= MAX_IMAGE_BYTES:
        try:
            content = path.read_bytes()
            width, height, digest = _image_details(content)
            expected = scene.get("preview", {}).get("sha256")
            if expected and expected != digest:
                content = None
        except (OSError, ValueError, SyntaxError):
            content = None
    cached = content is not None
    if not cached:
        with session.get(scene["browse_url"], stream=True, timeout=TIMEOUT, allow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200 or response.headers.get("Content-Type", "").split(";")[0] != "image/png":
                raise ValueError("NSOAS returned a non-PNG response")
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise ValueError("NSOAS preview exceeds size limit")
            content = bytes(content)
    width, height, digest = _image_details(content)
    if not cached:
        _atomic_write(path, content)
    first_download = scene.get("preview", {}).get("downloaded_at") if cached else None
    if not first_download:
        first_download = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if cached else now.isoformat()
    return {"file": str(path.relative_to(directory)), "sha256": digest, "bytes": len(content),
            "width": width, "height": height, "downloaded_at": first_download, "validated_at": now.isoformat()}


def due(snapshot=None, *, directory=None, now=None):
    directory = Path(directory or DIRECTORY)
    snapshot = load_snapshot(directory) if snapshot is None else snapshot
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    try:
        if now-stamp(snapshot["checked_at"]) >= INTERVAL:
            return True
        for row in snapshot["points"].values():
            if row.get("scene") and not preview_path(row["scene"], directory).is_file():
                return True
        return False
    except (KeyError, ValueError, TypeError):
        return True


def update(*, directory=None, now=None, days=90, point_ids=None, session=None):
    """Keep last good scenes on failures; write PNGs before publishing metadata."""
    from .hydro_ingest import update_lock
    if not 1 <= days <= 366:
        raise ValueError("days must be in 1..366")
    directory = Path(directory or DIRECTORY)
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    targets = points()
    if point_ids is not None:
        targets = {key: targets[key] for key in point_ids}
    owned = session is None
    session = requests.Session() if owned else session
    try:
        with update_lock(directory) as locked:
            if not locked:
                return load_snapshot(directory)
            snapshot = load_snapshot(directory)
            saved = dict(snapshot.get("points", {}))
            for key, point in targets.items():
                row = {**saved.get(key, {}), **point, "checked_at": now.isoformat()}
                try:
                    catalog, count = search_point(point, now=now, days=days, session=session)
                    row.update(catalog=catalog, catalog_count=count, covered_count=len(catalog))
                    if catalog:
                        candidate = catalog[0]
                        previous = row.get("scene")
                        if previous and previous["acquired_at_source"] > candidate["acquired_at_source"]:
                            candidate = previous.copy()
                        elif previous and previous["id"] == candidate["id"] and previous.get("preview"):
                            candidate = {**candidate, "preview": previous["preview"]}
                        candidate["preview"] = download_preview(candidate, directory=directory, session=session, now=now)
                        row.update(scene=candidate, status="ready", last_success_at=now.isoformat())
                    else:
                        row["status"] = "no_full_coverage"
                    row.pop("error", None)
                except Exception as exc:  # one provider/point must not erase other valid images
                    row.update(status="source_unavailable", error=type(exc).__name__)
                saved[key] = row
            payload = {"schema_version": 1, "provider": "NSOAS", "provider_url": BASE,
                       "checked_at": now.isoformat(), "query_days": days, "asset_kind": "browse_png",
                       "ml_ready": False, "points": saved}
            _atomic_write(directory / "snapshot.json", (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode())
            return payload
    finally:
        if owned:
            session.close()
