"""Optical branch of M2, trained separately from the Sentinel-1 radar model."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MODELS
from . import landsat as L
from .gaofen import _atomic_write, stamp

MODEL = MODELS / "m2_landsat_ice.cbm"
MASKS = MODELS / "m2_landsat_river_masks.npz"
META = MODELS / "m2_landsat.json"
MIN_PIXELS, MIN_CLEAR = 30, 0.35
FEATURES = [f"{b}_{stat}" for b in L.BANDS[:6] for stat in ("p10", "p50", "p90", "std")]
FEATURES += [f"{b}_{stat}" for b in ("ndsi", "ndwi", "ndvi") for stat in ("p10", "p50", "p90")]


def metadata():
    try:
        return json.loads(META.read_text())
    except (OSError, ValueError):
        return {}


def load_masks():
    if not MASKS.exists():
        return {}
    expected = metadata().get("mask_sha256")
    if expected and hashlib.sha256(MASKS.read_bytes()).hexdigest() != expected:
        raise ValueError("Landsat river-mask checksum differs from manifest")
    with np.load(MASKS, allow_pickle=False) as masks:
        return {k: masks[k].astype(bool) for k in masks.files}


def normalized_difference(a, b):
    out = np.full_like(a, np.nan, dtype=np.float32)
    np.divide(a-b, a+b, out=out, where=(a+b) > 0.02)
    return np.clip(out, -1, 1)


def features(stack, mask):
    if mask is None or mask.shape != stack.shape[1:] or int(mask.sum()) < MIN_PIXELS:
        return None, {"quality": "no_river_mask", "valid_fraction": 0.0, "valid_pixels": 0}
    valid = L.clear_pixels(stack) & mask
    n = int(valid.sum())
    fraction = n/int(mask.sum())
    quality = {"quality": "usable" if n >= MIN_PIXELS and fraction >= MIN_CLEAR else "cloud_or_nodata",
               "valid_fraction": round(fraction, 4), "valid_pixels": n, "river_pixels": int(mask.sum())}
    if quality["quality"] != "usable":
        return None, quality
    reflectance = L.reflectance(stack)
    result = {}
    for name, array in zip(L.BANDS[:6], reflectance):
        values = array[valid]
        for stat, value in zip(("p10", "p50", "p90"), np.percentile(values, [10, 50, 90])):
            result[f"{name}_{stat}"] = float(value)
        result[f"{name}_std"] = float(values.std())
    green, red, nir, swir = reflectance[1], reflectance[2], reflectance[3], reflectance[4]
    for name, array in (("ndsi", normalized_difference(green, swir)),
                        ("ndwi", normalized_difference(green, nir)),
                        ("ndvi", normalized_difference(nir, red))):
        values = array[valid]
        values = values[np.isfinite(values)]
        quantiles = np.percentile(values, [10, 50, 90]) if len(values) else [np.nan]*3
        result.update({f"{name}_{stat}": float(v) for stat, v in zip(("p10", "p50", "p90"), quantiles)})
    return result, quality


def build_mask(stacks, seed=None):
    """QA-clear summer water consensus; labels/held-out winters are not used."""
    from scipy.ndimage import binary_dilation, label
    if len(stacks) < 2:
        return None
    count = np.zeros((L.SIZE, L.SIZE), dtype=np.uint16)
    votes = np.zeros_like(count)
    for stack in stacks:
        clear = L.clear_pixels(stack)
        r = L.reflectance(stack)
        water = (normalized_difference(r[1], r[4]) > 0.15) & (normalized_difference(r[3], r[2]) < 0.15) & (r[3] < 0.18)
        count += clear
        votes += clear & water
    water = (count >= 2) & (votes >= np.maximum(1, count*0.5))
    components, n = label(water)
    candidates = []
    target = np.zeros_like(water)
    if seed is not None:
        from PIL import Image
        small = np.array(Image.fromarray(seed).resize((100, 100), Image.Resampling.NEAREST)) > 0
        target[50:150, 50:150] = small
        target = binary_dilation(target, iterations=4)
    for i in range(1, n+1):
        component = components == i
        size = int(component.sum())
        if size < MIN_PIXELS:
            continue
        y, x = np.where(component)
        distance = float(np.min((x-L.SIZE/2)**2+(y-L.SIZE/2)**2))
        overlap = int((component & target).sum())
        candidates.append((overlap, -distance, size, component))
    if not candidates:
        return None
    return max(candidates, key=lambda r: r[:3])[3]


def reference_label(day, dates, margin=14):
    """Phenology-derived class; uncertain boundaries and implausible dates abstain."""
    day = pd.Timestamp(day).normalize()
    hy = day.year+int(day.month >= 9)
    current, following = dates.get(hy, {}), dates.get(hy+1, {})
    freeze, breakup = current.get("freezeup"), current.get("breakup")
    if freeze is None or breakup is None:
        return "unknown", hy
    freeze, breakup = pd.Timestamp(freeze), pd.Timestamp(breakup)
    if freeze.month not in (9, 10, 11, 12, 1) or breakup.month not in (4, 5, 6, 7) or not freeze < breakup:
        return "unknown", hy
    buffer = pd.Timedelta(days=margin)
    if min(abs(day-freeze), abs(day-breakup)) <= buffer:
        return "transition", hy
    if freeze+buffer < day < breakup-buffer:
        return "ice", hy
    if day < freeze-buffer:
        return "water", hy
    nxt = following.get("freezeup")
    if day > breakup+buffer and (nxt is not None and day < pd.Timestamp(nxt)-buffer or nxt is None and day.month < 9):
        return "water", hy
    return "unknown", hy


def make_model():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, random_seed=42,
                              auto_class_weights="Balanced", verbose=False, allow_writing_files=False, thread_count=2)


def load_model():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(MODEL))
    return model


def predict_index(*, directory=None, now=None):
    directory = Path(directory or L.DIRECTORY)
    meta = metadata()
    if not meta.get("enabled") or not MODEL.exists():
        return pd.DataFrame()
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != meta.get("model_sha256"):
        raise ValueError("Landsat model checksum differs from validation manifest")
    index = L.load_index(directory)
    if index.empty:
        return pd.DataFrame()
    masks, model = load_masks(), load_model()
    prediction_file = directory/"predictions.csv"
    previous = pd.read_csv(prediction_file).to_dict("records") if prediction_file.exists() else []
    version = meta["model_sha256"][:12]
    existing = {(r["point"], r["item_id"]): r for r in previous if r.get("model_version") == version}
    rows = dict(existing)
    now = datetime.now(timezone.utc) if now is None else stamp(now)
    for row in index.to_dict("records"):
        key = (row["point"], row["item_id"])
        if key in existing:
            continue
        try:
            path = L.chip_path(row["point"], row["item_id"], directory)
            feat, quality = features(L.load_chip(path), masks.get(row["point"]))
            probability = float(model.predict_proba(pd.DataFrame([feat], columns=FEATURES))[0, 1]) if feat is not None else None
            available = max(stamp(row["available_at"]), stamp(meta["trained_at"]), now)
            rows[key] = {"point": row["point"], "item_id": row["item_id"], "date": row["date"],
                         "observed_at": row["observed_at"], "available_at": available.isoformat(),
                         "platform": row["platform"], "source": row["platform"].replace("landsat-", "Landsat "),
                         "p_ice": probability, **quality, "model_version": version}
        except (OSError, ValueError, KeyError):
            continue
    result = pd.DataFrame(list(rows.values()))
    if len(result):
        _atomic_write(directory/"predictions.csv", result.to_csv(index=False, lineterminator="\n").encode())
    return result
