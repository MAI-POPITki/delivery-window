"""М2 открытой воды: спектральное правило, маска неизвестного и сравнение с прошлым."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/flood/satellite'
ALGORITHM = 's2-water-v1.1'
PARAMS = dict(mndwi_min=.1, ndvi_max=.3, nir_max=.15, buffer_pixels=2,
              pixel_size_m=20, baseline_min_observations=2, abstain_scl_water_conflict=True)
# 0 неизвестно, 1 наблюдаемая суша, 2 вода, 3 новая вода относительно базы
COLORS = {0: '#9e9e9e', 1: '#a6d96a', 2: '#0b3d91', 3: '#a50026'}


def classify(bands, scl, abstain=True):
    """Отражательная способность (не DN). Неизвестное никогда не равно суше."""
    green, red, nir, swir = [np.asarray(bands[k], dtype=float) for k in ('B03', 'B04', 'B08', 'B11')]
    if any(a.shape != scl.shape for a in (green, red, nir, swir)):
        raise ValueError('Каналы должны быть на одной сетке')
    bad = np.isin(scl, [1, 2, 3, 7, 8, 9, 10, 11])
    bad = binary_dilation(bad, iterations=PARAMS['buffer_pixels'])
    valid = np.isin(scl, [4, 5, 6]) & ~bad
    valid &= np.isfinite(np.stack([green, red, nir, swir])).all(axis=0)
    valid &= (green + swir > 0) & (nir + red > 0)
    mndwi = np.divide(green - swir, green + swir, out=np.full(scl.shape, np.nan), where=green + swir > 0)
    ndvi = np.divide(nir - red, nir + red, out=np.full(scl.shape, np.nan), where=nir + red > 0)
    water = valid & (mndwi > PARAMS['mndwi_min']) & (ndvi < PARAMS['ndvi_max']) & (nir < PARAMS['nir_max'])
    if abstain:
        # Несогласие с водным классом поставщика не является доказательством суши.
        valid &= (scl != 6) | water
    return valid, water


def baseline(masks):
    if len(masks) < 2:
        raise ValueError('Для базы нужны минимум два разных снимка')
    valid = np.sum([v for v, _ in masks], axis=0)
    water = np.sum([w for _, w in masks], axis=0)
    rate = np.divide(water, valid, out=np.full(valid.shape, np.nan), where=valid > 0)
    result = np.zeros(valid.shape, dtype=np.uint8)
    observed = valid >= PARAMS['baseline_min_observations']
    result[observed & (rate <= 1 / 3)] = 1
    result[observed & (rate >= 2 / 3)] = 2
    return result


def compare(valid, water, reference):
    if valid.shape != water.shape or reference.shape != valid.shape:
        raise ValueError('База и снимок должны быть совмещены')
    out = np.zeros(valid.shape, dtype=np.uint8)
    out[valid] = 1
    out[valid & water] = 2
    out[valid & water & (reference == 1)] = 3
    pixel_km2 = PARAMS['pixel_size_m'] ** 2 / 1e6
    stats = dict(total_km2=float(out.size * pixel_km2), observed_fraction=float(valid.mean()),
        observed_km2=float(valid.sum() * pixel_km2), water_km2=float((valid & water).sum() * pixel_km2),
        new_water_km2=float((out == 3).sum() * pixel_km2),
        comparable_km2=float((valid & (reference > 0)).sum() * pixel_km2),
        water_unknown_baseline_km2=float((valid & water & (reference == 0)).sum() * pixel_km2))
    return out, stats


def require_past(current, references):
    now = datetime.fromisoformat(current['properties']['datetime'].replace('Z', '+00:00'))
    dates = []
    generation = datetime.fromisoformat(current['properties']['s2:generation_time'].replace('Z', '+00:00'))
    for ref in references:
        d = datetime.fromisoformat(ref['properties']['datetime'].replace('Z', '+00:00'))
        if d >= now:
            raise ValueError('База содержит снимок из будущего')
        created = datetime.fromisoformat(ref['properties']['s2:generation_time'].replace('Z', '+00:00'))
        if created > generation:
            raise ValueError('База содержит более позднюю переработку продукта')
        dates.append(d.date())
    if len(dates) != len(set(dates)):
        raise ValueError('База должна содержать разные дни съёмки')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta.get('algorithm') != ALGORITHM or meta.get('parameters') != PARAMS:
        raise ValueError('Неизвестная версия расчёта М2')
    for name, digest in meta['files'].items():
        if Path(name).name != name or sha256(path.parent / name) != digest:
            raise ValueError('Нарушена целостность результата М2')
    destination = path.parents[2].resolve()
    for scene in [meta['observation'], *meta['references']]:
        for name, digest in scene['files'].items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or sha256(target) != digest:
                raise ValueError('Нарушена целостность исходного снимка М2')
    observation = {'properties': {'datetime': meta['acquired_at'], 's2:generation_time': meta['product_generated_at']}}
    references = [{'properties': {'datetime': r['acquired_at'], 's2:generation_time': r['generated_at']}} for r in meta['references']]
    require_past(observation, references)
    return meta


def available_runs(pid):
    return sorted((DATA / pid / 'runs').glob('*/manifest.json'))


def select_run(pid, as_of):
    cutoff = datetime.fromisoformat(str(as_of)[:10]).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    eligible = []
    for path in available_runs(pid):
        meta = load_run(path)
        acquired = datetime.fromisoformat(meta['acquired_at'].replace('Z', '+00:00'))
        generated = datetime.fromisoformat(meta['product_generated_at'].replace('Z', '+00:00'))
        if acquired <= cutoff and generated <= cutoff:
            eligible.append((acquired, path, meta))
    if not eligible:
        return None
    _, path, meta = max(eligible, key=lambda x: x[0])
    return path, meta
