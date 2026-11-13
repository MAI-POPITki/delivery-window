"""Waterbody beta: compare open water on a fixed jointly observed area.

Classification reuses the working Sentinel-2 M2, not a new trained model.
Missing pixels never count as dry land; no volume/depth is inferred.
"""
import json
from pathlib import Path

import numpy as np
from pyproj import CRS

from . import flood_m2 as M

DATA = M.ROOT / 'data/waterbodies/verkh_iset'
VERSION = 'waterbody-area-beta-v1'
MIN_COMMON_FRACTION = .8
COLORS = {0: (190,190,190), 1: (233,236,224), 2: (24,91,157),
          3: (230,149,57), 4: (41,175,191)}


def pixel_area_km2(grid):
    crs = CRS.from_user_input(grid['crs'])
    if not crs.is_projected or any(a.unit_conversion_factor != 1 for a in crs.axis_info[:2]):
        raise ValueError('Для площади нужна проекция в метрах, не градусах')
    a,b,_,d,e,_ = grid['transform']
    area = abs(a*e-b*d) / 1e6
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Некорректная площадь пикселя')
    return area


def compare_masks(before_valid, before_water, after_valid, after_water, aoi, pixel_km2):
    arrays = [np.asarray(a) for a in (before_valid,before_water,after_valid,after_water,aoi)]
    if any(a.ndim != 2 or a.shape != arrays[0].shape or a.dtype != bool for a in arrays):
        raise ValueError('Нужны совмещённые двумерные булевы маски одной сетки')
    bv,bw,av,aw,aoi = arrays
    if not aoi.any() or not np.isfinite(pixel_km2) or pixel_km2 <= 0:
        raise ValueError('Нужны непустая область и положительная площадь пикселя')
    if (bw & ~bv).any() or (aw & ~av).any():
        raise ValueError('Вода не может быть помечена в неизвестных пикселях')
    common = aoi & bv & av
    mask = np.zeros(aoi.shape,dtype=np.uint8)
    mask[common] = 1
    mask[common & bw & aw] = 2
    mask[common & bw & ~aw] = 3
    mask[common & ~bw & aw] = 4
    n = int(aoi.sum()); both = int(common.sum())
    b = int((common & bw).sum()); a = int((common & aw).sum())
    gain = int((mask == 4).sum()); loss = int((mask == 3).sum())
    # Bounds account only for unseen area, not spectral classification error.
    b_observed = int((aoi & bv & bw).sum()); a_observed = int((aoi & av & aw).sum())
    b_unknown = int((aoi & ~bv).sum()); a_unknown = int((aoi & ~av).sum())
    delta_bounds = ((a_observed-b_observed-b_unknown)*pixel_km2,
                    (a_observed+a_unknown-b_observed)*pixel_km2)
    stats = dict(aoi_km2=n*pixel_km2,common_km2=both*pixel_km2,common_fraction=both/n,
        before_common_km2=b*pixel_km2,after_common_km2=a*pixel_km2,delta_common_km2=(a-b)*pixel_km2,
        delta_common_percent=100*(a-b)/b if b else None,gain_km2=gain*pixel_km2,loss_km2=loss*pixel_km2,
        before_observed_fraction=float((aoi & bv).sum()/n),after_observed_fraction=float((aoi & av).sum()/n),
        before_observed_water_km2=b_observed*pixel_km2,after_observed_water_km2=a_observed*pixel_km2,
        before_full_area_bounds_km2=[b_observed*pixel_km2,(b_observed+b_unknown)*pixel_km2],
        after_full_area_bounds_km2=[a_observed*pixel_km2,(a_observed+a_unknown)*pixel_km2],
        full_area_delta_bounds_km2=list(delta_bounds),
        comparable=both/n >= MIN_COMMON_FRACTION,
        full_area_direction=('increase' if delta_bounds[0] > 0 else 'decrease' if delta_bounds[1] < 0 else 'unknown'))
    return mask,stats


def image(mask, aoi):
    out = np.full((*mask.shape,3),255,dtype=np.uint8)
    for code,color in COLORS.items():out[(mask == code) & aoi] = color
    return out


def load_sample(directory=None):
    folder = Path(directory or DATA)
    path = folder/'manifest.json'
    if not path.exists():
        raise ValueError('Сохранённый спутниковый пример ещё не загружен')
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta.get('version') != VERSION or meta.get('algorithm') != M.ALGORITHM or meta.get('parameters') != M.PARAMS:
        raise ValueError('Неизвестная версия спутникового расчёта')
    for name,digest in meta['files'].items():
        p = (folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or M.sha256(p) != digest:
            raise ValueError('Нарушена целостность спутникового примера')
    grid = meta['grid']; pixel_area_km2(grid)
    with np.load(folder/'masks.npz',allow_pickle=False) as raw:
        arrays = {k:raw[k].copy() for k in raw.files}
    shape = (grid['height'],grid['width'])
    if arrays['aoi'].shape != shape:
        raise ValueError('Область не соответствует сетке')
    scenes = meta['scenes']
    dates = [s['acquired_at'][:10] for s in scenes]
    if dates != sorted(set(dates)) or len(dates) < 2:
        raise ValueError('Нужны разные даты наблюдений по порядку')
    for i,scene in enumerate(scenes):
        if scene['grid'] != grid:
            raise ValueError('Снимки имеют разные сетки')
        if arrays[f'valid_{i}'].shape != shape or arrays[f'water_{i}'].shape != shape:
            raise ValueError('Маски не соответствуют сетке')
    return meta,arrays


def compare_sample(before=0,after=1,directory=None):
    meta,arrays = load_sample(directory)
    if not 0 <= before < after < len(meta['scenes']):
        raise ValueError('Первая дата должна быть раньше второй')
    mask,stats = compare_masks(arrays[f'valid_{before}'],arrays[f'water_{before}'],
        arrays[f'valid_{after}'],arrays[f'water_{after}'],arrays['aoi'],pixel_area_km2(meta['grid']))
    return meta,mask,stats,arrays
