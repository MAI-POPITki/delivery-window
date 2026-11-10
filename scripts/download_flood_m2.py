"""Реальные Sentinel-2 → маска воды, летняя база, PNG-слой и проверяемые метаданные М2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno import flood_m2 as M
from okno.flood import POINTS

STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COLLECTION = 'sentinel-2-l2a'
BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'SCL']
WINDOWS = {'lena_yakutsk': ('2024-05-15', '2024-05-31'),
    'ishim_ishim': ('2024-04-18', '2024-04-28'), 'tobol_kurgan': ('2024-04-15', '2024-04-25'),
    'ural_orenburg': ('2024-04-10', '2024-04-20'), 'ural_orsk': ('2024-04-08', '2024-04-15')}


def search(lon, lat, start, end, tile=None):
    query = dict(collections=[COLLECTION], intersects={'type': 'Point', 'coordinates': [lon, lat]},
                 datetime=f'{start}T00:00:00Z/{end}T23:59:59Z', limit=100)
    if tile:
        query['query']={'s2:mgrs_tile':{'eq':tile}}
    url, method, body = STAC + '/search', 'POST', query
    features = {}
    for _ in range(10):
        response = requests.request(method, url, json=body if method=='POST' else None, timeout=90)
        response.raise_for_status()
        result = response.json()
        features.update({item['id']:item for item in result['features']})
        link = next((l for l in result.get('links',[]) if l.get('rel')=='next'),None)
        if link is None:
            return list(features.values())
        url, method = link['href'], link.get('method','GET').upper()
        if not url.startswith(STAC+'/search') or method not in ('GET','POST'):
            raise ValueError('Неизвестная ссылка продолжения каталога')
        body = {**query,**link.get('body',{})} if link.get('merge') else link.get('body',query)
    raise ValueError('Более 10 страниц: сократите интервал запроса')


def radiometry(xml, processing_baseline):
    root = ET.fromstring(xml)
    elements = list(root.iter())
    def tag(e): return e.tag.split('}')[-1]
    quant = next((float(e.text) for e in elements if tag(e) == 'BOA_QUANTIFICATION_VALUE'), None)
    offsets = {int(e.attrib['band_id']): float(e.text) for e in elements if tag(e) == 'BOA_ADD_OFFSET'}
    if not quant or (float(processing_baseline) >= 4 and not offsets):
        raise ValueError('Нет подтверждённых коэффициентов радиометрии')
    ids = {'B02': 1, 'B03': 2, 'B04': 3, 'B08': 7, 'B11': 11}
    if offsets and any(i not in offsets for i in ids.values()):
        raise ValueError('Неполный список BOA_ADD_OFFSET')
    return dict(quantification=quant, offsets={b: offsets.get(i, 0.) for b, i in ids.items()})


def grid(lon, lat):
    from pyproj import Transformer
    from rasterio.transform import from_origin
    epsg = 32600 + int((lon + 180) // 6) + 1
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    # Привязка к 20-метровой сетке, один и тот же квадрат для всех дат.
    west, north = np.floor(x / 20) * 20 - 6000, np.ceil(y / 20) * 20 + 6000
    return dict(crs=f'EPSG:{epsg}', transform=list(from_origin(west, north, 20, 20))[:6], width=600, height=600)


def read_scene(item, location, destination):
    import planetary_computer as pc
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    folder = destination / 'scenes' / item['id']
    folder.mkdir(parents=True, exist_ok=True)
    raw, xmlpath, stacpath = folder / 'dn.npz', folder / 'product.xml', folder / 'stac.json'
    cachemeta = folder / 'crop.json'
    if raw.exists() and xmlpath.exists() and cachemeta.exists():
        cached = json.loads(cachemeta.read_text())
        if cached['grid'] != location or cached['sha256'] != M.sha256(raw):
            raise ValueError('Изменена сохранённая вырезка или сетка')
        arrays = dict(np.load(raw, allow_pickle=False))
    else:
        xml = requests.get(pc.sign(item['assets']['product-metadata']['href']), timeout=90)
        xml.raise_for_status()
        xmlpath.write_bytes(xml.content)
        def one(band):
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=2):
                with rasterio.open(pc.sign(item['assets'][band]['href'])) as src:
                    if src.crs is None:
                        raise ValueError('У исходного растра отсутствует CRS')
                    with WarpedVRT(src, crs=location['crs'], transform=Affine(*location['transform']),
                        width=location['width'], height=location['height'], src_nodata=0, nodata=0,
                        resampling=Resampling.nearest if band == 'SCL' else Resampling.bilinear) as vrt:
                        return band, vrt.read(1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            arrays = dict(pool.map(one, BANDS))
        np.savez_compressed(raw, **arrays)
        cachemeta.write_text(json.dumps(dict(grid=location, sha256=M.sha256(raw),
            resampling='bilinear spectral DN / nearest SCL', nodata=0,
            fetched_at=datetime.now(timezone.utc).isoformat()), indent=2), encoding='utf-8')
    stacpath.write_text(json.dumps(item, indent=2), encoding='utf-8')
    radi = radiometry(xmlpath.read_bytes(), item['properties']['s2:processing_baseline'])
    reflectance = {b: np.where(arrays[b] > 0, (arrays[b].astype('float64') + radi['offsets'][b]) / radi['quantification'], np.nan)
                   for b in BANDS if b != 'SCL'}
    return reflectance, arrays['SCL'], dict(item_id=item['id'], acquired_at=item['properties']['datetime'],
        fetched_at=json.loads(cachemeta.read_text()).get('fetched_at'),
        generated_at=item['properties']['s2:generation_time'], cloud_scene_percent=item['properties']['eo:cloud_cover'],
        stac_url=f'{STAC}/collections/{COLLECTION}/items/{item["id"]}',
        files={p.relative_to(destination).as_posix(): M.sha256(p) for p in (raw, xmlpath, stacpath, cachemeta)}, radiometry=radi)


def rgb(bands):
    colors = np.stack([bands[b] for b in ('B04', 'B03', 'B02')], axis=-1)
    return (np.nan_to_num(np.clip(colors / .3, 0, 1)) ** (1 / 1.5) * 255).astype('uint8')


def overlay(mask, location, out):
    from affine import Affine
    from rasterio.transform import array_bounds, from_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    transform = Affine(*location['transform'])
    bounds = array_bounds(location['height'], location['width'], transform)
    dest, width, height = calculate_default_transform(location['crs'], 'EPSG:4326', location['width'], location['height'], *bounds)
    # Картографический слой в lon/lat; не растягиваем UTM как прямоугольник на глобусе.
    warped = np.zeros((height, width), dtype='uint8')
    reproject(mask, warped, src_transform=transform, src_crs=location['crs'], dst_transform=dest,
              dst_crs='EPSG:4326', resampling=Resampling.nearest)
    rgba = np.zeros((*warped.shape, 4), dtype='uint8')
    for code, color in M.COLORS.items():
        rgba[warped == code] = [int(color[i:i + 2], 16) for i in (1, 3, 5)] + [100 if code == 0 else 35 if code == 1 else 185]
    Image.fromarray(rgba).save(out)
    west, south, east, north = array_bounds(height, width, dest)
    return [[west, north], [east, north], [east, south], [west, south]]


def process(pid, start, end, selected=None):
    station = json.loads((M.ROOT / f'data/flood/levels/{pid}.manifest.json').read_text(encoding='utf-8'))
    lon, lat = station['station']['geometry']['coordinates']
    if selected is not None:
        current,chosen=selected
    else:
        scenes = search(lon, lat, start, end)
        if not scenes:
            raise ValueError('Нет сцен за указанный период')
        current = min(scenes, key=lambda i: (i['properties']['eo:cloud_cover'], i['id']))
        tile = current['properties']['s2:mgrs_tile']
        year = int(current['properties']['datetime'][:4]) - 1
        references = search(lon, lat, f'{year}-06-01', f'{year}-08-31', tile=tile)
        generated = datetime.fromisoformat(current['properties']['s2:generation_time'].replace('Z', '+00:00'))
        references = sorted([i for i in references if i['properties']['s2:mgrs_tile'] == tile
                             and datetime.fromisoformat(i['properties']['s2:generation_time'].replace('Z', '+00:00')) <= generated],
                            key=lambda i: (i['properties']['eo:cloud_cover'], i['id']))
        chosen, dates = [], set()
        for item in references:
            d = item['properties']['datetime'][:10]
            if d not in dates:
                chosen.append(item); dates.add(d)
            if len(chosen) == 3:
                break
    if len(chosen) < 3:
        raise ValueError('Недостаточно разных дат для летней базы')
    M.require_past(current, chosen)
    destination = M.DATA / pid
    location = grid(lon, lat)
    masks, provenance, base_rgb = [], [], []
    for item in chosen:
        bands, scl, meta = read_scene(item, location, destination)
        valid, water = M.classify(bands, scl)
        masks.append((valid, water)); provenance.append(meta)
        base_rgb.append(np.where(valid[..., None], rgb(bands).astype(float), np.nan))
        print(pid, item['id'], 'baseline', f'{valid.mean():.1%}', flush=True)
    reference = M.baseline(masks)
    bands, scl, meta = read_scene(current, location, destination)
    valid, water = M.classify(bands, scl)
    mask, stats = M.compare(valid, water, reference)
    folder = destination / 'runs' / current['id']
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'mask.npz', mask=mask, reference=reference)
    Image.fromarray(rgb(bands)).save(folder / 'rgb.png')
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', RuntimeWarning)
        composite = np.nanmedian(np.stack(base_rgb), axis=0)
    Image.fromarray(np.nan_to_num(composite).astype('uint8')).save(folder / 'baseline.png')
    coordinates = overlay(mask, location, folder / 'overlay.png')
    quality,_=M.classify(bands,scl,abstain=False)
    scl_water = quality & (scl == 6)
    union = (water | scl_water).sum()
    stats['scl_water_agreement_iou'] = float((water & scl_water).sum() / union) if union else None
    manifest = dict(algorithm=M.ALGORITHM, parameters=M.PARAMS, pid=pid,
        acquired_at=meta['acquired_at'], product_generated_at=meta['generated_at'],
        processed_at=datetime.now(timezone.utc).isoformat(), source='Copernicus Sentinel-2 L2A / Microsoft Planetary Computer',
        station=station['station'], grid=location, overlay_coordinates=coordinates, statistics=stats,
        observation=meta, references=provenance, selection_window=[start, end],
        selection='minimum scene cloud cover, no tuning of water thresholds',
        files={name: M.sha256(folder / name) for name in ['mask.npz', 'rgb.png', 'baseline.png', 'overlay.png']})
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    M.load_run(folder / 'manifest.json')
    print(pid, current['id'], json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', nargs='+', choices=list(POINTS), default=list(POINTS))
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--recent', action='store_true')
    parser.add_argument('--rebuild',action='store_true',help='Пересчитать сохранённые сцены без сетевых запросов')
    parser.add_argument('--days', type=int, default=14)
    args = parser.parse_args()
    if bool(args.start) != bool(args.end):
        parser.error('--start и --end задаются вместе')
    failures = []
    for pid in args.points:
        start, end = WINDOWS[pid]
        if args.recent:
            end = date.today().isoformat(); start = (date.today() - timedelta(days=args.days)).isoformat()
        if args.start:
            start, end = args.start, args.end
        try:
            if args.rebuild:
                for manifest in M.available_runs(pid):
                    record=json.loads(manifest.read_text(encoding='utf-8'))
                    destination=M.DATA/pid
                    selected=[]
                    for scene in [record['observation'],*record['references']]:
                        for name,digest in scene['files'].items():
                            target=(destination/name).resolve()
                            if not target.is_relative_to(destination.resolve()) or M.sha256(target)!=digest:
                                raise ValueError('Повреждены исходные файлы')
                        selected.append(json.loads((destination/'scenes'/scene['item_id']/'stac.json').read_text(encoding='utf-8')))
                    process(pid,*record['selection_window'],selected=(selected[0],selected[1:]))
            else:
                process(pid, start, end)
        except Exception as exc:
            failures.append(pid)
            # В исключении rasterio бывает временный SAS-токен; не выводим URL.
            print(pid, 'FAILED', type(exc).__name__, flush=True)
    if failures:
        raise SystemExit('Не обработаны: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
