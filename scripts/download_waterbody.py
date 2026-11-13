"""Reproduce one archived waterbody beta from public Earth Search COGs.

Only pre-baseline-04 products are admitted until offset harmonization of newer
Earth Search products is independently verified. Existing flood download is unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import binary_dilation

from okno import flood_m2 as M
from okno import waterbodies as W
from okno.http_range import RangeReader
from scripts.download_flood_m2 import grid, rgb

STAC='https://earth-search.aws.element84.com/v1'
QUERY=dict(collections=['sentinel-2-l2a'],intersects=dict(type='Point',coordinates=[60.478,56.857]),
    datetime='2021-06-01T00:00:00Z/2021-09-30T23:59:59Z',query={'eo:cloud_cover':{'lt':10}},limit=60)
IDS=['S2A_41VLD_20210601_0_L2A','S2B_41VLD_20210921_0_L2A']
BANDS=dict(B02='blue',B03='green',B04='red',B08='nir',B11='swir16',SCL='scl')


def scale_offset(item,asset):
    if float(item['properties']['s2:processing_baseline'])>=4:
        raise ValueError('Новые baseline Earth Search требуют отдельной проверки BOA offset')
    band=asset['raster:bands'][0]
    scale,offset=float(band['scale']),float(band.get('offset',0))
    if not np.isfinite([scale,offset]).all() or scale<=0 or offset!=0:
        raise ValueError('Непроверенная радиометрия старого продукта')
    return scale,offset


def read_scene(item,location,folder):
    # GDAL's curl does not inherit requests/ssl CA configuration automatically.
    # Reuse the environment trust bundle; TLS verification stays enabled.
    if os.environ.get('SSL_CERT_FILE'):
        os.environ.setdefault('CURL_CA_BUNDLE',os.environ['SSL_CERT_FILE'])
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from affine import Affine
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'stac.json').write_text(json.dumps(item,ensure_ascii=False,indent=2))
    path=folder/'dn.npz'
    if path.exists():
        cached=json.loads((folder/'crop.json').read_text())
        if cached['grid']!=location or M.sha256(path)!=cached['sha256']:
            raise ValueError('Сохранённая вырезка изменилась')
        with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    else:
        def read(pair):
            band,key=pair
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT=45,GDAL_HTTP_MAX_RETRY=1,
                              GDAL_HTTP_CAINFO=os.environ.get('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt')):
                with rasterio.open(item['assets'][key]['href'],opener=RangeReader) as source:
                    with WarpedVRT(source,crs=location['crs'],transform=Affine(*location['transform']),
                        width=location['width'],height=location['height'],src_nodata=0,nodata=0,
                        resampling=Resampling.nearest if band=='SCL' else Resampling.bilinear) as vrt:
                        return band,vrt.read(1)
        with ThreadPoolExecutor(max_workers=3) as pool:arrays=dict(pool.map(read,BANDS.items()))
        np.savez_compressed(path,**arrays)
        (folder/'crop.json').write_text(json.dumps(dict(grid=location,sha256=M.sha256(path),
            resampling='bilinear spectral / nearest SCL',fetched_at=datetime.now(timezone.utc).isoformat()),indent=2))
    reflectance={}
    for band,key in BANDS.items():
        if band=='SCL':continue
        scale,offset=scale_offset(item,item['assets'][key])
        reflectance[band]=np.where(arrays[band]>0,arrays[band].astype(float)*scale+offset,np.nan)
    return reflectance,arrays['SCL']


def main():
    import rasterio.features
    from rasterio.warp import transform_geom
    from affine import Affine
    root=W.DATA;root.mkdir(parents=True,exist_ok=True)
    osm_path=root/'osm_response.json'
    if not osm_path.exists():
        r=requests.get('https://nominatim.openstreetmap.org/search',params=dict(q='Верх-Исетское водохранилище',format='jsonv2',polygon_geojson=1),
            headers={'User-Agent':'OknoZavoza research beta'},timeout=30);r.raise_for_status();osm_path.write_bytes(r.content)
    osm=next(x for x in json.loads(osm_path.read_text()) if x['geojson']['type']=='Polygon' and 'Верх-Исетское водохранилище' in x['display_name'])
    location=grid(60.478,56.857)
    catalog=root/'search.json'
    if not catalog.exists():
        r=requests.post(STAC+'/search',json=QUERY,timeout=45);r.raise_for_status();catalog.write_bytes(r.content)
    features=json.loads(catalog.read_text())['features']
    aoi=rasterio.features.rasterize([(transform_geom('EPSG:4326',location['crs'],osm['geojson']),1)],
        out_shape=(location['height'],location['width']),transform=Affine(*location['transform'])).astype(bool)
    aoi=binary_dilation(aoi,iterations=8)
    arrays={'aoi':aoi};scenes=[]
    for i,item_id in enumerate(IDS):
        item=next(f for f in features if f['id']==item_id)
        print('Download',item_id,flush=True)
        bands,scl=read_scene(item,location,root/'scenes'/item_id)
        valid,water=M.classify(bands,scl)
        arrays[f'valid_{i}'],arrays[f'water_{i}']=valid,water
        Image.fromarray(rgb(bands)).save(root/f'rgb_{i}.jpg',quality=92)
        scenes.append(dict(item_id=item_id,acquired_at=item['properties']['datetime'],grid=location,
            product_generated_at=item['properties']['s2:generation_time'],processing_baseline=item['properties']['s2:processing_baseline'],
            source_url=f'{STAC}/collections/sentinel-2-l2a/items/{item_id}',rgb=f'rgb_{i}.jpg'))
        print('Saved',item_id,flush=True)
    np.savez_compressed(root/'masks.npz',**arrays)
    included=[osm_path,catalog,root/'masks.npz',*[root/f'rgb_{i}.jpg' for i in range(len(IDS))]]
    for item_id in IDS:included.extend((root/'scenes'/item_id/n) for n in ['stac.json','dn.npz','crop.json'])
    meta=dict(version=W.VERSION,algorithm=M.ALGORITHM,parameters=M.PARAMS,name='Верх-Исетское водохранилище',
        region='Свердловская область',grid=location,scenes=scenes,
        aoi=dict(source='OpenStreetMap / Nominatim',osm_type=osm['osm_type'],osm_id=osm['osm_id'],
            source_url=f'https://www.openstreetmap.org/{osm["osm_type"]}/{osm["osm_id"]}',
            note='Фиксированный полигон OSM с буфером 8 пикселей (приблизительно 160 м); не историческая разметка берега'),
        radiometry='pre-04 baseline, explicit asset scale, zero offset; no additional BOA subtraction',
        fetched_at=datetime.now(timezone.utc).isoformat(),files={p.relative_to(root).as_posix():M.sha256(p) for p in included})
    (root/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    _,_,stats,_=W.compare_sample();print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
