"""Сохранить реальные метаданные новых постов без выдуманных рядов уровня."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okno.flood_sites import EXTRA_SITES

ROOT=Path(__file__).resolve().parents[1]


def main():
    url='https://level-satellite.ru/api/posts_geojson'
    r=requests.post(url,json={},timeout=60);r.raise_for_status()
    raw=r.content;features=r.json()['features']
    folder=ROOT/'data/flood/sites';folder.mkdir(exist_ok=True)
    (folder/'posts_geojson.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    for pid,site in EXTRA_SITES.items():
        matches=[f for f in features if f['properties']['post_code']==site['gauge_code']]
        if len(matches)!=1:raise ValueError('Неоднозначный пост')
        station=matches[0];lon,lat=station['geometry']['coordinates']
        if abs(lon-site['lon'])>.001 or abs(lat-site['lat'])>.001:raise ValueError('Изменились координаты')
        meta=dict(source=url,request={},fetched_at=datetime.now(timezone.utc).isoformat(),station=station,
            sha256=hashlib.sha256(raw).hexdigest(),raw_file='data/flood/sites/posts_geojson.json.gz',
            summary={},coverage='satellite only; no measured level series downloaded')
        (ROOT/f'data/flood/levels/{pid}.manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        print(pid,site['gauge_code'])


if __name__=='__main__':main()
