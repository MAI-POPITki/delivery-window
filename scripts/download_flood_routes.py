"""Сохранение реальных геометрий OSM для карты паводкового планирования."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import requests
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno.flood import DATA
from okno.flood_sites import SITES as POINTS
from okno.flood_routes import parse_roads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', nargs='+', choices=list(POINTS), default=list(POINTS))
    args = parser.parse_args()
    folder = DATA / 'routes'
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'delivery-window/1.0 (+https://github.com/MAI-POPITki/delivery-window)'
    failures = []
    for pid, point in POINTS.items():
        if pid not in args.points:
            continue
        lat, lon = point['lat'], point['lon']
        # Рамка охватывает берега и городские подъезды; это не граница затопления.
        dx, dy = .02, .015
        if pid == 'lena_yakutsk':
            lon = 129.75  # городской берег Лены; точка расхода находится в русле
        bbox = f'{lon-dx},{lat-dy},{lon+dx},{lat+dy}'
        try:
            response = session.get('https://www.openstreetmap.org/api/0.6/map', params={'bbox': bbox}, timeout=45)
            response.raise_for_status()
            routes = parse_roads(response.content, pid)
            if not routes:
                raise ValueError('В ответе нет подходящих дорог')
            (folder / f'{pid}.osm.gz').write_bytes(gzip.compress(response.content, mtime=0))
            meta = dict(source_url=response.url, fetched_at=datetime.now(timezone.utc).isoformat(),
                        sha256=hashlib.sha256(response.content).hexdigest(), count=len(routes),
                        attribution='© OpenStreetMap contributors, ODbL 1.0')
            (folder / f'{pid}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{pid}: {len(routes)} road segments')
        except (requests.RequestException, ValueError) as exc:
            failures.append(f'{pid}: {exc}')
    if failures:
        raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
