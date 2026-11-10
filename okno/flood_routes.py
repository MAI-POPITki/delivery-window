"""Геометрии реальных дорог OSM; близость к реке не доказывает затопление."""
import gzip
import hashlib
import json
from functools import lru_cache
import xml.etree.ElementTree as ET

from .flood import DATA
from .flood_sites import SITES as POINTS


def parse_roads(raw, pid):
    root = ET.fromstring(raw)
    nodes = {n.attrib['id']: [float(n.attrib['lon']), float(n.attrib['lat'])] for n in root.findall('node')}
    routes = []
    for way in root.findall('way'):
        tags = {t.attrib['k']: t.attrib['v'] for t in way.findall('tag')}
        highway = tags.get('highway')
        bridge = tags.get('bridge') not in (None, 'no')
        ferry = tags.get('route') == 'ferry'
        if highway not in {'motorway','trunk','primary','secondary','tertiary'} and not (bridge and highway) and not ferry:
            continue
        refs = [n.attrib['ref'] for n in way.findall('nd')]
        if len(refs) < 2 or any(ref not in nodes for ref in refs):
            continue
        osm_id = way.attrib['id']
        routes.append(dict(id=f'{pid}_{osm_id}', point_id=pid, osm_id=osm_id,
                           name=tags.get('name') or tags.get('ref') or f'Участок OSM {osm_id}',
                           kind='Мост' if bridge else 'Переправа' if ferry else 'Дорога',
                           coordinates=[nodes[ref] for ref in refs], source=f'https://www.openstreetmap.org/way/{osm_id}'))
    return routes


@lru_cache(maxsize=16)
def _load(pid, meta_mtime, raw_mtime, size):
    folder = DATA / 'routes'
    metadata = json.loads((folder / f'{pid}.json').read_text(encoding='utf-8'))
    raw = gzip.decompress((folder / f'{pid}.osm.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != metadata['sha256']:
        raise ValueError('Повреждена геометрия OSM')
    return parse_roads(raw, pid), metadata


def load_routes(pid):
    if pid not in POINTS:
        raise ValueError('Неизвестная речная точка')
    folder = DATA / 'routes'
    try:
        meta, raw = (folder / f'{pid}.json').stat(), (folder / f'{pid}.osm.gz').stat()
        return _load(pid, meta.st_mtime_ns, raw.st_mtime_ns, raw.st_size)
    except FileNotFoundError:
        return [], {}
