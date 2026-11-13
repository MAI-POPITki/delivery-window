"""ERA5 wind matched to radar acquisition times, for an isolated M2 experiment."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

API = 'https://archive-api.open-meteo.com/v1/archive'
SPEED_FEATURES = ['wind_speed_ms', 'wind_mean6_ms', 'wind_max6_ms']
WIND_FEATURES = SPEED_FEATURES + ['wind_u_ms', 'wind_v_ms']


def acquisition_time(item_id):
    match = re.search(r'_(\d{8}T\d{6})_', str(item_id))
    if match is None:
        raise ValueError('Нет времени съёмки Sentinel-1 в item_id')
    return pd.Timestamp(datetime.strptime(match[1], '%Y%m%dT%H%M%S'), tz='UTC')


def parse_hourly(payload):
    if payload.get('timezone') != 'GMT' or payload.get('utc_offset_seconds') != 0:
        raise ValueError('Ожидались часы UTC')
    units = payload.get('hourly_units', {})
    if units.get('wind_speed_10m') != 'm/s' or units.get('wind_direction_10m') != '°':
        raise ValueError('Неверные единицы ветра')
    data = payload['hourly']
    times = pd.to_datetime(data['time'], utc=True, errors='raise')
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Неверные или повторяющиеся часы ветра')
    if not (times == times.floor('h')).all():
        raise ValueError('Ожидались полные часы')
    speed = np.asarray(data['wind_speed_10m'], dtype=float)
    direction = np.asarray(data['wind_direction_10m'], dtype=float)
    if len(speed) != len(times) or len(direction) != len(times):
        raise ValueError('Несогласованные длины почасовых данных')
    if np.isinf(speed).any() or np.isinf(direction).any():
        raise ValueError('Бесконечные значения ветра')
    if ((speed < 0) | (direction < 0) | (direction > 360)).any():
        raise ValueError('Значения ветра вне допустимых границ')
    return pd.DataFrame({'speed': speed, 'direction': direction}, index=times)


def wind_features(hourly, acquired_at):
    stamp = pd.Timestamp(acquired_at)
    if stamp.tzinfo is None:
        raise ValueError('Время съёмки должно иметь часовой пояс')
    last = stamp.tz_convert('UTC').floor('h')
    window = hourly.reindex(pd.date_range(end=last, periods=6, freq='h'))
    if window.isna().any().any():
        raise ValueError('Нет шести полных часов ветра перед съёмкой')
    speed, direction = float(window.speed.iloc[-1]), float(window.direction.iloc[-1])
    radians = np.deg2rad(direction)
    return dict(wind_speed_ms=speed, wind_mean6_ms=float(window.speed.mean()),
                wind_max6_ms=float(window.speed.max()),
                wind_u_ms=float(-speed*np.sin(radians)),
                wind_v_ms=float(-speed*np.cos(radians)),
                wind_hour_utc=last.isoformat())


def save_response(path, raw, url, point):
    """Preserve exact response bytes and explicit request semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0))
    metadata = dict(source='Open-Meteo Historical Weather API / ERA5', model='era5',
                    url=url, requested_point=point, fetched_at=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(raw).hexdigest(), timezone='UTC', speed_unit='m/s')
    path.with_suffix('.manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def load_response(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes())
    meta = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError('Нарушена целостность ответа ERA5')
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(meta['url'])
    query = parse_qs(parsed.query)
    if parsed.scheme != 'https' or parsed.netloc != 'archive-api.open-meteo.com' or parsed.path != '/v1/archive':
        raise ValueError('Неизвестный источник ветра')
    if any(query.get(k) != [v] for k,v in [('models','era5'),('timezone','UTC'),('wind_speed_unit','ms')]):
        raise ValueError('Не зафиксирован ERA5/UTC/м/с')
    payload = json.loads(raw)
    requested = meta['requested_point']
    if abs(payload['latitude']-requested['lat'])>.3 or abs(payload['longitude']-requested['lon'])>.3:
        raise ValueError('Слишком далёкая ячейка ERA5')
    return parse_hourly(payload), meta
