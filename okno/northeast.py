"""PANGAEA river gauges with observations extending beyond 2012.

IDs are the original station numbers, not aliases of existing Arctic Data Center
gauges. Administrative regions are assigned by the station settlement; the Anyuy
gauges (including Konstantinovskaya) belong to Chukotka.
"""
from io import StringIO
import numpy as np
import pandas as pd
from .paths import RAW

SOURCE = 'Makarieva et al., PANGAEA, doi:10.1594/PANGAEA.925460'
DATA_URL = 'https://doi.org/10.1594/PANGAEA.925460'
META_URL = 'https://doi.org/10.1594/PANGAEA.925456'
DIRECTORY = RAW / 'pangaea_river_ice'
# station ID: (Russian name, administrative region key, basin)
STATIONS = {
    1001: ('Колыма — Оротук', 'magadan', 'Kolyma'),
    1010: ('Колыма — Усть-Среднекан', 'magadan', 'Kolyma'),
    1018: ('Колыма — Зырянка', 'yakutia', 'Kolyma'),
    1176: ('Бохапча — 5,4 км от устья', 'magadan', 'Kolyma'),
    1367: ('Берёзовка — Берёзовка', 'yakutia', 'Kolyma'),
    1397: ('Малый Анюй — Островное', 'chukotka', 'Kolyma'),
    1398: ('Малый Анюй — Анюйск', 'chukotka', 'Kolyma'),
    1496: ('Анадырь — Ламутское', 'chukotka', 'Anadyr'),
    1497: ('Анадырь — Новый Еропол', 'chukotka', 'Anadyr'),
    1500: ('Анадырь — Усть-Белая', 'chukotka', 'Anadyr'),
    1502: ('Еропол — Чуванское', 'chukotka', 'Anadyr'),
    1504: ('Майн — Ваеги', 'chukotka', 'Anadyr'),
    1508: ('Энмываам — Мухоморное', 'chukotka', 'Anadyr'),
    1566: ('Тауй — Талон', 'magadan', 'Tauy'),
    1578: ('Ясачная — Нелемное', 'yakutia', 'Kolyma'),
    1596: ('Большой Анюй — Константиновская', 'chukotka', 'Kolyma'),
    1801: ('Колыма — Среднеколымск', 'yakutia', 'Kolyma'),
    1802: ('Колыма — Колымское', 'yakutia', 'Kolyma'),
    1805: ('Колыма — Черский', 'yakutia', 'Kolyma'),
}
REGION_NAMES = {'yakutia':'Respublika Sakha (Yakutiya)',
                'magadan':'Magadanskaya Oblast', 'chukotka':'Chukotskiy Avtonomnyy Okrug'}


def read_table(path):
    return pd.read_csv(StringIO(path.read_text(encoding='utf-8').split('*/', 1)[-1].strip()), sep='\t')


def load_observations(directory=DIRECTORY):
    """Normalise original metres to cm, preserving gaps and author quality notes."""
    d = read_table(directory/'ice_snow.tsv')
    meta = read_table(directory/'stations.tsv').set_index('Event')
    d = d[d.Station.isin(STATIONS)].copy()
    # The original file repeats 21 dates in the 1998/99 Enmyvaam season with
    # conflicting measurements. Quarantine both versions; do not average them
    # or guess a corrected year. The raw source remains available for audit.
    duplicate = d.duplicated(['Station','Date/Time'],keep=False)
    d = d[~duplicate].copy()
    dates = pd.to_datetime(d['Date/Time'])
    ice = pd.to_numeric(d['Ice thick [m] (river)'], errors='raise')*100
    snow = pd.to_numeric(d['Snow thick [m] (in the middle of the river)'], errors='raise')*100
    if (ice.dropna()<0).any() or (snow.dropna()<0).any():
        raise ValueError('Negative ice or snow thickness')
    out = pd.DataFrame(dict(gauge_code=d.Station.astype(int),date=dates.dt.strftime('%Y-%m-%d'),
        hydro_year=dates.dt.year+(dates.dt.month>=9).astype(int),ice_cm=ice,
        snow_on_ice_cm=snow,source_event=d.Event,source_type='наблюдения',source=SOURCE))
    # Explicit interpolation identified by the authors. Keep the original value
    # for audit, but do not turn it into a new independent ice observation.
    out['author_interpolated'] = out.gauge_code.eq(1578) & out.date.eq('2013-01-10')
    out['ice_cm_raw'] = out.ice_cm
    out.loc[out.author_interpolated,'ice_cm'] = np.nan
    out['frozen_to_bottom'] = False
    out['frozen_status_reported'] = False
    stations=[]
    for code,(name,region,basin) in STATIONS.items():
        rows=d[d.Station.eq(code)]
        if rows.empty:
            raise ValueError(f'Missing station {code}')
        event=rows.Event.iloc[0];m=meta.loc[event]
        comments='; '.join(str(v) for col,v in m.items() if col.startswith('Comment') and pd.notna(v))
        stations.append(dict(gauge_code=code,name=name,lat=m.Latitude,lon=m.Longitude,
            drainage_km2=m['Area [km**2]'],region=REGION_NAMES[region],region_key=region,
            basin=basin,yakutia=region=='yakutia',source_type='наблюдения',source=SOURCE,
            source_event=event,quality_notes=comments))
    return pd.DataFrame(stations),out.sort_values(['gauge_code','date']).reset_index(drop=True)
