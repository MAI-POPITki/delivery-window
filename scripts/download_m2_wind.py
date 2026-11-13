"""Download immutable ERA5 wind responses and match them to saved M2 features."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno import m2, m2_wind as W

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/s1/wind'


def inputs():
    features = pd.read_csv(m2.S1/'predictions.csv')
    features = features[features.label.isin(['water','ice']) & (features.hydro_year<=2022)].copy()
    if features.duplicated(['point','item_id']).any():
        raise ValueError('Дубли радарных признаков')
    index = pd.read_csv(m2.S1/'index.csv')
    if index.duplicated(['point','item_id']).any():
        raise ValueError('Дубли в индексе съёмок')
    coordinates = index[['point','item_id','lat','lon']]
    result = features.merge(coordinates, on=['point','item_id'], how='left', validate='one_to_one')
    if result[['lat','lon']].isna().any().any():
        raise ValueError('Нет координат съёмки')
    result['acquired_at'] = result.item_id.map(W.acquisition_time)
    if not result.acquired_at.dt.strftime('%Y-%m-%d').eq(result.date).all():
        raise ValueError('Дата индекса расходится со временем съёмки')
    years=result.acquired_at.dt.year+(result.acquired_at.dt.month>=9).astype(int)
    if not years.eq(result.hydro_year).all():
        raise ValueError('Неверный гидрологический год съёмки')
    return result


def collect(group, fetch=False):
    pid, rows = group
    if len(rows[['lat','lon']].drop_duplicates())!=1:
        raise ValueError(f'Меняющиеся координаты {pid}')
    point = dict(lat=float(rows.lat.iloc[0]), lon=float(rows.lon.iloc[0]))
    start = (rows.acquired_at.min()-pd.Timedelta(hours=6)).date().isoformat()
    end = rows.acquired_at.max().date().isoformat()
    path = OUT/f'{pid}-{start}-{end}.json.gz'
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f'Нет архива {pid}; загрузите с --fetch')
        params = dict(latitude=point['lat'],longitude=point['lon'],start_date=start,end_date=end,
                      hourly='wind_speed_10m,wind_direction_10m', models='era5', timezone='UTC',
                      wind_speed_unit='ms')
        response = requests.get(W.API, params=params, timeout=(15,150))
        response.raise_for_status()
        W.parse_hourly(response.json())
        W.save_response(path,response.content,response.url,point)
    hourly, meta = W.load_response(path)
    matched, missing = [], []
    for row in rows.itertuples():
        try:
            wind = W.wind_features(hourly,row.acquired_at)
            matched.append(dict(point=pid,item_id=row.item_id,acquired_at=row.acquired_at.isoformat(),**wind))
        except ValueError as exc:
            missing.append(dict(point=pid,item_id=row.item_id,reason=str(exc)))
    print(f'{pid}: {len(matched)}/{len(rows)} matched',flush=True)
    return matched,missing,dict(path=path.relative_to(ROOT).as_posix(),**meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true',help='Разрешить загрузку отсутствующих ответов ERA5')
    args = parser.parse_args()
    rows = inputs()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda group:collect(group,args.fetch),rows.groupby('point')))
    OUT.mkdir(parents=True,exist_ok=True)
    matched = [row for matches,_,_ in results for row in matches]
    missing = [row for _,misses,_ in results for row in misses]
    pd.DataFrame(matched).sort_values(['point','item_id']).to_csv(OUT/'features.csv',index=False)
    frozen_columns=['point','item_id','label','hydro_year',*m2.FEATURES]
    rows[frozen_columns].to_csv(OUT/'radar_features.csv',index=False)
    source_files = ['data/s1/predictions.csv','data/s1/index.csv']
    manifest = dict(protocol='docs/M2_WIND_EXPERIMENT.md',records=len(rows),matched=len(matched),
                    missing=missing,sources=[source for _,_,source in results],
                    inputs={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
                    features_sha256=hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest(),
                    radar_features_sha256=hashlib.sha256((OUT/'radar_features.csv').read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Total: {len(matched)}/{len(rows)} matched; {len(missing)} missing',flush=True)


if __name__=='__main__':
    main()
