"""Пересчёт М2 из сохранённых спектральных вырезок, независимый от готовой маски."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno import flood_m2 as M
from okno.flood import POINTS


def read_crop(destination, scene):
    raw_path=next(name for name in scene['files'] if name.endswith('/dn.npz'))
    with np.load(destination/raw_path,allow_pickle=False) as z:
        radi=scene['radiometry']
        bands={b:np.where(z[b]>0,(z[b].astype(float)+radi['offsets'][b])/radi['quantification'],np.nan)
               for b in ['B03','B04','B08','B11']}
        return M.classify(bands,z['SCL'])


def validate(path):
    meta=M.load_run(path)
    destination=path.parents[2]
    refs=[read_crop(destination,scene) for scene in meta['references']]
    reference=M.baseline(refs)
    valid,water=read_crop(destination,meta['observation'])
    mask,stats=M.compare(valid,water,reference)
    with np.load(path.parent/'mask.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(mask,saved['mask'])
        np.testing.assert_array_equal(reference,saved['reference'])
    for key,value in stats.items():
        np.testing.assert_allclose(value,meta['statistics'][key],rtol=1e-7,atol=1e-7)
    return dict(pid=meta['pid'],acquired_at=meta['acquired_at'],**stats,
                scl_water_agreement_iou=meta['statistics'].get('scl_water_agreement_iou'))


def main():
    rows=[]
    for pid in POINTS:
        for path in M.available_runs(pid):
            rows.append(validate(path))
    if not rows:
        raise SystemExit('Нет обработанных спутниковых наблюдений')
    table=pd.DataFrame(rows)
    table.to_csv(M.ROOT/'reports/flood_m2_checks.csv',index=False)
    lines=['# М2 открытой воды: проверка сохранённых наблюдений','',
        'Все исходные файлы и результаты проверены по SHA-256. Маски и площади '
        'пересчитаны из сохранённых DN с коэффициентами XML и совпали с результатами. '
        'Это проверка воспроизводимости алгоритма, не независимая оценка точности.', '',
        '| Пост | Съёмка UTC | Видно | Вода, км² | Новая вода, км² | Сопоставимо, км² | IoU с SCL |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in table.itertuples():
        agreement=f'{r.scl_water_agreement_iou:.2f}' if pd.notna(r.scl_water_agreement_iou) else 'нет воды'
        lines.append(f'| {r.pid} | {r.acquired_at[:10]} | {r.observed_fraction:.0%} | {r.water_km2:.2f} | '
                     f'{r.new_water_km2:.2f} | {r.comparable_km2:.1f} | {agreement} |')
    lines+=['','Квадрат каждой вырезки — 144 км², сетка 20 м. Площади не распространяются '
        'на весь город и не экстраполируются под облака. Новая вода сравнивается с тремя '
        'летними снимками предыдущего года; сезонное изменение тоже может попасть в этот слой. '
        'v1.1: водный SCL, не подтверждённый правилом, исключён из пригодной территории. '
        'IoU считается до этого воздержания, чтобы не скрывать несогласие.', '',
        'IoU с SCL показывает согласие двух способов выделения воды из одного снимка. '
        'Это не независимая разметка затопления и не точность М2. '
        'Спутниковая карта не измеряет глубину, не подтверждает непроходимость дороги '
        'и пока не изменяет прогноз М1. Архивные снимки не выдаются за текущие.', '',
        'Протокол: docs/FLOOD_M2_PLAN.md; реализация: okno/flood_m2.py; '
        'источники/радиометрия: manifest.json каждого наблюдения и scenes/*/{stac.json,product.xml,crop.json,dn.npz}.']
    (M.ROOT/'reports/flood_m2_checks.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(table.to_string(index=False))


if __name__=='__main__':main()
