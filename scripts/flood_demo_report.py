"""Воспроизводимая сводка демонстрационного охвата и проверенной пользы."""
from pathlib import Path
import json
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okno.flood_sites import SITES,EXTRA_SITES
from okno import flood_m2 as S
from okno.flood_routes import load_routes


def main():
    root=Path(__file__).resolve().parents[1]
    validation=json.loads((root/'models/flood_probability/validation.json').read_text())
    benefit=pd.read_csv(root/'reports/flood_decision_value.csv')
    wins=(benefit.model<benefit.persistence-1e-12)&(benefit.model<benefit.climatology-1e-12)
    bench=pd.read_csv(root/'reports/flood_m2_benchmark_chips.csv')
    lines=['# Сводка для демонстрации паводков','',
        f'Охват: {len(SITES)} участков, {len({s["region"] for s in SITES.values()})} регионов. '
        'Собственный М1 — 4 поста; новые Омск/Томск — только спутник. '
        'Количество регионов показывает расширяемость, а не точность.','',
        '| Новый участок | Дата снимка | Пригодная территория | Дорог OSM |',
        '| --- | --- | ---: | ---: |']
    for pid in EXTRA_SITES:
        runs=S.available_runs(pid)
        meta=S.load_run(runs[-1]);roads,_=load_routes(pid)
        lines.append(f"| {SITES[pid]['name']} | {meta['acquired_at'][:10]} | {meta['statistics']['observed_fraction']:.0%} | {len(roads)} |")
    lines+=['','## Тесты смысла и эффективности','',
        f'- Вероятности М1: {sum(v["skill"] for v in validation.values())}/{len(validation)} сочетаний '
        'прошли критерий преимущества над двумя простыми правилами. Это повторная '
        'архивная оценка, не точность классификации и не доля успешных рейсов.',
        f'- Условная ценность предупреждений: расходы ниже обоих правил в {int(wins.sum())}/{len(wins)} '
        'комбинаций события/горизонта/отношения стоимости действия к потере. '
        'Комбинации зависимы; стоимость условная, действие предполагается полностью '
        'предотвращающим потерю. Это не реальные рубли или экономия перевозчика.']
    for split in ['test','bolivia']:
        for method in ['candidate','current_spectral']:
            g=bench[(bench.split==split)&(bench.method==method)]
            totals=g[['tp','fp','fn','tn','n']].sum();f1=2*totals.tp/(2*totals.tp+totals.fp+totals.fn)
            iou=totals.tp/(totals.tp+totals.fp+totals.fn)
            lines.append(f'- М2, {split}, {method}: F1 воды {f1:.3f}, IoU {iou:.3f}.')
    lines+=['','М2 проверен на ручной разметке Sen1Floods11. Кандидат использует L1C TOA '
        'и не заменяет автоматически рабочую карту L2A BOA. Это проверка компонента '
        'выделения воды, а не глубины или непроходимости дороги.', '',
        'М3 проверен на сохранение объёма, мощности, полный период проезда, '
        'отсутствие будущих подтверждений и минимизацию заданных расходов. '
        'Это подтверждает алгоритм; транспортная польза по фактическим рейсам пока не измерена.', '',
        '## Практический смысл', '',
        'Полезно для демонстрации открытых данных, проверки высокой воды, '
        'сопоставления снимка и прогноза и условного планирования. '
        'Слабое место — переход от уровня/водной маски к доступности конкретной дороги. '
        'Дополнительные области расширяют наблюдение, но не устраняют этот пробел.']
    (root/'reports/flood_demo_value.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
