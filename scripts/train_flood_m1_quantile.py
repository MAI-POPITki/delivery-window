"""Обучить М1 v2; сохранить прогнозы и сравнение на совпадающих датах с v1."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno import flood_m1_quantile as M

REPORTS = M.ROOT / 'reports'


def metrics(predictions):
    predictions = predictions.assign(year=predictions.issue.dt.year)
    result = []
    for (pid, year, lead), g in predictions.groupby(['pid', 'year', 'lead']):
        for season in ('all', 'spring'):
            part = g if season == 'all' else g[g.issue.dt.month.between(4, 7)]
            if part.empty:
                continue
            width = part.upper - part.lower
            penalty = 10 * ((part.lower - part.target).clip(lower=0) + (part.target - part.upper).clip(lower=0))
            row = dict(pid=pid, year=int(year), lead=int(lead), season=season, n=len(part),
                coverage=float(part.target.between(part.lower, part.upper).mean()),
                width_cm=float(width.mean()), interval_score=float((width + penalty).mean()))
            for method in ('m1', 'catboost', 'persistence', 'calendar', 'last_year'):
                row[method] = float((part.target - part[method]).abs().mean())
            result.append(row)
    return pd.DataFrame(result)


def report(predictions):
    scores = metrics(predictions)
    scores.to_csv(REPORTS / 'flood_m1_quantile_scores.csv', index=False)
    old = pd.read_csv(REPORTS / 'flood_m1_predictions.csv.gz', parse_dates=['issue', 'target_date'])
    keys = ['pid', 'issue', 'target_date', 'lead']
    joined = predictions.merge(old[keys + ['target', 'm1', 'lower', 'upper']], on=keys,
                               suffixes=('', '_v1'), validate='one_to_one')
    if not np.allclose(joined.target, joined.target_v1):
        raise ValueError('Версии сравниваются с разными измерениями')
    paired_v2 = metrics(joined)
    paired_v1 = joined.copy()
    for col in ('m1', 'lower', 'upper'):
        paired_v1[col] = joined[col + '_v1']
    paired_v1 = metrics(paired_v1)
    statuses = {}
    rng = np.random.default_rng(42)
    for (pid, lead), group in paired_v2[paired_v2.season.eq('all')].groupby(['pid', 'lead']):
        wet = paired_v2[(paired_v2.pid == pid) & (paired_v2.lead == lead) & paired_v2.season.eq('spring')]
        bounds = {}
        for base in ('persistence', 'calendar', 'last_year'):
            gains = (group[base] - group.m1).to_numpy()
            boots = rng.choice(gains, (2000, len(gains)), replace=True).mean(axis=1)
            bounds[base] = np.quantile(boots, [.05, .95]).tolist()
        point = all(v[0] > 0 for v in bounds.values()) and all(wet.m1.mean() < wet[b].mean() for b in bounds)
        interval = group.coverage.mean() >= .75 and wet.coverage.mean() >= .75
        worst = wet.loc[wet.coverage.idxmin()]
        statuses[f'{pid}:{lead}'] = dict(success=bool(point and interval), point_skill_success=bool(point),
            interval_valid=bool(interval), spring_coverage=float(wet.coverage.mean()),
            worst_spring_year=int(worst.year), worst_spring_coverage=float(worst.coverage),
            gain_ci90_cm=bounds, years=group.year.tolist(), evaluation='reused 2019–2024 archive')
    (M.MODELS / 'validation.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')
    lines = ['# М1 паводков v2: квантильная модель', '',
        'Протокол: docs/FLOOD_M1_V2_PLAN.md. Реальные уровни Level-Sputnik, те же исходные файлы, что у v1. '
        '2019–2024 уже изучались при создании v1: это повторная оценка. '
        '2025 Оренбурга/Орска показан отдельно, до 09.09; это неполный год. '
        'Ни один проверяемый год не входит в обучение или калибровку его модели.', '',
        '## Сравнение с v1 на одинаковых датах: апрель–июль, 7 суток', '',
        '| Пост | MAE v1/v2, см | Покрытие v1/v2 | Ширина v1/v2, см | Interval score v1/v2 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for pid in M.STATIONS:
        def mean(frame):
            return frame[(frame.pid == pid) & frame.lead.eq(7) & frame.season.eq('spring')].select_dtypes('number').mean()
        a, b = mean(paired_v1), mean(paired_v2)
        lines.append(f'| {pid} | {a.m1:.1f} / {b.m1:.1f} | {a.coverage:.0%} / {b.coverage:.0%} | '
                     f'{a.width_cm:.1f} / {b.width_cm:.1f} | {a.interval_score:.1f} / {b.interval_score:.1f} |')
    lines += ['', 'Interval score: ширина диапазона + 10× ошибка за его пределами; меньше лучше. '
        'Большая ширина может улучшить покрытие без улучшения полезности.', '',
        '## Все горизонты, повторная оценка 2019–2024', '',
        '| Пост | Сутки | MAE М1 | Сохранить | Календарь | Прошлый год | Покрытие год/весна | Критерии |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for (pid, lead), group in paired_v2[paired_v2.season.eq('all')].groupby(['pid', 'lead']):
        g = group.select_dtypes('number').mean()
        status = statuses[f'{pid}:{lead}']
        lines.append(f"| {pid} | {lead} | {g.m1:.1f} | {g.persistence:.1f} | {g.calendar:.1f} | {g.last_year:.1f} | "
                     f"{g.coverage:.0%} / {status['spring_coverage']:.0%} | {'пройдены на повторной оценке' if status['success'] else 'эксперимент'} |")
    lines += ['', '## Стресс-проверка: паводок 2024 года, апрель–июль, 7 суток', '',
        '| Пост | MAE М1 / сохранить, см | Покрытие | Ширина, см |',
        '| --- | ---: | ---: | ---: |']
    for r in scores[scores.year.eq(2024) & scores.season.eq('spring') & scores.lead.eq(7)].itertuples():
        lines.append(f'| {r.pid} | {r.m1:.1f} / {r.persistence:.1f} | {r.coverage:.0%} | {r.width_cm:.1f} |')
    lines += ['', '**Среднее покрытие скрывает провалы отдельных паводков.** '
        'В 2024 году Оренбург и Орск остаются ниже критерия 75%. '
        'Улучшение относительно v1 не означает надёжности на экстремальном событии.', '',
        '## Дополнительный 2025 год: апрель–июль', '',
        '| Пост | Сутки | MAE М1 / сохранить, см | Покрытие | Ширина, см | Interval score |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in scores[scores.year.eq(2025) & scores.season.eq('spring') & scores.lead.isin([1, 3, 7])].itertuples():
        lines.append(f'| {r.pid} | {r.lead} | {r.m1:.1f} / {r.persistence:.1f} | {r.coverage:.0%} | {r.width_cm:.1f} | {r.interval_score:.1f} |')
    lines += ['', 'Все строки и сезоны: flood_m1_quantile_scores.csv; исходные прогнозы: '
        'flood_m1_quantile_predictions.csv.gz. Статусы рассчитаны по повторному периоду, '
        'не являются свидетельством независимого подтверждения новой версии. '
        'Суточные ошибки зависимы; статистическая гарантия покрытия не заявляется. '
        'Свежих наблюдений за 2026 год нет. Опасные отметки, площадь затопления и проезд дорог не рассчитываются.']
    (REPORTS / 'flood_m1_quantile.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    path = REPORTS / 'flood_m1_quantile_predictions.csv.gz'
    if args.report_only:
        report(pd.read_csv(path, parse_dates=['issue', 'target_date']))
        return
    M.MODELS.mkdir(parents=True, exist_ok=True)
    predictions = []
    for pid, (code, source) in M.STATIONS.items():
        series = M.load_levels(pid)
        table = M.dataset(series)
        manifest = json.loads((M.DATA / f'{pid}.manifest.json').read_text(encoding='utf-8'))
        for year in range(2019, series.last_valid_index().year + 2):
            try:
                model, meta = M.fit_year(series, year)
            except ValueError as exc:
                print(pid, year, ascii(str(exc)), flush=True)
                continue
            meta.update(point_id=pid, gauge_code=code, observation_source=source, data_sha256=manifest['sha256'])
            model.save_model(str(M.MODELS / f'{pid}_{year}.cbm'))
            (M.MODELS / f'{pid}_{year}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
            test = table[(table.issue.dt.year == year) & (table.target_date.dt.year == year)].copy()
            if not test.empty:
                p = M.predict_rows(series, test, year, model, meta)
                valid = p.notna().all(axis=1)
                out = test.loc[valid, ['issue', 'target_date', 'lead', 'target']].copy()
                out['pid'] = pid
                for col in p:
                    out[col] = p.loc[valid, col]
                predictions.append(out)
            print(pid, year, 'trained', flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
    report(predictions)


if __name__ == '__main__':
    main()
