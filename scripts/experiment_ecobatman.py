"""Offline monthly-discharge comparison; see docs/ECOBATMAN_EXPERIMENT.md."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from okno import flood as F

SITES = {'19063': ('ural_orenburg', 'Урал — Оренбург'),
         '11412': ('ishim_ishim', 'Ишим — Ишим')}


def parse_monthly(raw, code):
    soup = BeautifulSoup(raw, 'html.parser')
    code_rows = [r for r in soup.select('tr')
                 if [c.get_text(' ', strip=True) for c in r.select('td,th')][:1] == ['Код поста']]
    if len(code_rows) != 1 or code_rows[0].select('td,th')[-1].get_text(strip=True) != code:
        raise ValueError('Station code mismatch')
    tables = [t for t in soup.select('table') if t.select('tr') and
              [c.get_text(' ', strip=True) for c in t.select('tr')[0].select('td,th')]
              == ['Период', *map(str, range(1, 13)), 'Год']]
    if len(tables) != 1:
        raise ValueError('Monthly table missing or ambiguous')
    records = []
    for row in tables[0].select('tr')[1:]:
        cells = [c.get_text(' ', strip=True) for c in row.select('td,th')]
        if not cells or not cells[0].isdigit() or len(cells[0]) != 4:
            continue
        if len(cells) != 14:
            raise ValueError('Malformed monthly row')
        for month, text in enumerate(cells[1:13], 1):
            value = np.nan if text in ('', '-', '—', '–') else float(text.replace(',', '.').replace('\xa0', ''))
            if not np.isnan(value) and (not np.isfinite(value) or value < 0):
                raise ValueError('Invalid discharge')
            records.append((pd.Timestamp(int(cells[0]), month, 1), value))
    df = pd.DataFrame(records, columns=['date', 'observed_m3s']).set_index('date')
    if df.empty or df.index.has_duplicates:
        raise ValueError('Empty or duplicate monthly observations')
    return df.sort_index()


def complete_months(daily):
    values = daily['river_discharge']
    mean = values.resample('MS').mean()
    count = values.resample('MS').count()
    return mean.where(count == mean.index.days_in_month).rename('glofas_m3s')


def prepare_pairs(observed, daily):
    pairs = observed.join(complete_months(daily), how='inner').loc['1997':'2020'].dropna()
    train = pairs.loc['1997':'2010']
    counts = train.groupby(train.index.month).size().reindex(range(1, 13), fill_value=0)
    test = pairs.loc['2011':'2020'].copy()
    if test.empty:
        raise ValueError('Empty holdout')
    if (counts < 8).any():
        # Preserve the predeclared sample requirement; do not relax it after seeing gaps.
        test['climatology_m3s'] = np.nan
        test['corrected_m3s'] = np.nan
        return test, pd.DataFrame(), len(train)
    climate = train.groupby(train.index.month).mean()
    if (climate.glofas_m3s <= 0).any():
        raise ValueError('Nonpositive model climatology')
    factor = climate.observed_m3s / climate.glofas_m3s
    test['climatology_m3s'] = test.index.month.map(climate.observed_m3s)
    test['corrected_m3s'] = test.glofas_m3s * test.index.month.map(factor)
    calibration = climate.assign(factor=factor, train_pairs=counts)
    return test, calibration, len(train)


def metrics(frame, column):
    obs = frame.observed_m3s.to_numpy()
    err = frame[column].to_numpy() - obs
    denominator = ((obs - obs.mean()) ** 2).sum()
    return dict(months=len(obs), mae_m3s=float(abs(err).mean()),
                rmse_m3s=float(np.sqrt(np.mean(err**2))), bias_m3s=float(err.mean()),
                nse=float(1 - (err**2).sum()/denominator) if denominator > 0 else None)


def paired_interval(test):
    years = sorted(set(test.index.year))
    groups = [test[test.index.year == y] for y in years]
    delta = [((g.glofas_m3s-g.observed_m3s).abs() -
              (g.corrected_m3s-g.observed_m3s).abs()).sum() for g in groups]
    counts = [len(g) for g in groups]
    idx = np.random.default_rng(42).integers(0, len(years), (10000, len(years)))
    draws = np.asarray(delta)[idx].sum(axis=1) / np.asarray(counts)[idx].sum(axis=1)
    return list(map(float, np.quantile(draws, [.025, .975])))


def distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371 * 2 * math.asin(math.sqrt(h))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ROOT / 'reports'
    raw_dir = ROOT / 'data/flood/ecobatman_research'
    scores, all_pairs, factors, details = [], [], [], {}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained')
    lines = ['# Архив Ecobatman и исторический GloFAS', '',
             'Исследование среднемесячного расхода, м³/с. Это не оценка оперативного прогноза.', '',
             'Калибровка: 1997–2010; отложенный архив: 2011–2020. '
             'Пропуски не заполняются, метрики всех методов считаются на одинаковых месяцах.', '',
             '| Пост | Месяцев обучения / теста | Метод | MAE, м³/с | RMSE, м³/с | Смещение, м³/с | NSE |',
             '|---|---:|---|---:|---:|---:|---:|']
    labels = {'glofas_m3s': 'Исходный GloFAS', 'climatology_m3s': 'Календарное среднее',
              'corrected_m3s': 'Месячная коррекция GloFAS'}
    for ax, (code, (pid, name)) in zip(axes, SITES.items()):
        meta = json.loads((raw_dir / f'{code}.json').read_text())
        raw = gzip.decompress((raw_dir / f'{code}.html.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise ValueError('HTML checksum mismatch')
        observed = parse_monthly(raw, code)
        env = F.read_response(F.DATA / 'history' / f'{pid}.json')
        daily = F.daily_frame(env['payload'], F.POINTS[pid])
        test, calibration, ntrain = prepare_pairs(observed, daily)
        all_pairs.append(test.assign(post_code=code))
        if not calibration.empty:
            factors.append(calibration.rename_axis('month').reset_index().assign(post_code=code))
        for season, selected in [('all', test), ('apr_jul', test[test.index.month.isin([4, 5, 6, 7])])]:
            for column, label in labels.items():
                if selected[column].isna().any():
                    continue
                values = metrics(selected, column)
                scores.append(dict(post_code=code, pid=pid, subset=season, method=column, **values))
                if season == 'all':
                    lines.append(f"| {name} | {ntrain} / {len(test)} | {label} | "
                                 f"{values['mae_m3s']:.2f} | {values['rmse_m3s']:.2f} | "
                                 f"{values['bias_m3s']:+.2f} | {values['nse']:.3f} |")
        coords = {'19063': (51.68, 55.1), '11412': (56.1, 69.47)}[code]
        point = F.POINTS[pid]
        details[code] = dict(name=name, train_months=ntrain, test_months=len(test),
                             test_years=sorted(set(map(int, test.index.year))),
                             calibration_available=not calibration.empty,
                             mae_reduction_ci95_m3s=paired_interval(test) if not calibration.empty else None,
                             approximate_distance_km=distance_km(coords, (point['lat'], point['lon'])),
                             source_url=meta['url'], source_sha256=meta['sha256'],
                             glofas_url=env['url'], glofas_sha256=env['sha256'])
        for column, label in [('observed_m3s', 'Архив поста'), ('glofas_m3s', 'GloFAS'),
                              ('corrected_m3s', 'GloFAS с коррекцией')]:
            if test[column].notna().all():
                ax.plot(test.index, test[column], label=label, linewidth=1)
        ax.set_title(name + ' · отложенные годы 2011–2020')
        ax.set_ylabel('м³/с'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    lines += ['', 'Положительное уменьшение MAE означает улучшение после коррекции. '
              '95%-й интервал получен парным bootstrap целых тестовых лет (10 000, seed 42).', '']
    for code, d in details.items():
        interval = d['mae_reduction_ci95_m3s']
        result = (f'интервал уменьшения MAE [{interval[0]:.2f}; {interval[1]:.2f}] м³/с' if interval else
                  'коррекция и календарный базовый метод не рассчитаны: менее восьми обучающих пар на месяц')
        lines.append(f"- {d['name']}: {result}; "
                     f"расстояние между приблизительной точкой поста и ячейкой приложения {d['approximate_distance_km']:.1f} км. "
                     f"[Исходная карточка]({d['source_url']}).")
    lines += ['', '## Ограничения и решение', '',
              'Для Оренбурга MAE уменьшилась с 28,78 до 27,83 м³/с (около 3,3%), '
              'но 95%-й интервал включает ухудшение. Устойчивый выигрыш не подтверждён; '
              'переносить коррекцию в рабочий прогноз оснований пока нет.', '',
              'Координаты сайта приблизительны. Совпадение створа и модельного водосбора не подтверждено. '
              'Исследование характеризует именно текущие ячейки приложения; ошибка может быть связана '
              'и с выбором ячейки, и с моделью, и с наблюдениями. Простая коррекция не исправляет выбор русла.', '',
              'Календарное среднее — базовый метод без информации о погоде конкретного года. '
              'Месячные средние скрывают суточные пики и заторы. NSE может быть высоким из-за сезонности; '
              'он не является процентом точности. Результаты апреля–июля сохранены отдельно в CSV.', '',
              'Данные используются только в исследовании согласно опубликованным условиям Ecobatman; '
              'это не официальный источник. Для публичного коммерческого использования нужны подходящие условия.', '',
              'Повторный запрос карточек вернул HTTP 403. Использованы HTML-снимки, сохранённые '
              'при предшествующей проверке источника 12.11.2026; точные HTTP-заголовки и время получения '
              'не сохранены. В метаданных указано время сохранения файла и SHA-256 сохранённого HTML. '
              'У Ишима отсутствуют месячные расходы за 2000–2009: пропуски сохранены, '
              'условие восьми обучающих пар не ослаблялось.', '',
              'Рабочие пороги, модели, уровни М1 и интерфейс не изменены. Для переноса коррекции в работу '
              'нужны проверка речной ячейки и суточные парные наблюдения на отдельном периоде. '
              'Эти месячные расходы не обеспечивают динамический вход уровней для М1.', '',
              'Воспроизведение: `python scripts/experiment_ecobatman.py`. '
              'Протокол: `docs/ECOBATMAN_EXPERIMENT.md`. Источники и хеши: JSON рядом с отчётом.']
    pd.DataFrame(scores).to_csv(out/'ecobatman_glofas_metrics.csv', index=False)
    pd.concat(all_pairs).rename_axis('date').to_csv(out/'ecobatman_glofas_pairs.csv')
    pd.concat(factors).to_csv(out/'ecobatman_glofas_calibration.csv', index=False)
    (out/'ecobatman_glofas_experiment.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n')
    (out/'ecobatman_glofas_experiment.md').write_text('\n'.join(lines)+'\n')
    fig.savefig(out/'ecobatman_glofas_holdout.png', dpi=150)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
