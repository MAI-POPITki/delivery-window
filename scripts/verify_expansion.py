"""Reproduce synthetic transfer checks. Does not fit or calibrate any model.

Run: python scripts/verify_expansion.py --output reports/expansion
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from okno.corridor import CorridorConfig
from okno.regions import REGIONS
from okno.supply_scenarios import Closure, SeasonalWindow, ServiceDays, compare_supply


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    cfg = CorridorConfig(horizon_days=60, daily_demand_tons=10,
        initial_yak_tons=100, yak_warehouse_cap_tons=600,
        rail_lead_time_days=3, rail_cap_tons=100,
        crossing_cap_normal_tons=100, crossing_cap_alt_tons=2,
        rail_cost_per_ton=5000, crossing_cost_normal=2000,
        crossing_cost_alt=30000, nb_holding_cost_per_ton_day=50,
        yak_holding_cost_per_ton_day=50, shortage_penalty_per_ton_day=1000)
    cases = [(key, Closure(20, 40)) for key in REGIONS]
    cases += [('navigation', SeasonalWindow(20, 40)),
              ('remote', ServiceDays(tuple(range(10, 50, 7))))]
    rows, manifests = [], []
    for case, schedule in cases:
        mask = schedule.calendar(cfg.horizon_days)
        manifests.append(dict(case=case, parameters=asdict(cfg), calendar=mask.astype(int).tolist(),
            assumption='Synthetic parameters; full calendar known at start; no calibration'))
        for policy, (summary, daily) in compare_supply(cfg, schedule).items():
            assert abs(summary['mass_balance_error_tons']) < 1e-5
            assert (daily.loc[~mask, 'shipped_norm_t'].abs() < 1e-6).all()
            assert (daily.inv_yak_t <= cfg.yak_warehouse_cap_tons+1e-5).all()
            np.testing.assert_allclose(daily.rail_arrived_nb_t.iloc[3:],
                                       daily.hub_dispatched_t.iloc[:-3], atol=1e-5)
            policy_id = 'economic' if policy == 'Экономический план' else 'baseline'
            filename = f'{case}_{policy_id}.csv'
            content = daily.to_csv(index=False).encode('utf-8')
            (output/filename).write_bytes(content)
            rows.append(dict(case=case, policy=policy_id,
                total_cost_rub=summary['total_cost_rub'],
                served_tons=summary['served_tons'], backlog_tons=summary['terminal_backlog_tons'],
                waiting_ton_days=summary['shortage_ton_days'],
                mass_error_tons=summary['mass_balance_error_tons'],
                csv=filename, sha256=hashlib.sha256(content).hexdigest()))
    pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
    (output/'manifest.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Проверка реализации расширения', '',
        'Все случаи синтетические. Калибровка не выполнялась. Региональные случаи используют',
        'одинаковые параметры: они проверяют повторное использование ядра, а не местную экономику.',
        'Навигация — окно доступности, удалённый объект — отдельные дни обслуживания.', '',
        '| Случай | Правило | Расходы, ₽ | Выдано, т | Очередь, т | Ожидание, т·сут |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['policy']} | {r['total_cost_rub']:.2f} | {r['served_tons']:.2f} | {r['backlog_tons']:.2f} | {r['waiting_ton_days']:.2f} |")
    lines += ['', 'Проверены баланс массы, календарь, лаг первого участка и ёмкость склада.',
        'CSV содержат дневные решения; summary.csv — контрольные суммы SHA-256;',
        'manifest.json — все параметры и календари для воспроизведения.',
        'Это доказательство работы кода в заданной модели, не доказательство реальной экономии.']
    (output/'verification.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'PASS: {len(cases)} scenarios, {len(rows)} policy runs; output: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/expansion'))
    run(parser.parse_args().output)
