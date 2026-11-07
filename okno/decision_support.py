"""Проверяемые варианты начала отправок при одном доступном прогнозе."""
from dataclasses import replace
import math
import pandas as pd
from okno.planner import compare_ml_weekly_plans


def start_options(forecast, issue_day, params, planned_start, earliest_start):
    if not all(math.isfinite(x) for x in (issue_day, params.start_day, planned_start, earliest_start)):
        raise ValueError('Даты расчёта должны быть конечными числами')
    if earliest_start < max(issue_day, params.start_day) or planned_start < earliest_start:
        raise ValueError('Начало должно быть не раньше даты решения и начала сезона')
    rows = []
    for start in sorted(set([planned_start, earliest_start, max(earliest_start, planned_start-7)]), reverse=True):
        _, summary = compare_ml_weekly_plans(forecast, issue_day, replace(params, start_day=start))
        row = summary.loc['economic_ml'].to_dict()
        rows.append(dict(start_day=start, **row))
    return pd.DataFrame(rows)


def start_sensitivity(forecast, issue_day, params, planned_start, earliest_start, spread):
    if not 0 <= spread <= 1:
        raise ValueError('Диапазон должен быть от 0 до 1')
    rows = []
    for name in ('c_hold_week', 'c_exp', 'c_extra'):
        for factor in sorted(set([1-spread, 1., 1+spread])):
            table = start_options(forecast, issue_day, replace(params, **{name: getattr(params, name)*factor}), planned_start, earliest_start)
            before, after = table.iloc[0], table.iloc[-1]
            rows.append(dict(parameter=name, factor=factor, saving_rub=before.expected_cost_rub-after.expected_cost_rub,
                             backup_reduction_t=before.expected_backup_t-after.expected_backup_t))
    return pd.DataFrame(rows)
