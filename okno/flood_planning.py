"""Решения о доставке: гидрология не подменяет разрешение движения."""
import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def passage_risks(result, travel_days=1):
    if isinstance(travel_days, bool) or int(travel_days) != travel_days or travel_days < 1:
        raise ValueError("Время проезда должно быть целым положительным числом суток")
    frame = result["scenarios"]
    rows = []
    for start in range(len(frame)):
        stop = start + travel_days
        complete = stop <= len(frame)
        fraction = float((frame.iloc[start:stop].max() > result["threshold"]).mean()) if complete else None
        rows.append(dict(date=frame.index[start].date(), arrival=(frame.index[start] + pd.Timedelta(days=travel_days-1)).date(),
                         fraction=fraction, complete=complete))
    return pd.DataFrame(rows)


def confirmed_calendar(records, risks, decision, travel_days=1, risk_limit=1., allow_without_forecast=False):
    """Весь период рейса должен иметь подтверждение, известное на дату решения."""
    if not 0 <= risk_limit <= 1 or not math.isfinite(risk_limit):
        raise ValueError('Предел гидрологического индикатора должен быть от 0 до 100%')
    if travel_days < 1 or int(travel_days) != travel_days:
        raise ValueError('Время проезда должно быть целым положительным')
    data = records.copy().sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(data.date)
    if dates.duplicated().any() or len(data) == 0 or (dates.dt.date <= decision).any():
        raise ValueError('Календарь отправок должен содержать будущие уникальные даты')
    published = pd.to_datetime(data.published, errors='coerce')
    source = data.source.fillna('').astype(str).str.strip().ne('')
    known = data.confirmed.eq(True) & source & published.notna() & (published.dt.date <= decision)
    if (data.confirmed.eq(True) & ~known).any():
        raise ValueError('Для подтверждённого проезда нужны основание и дата публикации не позже решения')
    daily_risk = {} if risks is None else risks.set_index('date').fraction.to_dict()
    result = []
    for i, day in enumerate(dates):
        end = i + travel_days
        window = dates.iloc[i:end]
        confirmed = (len(window) == travel_days and known.iloc[i:end].all()
                     and window.iloc[-1] == day + pd.Timedelta(days=travel_days-1))
        risk = daily_risk.get(day.date())
        policy = (risk is not None and not pd.isna(risk) and risk <= risk_limit) or (
                  allow_without_forecast and (risk is None or pd.isna(risk)))
        result.append(dict(date=day.date(), arrival=(day+pd.Timedelta(days=travel_days-1)).date(),
                           capacity_t=data.capacity_t.iloc[i], confirmed=bool(confirmed),
                           policy_allowed=bool(policy), fraction=risk))
    return pd.DataFrame(result)


def optimize_dispatch(calendar, quantity, road_cost, backup_cost, holding_cost):
    """Точный линейный план по введённым мощностям; неизвестные дни запрещены.

    calendar: date, capacity_t, confirmed, policy_allowed. Тарифы и мощности —
    входы оператора. Гидрологический порог используется только как выбранная
    оператором политика предосторожности, не как вероятность срыва рейса.
    """
    values = (quantity, road_cost, backup_cost, holding_cost)
    if any(v is None or not math.isfinite(v) or v < 0 for v in values) or quantity <= 0:
        raise ValueError("Введите положительный объём и неотрицательные конечные тарифы")
    d = calendar.copy().sort_values("date").reset_index(drop=True)
    if d.empty or d.date.duplicated().any():
        raise ValueError("Календарь должен содержать уникальные даты")
    if not pd.api.types.is_bool_dtype(d.confirmed) or not pd.api.types.is_bool_dtype(d.policy_allowed):
        raise ValueError("Допуск и политика должны быть логическими значениями")
    caps = pd.to_numeric(d.capacity_t, errors="coerce")
    if ((caps < 0) | np.isinf(caps)).any():
        raise ValueError("Некорректные мощности")
    caps = caps.where(d.confirmed & d.policy_allowed, 0).fillna(0).to_numpy(float)
    dates = pd.to_datetime(d.date)
    arrivals = pd.to_datetime(d.arrival) if 'arrival' in d else dates
    early = (dates.max() - arrivals).dt.days.clip(lower=0).to_numpy()
    costs = road_cost + holding_cost * early
    solution = linprog(np.r_[costs, backup_cost], A_eq=np.ones((1, len(d)+1)),
                       b_eq=[quantity], bounds=[*( (0, v) for v in caps), (0, quantity)], method="highs")
    if not solution.success:
        raise ValueError("Оптимизатор не нашёл допустимый план")
    d["optimized_t"] = solution.x[:-1]
    # Соперник использует те же допуски, мощности и политику.
    base_total = min(quantity, caps.sum())
    d["uniform_t"] = caps / caps.sum() * base_total if caps.sum() else np.zeros(len(d))
    baseline_backup = quantity - base_total
    baseline_cost = float(np.dot(d.uniform_t, costs) + baseline_backup * backup_cost)
    return d, dict(cost=float(solution.fun), backup_t=float(solution.x[-1]),
                   road_t=float(solution.x[:-1].sum()), baseline_cost=baseline_cost,
                   baseline_backup_t=float(baseline_backup))
