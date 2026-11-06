"""
Симулятор сквозного 4-звенного логистического коридора (Кейс А3, `okno/corridor.py`).

Моделирует посуточное движение грузопотока (t = 0 ... horizon_days - 1) по всему пути:
1. Магистральное ж/д плечо (Оптовый хаб Сибири -> ст. Нижний Бестях, tau_rail = 6 сут).
2. Буферный узел «Правый берег» (терминал Нижний Бестях: ожидание переправы / затор).
3. Сезонная переправа через р. Лену (штатный паром / лёд 60 т/сут по 1 800 ₽/т vs
   аварийные СВП и вертолёты Ми-8 в распутицу до 8 т/сут по 35 000 ₽/т).
4. Целевой склад ТЛЦ в г. Якутске («Левый берег»: потребление 25 т/сут, ёмкость 650 т,
   плановое хранение 220 ₽/т·сут, переполнение 600 ₽/т·сут, штраф за дефицит 12 000 ₽/т·сут).

Явно учитывает эффект «бутылочного горлышка» подходов к переправе:
    N_pre = ceil( V_pre / (Q_max - d) ),   t_start_hub = t_close_plan - N_pre - tau_rail
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class CorridorConfig:
    daily_demand_tons: float = 25.0               # Суточная потребность ТЛЦ / потока заказов (т/сут)
    horizon_days: int = 90                        # Горизонт симуляции вокруг межсезонья (дней)
    rail_lead_time_days: int = 6                  # Время в пути по ж/д (сутки)
    rail_cost_per_ton: float = 8500.0             # Тариф ж/д (руб/т)
    nb_holding_cost_per_ton_day: float = 450.0    # Простой/хранение в Н.Бестяхе (руб/т/сут)
    crossing_cap_normal_tons: float = 60.0        # Пропускная способность парома/льда (т/сут)
    crossing_cap_alt_tons: float = 8.0            # Лимит СВП / вертолётов в распутицу (т/сут)
    crossing_cost_normal: float = 1800.0          # Штатный тариф переправы (руб/т)
    crossing_cost_alt: float = 35000.0            # Аварийный тариф (СВП и авиа 55 руб/кг) (руб/т)
    yak_warehouse_cap_tons: float = 650.0         # Свободная ёмкость склада в Якутске (т)
    yak_holding_cost_per_ton_day: float = 220.0   # Плановое хранение на ТЛЦ Якутска (руб/т/сут)
    yak_overflow_cost_per_ton_day: float = 600.0  # Аренда доп. площадей при переполнении (руб/т/сут)
    shortage_penalty_per_ton_day: float = 12000.0 # Штраф за дефицит / срыв сроков (руб/т/сут)

    def scaled(self, param_name: str, factor: float) -> "CorridorConfig":
        d = asdict(self)
        val = d[param_name] * factor
        if isinstance(getattr(self, param_name), int):
            val = max(1, int(round(val)))
        d[param_name] = val
        return CorridorConfig(**d)


SENSITIVITY_PARAMS: tuple[tuple[str, str], ...] = (
    ("yak_holding_cost_per_ton_day", "Хранение на ТЛЦ Якутска (h_yak)"),
    ("yak_overflow_cost_per_ton_day", "Аренда при переполнении ТЛЦ (h_over)"),
    ("nb_holding_cost_per_ton_day", "Простой в Нижнем Бестяхе (h_nb)"),
    ("crossing_cost_alt", "Аварийный тариф СВП / авиа (c_alt)"),
    ("shortage_penalty_per_ton_day", "Штраф за дефицит / срыв SLA (p_def)"),
    ("crossing_cap_normal_tons", "Пропускная способность переправы (Q_max)"),
    ("crossing_cap_alt_tons", "Лимит СВП / вертолётов в распутицу (Q_alt)"),
    ("yak_warehouse_cap_tons", "Свободная ёмкость склада Якутска (V_max)"),
)


def compute_prestock_schedule(cfg: CorridorConfig, planned_close_day: int, planned_open_day: int) -> dict:
    """Рассчитывает окно предзавоза N_pre и день старта отгрузки из ж/д хаба Сибири."""
    d = cfg.daily_demand_tons
    expected_closed_days = max(0, int(planned_open_day) - int(planned_close_day))
    target_prestock = expected_closed_days * d
    spare_crossing_cap = max(1.0, cfg.crossing_cap_normal_tons - d)
    prestock_duration = int(np.ceil(target_prestock / spare_crossing_cap)) if expected_closed_days > 0 else 0
    prestock_start_nb = max(cfg.rail_lead_time_days, int(planned_close_day) - prestock_duration)
    hub_dispatch_start_day = prestock_start_nb - cfg.rail_lead_time_days
    effective_days = max(1, int(planned_close_day) - prestock_start_nb)
    daily_extra_ship = target_prestock / effective_days if expected_closed_days > 0 else 0.0
    return {
        "expected_closed_days": expected_closed_days,
        "target_prestock_tons": round(target_prestock, 1),
        "spare_crossing_cap_tons": round(spare_crossing_cap, 1),
        "prestock_duration_days": prestock_duration,
        "prestock_start_nb_day": prestock_start_nb,
        "hub_dispatch_start_day": hub_dispatch_start_day,
        "daily_extra_ship_tons": round(daily_extra_ship, 2),
    }


def simulate_corridor(
    cfg: CorridorConfig,
    actual_close_day: int,
    actual_open_day: int,
    planned_close_day: int,
    planned_open_day: int,
    use_prestock: bool = True,
    return_daily: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Посуточная симуляция 4 звеньев сквозного маршрута за квартальный горизонт."""
    d = cfg.daily_demand_tons
    p_open = min(max(0, int(planned_open_day)), cfg.horizon_days - 1)
    p_close = max(1, min(int(planned_close_day), p_open))
    expected_closed_days = max(0, p_open - p_close)

    if use_prestock and expected_closed_days > 0:
        target_prestock = expected_closed_days * d
        spare_crossing_cap = max(1.0, cfg.crossing_cap_normal_tons - d)
        prestock_duration = int(np.ceil(target_prestock / spare_crossing_cap))
        prestock_start_nb = min(p_close - 1, max(cfg.rail_lead_time_days, p_close - prestock_duration))
        prestock_start_nb = max(0, prestock_start_nb)
        daily_extra_ship = target_prestock / max(1, (p_close - prestock_start_nb))
    else:
        prestock_duration = 0
        prestock_start_nb = 999
        daily_extra_ship = 0.0

    rail_arrivals = np.zeros(cfg.horizon_days + cfg.rail_lead_time_days + 10, dtype=float)

    for t_nb in range(cfg.horizon_days):
        if not use_prestock:
            if t_nb < p_close or t_nb >= p_open:
                rail_arrivals[t_nb] = d
            if t_nb == p_open:
                rail_arrivals[t_nb] += expected_closed_days * d
        else:
            if prestock_start_nb <= t_nb < p_close:
                rail_arrivals[t_nb] = d + daily_extra_ship
            elif t_nb < prestock_start_nb or t_nb >= p_open:
                rail_arrivals[t_nb] = d
            else:
                rail_arrivals[t_nb] = 0.0

    inv_nb = 0.0
    inv_yak = d * 3.0

    total_rail_tons = 0.0
    tons_crossed_norm = 0.0
    tons_crossed_alt = 0.0
    nb_holding_ton_days = 0.0
    yak_holding_ton_days = 0.0
    yak_overflow_ton_days = 0.0
    shortage_ton_days = 0.0
    delayed_order_days = 0
    peak_inv_yak = inv_yak
    peak_inv_nb = inv_nb

    daily_records = []

    for t in range(cfg.horizon_days):
        arrived_nb = float(rail_arrivals[t])
        total_rail_tons += arrived_nb
        inv_nb += arrived_nb

        is_river_open = (t < actual_close_day) or (t >= actual_open_day)

        if is_river_open:
            shipped_norm = min(inv_nb, cfg.crossing_cap_normal_tons)
            shipped_alt = 0.0
        else:
            shipped_norm = 0.0
            need_for_yak = max(0.0, d * 2.0 - inv_yak)
            shipped_alt = min(inv_nb, cfg.crossing_cap_alt_tons, need_for_yak)

        inv_nb -= (shipped_norm + shipped_alt)
        inv_yak += (shipped_norm + shipped_alt)

        tons_crossed_norm += shipped_norm
        tons_crossed_alt += shipped_alt

        satisfied = min(inv_yak, d)
        inv_yak -= satisfied
        unmet = d - satisfied

        if unmet > 1e-6:
            shortage_ton_days += unmet
            delayed_order_days += 1

        nb_holding_ton_days += inv_nb
        yak_holding_ton_days += min(inv_yak, cfg.yak_warehouse_cap_tons)
        yak_overflow_ton_days += max(0.0, inv_yak - cfg.yak_warehouse_cap_tons)
        peak_inv_yak = max(peak_inv_yak, inv_yak)
        peak_inv_nb = max(peak_inv_nb, inv_nb)

        if return_daily:
            daily_records.append({
                "day": t,
                "river_open": int(is_river_open),
                "rail_arrived_nb_t": round(arrived_nb, 2),
                "shipped_norm_t": round(shipped_norm, 2),
                "shipped_alt_t": round(shipped_alt, 2),
                "inv_nb_t": round(inv_nb, 2),
                "inv_yak_t": round(inv_yak, 2),
                "unmet_demand_t": round(unmet, 2),
            })

    cost_rail = total_rail_tons * cfg.rail_cost_per_ton
    cost_cross_norm = tons_crossed_norm * cfg.crossing_cost_normal
    cost_cross_alt = tons_crossed_alt * cfg.crossing_cost_alt
    cost_hold_nb = nb_holding_ton_days * cfg.nb_holding_cost_per_ton_day
    cost_hold_yak_base = yak_holding_ton_days * cfg.yak_holding_cost_per_ton_day
    cost_hold_yak_over = yak_overflow_ton_days * cfg.yak_overflow_cost_per_ton_day
    cost_hold_yak = cost_hold_yak_base + cost_hold_yak_over
    cost_shortage = shortage_ton_days * cfg.shortage_penalty_per_ton_day
    total_cost = cost_rail + cost_cross_norm + cost_cross_alt + cost_hold_nb + cost_hold_yak + cost_shortage

    summary = {
        "total_cost_rub": round(total_cost, 0),
        "cost_per_ton_rub": round(total_cost / max(1.0, total_rail_tons), 0),
        "cost_rail_rub": round(cost_rail, 0),
        "cost_crossing_normal_rub": round(cost_cross_norm, 0),
        "cost_crossing_alt_rub": round(cost_cross_alt, 0),
        "cost_holding_nb_rub": round(cost_hold_nb, 0),
        "cost_holding_yak_rub": round(cost_hold_yak, 0),
        "cost_holding_yak_base_rub": round(cost_hold_yak_base, 0),
        "cost_holding_yak_overflow_rub": round(cost_hold_yak_over, 0),
        "cost_shortage_rub": round(cost_shortage, 0),
        "total_rail_tons": round(total_rail_tons, 1),
        "tons_normal_transport": round(tons_crossed_norm, 1),
        "tons_alt_transport": round(tons_crossed_alt, 1),
        "tons_stuck_nb_ton_days": round(nb_holding_ton_days, 1),
        "yak_holding_ton_days": round(yak_holding_ton_days + yak_overflow_ton_days, 1),
        "shortage_ton_days": round(shortage_ton_days, 1),
        "days_with_deficit": int(delayed_order_days),
        "peak_inv_nb_tons": round(peak_inv_nb, 1),
        "peak_inv_yak_tons": round(peak_inv_yak, 1),
        "prestock_duration_days": int(prestock_duration),
        "hub_dispatch_start_day": int(prestock_start_nb - cfg.rail_lead_time_days) if use_prestock and prestock_duration > 0 else -1,
    }
    if return_daily:
        return summary, pd.DataFrame(daily_records)
    return summary


def run_sensitivity_check(
    base_cfg: CorridorConfig | None = None,
    actual_close_day: int = 30,
    actual_open_day: int = 57,
    cal_close_day: int = 36,
    cal_open_day: int = 63,
    model_close_day: int = 29,
    model_open_day: int = 58,
    delta: float = 0.30,
) -> pd.DataFrame:
    """Стресс-тест на чувствительность ±30% по всем синтетическим и операционным параметрам."""
    cfg0 = base_cfg or CorridorConfig()
    cols = [
        "parameter", "label", "variation", "param_value",
        "before_mln_rub", "after_mln_rub", "saving_mln_rub",
        "saving_pct", "yak_storage_delta_mln_rub", "effect_positive",
    ]
    rows = []
    for param, label in SENSITIVITY_PARAMS:
        for tag, factor in ((f"-{int(delta * 100)}%", 1.0 - delta), ("база", 1.0), (f"+{int(delta * 100)}%", 1.0 + delta)):
            cfg = cfg0.scaled(param, factor) if factor != 1.0 else cfg0
            # Пропускаем физически невозможный режим Q_max <= d
            if cfg.crossing_cap_normal_tons <= cfg.daily_demand_tons + 1.0:
                continue
            before = simulate_corridor(cfg, actual_close_day, actual_open_day, cal_close_day, cal_open_day, use_prestock=False)
            after = simulate_corridor(cfg, actual_close_day, actual_open_day, model_close_day, model_open_day, use_prestock=True)
            diff_rub = after["total_cost_rub"] - before["total_cost_rub"]
            saving_pct = -100.0 * diff_rub / before["total_cost_rub"] if before["total_cost_rub"] > 0 else 0.0
            rows.append({
                "parameter": param,
                "label": label,
                "variation": tag,
                "param_value": round(float(getattr(cfg, param)), 2),
                "before_mln_rub": round(before["total_cost_rub"] / 1e6, 2),
                "after_mln_rub": round(after["total_cost_rub"] / 1e6, 2),
                "saving_mln_rub": round(-diff_rub / 1e6, 2),
                "saving_pct": round(saving_pct, 1),
                "yak_storage_delta_mln_rub": round((after["cost_holding_yak_rub"] - before["cost_holding_yak_rub"]) / 1e6, 2),
                "effect_positive": bool(diff_rub < 0),
            })
    return pd.DataFrame(rows, columns=cols)

