"""
М3 — планировщик отправок по зимнику / ледовой переправе.

Задача склада: завезти Q т до закрытия пути. Каждую неделю решаем, сколько отправить
своей (базовой) техникой и сколько — привлечённой дополнительной (дороже на c_extra за тонну).
Груз, не доехавший до закрытия C, везут дорогим каналом (авиация / СВП / перенос на навигацию).
Ранний завоз стоит хранения до опорной даты R (начало расхода запаса).

Стоимость тонны в неделю w на «ярусе» техники j (линейно ⇒ точное решение сортировкой):
    cost(w, j) = P(C > w) · [c_road + c_extra·[j = доп.] + c_hold · (R − w)] + P(C ≤ w) · c_exp
Ограничения: x_base(w) ≤ K, x_extra(w) ≤ K_extra. P(C ≤ w) — Монте-Карло, 1 000 сценариев из М1.

Правила для сравнения:
- calendar («до»):  план Q равномерно до календарной даты закрытия; если базовой техники не хватает
                    на такой темп — заранее нанимают дополнительную (рациональный календарный план);
- forecast («после»): еженедельный пересчёт по свежему прогнозу М1;
- asap (контроль):   везти на пределе (база + доп. техника) с первой недели, без прогноза.

ВСЕ ЭКОНОМИЧЕСКИЕ ПАРАМЕТРЫ — СИНТЕТИКА/ДОПУЩЕНИЯ, откалиброванные по порядкам величин из
открытых источников; вывод проверяется изменением каждого на ±30% (scripts/before_after.py).
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    Q_t: float = 12_000              # потребность склада на сезон, т (синтетика)
    K_t_week: float = 1_000          # своя провозная способность, т/нед: план впритык к окну (синтетика)
    K_extra_t_week: float = 600      # максимум дополнительной техники, т/нед (синтетика)
    c_road: float = 20_000           # ₽/т по зимнику (допущение)
    c_extra: float = 10_000          # надбавка за привлечённую технику, ₽/т (допущение)
    c_exp: float = 120_000           # ₽/т дорогим каналом (допущение)
    c_hold_week: float = 400         # ₽/т·нед хранения при раннем завозе (допущение)
    close_margin_days: float = 21    # закрытие пути = начало таяния − запас (допущение)
    start_day: int = 150             # начало завоза ≈ 29 января (день сезона; 0 = 1 сентября)
    transit_days: float = 7          # рейс по зимнику от базы до склада, сут: груз «успел», если доехал до закрытия (допущение)
    backup_arrival_day: int = 317    # недовезённое приходит запасным каналом (летняя навигация) ≈ 15 июля (допущение)
    n_scenarios: int = 1000
    seed: int = 42

    def scaled(self, name: str, factor: float) -> "Params":
        d = asdict(self)
        d[name] = d[name] * factor
        return Params(**d)


SOURCE_TYPE = {"Q_t": "синтетика", "K_t_week": "синтетика", "K_extra_t_week": "синтетика",
               "c_road": "допущение", "c_extra": "допущение", "c_exp": "допущение",
               "c_hold_week": "допущение", "close_margin_days": "допущение", "transit_days": "допущение"}
SENSITIVE = list(SOURCE_TYPE)
WEEK = 7


def sample_quantiles(q10, q50, q90, n, rng):
    """Сценарии из кусочно-линейной функции квантилей; хвосты продолжают наклон соседнего отрезка."""
    u = rng.uniform(0, 1, n)
    lo_slope = (q50 - q10) / 0.4
    hi_slope = (q90 - q50) / 0.4
    return np.where(u < 0.5, q50 - (0.5 - u) * lo_slope, q50 + (u - 0.5) * hi_slope)


def plan_week_volumes(weeks, p_closed, remaining, p: Params, R):
    """Оптимальный при данном распределении план на оставшиеся недели: (x_base, x_extra)."""
    hold = p.c_hold_week * np.maximum(R - weeks, 0) / WEEK
    opts = []
    for j, (cap, extra) in enumerate(((p.K_t_week, 0.0), (p.K_extra_t_week, p.c_extra))):
        cost = (1 - p_closed) * (p.c_road + extra + hold) + p_closed * p.c_exp
        opts += [(c, i, j, cap) for i, c in enumerate(cost)]
    x = np.zeros((len(weeks), 2))
    left = remaining
    for c, i, j, cap in sorted(opts, key=lambda o: (o[0], o[1], o[2])):
        if left <= 0 or c >= p.c_exp:  # дороже, чем сразу дорогой канал — не везём
            break
        x[i, j] = min(cap, left)
        left -= x[i, j]
    return x[:, 0], x[:, 1]


def calendar_plan(weeks, cal_close, p: Params):
    n = max(1, int(np.sum(weeks + p.transit_days <= cal_close)))
    per = p.Q_t / n
    base, extra = np.zeros(len(weeks)), np.zeros(len(weeks))
    left = p.Q_t
    for i in range(len(weeks)):
        want = min(per if i < n else left, left)
        base[i] = min(p.K_t_week, want)
        extra[i] = min(p.K_extra_t_week, want - base[i])
        left -= base[i] + extra[i]
        if left <= 1e-9:
            break
    return base, extra


def simulate_season(actual_close, forecasts, cal_close, p: Params, rule: str) -> dict:
    """
    actual_close — фактическая дата закрытия пути (день сезона).
    forecasts — прогнозы М1 по датам выпуска: issue_day, q10, q50, q90 (дата события до вычета запаса).
    """
    rng = np.random.default_rng(p.seed)
    weeks = np.arange(p.start_day, 300, WEEK, dtype=float)
    # опорная дата начала расхода запаса — задаётся заранее, по календарю; фактическая дата закрытия
    # в момент планирования неизвестна и в план не попадает
    R = cal_close + 14
    base, extra = np.zeros(len(weeks)), np.zeros(len(weeks))
    if rule == "asap":
        left = p.Q_t
        for i in range(len(weeks)):
            base[i] = min(p.K_t_week, left)
            extra[i] = min(p.K_extra_t_week, left - base[i])
            left -= base[i] + extra[i]
    elif rule == "calendar":
        base, extra = calendar_plan(weeks, cal_close, p)
    else:
        remaining = p.Q_t
        fc = forecasts.set_index("issue_day").sort_index()
        for i, w in enumerate(weeks):
            if w >= actual_close or remaining <= 0:
                break
            avail = fc[fc.index <= w]
            if avail.empty:
                p_closed = (weeks[i:] + p.transit_days > cal_close).astype(float)
            else:
                f = avail.iloc[-1]
                sc = sample_quantiles(f.q10, f.q50, f.q90, p.n_scenarios, rng) - p.close_margin_days
                sc = np.maximum(sc, w)  # путь сегодня точно открыт — мы на нём
                p_closed = np.array([(sc < ww + p.transit_days).mean() for ww in weeks[i:]])
            xb, xe = plan_week_volumes(weeks[i:], p_closed, remaining, p, R)
            base[i], extra[i] = xb[0], xe[0]
            remaining -= xb[0] + xe[0]
    ok = weeks + p.transit_days <= actual_close  # рейс «успел»: груз выехал и доехал до закрытия
    delivered = float((base + extra)[ok].sum())
    extra_used = float(extra[ok].sum())
    undelivered = p.Q_t - delivered
    hold = float(((base + extra)[ok] * np.maximum(R - weeks[ok], 0) / WEEK).sum()) * p.c_hold_week
    cost = delivered * p.c_road + extra_used * p.c_extra + hold + undelivered * p.c_exp
    # весь путь до получателя: недовезённое по зимнику приходит запасным каналом позже, чем нужно складу
    wait_days = max(0.0, p.backup_arrival_day - R)
    return {"rule": rule, "delivered_t": delivered, "share_delivered": delivered / p.Q_t,
            "recipient_wait_t_days": undelivered * wait_days, "avg_wait_days": undelivered * wait_days / p.Q_t,
            "undelivered_t": undelivered, "extra_t": extra_used,
            "late_shipped_t": float((base + extra)[~ok].sum()),
            "cost_rub": cost, "cost_rub_per_t": cost / p.Q_t, "hold_rub": hold}


def warning_lead(forecasts: pd.DataFrame, actual_event: float, cal_event: float, threshold: float = 5) -> float:
    """За сколько дней до события М1 впервые предупредила о раннем закрытии (медиана прогноза
    раньше календаря на ≥ threshold сут). Только для сезонов, где событие действительно
    наступило раньше календаря на ≥ threshold; иначе NaN. 0 — предупреждения не было."""
    if actual_event > cal_event - threshold:
        return np.nan
    fc = forecasts.sort_values("issue_day")
    hit = fc[fc.q50 <= cal_event - threshold]
    if hit.empty:
        return 0.0
    return float(actual_event - hit.issue_day.iloc[0])
