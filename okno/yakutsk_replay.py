"""Daily replay of a *hypothetical* order on the Yakutsk ice crossing, spring 2026.

Only today's observed legal limit is supplied to a decision. Future limits come
from the announced timetable or M1; actual future limits are used only by the
simulator. M1 predicts physical breakup, not a legally permitted truck mass.
The 21-day margin and 14/7-day phase offsets are explicit assumptions.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .ice import to_season_day
from .m2 import apply_observation
from .planner import sample_quantiles


def _date(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A valid date is required")
    return stamp if stamp.tzinfo is None else stamp.tz_convert("UTC").tz_localize(None)


@dataclass(frozen=True)
class Scenario:
    Q_t: float = 5000.0
    base_trucks: float = 10.0
    extra_trucks: float = 5.0
    tare_t: float = 12.0
    max_payload_t: float = 28.0
    trips_per_workday: float = 1.0
    weekdays: tuple = (0, 1, 2, 3, 4)
    c_road: float = 4000.0
    c_extra: float = 1500.0
    c_backup: float = 20000.0
    c_hold_week: float = 100.0
    start_date: str = "2026-03-16"
    demand_date: str = "2026-04-16"
    backup_date: str = "2026-05-26"
    hydro_year: int = 2026
    close_margin_days: float = 21.0
    phase30_days: float = 14.0
    phase20_days: float = 7.0
    climatology_halfwidth_days: float = 9.0
    n_scenarios: int = 1000
    m2_max_age_days: int = 14

    def __post_init__(self):
        positive = ("Q_t", "base_trucks", "max_payload_t", "trips_per_workday", "c_backup")
        nonnegative = ("extra_trucks", "tare_t", "c_road", "c_extra", "c_hold_week",
                       "close_margin_days", "phase30_days", "phase20_days", "climatology_halfwidth_days")
        for key in positive + nonnegative:
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key in positive and value == 0):
                raise ValueError(f"Invalid {key}: {value}")
        if self.tare_t >= 40:
            raise ValueError("The empty vehicle must fit within the initial 40 t limit")
        if not self.weekdays or not set(self.weekdays) <= set(range(7)):
            raise ValueError("weekdays must contain values between 0 and 6")
        if self.n_scenarios < 20 or int(self.n_scenarios) != self.n_scenarios:
            raise ValueError("n_scenarios must be an integer >= 20")
        if self.m2_max_age_days < 0 or self.phase30_days < self.phase20_days:
            raise ValueError("Invalid phase offsets or image freshness")
        if not _date(self.start_date) < _date(self.demand_date) <= _date(self.backup_date):
            raise ValueError("Expected start_date < demand_date <= backup_date")


@dataclass(frozen=True)
class RestrictionSchedule:
    date30: str = "2026-04-01"
    date20: str = "2026-04-08"
    last_open_date: str = "2026-04-15"

    def __post_init__(self):
        if not _date(self.date30) <= _date(self.date20) <= _date(self.last_open_date):
            raise ValueError("Restriction dates must be ordered")

    def limit(self, dates):
        d = pd.DatetimeIndex(dates)
        return np.select([d > _date(self.last_open_date), d >= _date(self.date20),
                          d >= _date(self.date30)], [0.0, 20.0, 30.0], default=40.0)


ANNOUNCED = RestrictionSchedule()
ACTUAL = RestrictionSchedule(date30="2026-03-27")
RULES = ("calendar", "climatology", "forecast", "forecast_m2", "asap")
RULE_LABELS = {
    "calendar": "Календарь с пересчётом",
    "climatology": "Планировщик + календарь ±9 сут",
    "forecast": "М1 + планировщик",
    "forecast_m2": "М1 + спутник + планировщик",
    "asap": "Максимально быстрый завоз",
}


@dataclass
class ReplayResult:
    summary: dict
    schedule: pd.DataFrame
    decisions: pd.DataFrame


def payload(limit, scenario):
    return np.clip(np.asarray(limit, dtype=float) - scenario.tare_t, 0, scenario.max_payload_t)


def capacity_table(scenario=Scenario(), schedule=ACTUAL):
    dates = pd.date_range(scenario.start_date, _date(scenario.demand_date) - pd.Timedelta(days=1))
    limits = schedule.limit(dates)
    work = np.isin(dates.day_of_week, scenario.weekdays)
    load = payload(limits, scenario) * scenario.trips_per_workday * work
    return pd.DataFrame({"date": dates, "workday": work, "gross_limit_t": limits,
                         "base_capacity_t": load * scenario.base_trucks,
                         "extra_capacity_t": load * scenario.extra_trucks})


def _forecasts(frame, scenario):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["issue_date", "q10", "q50", "q90"])
    f = frame.copy()
    if "issue_date" not in f:
        if "decision_date" in f:
            f["issue_date"] = f["decision_date"]
        elif "issue_day" in f:
            f["issue_date"] = pd.Timestamp(scenario.hydro_year - 1, 9, 1) + pd.to_timedelta(f.issue_day, unit="D")
        else:
            raise ValueError("Forecasts require issue_date or issue_day")
    f["issue_date"] = pd.to_datetime(f.issue_date, utc=True).dt.tz_localize(None)
    if f.issue_date.duplicated().any():
        raise ValueError("Duplicate forecast issue dates")
    q = f[["q10", "q50", "q90"]].to_numpy(dtype=float)
    if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        raise ValueError("Forecast quantiles must be finite and ordered")
    return f.sort_values("issue_date")


def _observations(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "available_date", "p_ice"])
    obs = frame.copy()
    for col in ("date", "available_date"):
        if col not in obs:
            raise ValueError(f"Observations require {col}")
        obs[col] = pd.to_datetime(obs[col], utc=True).dt.tz_localize(None)
    if (obs.available_date < obs.date).any() or not obs.p_ice.between(0, 1).all():
        raise ValueError("Invalid image publication time or probability")
    return obs.sort_values(["date", "available_date"])


def _quantiles(day, forecasts, observations, rule, scenario, announced):
    available = forecasts[forecasts.issue_date <= day]
    info = {"forecast_issue_date": None, "image_date": None, "image_available_date": None,
            "p_ice": np.nan, "m2_applied": False, "forecast_fallback": available.empty}
    if rule == "climatology" or available.empty:
        center = to_season_day(announced.last_open_date, scenario.hydro_year) + scenario.close_margin_days
        q = np.array([center - scenario.climatology_halfwidth_days, center,
                      center + scenario.climatology_halfwidth_days])
    else:
        f = available.iloc[-1]
        q = f[["q10", "q50", "q90"]].to_numpy(dtype=float)
        info["forecast_issue_date"] = f.issue_date
    if rule == "forecast_m2":
        obs = observations[(observations.available_date <= day) & (observations.date <= day)
                           & (observations.date >= day - pd.Timedelta(days=scenario.m2_max_age_days))]
        if len(obs):
            o = obs.iloc[-1]
            info.update(image_date=o.date, image_available_date=o.available_date, p_ice=float(o.p_ice))
            obs_day = to_season_day(o.date, scenario.hydro_year)
            q, info["m2_applied"] = fuse_observation(q, obs_day, float(o.p_ice))
    info.update(q10=float(q[0]), q50=float(q[1]), q90=float(q[2]))
    return q, info


def fuse_observation(quantiles, obs_day, p_ice):
    """Reuse M2 truncation, subtracting its finite-sample quantile reconstruction error.

    An image that excludes no scenarios must have exactly zero effect. Both
    reference and truncated samples use the same seed and size as M2.
    """
    q = np.asarray(quantiles, dtype=float)
    support = sample_quantiles(*q, 4000, np.random.default_rng(42))
    keep = support > obs_day if p_ice >= .8 else support <= obs_day if p_ice <= .2 else np.ones(len(support), dtype=bool)
    if keep.all() or keep.mean() < .02:
        return q.copy(), False
    truncated = np.asarray(apply_observation(*q, obs_day, p_ice, n=4000, seed=42))
    updated = np.sort(q + truncated - np.percentile(support, [10, 50, 90]))
    return updated, bool(not np.allclose(q, updated))


def _probabilistic_today(dates, today_limit, remaining, q, scenario):
    """M3 linear cost slots, split by payload tiers supported by each limit.

    Upper payload tiers have their own probability of being permitted. Thus a
    20 t gross limit leaves 8 t on a 12 t truck, rather than treating its entire
    original 28 t payload as delivered. Only today's chosen slots are executed.
    """
    samples = sample_quantiles(*q, scenario.n_scenarios, np.random.default_rng(42)) - scenario.close_margin_days
    ds = np.array([to_season_day(d, scenario.hydro_year) for d in dates])
    limits = np.select([ds[:, None] > samples,
                        ds[:, None] >= samples - scenario.phase20_days,
                        ds[:, None] >= samples - scenario.phase30_days], [0., 20., 30.], default=40.)
    limits = np.minimum(limits, today_limit)  # only monotonic restriction information already observed
    limits[0, :] = today_limit
    work = np.isin(dates.day_of_week, scenario.weekdays)
    possible = sorted(set([0.] + [float(payload(x, scenario)) for x in (20, 30, 40, today_limit)]))
    hold = np.maximum(0, (_date(scenario.demand_date) - dates).days.to_numpy()) * scenario.c_hold_week / 7
    options = []
    for lo, hi in zip(possible[:-1], possible[1:]):
        if hi <= lo or hi > float(payload(today_limit, scenario)):
            continue
        p = (payload(limits, scenario) >= hi - 1e-10).mean(axis=1)
        for tier, trucks in enumerate((scenario.base_trucks, scenario.extra_trucks)):
            cap = (hi - lo) * trucks * scenario.trips_per_workday
            costs = p * (scenario.c_road + tier * scenario.c_extra + hold) + (1-p) * scenario.c_backup
            options.extend((float(costs[i]), i, tier, cap) for i in range(len(dates)) if work[i] and p[i] > 0)
    out = np.zeros(2)
    left = remaining
    for cost, i, tier, cap in sorted(options, key=lambda x: (x[0], x[1], x[2])):
        if left <= 1e-9 or cost >= scenario.c_backup:
            break
        x = min(left, cap)
        left -= x
        if i == 0:
            out[tier] += x
    return out


def simulate(rule, forecasts=None, observations=None, scenario=None, *, actual_schedule=ACTUAL,
             announced_schedule=ANNOUNCED):
    """Replay one policy without revealing future actual limits to its planner.

    A trip completes on its departure working day (explicit one-round-trip/day
    scenario). Unshipped cargo uses the backup once, at c_backup per tonne.
    """
    if rule not in RULES:
        raise ValueError(f"Unknown rule {rule}")
    s = scenario or Scenario()
    forecasts, observations = _forecasts(forecasts, s), _observations(observations)
    truth = capacity_table(s, actual_schedule)
    remaining = s.Q_t
    records, decisions = [], []
    for i, r in enumerate(truth.itertuples()):
        day, current_limit = r.date, r.gross_limit_t
        future_dates = pd.DatetimeIndex(truth.date.iloc[i:])
        info = {"forecast_issue_date": None, "image_date": None, "image_available_date": None,
                "p_ice": np.nan, "m2_applied": False, "forecast_fallback": False,
                "q10": np.nan, "q50": np.nan, "q90": np.nan}
        xb = xe = 0.
        if r.workday and current_limit > s.tare_t and remaining > 1e-8:
            if rule == "asap":
                xb = min(remaining, r.base_capacity_t)
                xe = min(remaining-xb, r.extra_capacity_t)
            elif rule == "calendar":
                limits = np.minimum(announced_schedule.limit(future_dates), current_limit)
                limits[0] = current_limit
                loads = payload(limits, s) * s.trips_per_workday * np.isin(future_dates.day_of_week, s.weekdays)
                base = loads * s.base_trucks
                extra = loads * s.extra_trucks
                # A rational calendar policy hires exactly the planned shortfall,
                # updating this requirement when a new restriction becomes known.
                xb = min(remaining, base[0] * min(1., remaining/base.sum())) if base.sum() else 0.
                shortfall = max(0., remaining-base.sum())
                xe = min(remaining-xb, extra[0] * min(1., shortfall/extra.sum())) if extra.sum() else 0.
            else:
                q, info = _quantiles(day, forecasts, observations, rule, s, announced_schedule)
                xb, xe = _probabilistic_today(future_dates, current_limit, remaining, q, s)
            xb = min(xb, r.base_capacity_t, remaining)
            xe = min(xe, r.extra_capacity_t, remaining-xb)
        delivered = xb + xe
        remaining = max(0., remaining-delivered)
        hold = delivered * max(0., (_date(s.demand_date)-day).days) * s.c_hold_week / 7
        records.append({"rule": rule, "date": day, "workday": r.workday,
                        "gross_limit_t": current_limit, "payload_limit_t": float(payload(current_limit, s)),
                        "base_capacity_t": r.base_capacity_t, "extra_capacity_t": r.extra_capacity_t,
                        "base_t": xb, "extra_t": xe, "delivered_t": delivered, "remaining_t": remaining,
                        "road_rub": delivered*s.c_road, "extra_rub": xe*s.c_extra, "hold_rub": hold})
        decisions.append({"rule": rule, "date": day, **info})
    table = pd.DataFrame(records)
    costs = {k: float(table[k].sum()) for k in ("road_rub", "extra_rub", "hold_rub")}
    costs["backup_rub"] = remaining * s.c_backup
    delivered = float(table.delivered_t.sum())
    completed = table[table.remaining_t <= 1e-7]
    wait = max(0, (_date(s.backup_date)-_date(s.demand_date)).days)
    summary = {"rule": rule, "Q_t": s.Q_t, "delivered_t": delivered, "backup_t": remaining,
               "undelivered_t": remaining, "extra_t": float(table.extra_t.sum()),
               "share_delivered": delivered/s.Q_t, "delivery_share": delivered/s.Q_t,
               **costs, "cost_rub": sum(costs.values()), "cost_rub_per_t": sum(costs.values())/s.Q_t,
               "recipient_wait_t_days": remaining*wait, "avg_wait_days": remaining*wait/s.Q_t,
               "completion_date": completed.date.iloc[0].date().isoformat() if len(completed) else s.backup_date}
    return ReplayResult(summary, table, pd.DataFrame(decisions))


def simulate_all(forecasts, observations=None, scenario=None, *, actual_schedule=ACTUAL,
                 announced_schedule=ANNOUNCED):
    return {rule: simulate(rule, forecasts, observations, scenario,
                           actual_schedule=actual_schedule, announced_schedule=announced_schedule)
            for rule in RULES}
