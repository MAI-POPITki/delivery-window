"""Declared availability scenarios, not predictions of legal transport access.

All calendars are assumed known on day zero. Historical facts must never be
silently used as an operational forecast. The two-leg continuous-tonne model
does not represent individual sailings, vessel turnaround or vehicle fleets.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .corridor import simulate_corridor


class Availability(Protocol):
    def calendar(self, horizon: int) -> np.ndarray: ...


@dataclass(frozen=True)
class SeasonalWindow:
    start: int
    end: int

    def calendar(self, horizon):
        if not 0 <= self.start < self.end <= horizon:
            raise ValueError('Window must be nonempty and within the horizon')
        days = np.arange(horizon)
        return (days >= self.start) & (days < self.end)


@dataclass(frozen=True)
class Closure:
    start: int
    end: int

    def calendar(self, horizon):
        return ~SeasonalWindow(self.start, self.end).calendar(horizon)


@dataclass(frozen=True)
class ServiceDays:
    days: tuple[int, ...]

    def calendar(self, horizon):
        if not self.days or any(d < 0 or d >= horizon for d in self.days):
            raise ValueError('Service dates must be within the horizon')
        return np.isin(np.arange(horizon), self.days)


@dataclass(frozen=True)
class DeclaredCalendar:
    """External timetable explicitly known at the start, not a future fact log."""
    available: tuple[bool, ...]

    def calendar(self, horizon):
        if len(self.available) != horizon or any(type(v) is not bool for v in self.available):
            raise ValueError('Calendar requires one boolean per day of the horizon')
        return np.array(self.available, dtype=bool)


def calendar_from_frame(frame, horizon):
    """Import day/available CSV; refuse missing days, duplicates and coercion.

    Caller must mark the input as a declared scenario, not an as-of forecast.
    """
    if not {'day', 'available'} <= set(frame):
        raise ValueError('Required columns: day, available (0 or 1)')
    ordered = frame.sort_values('day')
    if len(ordered) != horizon or ordered.day.tolist() != list(range(horizon)):
        raise ValueError('Days must cover 0..horizon-1 exactly once')
    if not ordered.available.isin([0, 1]).all():
        raise ValueError('Availability must be 0 or 1')
    return DeclaredCalendar(tuple(bool(v) for v in ordered.available))


def compare_supply(cfg, schedule: Availability):
    """Same known calendar and costs for reactive and anticipatory policies.

    Baseline: existing stock-replenishment rule, no seasonal prestocking.
    Optimiser: existing LP, with declared daily availability. Neither receives
    an ML forecast or a claim of measured economic benefit.
    """
    mask = schedule.calendar(cfg.horizon_days)
    return {
        name: simulate_corridor(cfg, 0, cfg.horizon_days, 0, cfg.horizon_days,
                                use_prestock=False, economic=economic,
                                availability=mask, return_daily=True)
        for name, economic in [('Пополнение по остатку', False),
                               ('Экономический план', True)]
    }
