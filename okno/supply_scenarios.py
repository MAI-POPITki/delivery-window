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
