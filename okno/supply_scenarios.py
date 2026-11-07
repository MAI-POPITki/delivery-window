"""Сценарии заданной доступности, а не разрешение движения.

Календарь предполагается известным на старте. Исторические факты нельзя
выдавать за оперативный прогноз. Модель делимых тонн не описывает
отдельные рейсы, оборот судов и парк техники.
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
            raise ValueError('Период должен быть непустым и находиться внутри горизонта')
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
            raise ValueError('Дни обслуживания должны находиться внутри горизонта')
        return np.isin(np.arange(horizon), self.days)


@dataclass(frozen=True)
class DeclaredCalendar:
    """Внешний календарь, заданный на старте; не журнал будущих фактов."""
    available: tuple[bool, ...]

    def calendar(self, horizon):
        if len(self.available) != horizon or any(type(v) is not bool for v in self.available):
            raise ValueError('На каждый день горизонта требуется одно логическое значение')
        return np.array(self.available, dtype=bool)


def calendar_from_frame(frame, horizon):
    """Импорт CSV day/available с проверкой пропусков и дублей.

    Вход должен быть обозначен как заданный сценарий, а не исторический прогноз.
    """
    if not {'day', 'available'} <= set(frame):
        raise ValueError('Нужны столбцы day и available (0 или 1)')
    ordered = frame.sort_values('day')
    if len(ordered) != horizon or ordered.day.tolist() != list(range(horizon)):
        raise ValueError('Нужен каждый день от 0 до конца горизонта без дублей')
    if not ordered.available.isin([0, 1]).all():
        raise ValueError('Доступность должна быть 0 или 1')
    return DeclaredCalendar(tuple(bool(v) for v in ordered.available))


def compare_supply(cfg, schedule: Availability):
    """Общие календарь и затраты для пополнения по остатку и оптимизатора.

    База — прежнее правило пополнения без сезонного предзавоза.
    Оптимизатор — прежняя линейная программа с календарём доступности.
    ML здесь не используется; фактическая экономия не заявляется.
    """
    mask = schedule.calendar(cfg.horizon_days)
    return {
        name: simulate_corridor(cfg, 0, cfg.horizon_days, 0, cfg.horizon_days,
                                use_prestock=False, economic=economic,
                                availability=mask, return_daily=True)
        for name, economic in [('Пополнение по остатку', False),
                               ('Экономический план', True)]
    }
