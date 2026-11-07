"""Связь существующего прогноза вскрытия Лены со сценарием речной доставки.

Дата физического события не является разрешением навигации. Сдвиг до начала
перевозок задаётся явно, а не подбирается по результату экономического расчёта.
"""
import pandas as pd

from .supply_scenarios import SeasonalWindow, compare_supply


def navigation_windows(forecast, origin, horizon, end_date, offset_days=0):
    """Одинаковый сдвиг и конец сезона для календаря и всех квантилей М1."""
    if forecast is None or 'error' in forecast:
        raise ValueError('Прогноз М1 недоступен')
    if not isinstance(offset_days, int) or offset_days < 0:
        raise ValueError('Сдвиг после вскрытия должен быть целым неотрицательным числом')
    origin = pd.Timestamp(origin)
    end = (pd.Timestamp(end_date)-origin).days
    result = {}
    for key in ('calendar', 'q10', 'q50', 'q90'):
        date = forecast.get(key+'_date')
        if date is None or pd.isna(date):
            raise ValueError('В прогнозе отсутствует дата '+key)
        start = (pd.Timestamp(date)-origin).days+offset_days
        window = SeasonalWindow(start, end)
        window.calendar(horizon)
        result[key] = window
    return result


def compare_navigation(cfg, windows):
    """Меняется только источник даты, оптимизатор и параметры общие."""
    return {key: compare_supply(cfg, window)['Экономический план']
            for key, window in windows.items()}


def observation_as_of(frame, point, date):
    """Последний снимок не позднее даты решения; будущие снимки исключены."""
    subset = frame[frame.point.eq(point)].copy()
    subset['date'] = pd.to_datetime(subset.date)
    subset = subset[subset.date <= pd.Timestamp(date)]
    return None if subset.empty else subset.sort_values('date').iloc[-1]
