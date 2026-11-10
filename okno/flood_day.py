"""Вероятность события на выбранные сутки, метрики на том же событии."""
import json
import pandas as pd
from . import flood_m1_quantile as M
from .flood_probability import exceedance

SOURCE='https://www.naai.ru/upload/iblock/a16/74xiojh2x2upvpwd1md2peasclw67vu9/%D0%9A%D1%80%D0%B8%D1%82%D0%B5%D1%80%D0%B8%D0%B8%20%D0%9E%D0%9F%D0%AF_25.02.2026_%D1%83%D1%82..pdf'
# НСА, критерии 2026, печатная страница 75: Оренбург 760;
# Орск 360–450. Для Орска показываем обе границы, не выдумываем точную отметку.
def event_definition(pid, series, year):
    if pid=='ural_orenburg':
        return dict(threshold=760.,lower_threshold=760.,label='Выход реки на пойму · разлив',physical=True,source=SOURCE,
                    note='Оренбург: начало выхода воды на пойму — 760 см. НСА, критерии 2026, стр. 75.')
    if pid=='ural_orsk':
        return dict(threshold=450.,lower_threshold=360.,label='Выход реки на пойму · разлив',physical=True,source=SOURCE,
                    note='Орск: начало выхода на пойму указано диапазоном 360–450 см. Вероятность показана диапазоном по этим двум порогам.')
    past=series[(series.index.year<year-1)&series.index.month.isin([4,5,6,7])].dropna()
    if len(past)<60:raise ValueError('Недостаточно прошлых наблюдений для порога')
    threshold=float(past.quantile(.9))
    return dict(threshold=threshold,lower_threshold=threshold,label='Очень высокая вода · q90',physical=False,
                source='https://level-satellite.ru',note='Для этого поста порог выхода на пойму не подтверждён. '
                'Процент относится к высокой воде относительно прошлых лет, а не к установленному факту разлива.')


def forecast_day(pid, issue, target):
    issue,target=pd.Timestamp(issue).normalize(),pd.Timestamp(target).normalize()
    lead=(target-issue).days
    if not 1<=lead<=7:raise ValueError('Проверенная вероятность доступна на 1–7 суток вперёд')
    series=M.load_levels(pid)
    definition=event_definition(pid,series,issue.year)
    forecast,_=M.forecast(pid,issue)
    selected=forecast[forecast.lead.eq(lead)]
    low=float(exceedance(pid,issue,selected,definition['threshold']).iloc[0].probability)
    high=float(exceedance(pid,issue,selected,definition['lower_threshold']).iloc[0].probability)
    return dict(**definition,probability=low,probability_upper=high,issue=issue.date(),target=target.date(),lead=lead,
                predicted_cm=float(selected.iloc[0].predicted_cm))


def efficiency(pid,lead):
    path=M.ROOT/'reports/flood_day_efficiency.json'
    return json.loads(path.read_text(encoding='utf-8')).get(f'{pid}:{lead}') if path.exists() else None
