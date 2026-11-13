"""М3 по собственному М1: условный план с проверкой всего периода рейса."""
import numpy as np
import pandas as pd
from .flood_planning import confirmed_calendar, optimize_dispatch


def calendar_for_levels(records, forecast, decision, travel_days, limit_cm, bound='upper_cm'):
    if not np.isfinite(limit_cm):
        raise ValueError('Нужен конечный порог уровня конкретного поста')
    if bound not in ('lower_cm','predicted_cm','upper_cm'):
        raise ValueError('Неизвестный вариант прогноза')
    f=forecast.copy()
    dates=pd.to_datetime(f.date).dt.normalize()
    expected=pd.date_range(pd.Timestamp(decision)+pd.Timedelta(days=1),periods=len(f))
    if not len(f) or not np.array_equal(dates.to_numpy(),expected.to_numpy()):
        raise ValueError('Прогноз должен содержать последовательные будущие даты после решения')
    values=f[['lower_cm','predicted_cm','upper_cm']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,0]>values[:,1]).any() or (values[:,1]>values[:,2]).any():
        raise ValueError('Некорректный диапазон прогноза')
    calendar=confirmed_calendar(records,None,decision,travel_days,allow_without_forecast=True)
    levels=dict(zip(dates.dt.date,f[bound]))
    allowed=[];maximum=[]
    for day in calendar.date:
        window=[levels.get((pd.Timestamp(day)+pd.Timedelta(days=i)).date()) for i in range(travel_days)]
        complete=all(v is not None for v in window)
        maximum.append(max(window) if complete else np.nan)
        allowed.append(complete and max(window)<limit_cm)
    calendar['policy_allowed']=allowed
    calendar['max_level_cm']=maximum
    return calendar


def plan_variants(records,forecast,decision,travel_days,limit_cm,quantity,road_cost,backup_cost,holding_cost,*,wind=None):
    result={}
    for name,bound in [('lower','lower_cm'),('median','predicted_cm'),('upper','upper_cm')]:
        calendar=calendar_for_levels(records,forecast,decision,travel_days,limit_cm,bound)
        if wind is not None:
            from .handling_wind import constrain
            calendar=constrain(calendar,*wind)
        plan,summary=optimize_dispatch(calendar,quantity,road_cost,backup_cost,holding_cost)
        result[name]=(plan,summary)
    return result
