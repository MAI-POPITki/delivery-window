"""Сценарии риска выбранного сезонного пути без построения географического маршрута."""
import pandas as pd
from .journey import evidence,model_risk


def scenarios(events,road,q,decision,passage,hy,margin,gross_t,earlier_days=7):
    decision=pd.Timestamp(decision).normalize();passage=pd.Timestamp(passage).normalize()
    if passage<decision:
        raise ValueError('Дата проезда не может быть раньше даты решения.')
    dates=[passage]
    dates.extend(passage-pd.Timedelta(days=n) for n in range(1,earlier_days+1) if passage-pd.Timedelta(days=n)>=decision)
    result=[]
    for date in dates:
        fact=evidence(events,road,decision,date,gross_t) if road else dict(status='unknown',limit_t=None,record=None)
        result.append(dict(date=date.date(),risk=model_risk(q,date,decision,hy,margin,gross_t),**fact))
    return result
