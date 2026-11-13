"""Общая оценка: согласованная дата М1/М2, без выдуманного шанса проезда."""
import pandas as pd
from . import flood_m2 as S
from . import flood_probability as P


def assess(pid,decision,forecast,threshold=None,probabilistic=True):
    issue=pd.Timestamp(decision).date()
    if forecast.empty:raise ValueError('Нет прогноза')
    selected=forecast.iloc[-1]
    if pd.Timestamp(selected.date).date()<=issue:raise ValueError('Цель прогноза не в будущем')
    result=dict(pid=pid,decision=issue,target=pd.Timestamp(selected.date).date(),
        predicted_cm=float(selected.predicted_cm),threshold_cm=threshold,probability=None,satellite=None,
        satellite_status='К дате решения нет обработанного спутникового наблюдения',passage_probability=None)
    if probabilistic and threshold is not None:
        result['probability']=float(P.exceedance(pid,issue,forecast,threshold).iloc[-1].probability)
    observation=S.select_run(pid,issue)
    if observation:
        path,meta=observation
        age=(issue-pd.Timestamp(meta['acquired_at']).date()).days
        stats=meta['statistics'];agreement=stats.get('scl_water_agreement_iou')
        usable=age<=10 and stats['observed_fraction']>=.5 and agreement is not None and agreement>=.5
        result['satellite']=dict(path=path,meta=meta,age_days=age,usable=usable)
        result['satellite_status']=('Есть пригодное спутниковое наблюдение' if usable else
            'Спутниковое наблюдение неполное, устарело или методы расходятся')
    return result
