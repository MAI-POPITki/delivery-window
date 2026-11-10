"""Пересчёт вероятностей и их эффективности для всех выбираемых горизонтов 1–7."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from okno import flood_m1_quantile as M
from okno import flood_probability as P
from okno.flood_day import event_definition


def main():
    predictions=pd.read_csv(M.ROOT/'reports/flood_m1_quantile_predictions.csv.gz',parse_dates=['issue','target_date'])
    output=[]
    for pid in M.STATIONS:
        s=M.load_levels(pid)
        for year,g in predictions[predictions.pid.eq(pid)&predictions.issue.dt.month.between(4,7)].groupby(predictions.issue.dt.year):
            definition=event_definition(pid,s,year);threshold=definition['threshold']
            errors=P.calibration(pid,year)
            for lead,t in g.groupby('lead'):
                c=errors[errors.lead.eq(lead)&errors.issue.dt.month.isin([4,5,6,7])]
                if len(c)<60:continue
                scale=np.maximum((t.upper-t.lower)/2,1.)
                p=[P.probability(c.normalized_error,f,w,threshold) for f,w in zip(t.m1,scale)]
                base=[P.probability(c.persistence_error,f,1.,threshold) for f in t.persistence]
                climate=(c.target.ge(threshold).sum()+1)/(len(c)+2)
                o=t[['pid','issue','target_date','lead']].copy()
                o['threshold_cm']=threshold;o['event']=t.target.ge(threshold).astype(int)
                o['probability']=p;o['persistence_probability']=base;o['climatology_probability']=climate
                output.append(o)
    raw=pd.concat(output,ignore_index=True)
    raw.to_csv(M.ROOT/'reports/flood_day_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    scores={}
    lines=['# Вероятность разлива на выбранную дату: эффективность','',
        'Все горизонты 1–7 суток, апрель–июль доступных тестовых лет. Проверяется фактическое '
        'превышение уровня в выбранные сутки, не контур затопления. Оренбург — 760 см, '
        'Орск — верхняя граница начала выхода на пойму 450 см. Источник: критерии НСА 2026, '
        'печатная страница 75; справочник применён ретроспективно. Якутск/Ишим — q90 прошлых '
        'учебных лет, только высокая вода. Прогнозы используют данные до решения; '
        'ошибки для вероятностей — предыдущий год. Архив уже использовался: это повторная, '
        'не новая независимая проверка. Brier меньше — лучше.','',
        '| Пост | Сутки | N | Событий | Brier | Сохранить уровень | Частота | TP / FN / FP / TN при 50% |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for (pid,h),g in raw.groupby(['pid','lead']):
        y=g.event.to_numpy();p=g.probability.to_numpy();warn=p>=.5
        bs=float(np.mean((p-y)**2));bp=float(np.mean((g.persistence_probability-y)**2));bc=float(np.mean((g.climatology_probability-y)**2))
        row=dict(n=len(g),events=int(y.sum()),first=str(g.target_date.min().date()),last=str(g.target_date.max().date()),
                 brier=bs,persistence_brier=bp,climatology_brier=bc,skill_climatology=1-bs/bc,skill_persistence=1-bs/bp,
                 tp=int((warn&(y==1)).sum()),fn=int((~warn&(y==1)).sum()),fp=int((warn&(y==0)).sum()),tn=int((~warn&(y==0)).sum()))
        scores[f'{pid}:{h}']=row
        lines.append(f"| {pid} | {h} | {len(g)} | {row['events']} | {bs:.4f} | {bp:.4f} | {bc:.4f} | {row['tp']} / {row['fn']} / {row['fp']} / {row['tn']} |")
    lines+=['','Сырые выпуски: `flood_day_predictions.csv.gz`; метрики: `flood_day_efficiency.json`. '
        'Отсутствие текущих измерений 2026 не скрывается. Вероятность выхода реки на пойму '
        'не является вероятностью закрытия дороги. Редкие события, серийная зависимость и '
        'непроверенный перенос порогов в прошлое ограничивают доказательства.']
    (M.ROOT/'reports/flood_day_efficiency.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
    (M.ROOT/'reports/flood_day_efficiency.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(raw)} прогнозов, {len(scores)} сочетаний пост/горизонт')


if __name__=='__main__':main()
