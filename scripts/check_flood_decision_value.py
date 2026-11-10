"""Ценность предупреждения на реальных превышениях в условной cost/loss задаче."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=pd.read_csv(ROOT/'reports/flood_probability_predictions.csv.gz')
    rows=[]
    for (pid,h,q),g in data.groupby(['pid','lead','threshold_quantile']):
        for ratio in np.arange(.1,1.,.1):
            row=dict(pid=pid,lead=h,threshold_quantile=q,cost_loss_ratio=ratio,n=len(g))
            for name,col in [('model','probability'),('persistence','persistence_probability'),('climatology','climatology_probability')]:
                action=g[col]>=ratio
                row[name]=float(np.mean(ratio*action+(~action)*g.event))
            row['oracle']=float(g.event.mean()*ratio)
            rows.append(row)
    scores=pd.DataFrame(rows)
    scores.to_csv(ROOT/'reports/flood_decision_value.csv',index=False)
    lines=['# Ценность вероятностного предупреждения: условная задача','',
        'Реальные события превышения из flood_probability_predictions.csv.gz. '
        'Допущение: защитное действие полностью предотвращает единичную потерю, '
        'но стоит C/L=0.1…0.9. Действовать, когда p≥C/L. Затраты в условных '
        'единицах: C/L при действии, иначе 1 при превышении. Это не реальные '
        'рубли, не перевозки и не доказанная экономия М3; это проверка полезности '
        'предупреждения при явно заданной цене ошибки. Архив повторно использован.','',
        '| Пост | Горизонт | Порог | Лучше обоих правил, отношений C/L из 9 |',
        '| --- | ---: | ---: | ---: |']
    for (pid,h,q),g in scores.groupby(['pid','lead','threshold_quantile']):
        count=int(((g.model<g.persistence-1e-12)&(g.model<g.climatology-1e-12)).sum())
        lines.append(f'| {pid} | {h} | q{int(q*100)} | {count} / 9 |')
    lines+=['','Все затраты, включая идеальное знание факта, сохранены в CSV. '
        'Нет гарантии выигрыша при каждой цене ошибки. При реальных рейсах действие '
        'может не предотвращать потерю полностью и иметь другие ограничения; '
        'поэтому эти числа не следует выдавать за подтверждённую эффективность доставки.']
    (ROOT/'reports/flood_decision_value.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
