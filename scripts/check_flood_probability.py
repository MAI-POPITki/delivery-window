"""Построить прошлогодние ошибки и проверить вероятности на следующих годах."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catboost import CatBoostRegressor
from okno import flood_m1_quantile as M
from okno import flood_probability as P
from okno.flood_m2 import sha256


def main():
    P.DATA.mkdir(parents=True,exist_ok=True)
    predictions=pd.read_csv(M.ROOT/'reports/flood_m1_quantile_predictions.csv.gz',parse_dates=['issue','target_date'])
    outputs=[]
    for pid in M.STATIONS:
        series=M.load_levels(pid);table=M.dataset(series)
        for path in sorted(M.MODELS.glob(f'{pid}_20*.json')):
            meta=json.loads(path.read_text(encoding='utf-8'));year=meta['year']
            model=CatBoostRegressor();model.load_model(str(path.with_suffix('.cbm')))
            cal=table[(table.issue.dt.year==year-1)&(table.target_date<pd.Timestamp(year,1,1))].copy()
            p=M.predict_rows(series,cal,year-1,model,meta)
            errors=pd.DataFrame(dict(issue=cal.issue,lead=cal.lead,target=cal.target,
                normalized_error=(cal.target-p.m1)/np.maximum((p.upper-p.lower)/2,1.),
                persistence_error=cal.target-cal.h0))
            out=P.DATA/f'{pid}_{year}.csv.gz'
            errors.to_csv(out,index=False,compression={'method':'gzip','mtime':0})
            source=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
            (P.DATA/f'{pid}_{year}.json').write_text(json.dumps(dict(year=year,calibration_year=year-1,
                sha256=sha256(out),model_sha256=sha256(path.with_suffix('.cbm')),
                metadata_sha256=P.metadata_hash(path),data_sha256=source['sha256']),indent=2),encoding='utf-8')
            test=predictions[(predictions.pid==pid)&predictions.issue.dt.year.eq(year)&predictions.issue.dt.month.between(4,7)]
            historical=series[(series.index.year<year-1)&series.index.month.isin([4,5,6,7])].dropna()
            for lead in [1,3,7]:
                c=errors[errors.lead.eq(lead)&errors.issue.dt.month.between(4,7)]
                t=test[test.lead.eq(lead)]
                if t.empty or len(c)<60:continue
                for quantile in [.8,.9]:
                    threshold=float(historical.quantile(quantile))
                    event=t.target.ge(threshold).to_numpy(float)
                    probs=[P.probability(c.normalized_error,r.m1,max((r.upper-r.lower)/2,1.),threshold) for r in t.itertuples()]
                    persist=[P.probability(c.persistence_error,r.persistence,1.,threshold) for r in t.itertuples()]
                    climate=(c.target.ge(threshold).sum()+1)/(len(c)+2)
                    o=t[['pid','issue','target_date','lead']].copy()
                    o['year']=year;o['threshold_quantile']=quantile;o['threshold_cm']=threshold
                    o['event']=event;o['probability']=probs;o['persistence_probability']=persist;o['climatology_probability']=climate
                    outputs.append(o)
            print(pid,year,flush=True)
    raw=pd.concat(outputs,ignore_index=True)
    raw.to_csv(M.ROOT/'reports/flood_probability_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    lines=['# Вероятности превышения уровня: повторная архивная проверка','',
        'Протокол: docs/FLOOD_PROBABILITY_PLAN.md. Это вероятность заданного события на гидропосту, '
        'не вероятность затопления/закрытия дороги. Все годы уже изучались; независимым тестом '
        'новой надстройки этот отчёт не является. Brier score меньше — лучше.','',
        '| Пост | Сутки | Порог | N | Событий | Brier М1 | Сохранить с ошибками | Частота | Выигрыш 90% против сохранения | Выигрыш 90% против частоты |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    statuses={};reliability=[];rng=np.random.default_rng(42)
    for (pid,lead,q),g in raw.groupby(['pid','lead','threshold_quantile']):
        losses={name:(g[col]-g.event)**2 for name,col in [('m1','probability'),('persistence','persistence_probability'),('climatology','climatology_probability')]}
        bounds={}
        for base in ['persistence','climatology']:
            gains=(losses[base]-losses['m1']).groupby(g.year).mean().to_numpy()
            bounds[base]=np.quantile(rng.choice(gains,(2000,len(gains))).mean(axis=1),[.05,.95]).tolist()
        key=f'{pid}:{lead}:{q}'
        statuses[key]=dict(repeated_evaluation=True,skill=all(v[0]>0 for v in bounds.values()),gain_ci90=bounds)
        lines.append(f"| {pid} | {lead} | q{int(q*100)} | {len(g)} | {int(g.event.sum())} | {losses['m1'].mean():.4f} | {losses['persistence'].mean():.4f} | {losses['climatology'].mean():.4f} | {bounds['persistence'][0]:+.4f} … {bounds['persistence'][1]:+.4f} | {bounds['climatology'][0]:+.4f} … {bounds['climatology'][1]:+.4f} |")
        for b in range(10):
            z=g[(g.probability>=b/10)&(g.probability<(b+1)/10)]
            if len(z):reliability.append(dict(pid=pid,lead=lead,threshold_quantile=q,bin=b,n=len(z),predicted=z.probability.mean(),observed=z.event.mean()))
    pd.DataFrame(reliability).to_csv(M.ROOT/'reports/flood_probability_reliability.csv',index=False)
    (P.DATA/'validation.json').write_text(json.dumps(statuses,indent=2),encoding='utf-8')
    lines+=['','Калибровка по корзинам: flood_probability_reliability.csv; все вероятности '
        'и реальные события: flood_probability_predictions.csv.gz. Положительная нижняя граница '
        'выигрыша против обоих соперников означает преимущество на повторной оценке; '
        'это не гарантия калибровки и не подтверждение эффективности доставки. '
        'Событий мало, соседние дни зависимы, годы различаются; отрицательные результаты не скрываются.']
    (M.ROOT/'reports/flood_probability.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
