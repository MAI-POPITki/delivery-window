"""Fixed M2 radar/wind comparisons; never replace production weights or predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okno import m2, m2_wind as W
from okno.report import md_table

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
SEEDS=(42,7,2026)
VARIANTS={'radar':m2.FEATURES, 'radar_speed':m2.FEATURES+W.SPEED_FEATURES,
          'radar_wind':m2.FEATURES+W.WIND_FEATURES, 'wind_only':W.WIND_FEATURES}


def load_inputs():
    folder=ROOT/'data/s1/wind'
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256((folder/'radar_features.csv').read_bytes()).hexdigest()!=meta['radar_features_sha256']:
        raise ValueError('Изменены зафиксированные радарные признаки')
    if hashlib.sha256((folder/'features.csv').read_bytes()).hexdigest()!=meta['features_sha256']:
        raise ValueError('Изменены признаки ветра')
    for source in meta['sources']:
        W.load_response(ROOT/source['path'])
    radar=pd.read_csv(folder/'radar_features.csv')
    radar=radar[radar.label.isin(['ice','water']) & (radar.hydro_year<=2022)].copy()
    wind=pd.read_csv(folder/'features.csv')
    if wind.duplicated(['point','item_id']).any():
        raise ValueError('Повторяющиеся признаки ветра')
    data=radar.merge(wind,on=['point','item_id'],how='left',validate='one_to_one')
    missing=data[VARIANTS['radar_wind']].isna().any(axis=1)
    coverage=dict(labelled=len(data),used=int((~missing).sum()),excluded=int(missing.sum()))
    data=data.loc[~missing].reset_index(drop=True)
    data['y']=data.label.map({'water':0,'ice':1})
    return data,coverage


def folds(data,scheme):
    years=range(2019,2023) if scheme=='past_only' else range(2017,2023)
    if scheme not in ('past_only','leave_year_out'):
        raise ValueError('Неизвестный протокол разбиения')
    for year in years:
        test=data.hydro_year.eq(year)
        train=data.hydro_year.lt(year) if scheme=='past_only' else ~test
        if data.loc[train,'y'].nunique()!=2 or data.loc[test,'y'].nunique()!=2:
            raise ValueError(f'Нет двух классов в {scheme}/{year}')
        yield year,train,test


def metrics(frame):
    y=frame.y.to_numpy()
    p=frame.p_ice.to_numpy()
    pred=(p>=.5).astype(int)
    return dict(n=len(y),water=int((y==0).sum()),ice=int((y==1).sum()),
                accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                f1_water=float(f1_score(y,pred,pos_label=0,zero_division=0)),
                f1_ice=float(f1_score(y,pred,pos_label=1,zero_division=0)),
                brier=float(brier_score_loss(y,p)),
                water_as_ice=int(((y==0)&(pred==1)).sum()),
                ice_as_water=int(((y==1)&(pred==0)).sum()))


def paired_interval(predictions,iterations=5000):
    selected=predictions[(predictions.scheme=='past_only') & (predictions.seed==42)]
    keys=['hydro_year','point','item_id','y']
    radar=selected[selected.variant=='radar'][keys+['p_ice']]
    candidate=selected[selected.variant=='radar_wind'][keys+['p_ice']]
    paired=radar.merge(candidate,on=keys,suffixes=('_radar','_wind'),validate='one_to_one')
    if len(paired)!=len(radar) or len(paired)!=len(candidate):
        raise ValueError('Непарные прогнозы для интервала')
    blocks=[]
    for _,g in paired.groupby('hydro_year'):
        row=[]
        for suffix in ['radar','wind']:
            y=g.y.to_numpy();pred=g['p_ice_'+suffix].to_numpy()>=.5
            row.extend([int(((y==1)&pred).sum()),int((y==1).sum()),
                        int(((y==0)&~pred).sum()),int((y==0).sum())])
        blocks.append(row)
    blocks=np.asarray(blocks)
    rng=np.random.default_rng(42)
    totals=blocks[rng.integers(0,len(blocks),size=(iterations,len(blocks)))].sum(axis=1)
    delta=.5*(totals[:,4]/totals[:,5]+totals[:,6]/totals[:,7]
              -totals[:,0]/totals[:,1]-totals[:,2]/totals[:,3])
    return dict(metric='balanced_accuracy',unit='fraction',candidate='radar_wind',baseline='radar',
                scheme='past_only',seed=42,cluster='hydro_year',clusters=len(blocks),
                iterations=iterations,low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)))


def evaluate(data):
    outputs=[]
    for scheme in ['past_only','leave_year_out']:
        for year,train,test in folds(data,scheme):
            for seed in SEEDS:
                for variant,features in VARIANTS.items():
                    model=m2.make_model().set_params(random_seed=seed,thread_count=2)
                    model.fit(data.loc[train,features],data.loc[train,'y'])
                    result=data.loc[test,['point','item_id','hydro_year','label','y','wind_speed_ms']].copy()
                    result['p_ice']=model.predict_proba(data.loc[test,features])[:,1]
                    result['scheme']=scheme;result['seed']=seed;result['variant']=variant
                    result['train_max_year']=int(data.loc[train,'hydro_year'].max())
                    outputs.append(result)
            print(f'{scheme} {year}: train={int(train.sum())}, test={int(test.sum())}',flush=True)
    return pd.concat(outputs,ignore_index=True)


def report(predictions,coverage):
    overall=[];by_year=[];by_wind=[]
    for (scheme,variant,seed),g in predictions.groupby(['scheme','variant','seed']):
        overall.append(dict(scheme=scheme,variant=variant,seed=int(seed),**metrics(g)))
        for year,part in g.groupby('hydro_year'):
            by_year.append(dict(scheme=scheme,variant=variant,seed=int(seed),hydro_year=int(year),**metrics(part)))
        if scheme=='past_only' and seed==42:
            bins=pd.cut(g.wind_speed_ms,[-np.inf,3,6,np.inf],right=False,labels=['<3','3–6','>=6'])
            for name,part in g.groupby(bins,observed=True):
                by_wind.append(dict(variant=variant,wind_bin=str(name),**metrics(part)))
    overall=pd.DataFrame(overall);years=pd.DataFrame(by_year);wind=pd.DataFrame(by_wind)
    interval=paired_interval(predictions)
    predictions.to_csv(OUT/'m2_wind_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    overall.to_csv(OUT/'m2_wind_metrics.csv',index=False)
    years.to_csv(OUT/'m2_wind_years.csv',index=False)
    wind.to_csv(OUT/'m2_wind_bins.csv',index=False)
    avg=overall.groupby(['scheme','variant'])[['accuracy','balanced_accuracy','f1_water','f1_ice','brier']].mean().reset_index()
    main=overall[overall.scheme=='past_only'].set_index(['variant','seed'])
    forward_n=int(main.loc[('radar',42),'n'])
    heldout_n=int(overall[(overall.scheme=='leave_year_out') & (overall.variant=='radar') & (overall.seed==42)].n.iloc[0])
    deltas=[float(main.loc[('radar_wind',s),'balanced_accuracy']-main.loc[('radar',s),'balanced_accuracy']) for s in SEEDS]
    metadata=dict(coverage=coverage,seeds=list(SEEDS),features=VARIANTS,interval=interval,
                  primary_delta_by_seed=dict(zip(map(str,SEEDS),deltas)),
                  source_manifest_sha256=hashlib.sha256((ROOT/'data/s1/wind/manifest.json').read_bytes()).hexdigest(),
                  model_params=dict(iterations=400,depth=4,learning_rate=.05,thread_count=2),
                  production_artifacts={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in [m2.MODEL,m2.MASKS,m2.S1/'predictions.csv']},
                  working_model_replaced=False,
                  independent_end_to_end_validation=False)
    (OUT/'m2_wind_experiment.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    yearly=years[(years.scheme=='past_only') & (years.seed==42)]
    lines=['# М2: эксперимент с ветром ERA5','',
           'Фиксированный протокол: [M2_WIND_EXPERIMENT](../docs/M2_WIND_EXPERIMENT.md). '
           'Повторная архивная оценка фиксированных радарных признаков; рабочая модель не заменена.','',
           f"Покрытие: {coverage['used']} из {coverage['labelled']} размеченных сцен до сезона 2022; "
           f"исключено {coverage['excluded']}. Один снимок 2024 года не используется. "
           'Тесты одинаковы для всех вариантов. ERA5 — оценки реанализа, а не местные измерения ветра.','',
           f'Основная проверка: {forward_n} сцен четырёх сезонов 2019–2022. '
           f'Дополнительная: {heldout_n} сцен шести сезонов 2017–2022. '
           'Balanced accuracy — среднее долей правильно распознанной воды и правильно распознанного льда.','',
           '## Среднее по трём seed','',md_table(avg.round(5)),'',
           'Основная метрика — balanced accuracy на past_only: обучение строго на предыдущих сезонах. '
           'leave_year_out дополнительно использует другие, в том числе более поздние сезоны. '
           'Среднее метрик разных seed не означает увеличение размера независимой выборки.','',
           '## Изменение основной метрики: radar_wind минус radar','',
           *[f'- Seed {seed}: {delta*100:+.3f} процентного пункта.' for seed,delta in zip(SEEDS,deltas)],'',
           f"Парный 95%-интервал seed=42, бутстрэп {interval['clusters']} целых сезонов: "
           f"{interval['low']*100:+.3f} … {interval['high']*100:+.3f} п.п. "
           'Мало сезонов; это ориентировочный интервал.','',
           f"Среднее изменение balanced accuracy по трём seed: {np.mean(deltas)*100:+.3f} п.п. "
           'Положительное среднее само по себе не доказывает устойчивого улучшения. '
           'Нужно учитывать интервалы и отдельные сезоны ниже.','',
           '## Каждый последующий сезон, seed=42','',
           md_table(yearly[['variant','hydro_year','n','water','balanced_accuracy','f1_water','brier','water_as_ice','ice_as_water']].round(5)),'',
           'water_as_ice — вода ошибочно названа льдом; ice_as_water — лёд ошибочно назван водой.','',
           '## Ошибки по скорости ветра, past_only, seed=42','',md_table(wind.round(5)),'',
           '## Ограничения и применение','',
           'Маски русла построены до эксперимента по общему архиву. Нет полного исключения тестового '
           'сезона из подготовки снимков, признаки округлены, метки сами имеют ошибку. '
           'Поэтому абсолютное качество не является новой независимой проверкой полного М2.','',
           'Скорость и направление получены из ERA5 задним числом. В час съёмки они могли ещё не быть '
           'опубликованы; нужен отдельный опыт с оперативными погодными выпусками. Ветер на сетке '
           'не измеряет локальные волны, течение подо льдом, толщину льда или допуск дороги.','',
           'Веса, маски и прогнозы действующего М2 сохранены. Результат допускает только исследовательский '
           'вывод; автоматическая замена модели и предупреждения по порогу ветра не выполнены.','',
           'Исходные ответы и SHA-256: data/s1/wind/. Полные прогнозы, метрики, сезоны и '
           'диапазоны ветра: reports/m2_wind_*. Повторение:','',
           '```bash','python scripts/download_m2_wind.py --fetch  # только при отсутствии сохранённых ответов',
           'python scripts/download_m2_wind.py          # повторное сопоставление без сети',
           'python scripts/experiment_m2_wind.py','```','']
    (OUT/'m2_wind_experiment.md').write_text('\n'.join(lines),encoding='utf-8')
    print(avg.to_string(index=False),flush=True)
    print('Primary deltas',deltas,'CI',interval,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    protected=[m2.MODEL,m2.MASKS,m2.S1/'predictions.csv',m2.S1/'index.csv']
    original={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    data,coverage=load_inputs()
    if args.report_only:
        predictions=pd.read_csv(OUT/'m2_wind_predictions.csv.gz')
    else:
        predictions=evaluate(data)
    if any(hashlib.sha256(path.read_bytes()).hexdigest()!=digest for path,digest in original.items()):
        raise ValueError('Рабочие артефакты М2 изменились во время опыта')
    report(predictions,coverage)


if __name__=='__main__':
    main()
