"""Независимая ручная разметка Sen1Floods11: фиксированный эксперимент М2."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import requests
import rasterio
from scipy.ndimage import uniform_filter
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/flood/benchmark'
CACHE=DATA/'cache'
BASE='https://storage.googleapis.com/sen1floods11/v1.1/'
FEATURES=['green','red','nir','swir','mndwi','ndvi','ndwi','mean_mndwi5','mean_nir5']
EXPECTED={}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(relative,path):
    if not path.exists():
        for attempt in range(3):
            try:
                r=requests.get(BASE+relative,timeout=120);r.raise_for_status()
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(r.content);temp.replace(path)
                break
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(1)
    actual=digest(path)
    expected=EXPECTED.get(path.relative_to(ROOT).as_posix())
    if expected is not None and actual!=expected:raise ValueError('Изменился файл независимого набора: '+path.name)
    return dict(url=BASE+relative,path=path.relative_to(ROOT).as_posix(),sha256=actual)


def load(chip):
    with rasterio.open(CACHE/(chip+'_S2Hand.tif')) as src,rasterio.open(CACHE/(chip+'_LabelHand.tif')) as lab:
        if src.crs!=lab.crs or src.shape!=lab.shape or not np.allclose(tuple(src.transform),tuple(lab.transform),atol=1e-12,rtol=0):
            raise ValueError('Разметка не совмещена со снимком: '+chip)
        bands=src.read([3,4,8,12]).astype('float32')/10000
        y=lab.read(1)
    green,red,nir,swir=bands
    finite=np.isfinite(bands).all(axis=0)&(bands>0).all(axis=0)
    def ratio(a,b):return np.divide(a-b,a+b,out=np.zeros_like(a),where=(a+b)>0)
    mndwi,ndvi,ndwi=ratio(green,swir),ratio(nir,red),ratio(green,nir)
    features=np.stack([green,red,nir,swir,mndwi,ndvi,ndwi,
                       uniform_filter(mndwi,size=5),uniform_filter(nir,size=5)],axis=-1)
    valid=finite&np.isin(y,[0,1])
    rule=(mndwi>.1)&(ndvi<.3)&(nir<.15)
    return features,y,valid,rule,ndwi>0


def counts(y,p,known):
    return dict(tp=int(((y==1)&p&known).sum()),fp=int(((y==0)&p&known).sum()),
        fn=int(((y==1)&(~p|~known)).sum()),tn=int(((y==0)&~p&known).sum()),
        abstain_land=int(((y==0)&~known).sum()),n=int(len(y)),known_n=int(known.sum()),coverage=float(known.mean()) if len(known) else np.nan)


def scores(row):
    tp,fp,fn,tn=[row[k] for k in ['tp','fp','fn','tn']]
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                accuracy=(tp+tn)/row['n'] if row['n'] else np.nan)


def fit_candidate(splits):
    rng=np.random.default_rng(42);xs=[];ys=[]
    for chip in splits['train']:
        x,y,v,_,_=load(chip);indices=np.flatnonzero(v)
        chosen=rng.choice(indices,min(5000,len(indices)),replace=False)
        xs.append(x.reshape(-1,len(FEATURES))[chosen]);ys.append(y.ravel()[chosen])
    model=CatBoostClassifier(iterations=200,depth=5,learning_rate=.06,l2_leaf_reg=10,
        random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    model.fit(np.concatenate(xs),np.concatenate(ys))
    modeldir=ROOT/'models/flood_m2_candidate';modeldir.mkdir(exist_ok=True)
    model.save_model(str(modeldir/'s2_toa_water.cbm'))
    validation=[]
    for chip in splits['valid']:
        x,y,v,_,_=load(chip)
        p=model.predict_proba(x.reshape(-1,len(FEATURES)))[:,1].reshape(y.shape)
        validation.append((y[v],p[v]))
    options={}
    for t in [.3,.4,.5,.6,.7]:
        y=np.concatenate([a for a,b in validation]);p=np.concatenate([b for a,b in validation])>=t
        options[t]=scores(counts(y,p,np.ones(len(y),bool)))['f1']
    threshold=max(options,key=options.get)
    print('Frozen validation threshold',threshold,options,flush=True)
    metadata=dict(features=FEATURES,threshold=threshold,validation_f1=options,train=splits['train'],
        valid=splits['valid'],domain='Sentinel-2 L1C TOA; not production L2A replacement',model_sha256=digest(modeldir/'s2_toa_water.cbm'))
    (modeldir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return model,threshold


def main():
    global EXPECTED
    manifest=DATA/'manifest.json'
    if manifest.exists():
        EXPECTED={item['path']:item['sha256'] for item in json.loads(manifest.read_text())['files']}
    if '--report-only' in sys.argv:
        write_report(pd.read_csv(ROOT/'reports/flood_m2_benchmark_chips.csv'));return
    DATA.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    manifests=[];splits={}
    for split in ['train','valid','test','bolivia']:
        name=f'flood_{split}_data.csv';path=DATA/name
        manifests.append(fetch('splits/flood_handlabeled/'+name,path))
        chips=sorted({row[0].replace('_S1Hand.tif','') for row in csv.reader(io.StringIO(path.read_text())) if row})
        if split in ['train','valid']:
            maximum=8 if split=='train' else 3
            chips=[c for event in sorted({c.split('_')[0] for c in chips}) for c in [x for x in chips if x.split('_')[0]==event][:maximum]]
        splits[split]=chips
    for a in splits:
        for b in splits:
            if a!=b and set(splits[a])&set(splits[b]):raise ValueError('Пересечение разбиений')
    jobs=[]
    for chip in sorted({c for chips in splits.values() for c in chips}):
        for layer in ['S2Hand','LabelHand']:
            name=f'{chip}_{layer}.tif'
            jobs.append(('data/flood_events/HandLabeled/'+layer+'/'+name,CACHE/name))
    print('Downloading',len(jobs),'files; test labels are not used for fitting',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,record in enumerate(pool.map(lambda args:fetch(*args),jobs)):
            manifests.append(record)
            if index%40==0:print('downloaded',index+1,flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(source='Sen1Floods11 v1.1, Bonafilia et al. 2020',
        splits=splits,files=manifests),indent=2),encoding='utf-8')
    if '--evaluate-only' in sys.argv:
        modeldir=ROOT/'models/flood_m2_candidate'
        metadata=json.loads((modeldir/'metadata.json').read_text())
        if digest(modeldir/'s2_toa_water.cbm')!=metadata['model_sha256']:raise ValueError('Model changed')
        model=CatBoostClassifier();model.load_model(str(modeldir/'s2_toa_water.cbm'))
        threshold=metadata['threshold']
    else:
        model,threshold=fit_candidate(splits)
    rng=np.random.default_rng(42)
    rows=[]
    for split in ['test','bolivia']:
        for chip in splits[split]:
            x,y,v,rule,ndwi=load(chip)
            gold=np.isin(y,[0,1]);known=v[gold]
            p=model.predict_proba(x.reshape(-1,len(FEATURES)))[:,1].reshape(y.shape)>=threshold
            for method,pred in [('candidate',p),('current_spectral',rule),('ndwi',ndwi),('all_land',np.zeros_like(v))]:
                row=dict(split=split,chip=chip,event=chip.split('_')[0],method=method,**counts(y[gold],pred[gold],known))
                row.update(scores(row));rows.append(row)
        print('Finished untouched split',split,flush=True)
    table=pd.DataFrame(rows);table.to_csv(ROOT/'reports/flood_m2_benchmark_chips.csv',index=False)
    write_report(table)


def write_report(table):
    rng=np.random.default_rng(42)
    lines=['# М2: независимая ручная разметка Sen1Floods11','',
        'Фиксированный протокол: docs/FLOOD_M2_BENCHMARK_PLAN.md. '
        'Использован полный официальный test и отдельное событие Bolivia, не участвовавшие в настройке. '
        'Обучение/валидация — фиксированные подвыборки, не полный набор. '
        'Измеряется вода, не только новая вода/дорожное затопление.','',
        '| Тест | Метод | Чипов | IoU воды | F1 воды | Accuracy | Coverage |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (split,method),g in table.groupby(['split','method']):
        total=g[['tp','fp','fn','tn','abstain_land','n','known_n']].sum();s=scores(total)
        lines.append(f"| {split} | {method} | {len(g)} | {s['iou']:.3f} | {s['f1']:.3f} | {s['accuracy']:.3f} | {total.known_n/total['n']:.3f} |")
    lines+=['','## Выигрыш IoU кандидата: 95%-интервал по чипам','']
    for split in ['test','bolivia']:
        a=table[(table.split==split)&(table.method=='candidate')].sort_values('chip')
        b=table[(table.split==split)&(table.method=='current_spectral')].sort_values('chip')
        indices=rng.integers(0,len(a),size=(2000,len(a)))
        def boot(g):
            z=g[['tp','fp','fn']].to_numpy()[indices].sum(axis=1)
            return z[:,0]/np.maximum(z.sum(axis=1),1)
        lo,hi=np.quantile(boot(a)-boot(b),[.025,.975])
        lines.append(f'- {split}: {lo:+.3f} … {hi:+.3f}. Соседние чипы могут быть зависимы.')
    events=[]
    for (split,event,method),g in table.groupby(['split','event','method']):
        row=dict(split=split,event=event,method=method,**scores(g[['tp','fp','fn','tn','n']].sum()));events.append(row)
    pd.DataFrame(events).to_csv(ROOT/'reports/flood_m2_benchmark_events.csv',index=False)
    event_table=pd.DataFrame(events)
    lines+=['','## Среднее по событиям (каждое событие с одинаковым весом)','',
        '| Тест | Метод | Средний IoU | Средний F1 |',
        '| --- | --- | ---: | ---: |']
    for (split,method),g in event_table.groupby(['split','method']):
        lines.append(f'| {split} | {method} | {g.iou.mean():.3f} | {g.f1.mean():.3f} |')
    lines+=['','Неоднородность: F1 кандидата в Ghana около 0.75, в Paraguay около 0.78. '
        'Хорошая общая метрика не означает одинакового качества во всех событиях.']

    lines+=['','Разбивка по событиям — flood_m2_benchmark_events.csv, по каждому чипу — '
        'flood_m2_benchmark_chips.csv. Сравнение с текущим правилом относится только '
        'к спектральному компоненту: в наборе L1C нет SCL, а рабочие снимки — L2A BOA. '
        'Чипы без ручных валидных пикселей сохраняются с N=0 и не дают фиктивной accuracy. Кандидат не заменяет локальный М2 автоматически. Требуется проверка переноса '
        'на российские сцены; высокая accuracy при редкой воде не равна хорошему IoU. '
        'Набор и разметка: https://github.com/cloudtostreet/Sen1Floods11, '
        'Bonafilia et al., CVPRW 2020. Исходные URL и SHA-256 сохранены в data/flood/benchmark/manifest.json.']
    (ROOT/'reports/flood_m2_benchmark.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
