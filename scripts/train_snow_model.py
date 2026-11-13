"""A/B snow-depth experiment, chronological year splits and independent old-model fallback."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from okno import features as F,snow
from okno.paths import PROCESSED,MODELS,REPORTS
from okno.regions import all_gauges
from okno.northeast import STATIONS
from okno.report import md_table

PARAMS=dict(iterations=400,depth=4,learning_rate=.04,l2_leaf_reg=5,thread_count=4,
            random_seed=42,verbose=False,allow_writing_files=False,
            loss_function='MultiQuantile:alpha=0.1,0.5,0.9')


def dataset(kind,codes=None,include_satellite=True):
    seasons=pd.read_csv(PROCESSED/'hydro_ice_seasons.csv')
    gauges=pd.read_csv(PROCESSED/'gauges.csv')
    rows=[]
    for g in gauges[gauges.gauge_code.isin(all_gauges() if codes is None else codes)].itertuples():
        pid=f'gauge_{g.gauge_code}'
        col=f'open_{F.OPEN_LOAD_T}t_day' if kind=='open' else 'melt_start_day'
        s=seasons[(seasons.gauge_code==g.gauge_code)&seasons.valid]
        targets=pd.DataFrame(dict(target_day=s[col].to_numpy(),freezeup_day=s.freezeup_day.to_numpy()),index=s.hydro_year)
        if targets.target_day.notna().sum()<6:
            continue
        if not (snow.DATA/'snow_depth'/f'{pid}.csv').exists():
            raise RuntimeError(f'Snow-depth series not downloaded: {pid}')
        # A fixed physical coefficient avoids the old leave-season-out alpha
        # using later observations in an otherwise chronological experiment.
        rows+=F.point_rows(pid,dict(lat=g.lat,lon=g.lon),targets,
                           {int(y):F.ALPHA_DEFAULT for y in targets.index},kind)
    if kind=='close' and include_satellite:
        sat=pd.read_csv(PROCESSED/'yakutsk_sat_seasons.csv')
        targets=pd.DataFrame(dict(target_day=sat.sat_breakup_day.to_numpy(),freezeup_day=sat.sat_freezeup_day.to_numpy()),index=sat.hydro_year)
        rows+=F.point_rows('yakutsk_crossing',dict(lat=62.,lon=129.82,is_satellite=1),targets,
                           {int(y):F.ALPHA_DEFAULT for y in targets.index},kind)
    df=pd.DataFrame(rows)
    extra=pd.DataFrame([snow.features(r.point_id,r.hydro_year,r.issue_day) for r in df.itertuples()],index=df.index)
    return pd.concat([df,extra],axis=1)


def fit(frame,columns):
    model=CatBoostRegressor(**PARAMS)
    model.fit(frame[columns],frame.target_day-frame.cal_target)
    return model


def main():
    metadata=dict(features={},feature_importance={},training_coverage={},baseline_features=F.FEATURES,issue_ranges={},models={},validation={},sources=dict(ground='Open-Meteo ERA5-Land snow depth, nominal 5-day lag',
        measured='PANGAEA.925460 river snow depth; measurement dates only, publication times unavailable'),
        physical_alpha=F.ALPHA_DEFAULT,parameters=PARAMS)
    predictions=[]
    for kind,years,fixed_issue in [('open',range(2010,2019),72),('close',range(2010,2025),192)]:
        df=dataset(kind);columns=F.FEATURES[kind]+snow.FEATURES
        metadata['issue_ranges'][kind]=[int(df.issue_day.min()),int(df.issue_day.max())]
        metadata['features'][kind]=columns
        metadata['training_coverage'][kind]=dict(rows=len(df),points=df.point_id.nunique(),
            first_year=int(df.hydro_year.min()),last_year=int(df.hydro_year.max()),
            ground_snow_rows=int(df.snow_ground_cm.notna().sum()),
            measured_river_snow_rows=int(df.snow_on_ice_cm.notna().sum()))
        df.to_csv(PROCESSED/f'm1_snow_dataset_{kind}.csv',index=False)
        print(kind,len(df),'rows',df.point_id.nunique(),'points',flush=True)
        parts=[]
        reference=None
        for year in years:
            train=df[df.hydro_year<year]
            test=df[(df.hydro_year==year)&(df.issue_day==fixed_issue)]
            if test.empty:
                continue
            baseline=fit(train,F.FEATURES[kind]);model=fit(train,columns)
            p0=np.sort(np.asarray(baseline.predict(test[F.FEATURES[kind]])),axis=1)+test.cal_target.to_numpy()[:,None]
            p1=np.sort(np.asarray(model.predict(test[columns])),axis=1)+test.cal_target.to_numpy()[:,None]
            ground_only=test[columns].copy()
            ground_only[['snow_on_ice_cm','snow_on_ice_age_days']]=np.nan
            ground_only['snow_on_ice_observed']=0.
            pg=np.sort(np.asarray(model.predict(ground_only)),axis=1)+test.cal_target.to_numpy()[:,None]
            p0=np.maximum(p0,fixed_issue);p1=np.maximum(p1,fixed_issue);pg=np.maximum(pg,fixed_issue)
            result=test[['point_id','hydro_year','issue_day','target_day','cal_target']].copy()
            result['kind']=kind;result['train_max_year']=int(train.hydro_year.max())
            result['source_group']=result.point_id.map(lambda p:'satellite' if p=='yakutsk_crossing' else
                'new_gauges' if int(p.split('_')[1]) in STATIONS else 'existing_gauges')
            for j,q in enumerate([10,50,90]):
                result[f'base_q{q}']=p0[:,j];result[f'snow_q{q}']=p1[:,j]
                result[f'without_river_q{q}']=pg[:,j]
            parts.append(result)
            print(kind,year,'test',len(test),'base',round(abs(p0[:,1]-test.target_day).mean(),3),
                  'snow',round(abs(p1[:,1]-test.target_day).mean(),3),flush=True)
            if reference is None:
                filename=f'm1_{kind}_snow_reference.cbm';model.save_model(str(MODELS/filename))
                basefile=f'm1_{kind}_updated_reference.cbm';baseline.save_model(str(MODELS/basefile))
                reference=dict(file=filename,baseline_file=basefile,train_max_year=int(train.hydro_year.max()),
                               correction_days=0.,baseline_correction_days=0.)
        checked=pd.concat(parts,ignore_index=True)
        e0=float((checked.base_q50-checked.target_day).abs().mean())
        e1=float((checked.snow_q50-checked.target_day).abs().mean())
        metadata['validation'][kind]=dict(n=len(checked),seasons=checked.hydro_year.nunique(),
            issue_day=fixed_issue,first_year=int(checked.hydro_year.min()),last_year=int(checked.hydro_year.max()),
            baseline_mae_days=e0,snow_mae_days=e1,improvement_pct=100*(1-e1/e0),
            sample_scope='future events at a fixed season-day; new gauges have no closing-phase labels')
        metadata['validation'][kind]['groups']={}
        for group,part in checked.groupby('source_group'):
            a=float((part.base_q50-part.target_day).abs().mean());b=float((part.snow_q50-part.target_day).abs().mean())
            metadata['validation'][kind]['groups'][group]=dict(n=len(part),seasons=part.hydro_year.nunique(),
                first_year=int(part.hydro_year.min()),last_year=int(part.hydro_year.max()),
                baseline_mae_days=a,snow_mae_days=b,improvement_pct=100*(1-b/a),
                validated_points=sorted(part.point_id.unique().tolist()))
        for part,quality in [(checked,metadata['validation'][kind])]+[
                (part,metadata['validation'][kind]['groups'][group]) for group,part in checked.groupby('source_group')]:
            a=float((part.base_q50-part.target_day).abs().mean())
            b=float((part.without_river_q50-part.target_day).abs().mean())
            quality['without_river']=dict(n=len(part),seasons=part.hydro_year.nunique(),
                first_year=int(part.hydro_year.min()),last_year=int(part.hydro_year.max()),
                baseline_mae_days=a,snow_mae_days=b,improvement_pct=100*(1-b/a),
                validated_points=sorted(part.point_id.unique().tolist()))
        correction=float(max(0,np.quantile(np.maximum(checked.snow_q10-checked.target_day,
            checked.target_day-checked.snow_q90),.8,method='higher')))
        full=fit(df,columns);filename=f'm1_{kind}_snow.cbm';full.save_model(str(MODELS/filename))
        metadata['feature_importance'][kind]=dict(zip(columns,map(float,full.get_feature_importance())))
        baseline=fit(df,F.FEATURES[kind]);basefile=f'm1_{kind}_updated.cbm';baseline.save_model(str(MODELS/basefile))
        base_correction=float(max(0,np.quantile(np.maximum(checked.base_q10-checked.target_day,
            checked.target_day-checked.base_q90),.8,method='higher')))
        metadata['models'][kind]=[reference,dict(file=filename,baseline_file=basefile,
            train_max_year=int(df.hydro_year.max()),correction_days=correction,baseline_correction_days=base_correction)]
        predictions.append(checked)
    root=Path(__file__).resolve().parents[1]
    artifacts=[MODELS/m[key] for versions in metadata['models'].values() for m in versions
               for key in ('file','baseline_file')]
    metadata['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__).resolve(),root/'okno/snow.py',root/'okno/features.py',root/'okno/ice.py',
        root/'okno/northeast.py',root/'okno/forecast.py',
        snow.DATA/'raw/pangaea_river_ice/ice_snow.tsv',
        snow.DATA/'raw/pangaea_river_ice/stations.tsv',
        PROCESSED/'gauges.csv',PROCESSED/'hydro_ice_seasons.csv',PROCESSED/'hydro_snow_on_ice.csv',
        PROCESSED/'m1_snow_dataset_open.csv',PROCESSED/'m1_snow_dataset_close.csv',*artifacts]}
    (MODELS/'m1_snow.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    pd.concat(predictions,ignore_index=True).to_csv(REPORTS/'snow_backtest.csv',index=False)
    rows=[]
    group_names=dict(existing_gauges='Прежние гидропосты',new_gauges='Новые гидропосты',satellite='Спутник: Якутск')
    for kind,quality in metadata['validation'].items():
        for group,s in [('все',quality),*quality['groups'].items()]:
            rows.append({'Событие':'Открытие' if kind=='open' else 'Таяние / вскрытие',
                'Группа':group_names.get(group,group),'Проверенных прогнозов':s['n'],'Сезонов':s['seasons'],
                'MAE без толщины снега, сут':round(s['baseline_mae_days'],3),
                'MAE со снегом, сут':round(s['snow_mae_days'],3),'Изменение ошибки, %':round(-s['improvement_pct'],2),
                'MAE при отсутствии замера на реке, сут':round(s['without_river']['snow_mae_days'],3),
                'Изменение без замера на реке, %':round(-s['without_river']['improvement_pct'],2)})
    lines=['# Толщина снега: проверка отдельной модели','',md_table(pd.DataFrame(rows)),'',
        'Обе версии обучены на одинаковых прошлых сезонах; отличаются только добавленными признаками толщины снега. '
        'Это не сравнение с замороженными моделями старого релиза. Выигрыш не задан заранее. '
        'Для будущих сезонов снежная поправка применяется только в проверенной группе источников '
        '(прежние гидропосты, новые гидропосты или спутник), где средняя ошибка снизилась. '
        'Проверка выполнена на фиксированном дне сезона: 72 для открытия и 192 для закрытия, только до будущего события.','',
        'Отдельно каждый прогноз пересчитан с недоступным замером снега на реке '
        '(поля замера и его возраста пусты, индикатор наличия равен нулю). '
        'Если текущего замера нет, применение снега зависит именно от результата этой проверки. '
        'Старая запись 2018 года не подставляется в текущий сезон.','',
        'В каждом проверяемом сезоне обучение использует только предыдущие годы; календарь и климатология '
        'также строятся по прошлым сезонам. В обеих версиях коэффициент Стефана фиксирован на 1,8, '
        'чтобы калибровка по более поздним замерам не попала в прошлый прогноз. '
        'Снижение средней ошибки на этой небольшой выборке не доказывает улучшение на каждом посту.','',
        'Открытие: 2010–2018 (после 2012 только новые посты). Закрытие: 2010–2012 для прежних гидропостов '
        'и 2010–2024 для спутниковой точки Якутска. Результаты разделены по происхождению цели; '
        'у новых постов в источнике нет дат начала таяния. Эти отсутствующие даты не восстановлены как факты. '
        'Спутниковое вскрытие тоже не равно официальному закрытию дороги.','',
        'В обучении закрытия нет совместных наблюдений снега на реке и целевой ледовой фазы: '
        'здесь проверяется только модельный снег на суше. Замеры снега на льду участвуют в модели открытия новых постов.','',
        'Толщина снега на суше из ERA5-Land отделена от замеров снега на реке. '
        'Реанализ доступен с номинальным лагом пять суток; замеры берутся только не позднее даты решения, '
        'не старше 30 суток и только из текущего сезона. Снег не заменяется суммой снегопадов.','',
        'Исторические версии оперативных данных и даты публикации замеров не восстановлены. '
        'Это исследовательская реконструкция, не архив реально выпущенных прогнозов. '
        'ERA5-Land описывает снег на суше и может завышать его высоту; прогноз экспериментальный. '
        'Достижение расчётной толщины льда не равно официальному открытию дороги.','',
        'Для исторических сезонов применяется отдельная модель, обученная до этого периода; '
        'это фиксированная референсная модель до 2010 года, а не отдельные модели каждой строки бэктеста. '
        'Финальная модель используется только после последнего года её обучения. '
        'Референсные интервалы не откалиброваны, поэтому 80% покрытия не обещается.','',
        '- [Результаты по каждому прогнозу](snow_backtest.csv)',
        '- [Параметры и контрольные суммы](../models/m1_snow.json)']
    (REPORTS/'snow_model.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
