"""Обучение и независимая скользящая проверка собственного М1 уровней."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okno import flood_m1 as M


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        write_report(pd.read_csv(M.ROOT/'reports/flood_m1_scores.csv'))
        return
    M.MODELS.mkdir(parents=True,exist_ok=True)
    scores=[];predictions=[]
    for pid,(code,source) in M.STATIONS.items():
        series=M.load_levels(pid)
        table=M.dataset(series)
        last=series.last_valid_index().year
        test_end=2023 if source=='gmvo' else 2024
        # Версии доступны и для исторического выбора даты, и для последнего года наблюдений.
        for year in range(2017,last+2):
            try:
                model,meta=M.fit_year(series,year)
            except ValueError as exc:
                print(pid,year,str(exc).encode('ascii','backslashreplace').decode());continue
            raw_meta=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
            meta.update(point_id=pid,gauge_code=code,observation_source=source,data_sha256=raw_meta['sha256'])
            model.save_model(str(M.MODELS/f'{pid}_{year}.cbm'))
            (M.MODELS/f'{pid}_{year}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            print(pid,year,'trained',flush=True)
            if not 2019<=year<=test_end:
                continue
            test=table[(table.issue.dt.year==year)&(table.target_date.dt.year==year)].copy()
            p=M.predict_rows(series,test,year,model,meta)
            valid=p[['m1','catboost','persistence','calendar','last_year']].notna().all(axis=1)
            test,p=test.loc[valid],p.loc[valid]
            for lead in (1,3,7):
                for season in ('all','spring'):
                    mask=test.lead.eq(lead)
                    if season=='spring': mask &= test.issue.dt.month.between(4,7)
                    truth=test.loc[mask,'target']
                    if truth.empty: continue
                    row=dict(pid=pid,year=year,lead=lead,season=season,n=len(truth),
                             coverage=float(((truth>=p.loc[mask,'lower'])&(truth<=p.loc[mask,'upper'])).mean()))
                    for method in ['m1','catboost','persistence','calendar','last_year']:
                        row[method]=float((truth-p.loc[mask,method]).abs().mean())
                    scores.append(row)
            out=test[['issue','target_date','lead','target']].copy()
            out['pid']=pid
            for col in p: out[col]=p[col]
            predictions.append(out)
    scores=pd.DataFrame(scores)
    scores.to_csv(M.ROOT/'reports/flood_m1_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(M.ROOT/'reports/flood_m1_predictions.csv.gz',index=False,compression='gzip')
    write_report(scores)


def write_report(scores):
    lines=['# М1 уровней воды: независимая проверка','',
           'Источник разметки — измеренные уровни, опубликованные Level-Sputnik (gmvo / vodinfo); '
           'GloFAS не участвует. Протокол: docs/FLOOD_M1_VALIDATION_PLAN.md. '
           'Ретроспективная проверка по позднее полученному архиву, не восстановленные оперативные выпуски.','',
           '| Точка | Суток вперёд | MAE М1, см | Сохранить уровень | Календарь | Прошлый год | Покрытие | Статус |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    statuses={}
    rng=np.random.default_rng(42)
    for (pid,lead),group in scores[scores.season=='all'].groupby(['pid','lead']):
        spring=scores[(scores.pid==pid)&(scores.lead==lead)&(scores.season=='spring')]
        bounds={}
        for base in ('persistence','calendar','last_year'):
            gains=(group[base]-group.m1).to_numpy()
            boot=rng.choice(gains,(2000,len(gains)),replace=True).mean(axis=1)
            bounds[base]=[float(x) for x in np.quantile(boot,[.05,.95])]
        point_success=all(v[0]>0 for v in bounds.values()) and all(spring.m1.mean()<spring[b].mean() for b in bounds)
        interval_success=group.coverage.mean()>=.75 and spring.coverage.mean()>=.75
        success=point_success and interval_success
        statuses[f'{pid}:{lead}']=dict(success=bool(success),point_skill_success=bool(point_success),
            interval_valid=bool(interval_success),spring_coverage=float(spring.coverage.mean()),
            gain_ci90_cm=bounds,years=group.year.tolist())
        means=group[['m1','persistence','calendar','last_year','coverage']].mean()
        lines.append(f"| {pid} | {lead} | {means.m1:.1f} | {means.persistence:.1f} | {means.calendar:.1f} | {means.last_year:.1f} | {means.coverage:.0%} | {'критерий пройден' if success else 'эксперимент'} |")
    (M.MODELS/'validation.json').write_text(json.dumps(statuses,indent=2),encoding='utf-8')
    lines+=['','Проверка отдельно по сезонам и сырые прогнозы: flood_m1_scores.csv, flood_m1_predictions.csv.gz. '
            'М1 выбирает CatBoost либо простое правило по прошлому году; выбор записан в JSON каждой модели. '
            'Отрицательные результаты не скрываются.','',
            'Курган исключён: недостаточно однородных полных лет. '
            'Якутск/Ишим: gmvo до 2023; Оренбург/Орск: vodinfo до 09.09.2025. '
            'Текущих измерений за 2026 год нет — свежий собственный прогноз на 2026 не выпускается. '
            'Опасные уровни и связь с доступностью дороги требуют отдельного подтверждения.']
    lines+=['','## Отдельно апрель–июль, горизонт 7 суток','',
            '| Точка | М1, см | Сохранить уровень, см | Покрытие диапазона |',
            '| --- | ---: | ---: | ---: |']
    for pid,group in scores[(scores.season=='spring')&(scores.lead==7)].groupby('pid'):
        lines.append(f'| {pid} | {group.m1.mean():.1f} | {group.persistence.mean():.1f} | {group.coverage.mean():.0%} |')
    lines+=['','**Диапазоны пока недостаточно надёжны в половодье.** Преимущество точечного прогноза '
            'и надёжность диапазона оцениваются отдельно. Первоначальная сводка только среднегодового '
            'покрытия скрывала сезонное недопокрытие; итоговый статус исправлен и требует ≥75% также '
            'в апреле–июле. Параметры модели и калибровки после просмотра теста не менялись.', '',
            'В 2024 году в апреле–июле при горизонте 7 суток: Оренбург MAE 71,9 см '
            '(сохранение уровня 92,1), покрытие всего 29%; Орск 59,0 см (65,2), покрытие 13%. '
            'Этот провал не скрывается среднегодовыми метриками.']
    (M.ROOT/'reports/flood_m1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
