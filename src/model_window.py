"""Five physical-breakup quantiles with expanding past-only training.

python src/model_window.py

Targets are satellite breakup, NOT legal closure or ferry reopening. Weather is
ERA5 reanalysis with a nominal five-day lag, not archived operational forecasts.
"""
from pathlib import Path
import sys
import json
import hashlib
import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from okno.features import FEATURES,past_clim,season_matrix,season_features,thaw_thresholds
from okno.ice import to_season_day
from okno.report import md_table

QUANTILES=(.1,.2,.5,.8,.9)
LAG=5


def feature_row(sat,T,S,year,issue_date):
    """Current target is deliberately never read; targets/calendars use y < year."""
    past=sat[sat.hydro_year<year].set_index("hydro_year")
    if len(past)<3:
        return None
    cutoff=pd.Timestamp(issue_date)-pd.Timedelta(days=LAG)
    day=int(to_season_day(cutoff,year))
    clim=past_clim(T,year)
    if clim is None or year not in T.index:
        return None
    # Estimate the physical threshold on completed prior seasons only.
    theta=thaw_thresholds(T[T.index<year],past.sat_breakup_day)
    prior_theta=[v for v in theta.values() if np.isfinite(v)]
    # thaw_thresholds itself is rolling: use the latest prior estimate.
    threshold=prior_theta[-1] if prior_theta else np.nan
    cal=float(past.sat_breakup_day.median())
    current=sat[sat.hydro_year==year]
    fz=float(current.iloc[0].sat_freezeup_day) if len(current) else np.nan
    t,s=T.loc[T.index<=year].copy(),S.loc[S.index<=year].copy()
    t.loc[year,t.columns>day]=np.nan
    s.loc[year,s.columns>day]=np.nan
    # 1.8 is a fixed hypothesis, not a coefficient fitted using future years.
    f=season_features(t,s,year,day,clim,1.8,fz,threshold)
    phys=f.pop("phys_close")
    f.pop("phys_open")
    phys=float(phys) if np.isfinite(phys) else cal
    return dict(issue_day=day,lat=62.,is_satellite=1,cal_target=cal,
                phys_pred=phys,phys_minus_cal=phys-cal,**f)


def dates_for_year(year):
    # Fixed issue schedule, chosen without consulting the held-out event date.
    dates=set(pd.date_range(f"{year}-02-01",f"{year}-05-31",freq="7D"))
    dates.add(pd.Timestamp(year,3,10))
    return sorted(dates)


def build_dataset(sat,T,S):
    rows=[]
    for row in sat.sort_values("hydro_year").itertuples():
        y=int(row.hydro_year)
        for date in dates_for_year(y):
            f=feature_row(sat,T,S,y,date)
            # Past training records are used only while breakup was not yet observed.
            if f is None or f["issue_day"]>=row.sat_breakup_day:
                continue
            rows.append(dict(hydro_year=y,issue_date=date.date().isoformat(),
                             weather_cutoff_date=(date-pd.Timedelta(days=LAG)).date().isoformat(),
                             target_day=float(row.sat_breakup_day),**f))
    return pd.DataFrame(rows)


def fit_model(train):
    model=CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.2,0.5,0.8,0.9",
        iterations=400,depth=3,learning_rate=.03,l2_leaf_reg=5,random_seed=42,
        verbose=False,allow_writing_files=False,thread_count=2)
    model.fit(train[FEATURES["close"]],train.target_day-train.cal_target)
    return model


def offset(scores,coverage):
    if not scores:
        return 0.
    values=np.sort(scores)
    return max(0.,float(values[min(len(values)-1,math.ceil((len(values)+1)*coverage)-1)]))


def walk_forward(data):
    predictions=[]
    scores80,scores60=[],[]
    for year in sorted(data.hydro_year.unique()):
        train=data[data.hydro_year<year]
        if train.hydro_year.nunique()<3:
            continue
        test=data[data.hydro_year==year].copy()
        model=fit_model(train)
        raw=np.sort(np.asarray(model.predict(test[FEATURES["close"]])),axis=1)+test.cal_target.to_numpy()[:,None]
        c80,c60=offset(scores80,.8),offset(scores60,.6)
        q=raw.copy()
        q[:,0]-=c80;q[:,1]-=c60;q[:,3]+=c60;q[:,4]+=c80
        q[:,0]=np.minimum(q[:,0],q[:,1]);q[:,4]=np.maximum(q[:,4],q[:,3])
        for i,name in enumerate(("q10","q20","q50","q80","q90")):
            test[name]=q[:,i]
            test[name+"_raw"]=raw[:,i]
        test["train_max_year"]=int(train.hydro_year.max())
        test["calibration_max_year"]=int(year-1) if scores80 else np.nan
        predictions.append(test)
        # One fixed-date residual per completed year avoids pretending that
        # weekly predictions of the same target are independent seasons.
        fixed=test[test.issue_date.eq(f"{year}-03-10")]
        if len(fixed):
            r=fixed.iloc[0]
            scores80.append(max(r.q10_raw-r.target_day,r.target_day-r.q90_raw))
            scores60.append(max(r.q20_raw-r.target_day,r.target_day-r.q80_raw))
        print(f"Окно А3: {year}, обучение до {int(train.hydro_year.max())}",flush=True)
    return pd.concat(predictions,ignore_index=True)


def main():
    sat=pd.read_csv(ROOT/"data/processed/yakutsk_sat_seasons.csv")
    T,S=season_matrix("yakutsk_crossing","t"),season_matrix("yakutsk_crossing","snow_cm")
    data=build_dataset(sat,T,S)
    forecasts=walk_forward(data)
    forecasts['weather_issue_day']=forecasts.issue_day
    forecasts['issue_day']=[int(to_season_day(date,int(year))) for date,year in zip(forecasts.issue_date,forecasts.hydro_year)]
    reports=ROOT/"reports";models=ROOT/"models"
    forecasts.to_csv(reports/"window_forecasts_a3.csv",index=False,float_format="%.8f")
    fixed=forecasts[forecasts.issue_date.str.endswith("-03-10")].copy()
    fixed["error_calendar_days"]=(fixed.cal_target-fixed.target_day).abs()
    fixed["error_model_days"]=(fixed.q50-fixed.target_day).abs()
    fixed["covered_10_90"]=(fixed.q10<=fixed.target_day)&(fixed.target_day<=fixed.q90)
    fixed.to_csv(reports/"window_backtest_yakutsk_2001_2024.csv",index=False)
    metric=pd.DataFrame([dict(seasons=fixed.hydro_year.nunique(),issue="10 марта каждого года",
        MAE_calendar=float(fixed.error_calendar_days.mean()),MAE_model=float(fixed.error_model_days.mean()),
        coverage=float(fixed.covered_10_90.mean()))])
    metric.to_csv(reports/"window_metrics_by_regime.csv",index=False)
    model=fit_model(data)
    model.save_model(str(models/"window_a3_close.cbm"))
    meta=dict(protocol="expanding past-only; fixed March 10 evaluation",target="satellite physical breakup",
        weather="ERA5 reanalysis; nominal lag 5 days; not operational vintage archive",
        satellite_label_vintage="Retrospective phenology product; historical publication of each label is not verified",
        train_max_year=int(data.hydro_year.max()),physical_alpha_hypothesis=1.8,
        calibration="Only previous walk-forward March 10 residuals; no current/future targets",
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                 ["data/processed/yakutsk_sat_seasons.csv","data/weather/yakutsk_crossing.csv"]})
    (models/"window_a3_conformal.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    lines=["# Модель физического вскрытия: проверка только на прошлых годах","",
        "Прежние LOSO-метрики и выигрыш 28,8% отозваны: они не воспроизводили доступность данных в прошлом.","",
        "Обучение и календарь используют только более ранние сезоны; 2001–2003 не дополнены будущими целями. "
        "Первые годы без достаточной истории пропущены. Пять квантилей относятся к спутниковому событию, "
        "а не к разрешению проезда или открытию паромов.","",md_table(metric.round(3)),"",
        "Оценка — на фиксированную дату 10 марта; выбор даты не зависит от фактического вскрытия. "
        "Калибровка использует только ошибки предыдущих лет. Архив ERA5 с задержкой 5 суток "
        "не восстанавливает исторические версии оперативной погоды; текущий код не был развёрнут в те годы. "
        "Спутниковая фенология — позднейшая обработка архива, доступность этих меток в момент каждого "
        "исторического решения не подтверждена. Это реконструкция, не журнал реально выпущенных прогнозов.","",
        "Полный датированный журнал: [window_forecasts_a3.csv](window_forecasts_a3.csv). "
        "В коридоре журнал применяется только после даты выпуска и при наличии прошлых проверенных "
        "дорожных фаз для преобразования. При недостатке истории ML явно заменяется календарным q20."]
    (reports/"window_model_report.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
