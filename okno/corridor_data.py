"""Verified operational events and strictly past-only satellite/road conversion."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EVENTS=ROOT/"data/rosstat_prices/yakutsk_mchs_crossing_dates.csv"


def load_events():
    events=pd.read_csv(EVENTS)
    for phase in ("close","open"):
        flag=f"verified_{phase}"
        events[flag]=events[flag].astype(str).str.lower().eq("true")
        col="ice_road_close_date" if phase=="close" else "ferry_open_date"
        url="ice_close_source_url" if phase=="close" else "ferry_open_source_url"
        pub="ice_close_published_date" if phase=="close" else "ferry_open_published_date"
        if (events[flag] & (~events[url].fillna("").str.startswith("https://") | events[col].isna()
                           | pd.to_datetime(events[pub],errors='coerce').isna())).any():
            raise ValueError("Verified event lacks date or source URL")
    return events


def past_shifts(year,events=None,sat=None,min_years=2):
    """Estimate each phase only from earlier verified paired observations."""
    events=load_events() if events is None else events
    sat=pd.read_csv(ROOT/"data/processed/yakutsk_sat_seasons.csv") if sat is None else sat
    pairs=events[events.hydro_year<year].merge(sat[["hydro_year","sat_breakup_date"]],on="hydro_year")
    out={}
    for phase,col in (("close","ice_road_close_date"),("open","ferry_open_date")):
        pub="ice_close_published_date" if phase=="close" else "ferry_open_published_date"
        published=pd.to_datetime(pairs[pub],errors='coerce')<pd.Timestamp(year,1,1)
        part=pairs[pairs[f"verified_{phase}"] & pairs[col].notna() & published]
        shift=(pd.to_datetime(part[col])-pd.to_datetime(part.sat_breakup_date)).dt.days
        out[phase]=float(shift.median()) if len(shift)>=min_years else None
        out[f"{phase}_years"]=part.hydro_year.astype(int).tolist()
    return out


def verified_years():
    e=load_events()
    return e.loc[e.verified_close & e.verified_open,"hydro_year"].astype(int).tolist()


def case_inputs(year,rasputitsa_days=38):
    if year=="synthetic":
        fc=pd.DataFrame([dict(issue_day=10,planned_close_day=49.,planned_open_day=88.)])
        return dict(label="Искусственный сценарий",start_date=None,ac=50,ao=50+rasputitsa_days,
                    cc=55,co=55+rasputitsa_days,qc=49,qo=55+rasputitsa_days,forecasts=fc,
                    model_available=True,note="Все даты и успешное раннее предупреждение заданы вручную; это не результат CatBoost.",sources=[])
    year=int(year)
    events=load_events()
    row=events[events.hydro_year==year].iloc[0]
    if not (row.verified_close and row.verified_open):
        raise ValueError("Operational dates are not verified for this year")
    start=pd.Timestamp(year,3,1)
    prior=events[events.hydro_year<year]
    def day_for_this_year(value):
        t=pd.Timestamp(value)
        return (pd.Timestamp(year,t.month,t.day)-start).days
    def prior_date(col,flag,q,fallback):
        pub='ice_close_published_date' if flag=='verified_close' else 'ferry_open_published_date'
        published=pd.to_datetime(prior[pub],errors='coerce')<pd.Timestamp(year,1,1)
        vals=prior.loc[prior[flag] & prior[col].notna() & published,col].map(day_for_this_year)
        return float(vals.quantile(q)) if len(vals) else (pd.Timestamp(f"{year}-{fallback}")-start).days
    cc=prior_date("ice_road_close_date","verified_close",.5,"04-15")
    qc=prior_date("ice_road_close_date","verified_close",.2,"04-15")
    co=prior_date("ferry_open_date","verified_open",.5,"05-25")
    path=ROOT/"reports/window_forecasts_a3.csv"
    fc=pd.read_csv(path) if path.exists() else pd.DataFrame()
    shifts=past_shifts(year,events)
    available=not fc.empty and year in set(fc.hydro_year) and shifts["close"] is not None and shifts["open"] is not None
    if available:
        fc=fc[fc.hydro_year==year].copy()
        origin=(pd.Timestamp(year-1,9,1)-start).days
        fc["issue_day"]=(pd.to_datetime(fc.issue_date)-start).dt.days
        fc["planned_close_day"]=origin+fc.q20+shifts["close"]
        fc["planned_open_day"]=np.maximum(fc.planned_close_day+1,origin+fc.q50+shifts["open"])
    else:
        fc=pd.DataFrame()
    return dict(label=f"{year}: подтверждённые даты переправы",start_date=start.date().isoformat(),
                ac=(pd.Timestamp(row.ice_road_close_date)-start).days,
                ao=(pd.Timestamp(row.ferry_open_date)-start).days,
                cc=cc,co=co,qc=qc,qo=co,forecasts=fc,model_available=available,
                note="Остальные параметры — допущения. Календарь — только прошлые подтверждённые годы; при отсутствии истории 15 апреля / 25 мая — явно заданный исходный сценарий. "
                     +(f"Преобразование спутника в дорожные даты: прошлые годы {shifts['close_years']} / {shifts['open_years']}." if available else "Для ML не хватает прошлых подтверждённых фаз: используется календарный q20."),
                sources=[row.ice_close_source_url,row.ferry_open_source_url])
