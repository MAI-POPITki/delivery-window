"""Causal daily hub → rail → Lena → Yakutsk inventory heuristic.

Today's dispatch sees only dated forecasts already available and today's river
status. Costs, demand, allocated capacities and overflow space are assumptions.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorridorConfig:
    daily_demand_tons: float = 25.
    horizon_days: int = 120
    rail_lead_time_days: int = 6
    rail_cap_tons: float = 120.
    rail_cost_per_ton: float = 8500.
    nb_holding_cost_per_ton_day: float = 450.
    crossing_cap_normal_tons: float = 60.
    crossing_cap_alt_tons: float = 8.
    crossing_cost_normal: float = 1800.
    crossing_cost_alt: float = 35000.
    yak_warehouse_cap_tons: float = 650.
    yak_overflow_capacity_tons: float = 0.
    yak_holding_cost_per_ton_day: float = 220.
    yak_overflow_cost_per_ton_day: float = 600.
    shortage_penalty_per_ton_day: float = 12000.
    safety_days: float = 3.
    initial_yak_tons: float = 225.

    def __post_init__(self):
        for name,value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if min(self.daily_demand_tons,self.rail_cap_tons,self.crossing_cap_normal_tons,self.yak_warehouse_cap_tons) <= 0:
            raise ValueError("Demand and normal capacities must be positive")
        if self.horizon_days < 1 or int(self.horizon_days) != self.horizon_days:
            raise ValueError("horizon_days must be a positive integer")
        if int(self.rail_lead_time_days) != self.rail_lead_time_days:
            raise ValueError("rail_lead_time_days must be an integer")
        if self.initial_yak_tons > self.yak_warehouse_cap_tons+self.yak_overflow_capacity_tons:
            raise ValueError("Initial stock does not fit available warehouse space")

    def scaled(self,name,factor):
        values=asdict(self)
        value=values[name]*factor
        if isinstance(values[name],int):
            value=max(0 if name=="rail_lead_time_days" else 1,int(round(value)))
        values[name]=value
        return CorridorConfig(**values)


SENSITIVITY_PARAMS=(
    ("daily_demand_tons","Спрос"),("rail_lead_time_days","Время ж/д"),
    ("rail_cap_tons","Лимит ж/д"),("rail_cost_per_ton","Тариф ж/д"),
    ("nb_holding_cost_per_ton_day","Хранение Н. Бестях"),
    ("crossing_cap_normal_tons","Лимит штатной переправы"),
    ("crossing_cap_alt_tons","Лимит резерва"),("crossing_cost_normal","Штатный тариф"),
    ("crossing_cost_alt","Резервный тариф"),("yak_warehouse_cap_tons","Ёмкость ТЛЦ"),
    ("yak_overflow_capacity_tons","Доступный дополнительный склад"),
    ("yak_holding_cost_per_ton_day","Хранение ТЛЦ"),
    ("yak_overflow_cost_per_ton_day","Хранение доп. склада"),
    ("shortage_penalty_per_ton_day","Штраф ожидания"),("safety_days","Страховой запас"),
    ("initial_yak_tons","Начальный запас"))
POLICY_LABELS={"reactive":"Без предзавоза","calendar":"Предзавоз: календарная медиана",
               "calendar_q20":"Предзавоз: календарный q20","ml":"Предзавоз: ML q20"}


def compute_prestock_schedule(cfg,planned_close_day,planned_open_day):
    gap=max(0.,float(planned_open_day)-float(planned_close_day))
    spare=min(cfg.crossing_cap_normal_tons,cfg.rail_cap_tons)-cfg.daily_demand_tons
    target=gap*cfg.daily_demand_tons
    duration=math.ceil(target/spare) if spare>0 and target>0 else 0 if not target else None
    start=float(planned_close_day)-duration if duration is not None else None
    return dict(expected_closed_days=gap,target_prestock_tons=target,spare_crossing_cap_tons=spare,
                prestock_duration_days=duration,prestock_start_nb_day=start,
                hub_dispatch_start_day=start-cfg.rail_lead_time_days if start is not None else None,
                feasible=duration is not None)


def _forecast_table(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["issue_day","planned_close_day","planned_open_day"])
    f=frame.copy().sort_values("issue_day")
    if not np.isfinite(f[["issue_day","planned_close_day","planned_open_day"]].to_numpy(dtype=float)).all() or f.issue_day.duplicated().any():
        raise ValueError("Forecast dates must be finite and unique")
    if (f.planned_open_day<f.planned_close_day).any():
        raise ValueError("Forecast reopening precedes closure")
    if {"train_max_year","hydro_year"} <= set(f) and (f.train_max_year>=f.hydro_year).any():
        raise ValueError("Forecast training overlaps test season")
    if {"calibration_max_year","hydro_year"} <= set(f) and (f.calibration_max_year>=f.hydro_year).any():
        raise ValueError("Forecast calibration overlaps test season")
    # The dispatch policy receives no future target columns from the study file.
    return f[["issue_day","planned_close_day","planned_open_day"]]


def simulate_corridor(cfg,actual_close_day,actual_open_day,planned_close_day,planned_open_day,
                      use_prestock=True,return_daily=False,*,forecasts=None):
    """Execute only current hub dispatch. Arrival is dispatch + rail lead.

    Unsatisfied demand persists as backlog; penalty is its daily area in t·days.
    Receiving space is finite. Opening inventory and terminal cargo are exposed.
    """
    if not 0<=actual_close_day<=actual_open_day or planned_open_day<planned_close_day:
        raise ValueError("Invalid river window")
    f=_forecast_table(forecasts)
    d,H,lag=cfg.daily_demand_tons,cfg.horizon_days,cfg.rail_lead_time_days
    space=cfg.yak_warehouse_cap_tons+cfg.yak_overflow_capacity_tons
    initial=cfg.initial_yak_tons
    total_order=d*H
    hub,nb,yak,backlog=total_order,0.,initial,0.
    arrivals=np.zeros(H+lag+1)
    in_transit=0.
    seen_closed=False
    records=[]
    for t in range(H):
        river_open=t<actual_close_day or t>=actual_open_day
        reopened=seen_closed and river_open
        seen_closed |= not river_open
        avail=f[f.issue_day<=t]
        close,op=float(planned_close_day),float(planned_open_day)
        forecast_issue=np.nan
        if not avail.empty:
            fc=avail.iloc[-1]
            close,op=float(fc.planned_close_day),float(fc.planned_open_day)
            forecast_issue=float(fc.issue_day)
        if not river_open:
            close=min(close,float(t))
            op=max(op,float(t+1))
        sched=compute_prestock_schedule(cfg,close,op)
        future_arrival=t+lag
        extra_target=0.
        if use_prestock and not reopened and sched["feasible"]:
            start=sched["prestock_start_nb_day"]
            if start<=future_arrival<close:
                extra_target=min(sched["target_prestock_tons"],(future_arrival-start+1)*sched["spare_crossing_cap_tons"])
            elif close<=future_arrival<op:
                extra_target=max(0.,op-future_arrival-1)*d
        target=min(space,d*cfg.safety_days+extra_target)+lag*d
        position=yak+nb+in_transit-backlog
        dispatch=min(hub,cfg.rail_cap_tons,max(0.,target-position))
        hub-=dispatch
        arrivals[t+lag]+=dispatch
        in_transit+=dispatch
        arrived=float(arrivals[t])
        in_transit=max(0.,in_transit-arrived)
        nb+=arrived
        backlog+=d
        receiving=max(0.,space-yak+backlog)
        norm=min(nb,cfg.crossing_cap_normal_tons,receiving) if river_open else 0.
        alt=min(nb,cfg.crossing_cap_alt_tons,receiving,max(0.,backlog+d*cfg.safety_days-yak)) if not river_open else 0.
        nb-=norm+alt
        yak+=norm+alt
        served=min(yak,backlog)
        yak-=served
        backlog=max(0.,backlog-served)
        records.append(dict(day=t,river_open=int(river_open),hub_dispatched_t=dispatch,
            rail_arrived_nb_t=arrived,in_transit_t=in_transit,hub_remaining_t=hub,
            shipped_norm_t=norm,shipped_alt_t=alt,inv_nb_t=nb,inv_yak_t=yak,
            demanded_t=d,served_t=served,backlog_t=backlog,unmet_demand_t=backlog,
            forecast_issue_day=forecast_issue,planned_close_day=close,planned_open_day=op,
            cost_rail_rub=dispatch*cfg.rail_cost_per_ton,
            cost_normal_rub=norm*cfg.crossing_cost_normal,cost_alt_rub=alt*cfg.crossing_cost_alt,
            cost_nb_rub=nb*cfg.nb_holding_cost_per_ton_day,
            cost_yak_rub=min(yak,cfg.yak_warehouse_cap_tons)*cfg.yak_holding_cost_per_ton_day,
            cost_overflow_rub=max(0.,yak-cfg.yak_warehouse_cap_tons)*cfg.yak_overflow_cost_per_ton_day,
            cost_shortage_rub=backlog*cfg.shortage_penalty_per_ton_day))
    daily=pd.DataFrame(records)
    cost=float(daily[[c for c in daily if c.startswith("cost_")]].to_numpy().sum())
    def total(name):
        return float(daily[name].sum())
    summary=dict(total_cost_rub=cost,cost_per_ton_rub=cost/total_order,
        cost_rail_rub=total("cost_rail_rub"),cost_crossing_normal_rub=total("cost_normal_rub"),
        cost_crossing_alt_rub=total("cost_alt_rub"),cost_holding_nb_rub=total("cost_nb_rub"),
        cost_holding_yak_base_rub=total("cost_yak_rub"),cost_holding_yak_overflow_rub=total("cost_overflow_rub"),
        cost_holding_yak_rub=total("cost_yak_rub")+total("cost_overflow_rub"),cost_shortage_rub=total("cost_shortage_rub"),
        demand_tons=total_order,total_rail_tons=total("hub_dispatched_t"),served_tons=total("served_t"),
        terminal_backlog_tons=backlog,initial_yak_tons=initial,terminal_yak_tons=yak,terminal_nb_tons=nb,
        terminal_in_transit_tons=in_transit,terminal_hub_tons=hub,
        tons_normal_transport=total("shipped_norm_t"),tons_alt_transport=total("shipped_alt_t"),
        tons_stuck_nb_ton_days=total("inv_nb_t"),yak_holding_ton_days=total("inv_yak_t"),
        shortage_ton_days=total("backlog_t"),days_with_deficit=int((daily.backlog_t>1e-7).sum()),
        peak_inv_nb_tons=float(daily.inv_nb_t.max()),peak_inv_yak_tons=float(daily.inv_yak_t.max()),
        mass_balance_error_tons=initial+total_order-(hub+nb+yak+in_transit+total("served_t")))
    return (summary,daily) if return_daily else summary


def compare_policies(cfg,ac,ao,cc,co,q20_close=None,q20_open=None,forecasts=None):
    qc,qo=(cc if q20_close is None else q20_close),(co if q20_open is None else q20_open)
    return {name:simulate_corridor(cfg,ac,ao,pc,po,use_prestock=pre,return_daily=True,
                                  forecasts=forecasts if name=="ml" else None)
            for name,pc,po,pre in (("reactive",cc,co,False),("calendar",cc,co,True),
                                   ("calendar_q20",qc,qo,True),("ml",qc,qo,True))}


def run_sensitivity_check(base_cfg=None,actual_close_day=30,actual_open_day=57,cal_close_day=36,
                          cal_open_day=63,model_close_day=29,model_open_day=58,delta=.30,*,
                          forecasts=None,q20_close=None,q20_open=None):
    """One base row and ± changes for each nonzero assumption; no success guarantee."""
    cfg=base_cfg or CorridorConfig()
    if forecasts is None:
        forecasts=pd.DataFrame([dict(issue_day=0,planned_close_day=model_close_day,planned_open_day=model_open_day)])
    variants=[("base","База",1.,cfg)]
    for name,label in SENSITIVITY_PARAMS:
        if getattr(cfg,name):
            variants += [(name,label,fac,cfg.scaled(name,fac)) for fac in (1-delta,1+delta)]
    rows=[]
    for name,label,fac,c in variants:
        results=compare_policies(c,actual_close_day,actual_open_day,cal_close_day,cal_open_day,q20_close,q20_open,forecasts)
        b,a=results["calendar"][0],results["ml"][0]
        rows.append(dict(parameter=name,label=label,variation="база" if name=="base" else f"{(fac-1)*100:+.0f}%",
            param_value=getattr(c,name) if name!="base" else np.nan,before_mln_rub=b["total_cost_rub"]/1e6,
            after_mln_rub=a["total_cost_rub"]/1e6,saving_mln_rub=(b["total_cost_rub"]-a["total_cost_rub"])/1e6,
            saving_pct=100*(b["total_cost_rub"]-a["total_cost_rub"])/b["total_cost_rub"],
            yak_storage_delta_mln_rub=(a["cost_holding_yak_rub"]-b["cost_holding_yak_rub"])/1e6,
            effect_positive=a["total_cost_rub"]<b["total_cost_rub"]))
    return pd.DataFrame(rows)
