"""Causal daily hub → rail → Lena → Yakutsk dispatch comparison.

Today's dispatch sees only dated forecasts already available and today's river
status. Costs, demand, allocated capacities and overflow space are assumptions.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import pandas as pd
from .economic_planner import EconomicPlanner


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
               "calendar_q20":"Предзавоз: календарный q20","ml":"Предзавоз: ML q20",
               "economic_calendar":"Экономический: календарная медиана",
               "economic_ml":"Экономический: ML q20"}


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
                      use_prestock=True,return_daily=False,*,forecasts=None,economic=False,availability=None):
    """Execute only current hub dispatch. Arrival is dispatch + rail lead.

    Unsatisfied demand persists as backlog; penalty is its daily area in t·days.
    Receiving space is finite. Opening inventory and terminal cargo are exposed.
    """
    if not 0<=actual_close_day<=actual_open_day or planned_open_day<planned_close_day:
        raise ValueError("Invalid river window")
    if availability is not None:
        availability = np.asarray(availability, dtype=bool).copy()
        if availability.shape != (cfg.horizon_days,):
            raise ValueError("Календарь должен покрывать весь горизонт")
    f=_forecast_table(forecasts)
    d,H,lag=cfg.daily_demand_tons,cfg.horizon_days,cfg.rail_lead_time_days
    space=cfg.yak_warehouse_cap_tons+cfg.yak_overflow_capacity_tons
    initial=cfg.initial_yak_tons
    total_order=d*H
    hub,nb,yak,backlog=total_order,0.,initial,0.
    arrivals=np.zeros(H+lag+1)
    in_transit=0.
    seen_closed=False
    planner=EconomicPlanner(cfg, availability=availability) if economic else None
    records=[]
    for t in range(H):
        river_open=bool(availability[t]) if availability is not None else t<actual_close_day or t>=actual_open_day
        reopened=seen_closed and river_open
        seen_closed |= not river_open
        avail=f[f.issue_day<=t]
        close,op=float(planned_close_day),float(planned_open_day)
        forecast_issue=np.nan
        if not avail.empty:
            fc=avail.iloc[-1]
            close,op=float(fc.planned_close_day),float(fc.planned_open_day)
            forecast_issue=float(fc.issue_day)
        action=None
        if planner is not None:
            action=planner.action(t,hub=hub,nb=nb,yak=yak,backlog=backlog,in_transit=in_transit,
                arrivals=arrivals[t:],river_open=river_open,reopened=reopened,close=close,opening=op,
                forecast_issue=None if np.isnan(forecast_issue) else forecast_issue)
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
        dispatch=min(hub,cfg.rail_cap_tons,max(0.,action.dispatch if action else target-position))
        hub-=dispatch
        arrivals[t+lag]+=dispatch
        in_transit+=dispatch
        arrived=float(arrivals[t])
        in_transit=max(0.,in_transit-arrived)
        nb+=arrived
        backlog+=d
        receiving=max(0.,space-yak+backlog)
        norm=min(nb,cfg.crossing_cap_normal_tons,receiving,action.normal if action else np.inf) if river_open else 0.
        alt=min(nb,cfg.crossing_cap_alt_tons,receiving,
                action.alternative if action else max(0.,backlog+d*cfg.safety_days-yak)) if not river_open else 0.
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
            economic_replanned=int(action.replanned) if action else 0,
            optimizer_status=action.status if action else 'heuristic',
            forecast_terminal_shortfall_t=action.terminal_shortfall if action else np.nan,
            cost_rail_rub=dispatch*cfg.rail_cost_per_ton,
            cost_normal_rub=norm*cfg.crossing_cost_normal,cost_alt_rub=alt*cfg.crossing_cost_alt,
            cost_nb_rub=nb*cfg.nb_holding_cost_per_ton_day,
            cost_yak_rub=min(yak,cfg.yak_warehouse_cap_tons)*cfg.yak_holding_cost_per_ton_day,
            cost_overflow_rub=max(0.,yak-cfg.yak_warehouse_cap_tons)*cfg.yak_overflow_cost_per_ton_day,
            cost_shortage_rub=backlog*cfg.shortage_penalty_per_ton_day))
    daily=pd.DataFrame(records)
    cash=float(daily[[c for c in daily if c.startswith("cost_")]].to_numpy().sum())
    # Same end valuation for all policies: unpaid regular transport of the
    # remaining committed order. This is an estimate, not a simulated shipment.
    terminal_rail=hub*cfg.rail_cost_per_ton
    terminal_crossing=(hub+in_transit+nb)*cfg.crossing_cost_normal
    terminal_commitment=terminal_rail+terminal_crossing
    cost=cash+terminal_commitment
    def total(name):
        return float(daily[name].sum())
    summary=dict(total_cost_rub=cost,cash_cost_rub=cash,cost_per_ton_rub=cost/total_order,
        terminal_commitment_rub=terminal_commitment,terminal_rail_rub=terminal_rail,
        terminal_crossing_rub=terminal_crossing,economic_plan_count=planner.solves if planner else 0,
        terminal_service_complete=backlog<1e-6,
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
                                  forecasts=forecasts if name in ('ml','economic_ml') else None,
                                  economic=name.startswith('economic_'))
            for name,pc,po,pre in (("reactive",cc,co,False),("calendar",cc,co,True),
                                   ("calendar_q20",qc,qo,True),("ml",qc,qo,True),
                                   ("economic_calendar",cc,co,False),("economic_ml",qc,qo,False))}


def weekly_dispatch_comparison(results,start_date):
    """Compare the old and economic ML policies in seven-day reporting periods.

    Flows and incurred costs are summed; stocks and backlog are end-of-week
    values. The last period ends on the last simulated day, even if incomplete.
    This aggregates sequential daily decisions, not a forecast issued on day 0.
    """
    start=pd.Timestamp(start_date) if start_date is not None else None
    frames=[]
    flows=('hub_dispatched_t','rail_arrived_nb_t','shipped_norm_t','shipped_alt_t',
           'demanded_t','served_t')
    stocks=('inv_nb_t','inv_yak_t','backlog_t','in_transit_t','hub_remaining_t')
    for policy in ('ml','economic_ml'):
        daily=results[policy][1].sort_values('day').copy()
        daily['week']=daily.day//7+1
        daily['weekly_cost_rub']=daily[[c for c in daily if c.startswith('cost_')]].sum(axis=1)
        weekly=daily.groupby('week',sort=True).agg(
            day_start=('day','min'),day_end=('day','max'),days=('day','size'),
            **{col:(col,'sum') for col in flows},
            **{col:(col,'last') for col in stocks},
            weekly_cost_rub=('weekly_cost_rub','sum')).reset_index()
        for column,day_column in (('week_start_date','day_start'),('week_end_date','day_end')):
            weekly[column]=(start+pd.to_timedelta(weekly[day_column],unit='D')).dt.strftime('%Y-%m-%d') if start is not None else ''
        weekly['shipped_total_t']=weekly.shipped_norm_t+weekly.shipped_alt_t
        frames.append(weekly.assign(policy=policy))
    return pd.concat(frames,ignore_index=True).sort_values(['week','policy']).reset_index(drop=True)


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
        ec,em=results['economic_calendar'][0],results['economic_ml'][0]
        rows.append(dict(parameter=name,label=label,variation="база" if name=="base" else f"{(fac-1)*100:+.0f}%",
            param_value=getattr(c,name) if name!="base" else np.nan,before_mln_rub=b["total_cost_rub"]/1e6,
            after_mln_rub=a["total_cost_rub"]/1e6,saving_mln_rub=(b["total_cost_rub"]-a["total_cost_rub"])/1e6,
            saving_pct=100*(b["total_cost_rub"]-a["total_cost_rub"])/b["total_cost_rub"],
            yak_storage_delta_mln_rub=(a["cost_holding_yak_rub"]-b["cost_holding_yak_rub"])/1e6,
            effect_positive=a["total_cost_rub"]<b["total_cost_rub"],
            economic_calendar_mln_rub=ec['total_cost_rub']/1e6,
            economic_ml_mln_rub=em['total_cost_rub']/1e6,
            # вклад именно прогноза при одном и том же оптимизаторе: > 0 — ML дешевле календаря
            economic_forecast_effect_pct=100*(ec['total_cost_rub']-em['total_cost_rub'])/ec['total_cost_rub'],
            economic_saving_vs_calendar_pct=100*(b['total_cost_rub']-em['total_cost_rub'])/b['total_cost_rub'],
            economic_saving_vs_ml_pct=100*(a['total_cost_rub']-em['total_cost_rub'])/a['total_cost_rub'],
            economic_terminal_backlog_tons=em['terminal_backlog_tons'],
            economic_shortage_ton_days=em['shortage_ton_days'],
            economic_effect_positive=em['total_cost_rub']<b['total_cost_rub']))
    return pd.DataFrame(rows)
