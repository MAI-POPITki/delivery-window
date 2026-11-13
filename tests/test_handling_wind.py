from datetime import datetime, timezone
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from okno import handling_wind as W
from okno.flood_planning import optimize_dispatch
from okno.planner import Params, compare_ml_weekly_plans


class WindHandlingTest(unittest.TestCase):
    as_of=datetime(2026,11,13,tzinfo=timezone.utc)

    def envelope(self,values=None,gusts=None):
        times=pd.date_range('2026-11-13',periods=72,freq='h')
        payload=dict(timezone='GMT',utc_offset_seconds=0,
            hourly_units=dict(wind_speed_10m='m/s',wind_gusts_10m='m/s'),
            hourly=dict(time=times.strftime('%Y-%m-%dT%H:%M').tolist(),
                        wind_speed_10m=[1.]*72 if values is None else values,
                        wind_gusts_10m=[2.]*72 if gusts is None else gusts))
        return dict(payload=payload,source=W.API,kind='weather_forecast',fetched_at=self.as_of.isoformat(),
                    sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())

    def daily(self,envelope,**kwargs):
        return W.capacity(envelope,[pd.Timestamp('2026-11-14').date()],as_of=self.as_of,
            tz='UTC',speed_limit=10,gust_limit=15,duration_h=2,batch_t=10,start_hour=8,end_hour=12,**kwargs)

    def test_puffs_at_limit_break_contiguous_operations(self):
        self.assertEqual(self.daily(self.envelope()).capacity_t.iloc[0],20)
        gusts=[2.]*72;gusts[24+9]=15
        # Gust stamped 09 covers 08–09; the remaining three hours fit one batch.
        day=self.daily(self.envelope(gusts=gusts))
        self.assertEqual(day.safe_hours.iloc[0],3)
        self.assertEqual(day.capacity_t.iloc[0],10)

    def test_unknown_hour_never_becomes_permission(self):
        values=[1.]*72;values[24+9]=None
        day=self.daily(self.envelope(values=values))
        self.assertFalse(day.known.iloc[0]);self.assertEqual(day.capacity_t.iloc[0],0)

    def test_temperature_precipitation_and_work_budget_reduce_capacity(self):
        p=self.envelope()
        p['payload']['hourly_units'].update(temperature_2m='°C',precipitation='mm')
        p['payload']['hourly'].update(temperature_2m=[-20.]*72,precipitation=[0.]*72)
        p['payload']['hourly']['temperature_2m'][24+8]=-51
        p['payload']['hourly']['precipitation'][24+12]=.2
        p['sha256']=hashlib.sha256(json.dumps(p['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(self.daily(p).capacity_t.iloc[0],20) # Optional conditions disabled.
        restricted=self.daily(p,min_temperature=-50,max_precipitation=0)
        self.assertEqual(restricted.safe_hours.iloc[0],2)
        self.assertEqual(restricted.capacity_t.iloc[0],10)
        self.assertEqual(self.daily(p,max_operation_hours=1).capacity_t.iloc[0],0)
        self.assertEqual(self.daily(p,max_operation_hours=2).capacity_t.iloc[0],10)

    def test_missing_optional_weather_is_unknown_when_requested(self):
        day=self.daily(self.envelope(),min_temperature=-50)
        self.assertFalse(day.known.iloc[0])
        self.assertEqual(day.capacity_t.iloc[0],0)

    def test_own_level_variants_share_handling_limits(self):
        from okno.flood_m3 import plan_variants
        dates=pd.date_range('2026-11-14',periods=2)
        records=pd.DataFrame(dict(date=dates.date,confirmed=True,capacity_t=100.,source='fixture',published=self.as_of.date()))
        forecast=pd.DataFrame(dict(date=dates,lower_cm=1.,predicted_cm=2.,upper_cm=3.))
        wind=pd.DataFrame(dict(date=dates.date,known=True,capacity_t=[0.,7.]))
        results=plan_variants(records,forecast,self.as_of.date(),1,10,20,1,100,0,wind=(wind,wind))
        for plan,summary in results.values():
            self.assertEqual(summary['road_t'],7)
            self.assertEqual(summary['backup_t'],13)

    def test_local_timezone_and_no_past_hours(self):
        day=W.capacity(self.envelope(),[pd.Timestamp('2026-11-13').date()],as_of=self.as_of,
            tz='Asia/Yakutsk',speed_limit=10,gust_limit=15,duration_h=1,batch_t=1,start_hour=8,end_hour=12)
        self.assertEqual(day.working_hours.iloc[0],3) # 09–12 local, 08 was before decision.
        self.assertEqual(day.capacity_t.iloc[0],3)

    def test_future_stale_reanalysis_and_hash_rejected(self):
        for field,value in [('fetched_at','2026-11-14T00:00:00Z'),
                            ('fetched_at','2026-11-11T00:00:00Z'),('kind','reanalysis'),('sha256','bad')]:
            p=self.envelope();p[field]=value
            with self.assertRaises(ValueError):W.parse(p,self.as_of)

    def test_unloading_date_is_arrival_and_both_plans_obey_capacity(self):
        dates=pd.date_range('2026-11-14',periods=3).date
        calendar=pd.DataFrame(dict(date=dates[:2],arrival=dates[1:],capacity_t=100.,confirmed=True,policy_allowed=True))
        load=pd.DataFrame(dict(date=dates[:2],known=True,capacity_t=[20.,30.]))
        unload=pd.DataFrame(dict(date=dates[1:],known=[False,True],capacity_t=[100.,10.]))
        constrained=W.constrain(calendar,load,unload)
        self.assertEqual(constrained.capacity_t.tolist(),[0.,10.])
        plan,summary=optimize_dispatch(constrained,20,1,10,0)
        self.assertEqual(summary['backup_t'],10)
        for column in ['uniform_t','optimized_t']:
            self.assertTrue((plan[column]<=constrained.capacity_t+1e-9).all())

    def test_winter_daily_limits_include_extra_vehicles_and_road_calendar(self):
        forecast=dict(q10=290,q50=300,q90=310,calendar=300,issue_day=200)
        p=Params(Q_t=100,K_t_week=70,K_extra_t_week=70,start_day=200,transit_days=1,close_margin_days=0)
        def capacity(starts,ends):return np.where(starts==201,5.,0.)
        plan,summary=compare_ml_weekly_plans(forecast,200,p,end_day=204,handling_capacity=capacity)
        self.assertEqual(plan.dispatch_day.nunique(),4)
        for rule in ['ml','economic_ml']:
            rows=plan[plan.rule==rule]
            self.assertEqual(rows.shipped_t.sum(),5)
            self.assertTrue((rows.shipped_t<=rows.handling_capacity_t).all())
        blocked,_=compare_ml_weekly_plans(forecast,200,p,end_day=204,handling_capacity=capacity,
                                         availability=lambda start,end:np.zeros(len(start),bool))
        self.assertEqual(blocked.shipped_t.sum(),0)

    def test_enabled_ui_consumes_saved_forecasts_for_both_terminals(self):
        from streamlit.testing.v1 import AppTest
        script="""from datetime import date
import streamlit as st
from app.handling_wind_view import render_wind
result=render_wind('test_wind',date(2026,11,13),[date(2026,11,14)],[date(2026,11,14)])
if result is not None:st.session_state['test_capacities']=[t.capacity_t.tolist() for t in result]
"""
        app=AppTest.from_string(script,default_timeout=30).run()
        app.checkbox(key='test_wind_enabled').check().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['test_capacities'],[[0.],[0.]])
        for terminal in ['load','unload']:
            prefix='test_wind_'+terminal
            for suffix,value in [('_lat',62.),('_lon',129.),('_speed',10.),('_gust',15.),('_batch',10.)]:
                app.number_input(key=prefix+suffix).set_value(value)
            app.text_input(key=prefix+'_basis').set_value('Test fixture equipment limit')
            app.session_state[prefix+'_data']=('62.0|129.0',self.envelope())
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['test_capacities'],[[120.],[120.]])


if __name__=='__main__':unittest.main()
