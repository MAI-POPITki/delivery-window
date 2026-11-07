"""Resource, mass, causality and attribution checks for the A3 corridor."""
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno.corridor import CorridorConfig,compute_prestock_schedule,simulate_corridor,compare_policies,run_sensitivity_check
from okno.corridor_data import case_inputs,load_events,past_shifts,verified_years
from okno.features import season_matrix
from src.model_window import feature_row

ROOT=Path(__file__).resolve().parents[1]


class CorridorSimulatorTests(unittest.TestCase):
    def test_rail_arrivals_are_delayed_and_both_capacities_hold(self):
        cfg=replace(CorridorConfig(),initial_yak_tons=0)
        s,d=simulate_corridor(cfg,46,84,45,85,return_daily=True)
        self.assertEqual(d.rail_arrived_nb_t.iloc[:6].sum(),0)
        np.testing.assert_allclose(d.rail_arrived_nb_t.iloc[6:],d.hub_dispatched_t.iloc[:-6])
        self.assertTrue((d.hub_dispatched_t<=cfg.rail_cap_tons).all())
        self.assertTrue((d.shipped_norm_t<=cfg.crossing_cap_normal_tons).all())
        self.assertTrue((d.shipped_alt_t<=cfg.crossing_cap_alt_tons).all())
        _,zero=simulate_corridor(replace(cfg,rail_lead_time_days=0),46,84,45,85,return_daily=True)
        np.testing.assert_allclose(zero.rail_arrived_nb_t,zero.hub_dispatched_t)
        self.assertFalse(zero.rail_arrived_nb_t.equals(d.rail_arrived_nb_t))

    def test_mass_backlog_and_cost_balance_each_day(self):
        c=case_inputs(2024);cfg=CorridorConfig()
        for name,(s,d) in compare_policies(cfg,c['ac'],c['ao'],c['cc'],c['co'],c['qc'],c['qo'],c['forecasts']).items():
            with self.subTest(policy=name):
                mass=d.hub_remaining_t+d.in_transit_t+d.inv_nb_t+d.inv_yak_t+d.served_t.cumsum()
                np.testing.assert_allclose(mass,cfg.daily_demand_tons*cfg.horizon_days+s['initial_yak_tons'])
                np.testing.assert_allclose(d.backlog_t,d.demanded_t.cumsum()-d.served_t.cumsum(),atol=1e-7)
                self.assertAlmostEqual(s['cash_cost_rub'],d[[col for col in d if col.startswith('cost_')]].to_numpy().sum())
                self.assertAlmostEqual(s['total_cost_rub'],s['cash_cost_rub']+s['terminal_commitment_rub'])
                self.assertAlmostEqual(s['cost_rail_rub'],d.hub_dispatched_t.sum()*cfg.rail_cost_per_ton)
                self.assertAlmostEqual(s['shortage_ton_days'],d.backlog_t.sum())
                self.assertAlmostEqual(s['mass_balance_error_tons'],0.)

    def test_unserved_orders_remain_and_are_charged_each_day(self):
        cfg=replace(CorridorConfig(),horizon_days=10,initial_yak_tons=0,crossing_cap_alt_tons=0)
        s,d=simulate_corridor(cfg,0,100,0,100,return_daily=True)
        self.assertEqual(s['terminal_backlog_tons'],250)
        self.assertEqual(s['shortage_ton_days'],25*sum(range(1,11)))
        self.assertEqual(s['cost_shortage_rub'],s['shortage_ton_days']*12000)
        self.assertEqual(s['served_tons'],0)

    def test_future_forecasts_and_future_status_cannot_change_earlier_dispatch(self):
        cfg=CorridorConfig()
        f=pd.DataFrame([dict(issue_day=0,planned_close_day=45,planned_open_day=85),
                        dict(issue_day=20,planned_close_day=40,planned_open_day=90)])
        poisoned=f.copy();poisoned.loc[1,['planned_close_day','planned_open_day']]=[25,100]
        _,a=simulate_corridor(cfg,46,84,45,85,forecasts=f,return_daily=True)
        _,b=simulate_corridor(cfg,46,84,45,85,forecasts=poisoned,return_daily=True)
        pd.testing.assert_frame_equal(a[a.day<20],b[b.day<20])
        _,b=simulate_corridor(cfg,40,75,45,85,forecasts=f,return_daily=True)
        pd.testing.assert_frame_equal(a[a.day<40],b[b.day<40])
        used=a[a.forecast_issue_day.notna()]
        self.assertTrue((used.forecast_issue_day<=used.day).all())

    def test_warehouse_is_hard_and_overflow_requires_explicit_capacity(self):
        for extra in (0.,100.):
            cfg=replace(CorridorConfig(),yak_warehouse_cap_tons=40,initial_yak_tons=0,yak_overflow_capacity_tons=extra)
            s,d=simulate_corridor(cfg,46,84,45,85,return_daily=True)
            self.assertTrue((d.inv_yak_t<=40+extra+1e-7).all())
            if not extra:
                self.assertEqual(s['cost_holding_yak_overflow_rub'],0)

    def test_no_fake_spare_capacity_and_invalid_parameters(self):
        cfg=replace(CorridorConfig(),crossing_cap_normal_tons=20)
        self.assertFalse(compute_prestock_schedule(cfg,30,57)['feasible'])
        self.assertIsNone(compute_prestock_schedule(cfg,30,57)['prestock_duration_days'])
        for kw in [dict(daily_demand_tons=0),dict(rail_lead_time_days=-1),dict(yak_overflow_capacity_tons=-1)]:
            with self.assertRaises(ValueError):
                CorridorConfig(**kw)

    def test_identical_forecast_and_calendar_have_identical_economics(self):
        f=pd.DataFrame([dict(issue_day=0,planned_close_day=45,planned_open_day=85)])
        rs=compare_policies(CorridorConfig(),46,84,45,85,45,85,f)
        a=rs['calendar_q20'][1].drop(columns='forecast_issue_day')
        b=rs['ml'][1].drop(columns='forecast_issue_day')
        pd.testing.assert_frame_equal(a,b)

    def test_sensitivity_has_one_base_and_does_not_assert_success(self):
        cfg=replace(CorridorConfig(),horizon_days=30)
        sens=run_sensitivity_check(cfg)
        self.assertEqual(int(sens.parameter.eq('base').sum()),1)
        self.assertTrue((sens[~sens.parameter.eq('base')].groupby('parameter').size()==2).all())
        for vals in [dict(issue_day=0,planned_close_day=50,planned_open_day=40),
                     dict(issue_day=0,planned_close_day=40,planned_open_day=50,hydro_year=2024,train_max_year=2024)]:
            with self.assertRaises(ValueError):
                simulate_corridor(cfg,40,50,40,50,forecasts=pd.DataFrame([vals]))


class CorridorEvidenceTests(unittest.TestCase):
    def test_real_windows_are_not_reconstructed_from_satellite_anomalies(self):
        c=case_inputs(2024)
        self.assertEqual(c['ao']-c['ac'],38)
        self.assertEqual(verified_years(),[2019,2023,2024])
        with self.assertRaises(ValueError):
            case_inputs(2022)
        e=load_events().set_index('hydro_year')
        self.assertEqual(e.loc[2020,'ferry_open_date'],'2020-05-22')
        self.assertEqual(e.loc[2022,'ferry_open_date'],'2022-05-30')

    def test_current_year_labels_do_not_fit_its_road_conversion(self):
        e=load_events();a=past_shifts(2024,e)
        e.loc[e.hydro_year==2024,['ice_road_close_date','ferry_open_date']]=['2024-01-01','2024-08-01']
        self.assertEqual(a,past_shifts(2024,e))
        e.loc[e.hydro_year==2019,'ice_close_published_date']='2026-01-01'
        self.assertNotIn(2019,past_shifts(2024,e)['close_years'])
        self.assertFalse(case_inputs(2019)['model_available'])
        self.assertTrue(case_inputs(2024)['model_available'])

    def test_saved_forecasts_have_strict_past_training_and_weather_cutoffs(self):
        f=pd.read_csv(ROOT/'reports/window_forecasts_a3.csv')
        self.assertTrue((f.train_max_year<f.hydro_year).all())
        self.assertTrue((f.calibration_max_year.dropna()<f.loc[f.calibration_max_year.notna(),'hydro_year']).all())
        self.assertTrue((pd.to_datetime(f.weather_cutoff_date)<=pd.to_datetime(f.issue_date)-pd.Timedelta(days=5)).all())
        fixed=pd.read_csv(ROOT/'reports/window_backtest_yakutsk_2001_2024.csv')
        self.assertTrue(fixed.issue_date.str.endswith('-03-10').all())
        self.assertNotIn(2001,f.hydro_year.unique())

    def test_current_targets_and_future_weather_cannot_change_features(self):
        sat=pd.read_csv(ROOT/'data/processed/yakutsk_sat_seasons.csv')
        T=season_matrix('yakutsk_crossing','t');S=season_matrix('yakutsk_crossing','snow_cm')
        date=pd.Timestamp('2024-03-10')
        a=feature_row(sat,T,S,2024,date)
        sat.loc[sat.hydro_year>=2024,'sat_breakup_day']+=100
        T.loc[2024,T.columns>a['issue_day']]+=50
        S.loc[2024,S.columns>a['issue_day']]+=100
        b=feature_row(sat,T,S,2024,date)
        pd.testing.assert_series_equal(pd.Series(a),pd.Series(b))


class CorridorDashboardTests(unittest.TestCase):
    def test_historical_and_synthetic_render_with_honest_controls(self):
        app=AppTest.from_file(str(ROOT/'app/dashboard.py'),default_timeout=120).run()
        app.button(key='open_dashboard').click().run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertEqual(app.selectbox(key='corr_scenario').value,2024)
        frame=next(d.value for d in app.dataframe if 'Получено потребителем, т' in d.value.columns)
        self.assertEqual(len(frame),6)
        self.assertEqual(app.selectbox(key='corr_focus').value,'economic_ml')
        self.assertEqual(app.number_input(key='corr_overflow_cap').value,0)
        app.slider(key='corr_calt').set_value(45000.).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        app.selectbox(key='corr_scenario').select('synthetic').run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        app.slider(key='corr_d').set_value(50.)
        app.slider(key='corr_qmax').set_value(35.)
        app.run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any('не превышает' in w.value for w in app.warning))


if __name__=='__main__':
    unittest.main()
