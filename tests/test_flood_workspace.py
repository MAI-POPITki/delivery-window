"""Причинность выпусков, пути из OSM и проверка оптимума независимым перебором."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import itertools
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno import flood as F
from okno.flood_routes import load_routes
from okno.flood_planning import confirmed_calendar, optimize_dispatch, passage_risks
from okno.map_style import AVAILABILITY_SCALE
from app.flood_view import make_map


class FloodWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now = max(datetime.fromisoformat(F.read_response(F.DATA/'latest'/f'{p}.json')['fetched_at']) for p in F.POINTS)
        cls.rows, cls.details = F.load_points(cls.now)

    def test_no_forecast_before_it_was_received(self):
        earliest = min(datetime.fromisoformat(F.read_response(F.DATA/'runs'/'2026-11-09'/f'{p}.json')['fetched_at']) for p in F.POINTS)
        with self.assertRaises(ValueError):
            F.forecast_as_of('lena_yakutsk', earliest-timedelta(days=1))
        rows, details = F.load_points(self.now, planning_date=date(2024,4,10))
        self.assertFalse(details)
        self.assertTrue(rows.fraction.isna().all())

    def test_asof_selects_latest_known_snapshot_not_future(self):
        envelope = F.read_response(F.DATA/'latest'/'lena_yakutsk.json')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            past=self.now-timedelta(hours=1)
            future=self.now+timedelta(hours=1)
            F.save_response(root/'runs'/'saved'/'lena_yakutsk.json',envelope['payload'],envelope['url'],past.isoformat())
            F.save_response(root/'latest'/'lena_yakutsk.json',envelope['payload'],envelope['url'],future.isoformat())
            selected=F.forecast_as_of('lena_yakutsk',self.now,root)
            self.assertEqual(selected['fetched_at'],past.isoformat())

    def test_real_roads_exist_for_every_point_and_palette_matches(self):
        for pid in F.POINTS:
            roads, metadata=load_routes(pid)
            self.assertTrue(roads)
            self.assertTrue(metadata['source_url'].startswith('https://www.openstreetmap.org/api/0.6/map?'))
            self.assertTrue(all(len(r['coordinates'])>=2 and r['source'].startswith('https://www.openstreetmap.org/way/') for r in roads))
        fig=make_map(self.rows)
        expected=getattr(__import__('plotly.colors',fromlist=['sequential']).diverging, AVAILABILITY_SCALE)
        self.assertEqual(fig.data[0].marker.colorscale[0][1],expected[0])
        self.assertAlmostEqual(fig.data[0].marker.color[0],100*(1-self.rows.fraction.iloc[0]))

    def test_trip_requires_forecast_for_every_day(self):
        risk=passage_risks(self.details['lena_yakutsk'],3)
        self.assertTrue(risk.complete.iloc[:5].all())
        self.assertTrue(risk.fraction.iloc[-2:].isna().all())

    def test_target_date_changes_real_discharge_even_when_exceedance_is_zero(self):
        d = self.details['lena_yakutsk']
        first = F.evaluate_window(d, d['curve'].index[0].date())
        last = F.evaluate_window(d, d['curve'].index[-1].date())
        self.assertNotEqual(first['median'], last['median'])
        self.assertEqual(first['fraction'], 0.)
        self.assertEqual(last['fraction'], 0.)
        self.assertIsNone(F.evaluate_window(d,d['curve'].index[-1].date(),2))

    def test_map_estimate_is_tied_to_target_date(self):
        d = self.details['lena_yakutsk']
        first = F.evaluate_points(self.rows,self.details,d['curve'].index[0].date())
        last = F.evaluate_points(self.rows,self.details,d['curve'].index[-1].date())
        self.assertNotEqual(first.iloc[0].median_m3s,last.iloc[0].median_m3s)
        self.assertNotEqual(first.iloc[0].start,last.iloc[0].start)

    def test_target_control_updates_displayed_forecast_in_ui(self):
        with patch('app.flood_view.load_points',side_effect=lambda **kw:F.load_points(self.now,**kw)):
            at=AppTest.from_string('from app.flood_view import render_flood\nrender_flood()',default_timeout=30)
            at.session_state['flood_decision']=self.now.date()
            at.session_state['flood_tab']='Прогноз'
            at.run()
            before=at.metric[0].value
            last=self.details['lena_yakutsk']['curve'].index[-1].date()
            next(w for w in at.sidebar.date_input if w.label=='Дата оценки / проезда').set_value(last).run()
            self.assertFalse(at.exception,[e.message for e in at.exception])
            self.assertNotEqual(before,at.metric[0].value)
            self.assertTrue(any(row.value['Выбранная дата'].iloc[-1] for row in at.dataframe))

    def test_march_date_explains_missing_archive_and_opens_available_forecast(self):
        with patch('app.flood_view.load_points',side_effect=lambda **kw:F.load_points(self.now,**kw)):
            at=AppTest.from_string('from app.flood_view import render_flood\nrender_flood()',default_timeout=30)
            at.session_state['flood_decision']=date(2026,3,20)
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(next(w for w in at.sidebar.date_input if w.label=='Дата оценки / проезда').disabled)
            at.button(key='flood_open_available').click().run()
            self.assertFalse(at.exception,[e.message for e in at.exception])
            self.assertEqual(at.sidebar.date_input(key='flood_decision').value,self.now.date())
            self.assertFalse(next(w for w in at.sidebar.date_input if w.label=='Дата оценки / проезда').disabled)
            self.assertTrue(at.metric)

    def records(self,n=3):
        return pd.DataFrame(dict(date=pd.date_range('2026-11-10',periods=n).date,
             confirmed=True, capacity_t=1., source='test-only operator record', published=date(2026,11,9)))

    def test_unconfirmed_or_future_permission_cannot_enable_route(self):
        records=self.records()
        records['confirmed']=False
        c=confirmed_calendar(records,None,date(2026,11,9),allow_without_forecast=True)
        self.assertFalse(c.confirmed.any())
        records['confirmed']=True
        records['published']=date(2026,11,10)
        with self.assertRaises(ValueError):
            confirmed_calendar(records,None,date(2026,11,9),allow_without_forecast=True)

    def test_permission_must_cover_entire_trip(self):
        records=self.records()
        records.loc[1,'confirmed']=False
        c=confirmed_calendar(records,None,date(2026,11,9),travel_days=2,allow_without_forecast=True)
        self.assertFalse(c.confirmed.any())

    def test_unknown_hydrology_only_allowed_by_explicit_operator_choice(self):
        records=self.records()
        c=confirmed_calendar(records,None,date(2026,11,9))
        self.assertFalse(c.policy_allowed.any())
        allowed=confirmed_calendar(records,None,date(2026,11,9),allow_without_forecast=True)
        self.assertTrue(allowed.policy_allowed.all())

    def test_hydrology_policy_excludes_exceedance_without_inventing_closure(self):
        records=self.records()
        risks=pd.DataFrame(dict(date=records.date,fraction=[.1,.9,None]))
        c=confirmed_calendar(records,risks,date(2026,11,9),risk_limit=.2)
        self.assertEqual(c.policy_allowed.tolist(),[True,False,False])
        self.assertTrue(c.confirmed.all())

    def test_linear_optimizer_matches_exhaustive_small_problems(self):
        for caps in itertools.product((0,1,2),repeat=3):
            calendar=pd.DataFrame(dict(date=pd.date_range('2026-11-10',periods=3).date,
                         capacity_t=caps,confirmed=True,policy_allowed=True))
            plan,result=optimize_dispatch(calendar,3.,2.,9.,2.)
            brute=min(sum(x*c for x,c in zip(amounts,[6,4,2]))+(3-sum(amounts))*9
                      for amounts in itertools.product(*(range(c+1) for c in caps)) if sum(amounts)<=3)
            self.assertAlmostEqual(result['cost'],brute)
            self.assertAlmostEqual(plan.optimized_t.sum()+result['backup_t'],3.)
            self.assertTrue((plan.optimized_t<=np.array(caps)+1e-8).all())
            self.assertLessEqual(result['cost'],result['baseline_cost']+1e-8)

    def test_closed_and_risky_days_excluded_from_both_plans(self):
        calendar=self.records().assign(policy_allowed=[True,False,True],confirmed=[False,True,True])
        plan,result=optimize_dispatch(calendar,5.,2.,9.,0.)
        self.assertEqual(plan.optimized_t.iloc[:2].sum(),0.)
        self.assertEqual(plan.uniform_t.iloc[:2].sum(),0.)
        self.assertEqual(result['backup_t'],4.)

    def test_all_workspace_tabs_render_with_real_saved_data(self):
        with patch('app.flood_view.load_points',side_effect=lambda **kw:F.load_points(self.now,**kw)):
            for tab in ['Карта путей','Прогноз воды','Риски и рекомендации','Планировщик','О данных']:
                at=AppTest.from_string('from app.flood_view import render_flood\nrender_flood()',default_timeout=30)
                at.session_state['flood_decision']=self.now.date()
                at.session_state['flood_tab']=tab
                at.run()
                self.assertFalse(at.exception,[e.message for e in at.exception])
                if tab=='Карта путей':
                    self.assertTrue(at.get('plotly_chart'))
                    self.assertTrue(at.sidebar.selectbox(key='flood_route').options)
                elif tab=='Риски и рекомендации':
                    self.assertTrue(any('проезда' in w.label for w in at.date_input))
                    self.assertTrue(at.dataframe)
                elif tab=='Планировщик':
                    self.assertIsNone(at.number_input(key='flood_Q').value)
                    at.button(key='flood_optimize').click().run()
                    self.assertFalse(at.exception)
                    self.assertTrue(any('Введите объём' in w.value for w in at.info))
                elif tab=='О данных':
                    self.assertTrue(at.dataframe)

    def test_optimizer_ui_calculates_and_keeps_inputs_between_tabs(self):
        records=pd.DataFrame(dict(date=pd.date_range(self.now.date()+timedelta(days=1),periods=7).date,
                         confirmed=True,capacity_t=2.,source='synthetic input for test only',published=self.now.date()))
        with patch('app.flood_view.load_points',side_effect=lambda **kw:F.load_points(self.now,**kw)), \
             patch('app.flood_workspace.st.data_editor',return_value=records):
            at=AppTest.from_string('from app.flood_view import render_flood\nrender_flood()',default_timeout=30)
            at.session_state['flood_decision']=self.now.date()
            at.session_state['flood_tab']='Планировщик'
            at.run()
            for key,value in [('flood_Q',5.),('flood_road_cost',2.),('flood_backup_cost',9.),('flood_holding',0.)]:
                at.number_input(key=key).set_value(value)
            at.button(key='flood_optimize').click().run()
            self.assertFalse(at.exception,[e.message for e in at.exception])
            self.assertEqual(at.metric[0].value,'5.0 т')
            self.assertTrue(at.get('plotly_chart'))
            next(w for w in at.sidebar.date_input if w.label=='Дата оценки / проезда').set_value(records.date.iloc[-1]).run()
            at.button(key='flood_optimize').click().run()
            self.assertFalse(at.exception,[e.message for e in at.exception])
            self.assertEqual(at.metric[0].value,'2.0 т')
            self.assertEqual(at.metric[1].value,'3.0 т')
            at.session_state['flood_tab']='Прогноз воды'
            at.run()
            at.session_state['flood_tab']='Планировщик'
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.number_input(key='flood_Q').value,5.)


if __name__=='__main__':
    unittest.main()
