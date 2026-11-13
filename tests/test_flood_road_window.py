from datetime import date
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from okno.flood_road_window import analog_paths, transitions, summarize


class RoadDatesTests(unittest.TestCase):
    def test_closure_equality_and_recovery_reset(self):
        d=pd.date_range('2024-04-02',periods=8)
        r=transitions([8,10,6,5,9,5,5,5],d,7,10,8,2,1)
        self.assertEqual(r['close'],pd.Timestamp('2024-04-03'))
        self.assertEqual(r['reopen'],pd.Timestamp('2024-04-09'))

    def test_censoring_and_already_above_are_not_fabricated_dates(self):
        d=pd.date_range('2024-04-02',periods=3)
        r=transitions([11,11,11],d,12,10,8)
        self.assertEqual(r['close'],pd.Timestamp('2024-04-01'))
        self.assertIsNone(r['reopen']);self.assertTrue(r['censored'])
        self.assertIsNone(transitions([1,2,3],d,1,10,8)['close'])

    def test_already_closed_road_can_reopen_below_closing_threshold(self):
        d=pd.date_range('2024-04-02',periods=3)
        r=transitions([6,5,5],d,7,10,8,3,0,initial_closed=True)
        self.assertEqual(r['reopen'],pd.Timestamp('2024-04-04'))
        self.assertIsNone(transitions([6,5,5],d,7,10,8,3,0,initial_closed=False)['reopen'])

    def test_full_trip_deadline_and_missing_scenarios_preserved(self):
        p=pd.DataFrame({'a':[8,10,6,5,5],'b':[5,5,5,5,5]},index=pd.date_range('2024-04-02',periods=5))
        r=summarize(p,7,10,8,2,0,1)
        self.assertEqual(r['close']['n'],1)
        self.assertEqual(r['last_departure'],date(2024,4,2))
        self.assertEqual(r['scenarios'],2)
        self.assertTrue(summarize(p,7,10,8,2,0,3)['departure_expired'])

    def test_future_observations_cannot_change_analogs(self):
        idx=pd.date_range('2000-01-01','2025-12-31')
        s=pd.Series(100+50*np.sin(2*np.pi*idx.day_of_year/365.25),index=idx)
        issue=pd.Timestamp('2020-04-01')
        p,m=analog_paths(s,issue)
        s.loc[issue+pd.Timedelta(days=1):]=1e9
        p2,m2=analog_paths(s,issue)
        pd.testing.assert_frame_equal(p,p2)
        self.assertTrue((m.source_end<pd.Timestamp('2020-01-01')).all())
        self.assertTrue(m.year.is_unique)
        self.assertEqual(len(p),60)

    def test_bad_data_and_no_history_fail(self):
        d=pd.date_range('2024-04-02',periods=3)
        for args in [([1,np.nan,3],d,1,10,8),([1,2,3],d,1,8,10),([1,2,3],d,1,10,8,0)]:
            with self.assertRaises(ValueError):transitions(*args)
        s=pd.Series(1.,index=pd.date_range('2023-01-01','2024-05-01'))
        with self.assertRaises(ValueError):analog_paths(s,'2024-04-01')

    def test_ui_road_dates_without_costs_and_high_water_example(self):
        app=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=90).run()
        self.assertFalse(app.exception)
        pid=app.sidebar.selectbox(key='m1_flood_point').value
        self.assertIsNone(app.number_input(key='road_window_close_'+pid).value)
        app.text_input(key='road_window_route_'+pid).input('Тестовый участок · не фактические сведения')
        app.text_input(key='road_window_evidence_'+pid).input('Тестовые пороги для проверки интерфейса')
        app.number_input(key='road_window_close_'+pid).set_value(1000.)
        app.number_input(key='road_window_reopen_'+pid).set_value(900.).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any('Ожидаемое закрытие' in m.value for m in app.markdown))
        app.radio(key='road_window_mode_'+pid).set_value('Пример по высокой воде · не дорожный прогноз').run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any('не установленный порог' in w.value for w in app.warning))
