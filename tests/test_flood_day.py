from datetime import date
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from okno.flood_day import forecast_day,efficiency,event_definition
from okno.flood_m1 import ROOT,load_levels


class DayTests(unittest.TestCase):
    def test_selected_date_controls_actual_forecast(self):
        a=forecast_day('ural_orenburg','2024-04-03','2024-04-04')
        b=forecast_day('ural_orenburg','2024-04-03','2024-04-10')
        self.assertEqual(a['lead'],1);self.assertEqual(b['lead'],7)
        self.assertEqual(a['threshold'],760.)
        self.assertNotEqual(a['probability'],b['probability'])
        self.assertLessEqual(a['probability'],a['probability_upper'])
        with self.assertRaises(ValueError):forecast_day('ural_orenburg','2024-04-03','2024-04-11')

    def test_no_substitution_of_quantile_for_bankfull(self):
        e=event_definition('lena_yakutsk',load_levels('lena_yakutsk'),2023)
        self.assertFalse(e['physical'])
        e=event_definition('ural_orsk',load_levels('ural_orsk'),2023)
        self.assertEqual((e['lower_threshold'],e['threshold']),(360.,450.))

    def test_efficiency_recomputed_from_saved_predictions(self):
        raw=pd.read_csv(ROOT/'reports/flood_day_predictions.csv.gz')
        self.assertEqual(set(raw.lead),set(range(1,8)))
        for (pid,h),g in raw.groupby(['pid','lead']):
            s=efficiency(pid,h)
            self.assertAlmostEqual(s['brier'],float(np.mean((g.probability-g.event)**2)))
            self.assertEqual(s['n'],s['tp']+s['fn']+s['fp']+s['tn'])
        self.assertTrue((raw.target_date>raw.issue).all())

    def test_ui_target_date_changes_probability_and_shows_efficiency(self):
        app=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=90)
        app.session_state['m1_flood_point']='ural_orenburg'
        app.session_state['m1_issue_ural_orenburg']=date(2024,4,3)
        app.run();self.assertFalse(app.exception,[e.message for e in app.exception])
        before=app.metric[0].value
        app.date_input(key='flood_target_date_ural_orenburg').set_value(date(2024,4,4)).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertNotEqual(before,app.metric[0].value)
        self.assertEqual(app.metric[0].label,'Вероятность разлива')
        self.assertTrue(any('Выигрыш относительно' in m.value for m in app.markdown))
