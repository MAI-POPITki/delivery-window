"""М1 по реальным уровням: причинность, источники и исторический интерфейс."""
from datetime import date
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno import flood_m1 as M


class FloodM1Tests(unittest.TestCase):
    def test_real_observations_have_expected_sources_and_negative_levels(self):
        for pid,(code,source) in M.STATIONS.items():
            s=M.load_levels(pid)
            self.assertGreater(s.notna().sum(),5000)
            meta=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['station']['properties']['post_code'],code)
            self.assertTrue(meta['sha256'])
        self.assertLess(M.load_levels('lena_yakutsk').min(),0)
        self.assertLess(M.load_levels('ishim_ishim').min(),0)

    def test_future_observations_cannot_change_forecast(self):
        original=M.load_levels('lena_yakutsk')
        first,_=M.forecast('lena_yakutsk','2023-05-01')
        changed=original.copy()
        changed.loc[changed.index>pd.Timestamp('2023-05-01')]+=100000
        with patch('okno.flood_m1.load_levels',return_value=changed):
            second,_=M.forecast('lena_yakutsk','2023-05-01')
        pd.testing.assert_frame_equal(first,second)

    def test_training_and_calibration_precede_forecast_year(self):
        for path in M.MODELS.glob('*.json'):
            if path.name=='validation.json':continue
            d=json.loads(path.read_text(encoding='utf-8'))
            self.assertLessEqual(pd.Timestamp(d['train_until']).year,d['year']-2)
            self.assertEqual(d['calibration_year'],d['year']-1)
            for lead,method in d['methods'].items():
                self.assertEqual(method,min(d['calibration_mae'][lead],key=d['calibration_mae'][lead].get))

    def test_calendar_lags_do_not_skip_missing_days(self):
        s=pd.Series(np.arange(60.),index=pd.date_range('2020-01-01',periods=60))
        s.loc['2020-01-15']=np.nan
        f=M.feature_frame(s,7)
        self.assertTrue(pd.isna(f.loc['2020-01-22','h7']))
        self.assertEqual(f.loc['2020-01-23','h7'],s.loc['2020-01-16'])

    def test_dates_change_prediction_and_horizons_are_correct(self):
        a,meta=M.forecast('ishim_ishim','2023-04-01')
        b,_=M.forecast('ishim_ishim','2023-04-05')
        self.assertEqual(a.lead.tolist(),list(range(1,8)))
        self.assertEqual(a.date.iloc[0],pd.Timestamp('2023-04-02'))
        self.assertNotEqual(a.predicted_cm.iloc[0],b.predicted_cm.iloc[0])
        self.assertTrue((a.lower_cm<=a.predicted_cm).all())
        self.assertTrue((a.predicted_cm<=a.upper_cm).all())

    def test_no_fake_current_forecast_or_transfer_to_kurgan(self):
        with self.assertRaisesRegex(ValueError,'нет измерения'):
            M.forecast('lena_yakutsk','2026-03-20')
        with self.assertRaisesRegex(ValueError,'недостаточно'):
            M.forecast('tobol_kurgan','2023-04-01')

    def test_report_uses_future_years_and_exposes_spring_undercoverage(self):
        scores=pd.read_csv(M.ROOT/'reports/flood_m1_scores.csv')
        self.assertEqual(set(scores.year),set(range(2019,2025)))
        spring=scores[(scores.pid=='ishim_ishim')&(scores.lead==7)&(scores.season=='spring')]
        self.assertLess(spring.m1.mean(),spring.persistence.mean())
        self.assertLess(spring.coverage.mean(),.75)
        status=json.loads((M.MODELS/'validation.json').read_text())['ishim_ishim:7']
        self.assertTrue(status['point_skill_success'])
        self.assertFalse(status['interval_valid'])
        self.assertFalse(status['success'])

    def test_replay_ui_changes_date_and_shows_model_provenance(self):
        at=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=60)
        at.session_state['m1_flood_tab']='М1: прогноз'
        at.session_state['m1_flood_version']='v1 · точечная модель'
        at.run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        before=at.metric[1].value
        at.sidebar.date_input(key='m1_issue_lena_yakutsk').set_value(date(2023,5,10)).run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertNotEqual(before,at.metric[1].value)
        self.assertTrue(at.get('plotly_chart'))
        self.assertTrue(at.dataframe)
        self.assertTrue(any('диапазон' in w.value.lower() for w in at.warning))

    def test_replay_ui_explains_missing_station(self):
        at=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=60).run()
        at.sidebar.selectbox(key='m1_flood_region').select('Курганская область').run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Курган' in w.value for w in at.warning))
        self.assertFalse(at.metric)


if __name__=='__main__':unittest.main()
