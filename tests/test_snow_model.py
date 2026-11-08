"""Saved experiment evidence, application fallbacks and new-station UI."""
import json
import hashlib
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno import forecast as FC,snow
from okno.northeast import STATIONS
from okno.regions import all_gauges

ROOT=Path(__file__).resolve().parents[1]


class SnowArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info=json.loads((ROOT/'models/m1_snow.json').read_text())
        cls.checked=pd.read_csv(ROOT/'reports/snow_backtest.csv')

    def test_training_inputs_and_saved_artifacts_are_present(self):
        for relative,digest in self.info['sha256'].items():
            with self.subTest(file=relative):
                self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),digest)
        for versions in self.info['models'].values():
            for version in versions:
                for key in ('file','baseline_file'):
                    self.assertTrue((ROOT/'models'/version[key]).is_file())

    def test_chronological_results_match_reported_metrics_without_invented_closing_targets(self):
        self.assertTrue((self.checked.train_max_year<self.checked.hydro_year).all())
        self.assertTrue((self.checked.issue_day<self.checked.target_day).all())
        self.assertFalse(((self.checked.kind=='close')&(self.checked.source_group=='new_gauges')).any())
        for kind,quality in self.info['validation'].items():
            part=self.checked[self.checked.kind.eq(kind)]
            self.assertEqual(len(part),quality['n'])
            self.assertAlmostEqual(float((part.base_q50-part.target_day).abs().mean()),quality['baseline_mae_days'])
            self.assertAlmostEqual(float((part.snow_q50-part.target_day).abs().mean()),quality['snow_mae_days'])
            self.assertTrue(part[f'base_q10'].le(part[f'base_q50']).all())
            self.assertTrue(part[f'snow_q50'].le(part[f'snow_q90']).all())

    def test_ground_data_is_available_for_every_supported_gauge_and_pilot(self):
        manifest=json.loads((ROOT/'data/snow_depth/manifest.json').read_text())
        expected={f'gauge_{c}.csv' for c in all_gauges()}|{'yakutsk_crossing.csv'}
        self.assertEqual(set(manifest['files']),expected)
        self.assertEqual(manifest['stored_units'],'cm')
        for c in STATIONS:
            path=ROOT/f'data/weather/gauges/gauge_{c}.csv'
            w=pd.read_csv(path)
            self.assertFalse(w.temperature_2m_mean.isna().any())
            self.assertFalse(w.date.duplicated().any())
            self.assertGreaterEqual(w.date.max(),'2018-05-20')

    def test_historical_snow_uses_past_model_and_leaves_absent_closing_unavailable(self):
        ordinary=FC.forecast('gauge_1018',2018,72,'open')
        experiment=FC.forecast('gauge_1018',2018,72,'open',use_snow=True)
        self.assertTrue(ordinary['updated_model_applied'])
        self.assertFalse(ordinary['snow_model_applied'])
        self.assertTrue(experiment['snow_model_applied'])
        self.assertIn('reference',experiment['prediction_model'])
        self.assertEqual(experiment['snow_features']['snow_on_ice_observed'],1.)
        self.assertGreaterEqual(experiment['snow_features']['snow_on_ice_age_days'],0.)
        self.assertAlmostEqual(experiment['q50']-ordinary['q50'],experiment['snow_delta_days'])
        self.assertTrue(experiment['q10']<=experiment['q50']<=experiment['q90'])
        self.assertIsNone(FC.forecast('gauge_1018',2018,192,'close',use_snow=True))

    def test_current_year_policy_follows_group_validation_and_never_uses_old_river_snow(self):
        result=FC.forecast('gauge_1018',2027,65,'open',use_snow=True)
        quality=self.info['validation']['open']['groups']['new_gauges']['without_river']
        self.assertEqual(result['snow_model_applied'],quality['improvement_pct']>0)
        if result['snow_model_applied']:
            self.assertEqual(result['snow_features']['snow_on_ice_observed'],0.)
            self.assertTrue(np.isnan(result['snow_features']['snow_on_ice_cm']))
        # The separate Lena tributary model must survive the update.
        from okno.regions import basin_points
        point=next(p for p in basin_points() if FC._weather_path(p).exists())
        original=FC.forecast(point,2024,192,'close')
        with_snow=FC.forecast(point,2024,192,'close',use_snow=True)
        self.assertFalse(with_snow['snow_model_applied'])
        self.assertEqual(with_snow['prediction_model'],'m1_close_lena.cbm')
        for q in ('q10','q50','q90'):
            self.assertEqual(original[q],with_snow[q])


class SnowDashboardTests(unittest.TestCase):
    def test_new_regions_and_archive_keep_missing_closing_explicit(self):
        app=AppTest.from_file(str(ROOT/'app/dashboard.py'),default_timeout=90)
        app.session_state['dashboard_open']=True
        app.run()
        for region,count in [('magadan',4),('chukotka',9)]:
            app.sidebar.selectbox(key='filter_region').select(region).run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(len(app.sidebar.selectbox(key='filter_point').options),count)
            self.assertTrue(any('нет дат начала таяния' in item.value for item in app.info))
            self.assertTrue(any('последний архивный сезон' in item.value for item in app.caption))
        app.sidebar.selectbox(key='filter_region').select('yakutia').run()
        app.sidebar.selectbox(key='filter_point').select('gauge_1018').run()
        app.sidebar.selectbox(key='filter_season').select(2018).run()
        import datetime as dt
        app.sidebar.date_input(key='filter_date').set_value(dt.date(2017,11,12)).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any('Сдвиг при учёте снега' in item.value for item in app.markdown))
        archive=next(d.value for d in app.dataframe if 'Снег на льду, см' in d.value.columns)
        self.assertTrue(pd.to_datetime(archive['Дата замера']).dt.year.isin([2017,2018]).all())
        app.sidebar.checkbox(key='snow_depth_enabled').uncheck().run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertFalse(any('Сдвиг при учёте снега' in item.value for item in app.markdown))


if __name__=='__main__':
    unittest.main()
