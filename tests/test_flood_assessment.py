from datetime import date
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from okno import flood_assessment as A
from okno import flood_m1_quantile as M


class AssessmentTests(unittest.TestCase):
    def test_no_future_satellite_in_joint_result(self):
        f,_=M.forecast('ishim_ishim','2023-05-01')
        result=A.assess('ishim_ishim',date(2023,5,1),f,500)
        self.assertIsNone(result['passage_probability'])
        if result['satellite']:
            meta=result['satellite']['meta']
            self.assertLessEqual(pd.Timestamp(meta['product_generated_at']).date(),date(2023,5,1))
            self.assertLessEqual(pd.Timestamp(meta['acquired_at']).date(),date(2023,5,1))

    def test_missing_satellite_does_not_fabricate_road_safety(self):
        f,_=M.forecast('ishim_ishim','2023-05-01')
        with patch.object(A.S,'select_run',return_value=None):
            result=A.assess('ishim_ishim',date(2023,5,1),f,500)
        self.assertIsNone(result['satellite']);self.assertIsNone(result['passage_probability'])
        self.assertTrue(0<result['probability']<1)

    def test_joint_page_has_forecast_observation_and_dispatch(self):
        app=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=60).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertEqual(app.tabs[0].label,'Общий результат')
        self.assertTrue(any('Что ожидается' in item.value for item in app.markdown))
        self.assertTrue(any('Что наблюдается' in item.value for item in app.markdown))
        self.assertTrue(app.button(key='own_m3_run'))

    def test_independent_benchmark_splits_and_frozen_model(self):
        manifest=json.loads((M.ROOT/'data/flood/benchmark/manifest.json').read_text())
        splits=manifest['splits']
        self.assertEqual(len(splits['test']),90);self.assertEqual(len(splits['bolivia']),15)
        self.assertFalse(set(splits['test'])&set(splits['train']))
        self.assertFalse(set(splits['test'])&set(splits['valid']))
        self.assertTrue(all(c.startswith('Bolivia_') for c in splits['bolivia']))
        folder=M.ROOT/'models/flood_m2_candidate'
        meta=json.loads((folder/'metadata.json').read_text())
        self.assertEqual(hashlib.sha256((folder/'s2_toa_water.cbm').read_bytes()).hexdigest(),meta['model_sha256'])
        self.assertEqual(set(meta['train']),set(splits['train']))

    def test_plan_inputs_survive_detail_tabs_and_threshold_cannot_move_to_other_post(self):
        app=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=60).run()
        app.number_input(key='own_m3_road').set_value(150.).run()
        app.number_input(key='own_m3_threshold').set_value(500.).run()
        app.session_state['m1_flood_tab']='М1: прогноз';app.run()
        app.session_state['m1_flood_tab']='Общий результат';app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key='own_m3_road').value,150.)
        self.assertEqual(app.number_input(key='own_m3_threshold').value,500.)
        app.sidebar.selectbox(key='m1_flood_region').select('Тюменская область').run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.number_input(key='own_m3_threshold').value)

    def test_benchmark_counts_support_claim_not_accuracy_alone(self):
        scores=pd.read_csv(M.ROOT/'reports/flood_m2_benchmark_chips.csv')
        for split in ['test','bolivia']:
            def iou(method):
                g=scores[(scores.split==split)&(scores.method==method)]
                totals=g[['tp','fp','fn']].sum()
                return totals.tp/totals.sum()
            self.assertGreater(iou('candidate'),iou('current_spectral'))
            self.assertEqual(iou('all_land'),0)
