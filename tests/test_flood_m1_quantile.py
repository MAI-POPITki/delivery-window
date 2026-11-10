"""Причинность квантильного прогноза, калибровка и воспроизводимость отчёта."""
from datetime import date
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno import flood_m1_quantile as M
from scripts.train_flood_m1_quantile import metrics


class QuantileM1Tests(unittest.TestCase):
    def test_finite_sample_correction_and_nonnegative_expansion(self):
        self.assertEqual(M.correction([1, 2, 3, 4]), 4)
        self.assertEqual(M.correction([-5, -4, -3, -2]), 0)
        with self.assertRaises(ValueError):
            M.correction([np.nan])

    def test_future_measurements_cannot_change_forecast(self):
        original = M.load_levels('lena_yakutsk')
        first, _ = M.forecast('lena_yakutsk', '2023-05-01')
        changed = original.copy()
        changed.loc[changed.index > '2023-05-01'] += 100000
        with patch.object(M, 'load_levels', return_value=changed):
            second, _ = M.forecast('lena_yakutsk', '2023-05-01')
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.lower_cm <= first.predicted_cm).all())
        self.assertTrue((first.predicted_cm <= first.upper_cm).all())

    def test_all_models_use_only_previous_years_and_record_seasonal_samples(self):
        paths = list(M.MODELS.glob('*_20*.json'))
        self.assertGreater(len(paths), 20)
        for path in paths:
            meta = json.loads(path.read_text())
            self.assertLessEqual(pd.Timestamp(meta['train_until']).year, meta['year'] - 2)
            self.assertEqual(meta['calibration_year'], meta['year'] - 1)
            for lead, method in meta['methods'].items():
                self.assertEqual(method, min(meta['calibration_mae'][lead], key=meta['calibration_mae'][lead].get))
                for entry in meta['corrections'][lead].values():
                    self.assertGreaterEqual(entry['cm'], 0)
                    self.assertGreaterEqual(entry['n'], 60)

    def test_seasons_are_decision_dates_not_future_target_dates(self):
        dates = pd.to_datetime(['2023-03-31', '2023-04-01', '2023-07-31', '2023-08-01'])
        self.assertEqual(M.spring(dates).tolist(), [False, True, True, False])
        series = pd.Series(100., index=pd.date_range('2021-01-01', '2023-08-31'))
        rows = pd.DataFrame({'issue': dates, 'target_date': dates + pd.Timedelta(days=7),
                             'lead': 7, 'h0': 100.})
        meta = {'methods': {'7': 'persistence'}, 'corrections': {'7': {'spring': {'cm': 20}, 'other': {'cm': 5}}}}
        with patch.object(M, 'quantiles', return_value=np.full((4, 3), 100.)):
            pred = M.predict_rows(series, rows, 2023, None, meta)
        self.assertEqual(pred.lower.tolist(), [95, 80, 80, 95])
        self.assertEqual(pred.upper.tolist(), [105, 120, 120, 105])

    def test_saved_scores_reproduce_from_raw_predictions(self):
        p = pd.read_csv(M.ROOT / 'reports/flood_m1_quantile_predictions.csv.gz', parse_dates=['issue', 'target_date'])
        actual = metrics(p)
        saved = pd.read_csv(M.ROOT / 'reports/flood_m1_quantile_scores.csv')
        pd.testing.assert_frame_equal(actual, saved, check_exact=False, rtol=1e-10)
        self.assertTrue((saved.width_cm >= 0).all())
        self.assertTrue((saved.interval_score >= saved.width_cm - 1e-8).all())
        self.assertEqual(set(saved.lead), set(range(1, 8)))
        self.assertEqual(set(saved[saved.year.eq(2025)].pid), {'ural_orenburg', 'ural_orsk'})

    def test_missing_live_data_and_missing_year_rejected(self):
        with self.assertRaisesRegex(ValueError, 'нет измерения'):
            M.forecast('lena_yakutsk', '2026-05-01')
        with self.assertRaisesRegex(ValueError, 'нет версии'):
            M.forecast('lena_yakutsk', '2018-05-01')

    def test_ui_switches_versions_dates_and_explains_calculation(self):
        app = AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()', default_timeout=60)
        app.session_state['m1_flood_tab'] = 'М1: прогноз'
        app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        before = app.metric[1].value
        app.sidebar.date_input(key='m1_issue_lena_yakutsk').set_value(date(2023, 5, 10)).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(before, app.metric[1].value)
        self.assertTrue(any('Как получен' in e.label for e in app.expander))
        self.assertTrue(any('поправ' in e.value.lower() or 'расширен' in e.value.lower() for e in app.markdown))
        app.sidebar.radio(key='m1_flood_version').set_value('v1 · точечная модель').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get('plotly_chart'))

    def test_v2_map_report_and_data_tabs_render(self):
        app = AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()', default_timeout=60)
        for tab in ('Карта', 'Проверка', 'Данные'):
            app.session_state['m1_flood_tab'] = tab
            app.run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            if tab == 'Карта':
                self.assertTrue(app.get('plotly_chart'))
            if tab == 'Проверка':
                self.assertTrue(any('2024' in e.value and 'покрыт' in e.value.lower() for e in app.markdown))
            if tab == 'Данные':
                self.assertTrue(any('CQR' in e.value for e in app.markdown))


if __name__ == '__main__':
    unittest.main()
