"""Boundary, leakage, completeness and UI checks for the summer beta."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from okno import low_water as L


def low_water_app():
    from app.low_water_view import render_low_water
    render_low_water()


class LowWaterTests(unittest.TestCase):
    def test_low_probability_includes_ties(self):
        self.assertEqual(L.below_probability(np.array([-1.,0.]),0.,0.),.75)
        self.assertEqual(L.below_probability(np.array([-1.,0.]),0.,-2.),.25)
        with self.assertRaises(ValueError):L.below_probability([],0.,0.)
        with self.assertRaises(ValueError):L.below_probability([0.],0.,np.nan)

    def test_threshold_never_uses_calibration_or_evaluation_levels(self):
        s=L.M.load_levels(L.PID)
        changed=s.copy();changed.loc[changed.index.year>=2022]=10000.
        self.assertEqual(L.threshold(s,2023),L.threshold(changed,2023))

    def test_calibration_ignores_future(self):
        s=L.M.load_levels(L.PID);model,_,cal,_=L.bundle(2023)
        changed=s.copy();changed.loc[changed.index.year>=2023]=-10000.
        other=L.fit_calibration(changed,2023,model)
        for h in range(1,8):
            self.assertEqual(cal[h]['method'],other[h]['method'])
            np.testing.assert_array_equal(cal[h]['errors'],other[h]['errors'])

    def test_forecast_ignores_future_and_returns_seven_dates(self):
        issue=pd.Timestamp('2023-08-20');forecast,meta=L.forecast(issue)
        s=L.M.load_levels(L.PID);s.loc[s.index>issue]=10000.
        with patch.object(L.M,'load_levels',return_value=s):other,_=L.forecast(issue)
        assert_frame_equal(forecast,other)
        self.assertEqual(forecast.lead.tolist(),list(range(1,8)))
        self.assertEqual(meta['calibration_year'],2022)
        self.assertTrue(forecast.probability.between(0,1).all())
        self.assertTrue((forecast.lower_cm<=forecast.upper_cm).all())

    def test_missing_history_is_not_interpolated(self):
        s=L.M.load_levels(L.PID);s.loc['2023-08-15']=np.nan
        with patch.object(L.M,'load_levels',return_value=s):
            with self.assertRaisesRegex(ValueError,'31 ежедневное'):L.forecast('2023-08-20')

    def test_season_end_and_invalid_inputs(self):
        f,_=L.forecast('2023-09-29');self.assertEqual(f.lead.tolist(),[1])
        for issue in ['2023-01-01','2023-09-30','2026-08-20']:
            with self.assertRaises(ValueError):L.forecast(issue)
        with self.assertRaises(ValueError):L.forecast('2023-08-20',float('inf'))

    def test_dashboard_deep_link(self):
        app=AppTest.from_file(str(L.M.ROOT/'app/dashboard.py'))
        app.query_params['module']='low_water'
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key='service_mode').value,'Маловодье · бета')
        self.assertEqual(app.title[0].value,'Маловодье · бета')

    def test_ui_archive_and_operator_threshold(self):
        app=AppTest.from_function(low_water_app).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn('Архивная проверка',app.info[0].value)
        self.assertEqual(len(app.metric),3)
        app.checkbox(key='low_water_custom').check().run(timeout=30)
        self.assertFalse(app.exception)
        app.number_input(key='low_water_limit').set_value(-50.).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[2].value,'-50.0 см')
        app.checkbox(key='low_water_show_fact').check().run(timeout=30)
        self.assertFalse(app.exception)


if __name__=='__main__':unittest.main()
