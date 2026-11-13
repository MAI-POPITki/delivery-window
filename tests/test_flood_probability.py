import json
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from okno import flood_probability as P


class FloodProbabilityTests(unittest.TestCase):
    def test_smoothed_probabilities_and_monotonicity(self):
        samples=np.arange(-10.,11.)
        values=[P.probability(samples,100,1,t) for t in range(80,122)]
        self.assertTrue(all(0<p<1 for p in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        with self.assertRaises(ValueError):P.probability([],100,1,100)
        with self.assertRaises(ValueError):P.probability(samples,100,0,100)

    def test_calibration_precedes_decision_and_matches_model(self):
        for path in P.DATA.glob('*_20*.json'):
            d=json.loads(path.read_text());pid=path.stem.rsplit('_',1)[0];year=d['year']
            c=P.calibration(pid,year)
            self.assertTrue(c.issue.dt.year.eq(year-1).all())
            self.assertTrue((c.issue+pd.to_timedelta(c.lead,unit='D')<pd.Timestamp(year,1,1)).all())

    def test_forecast_dates_and_sample_sizes(self):
        forecast,_=P.M.forecast('ishim_ishim','2023-05-01')
        p=P.exceedance('ishim_ishim','2023-05-01',forecast,500)
        self.assertTrue(p.probability.between(0,1).all());self.assertTrue((p.n>=60).all())
        with self.assertRaises(ValueError):P.exceedance('ishim_ishim','2023-05-02',forecast,500)

    def test_ui_shows_real_probabilities_after_threshold_entered(self):
        app=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=60).run()
        app.radio(key='own_probability_kind_lena_yakutsk').set_value('Своя отметка').run()
        app.number_input(key='own_probability_threshold_lena_yakutsk').set_value(300.).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any('Оценка превышения, %' in df.value.columns for df in app.dataframe))

    def test_saved_probability_report_recomputes_brier_and_keeps_failures(self):
        p=pd.read_csv(P.M.ROOT/'reports/flood_probability_predictions.csv.gz')
        self.assertTrue(p.probability.between(0,1).all())
        selected=p[(p.pid=='ural_orenburg')&(p.lead==7)&(p.threshold_quantile==.9)]
        self.assertGreater(((selected.probability-selected.event)**2).mean(),
                           ((selected.persistence_probability-selected.event)**2).mean())
        checks=json.loads((P.DATA/'validation.json').read_text())
        self.assertTrue(any(v['skill'] for v in checks.values()))
        self.assertTrue(any(not v['skill'] for v in checks.values()))
