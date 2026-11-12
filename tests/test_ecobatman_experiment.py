"""Checks for scientific failure modes: missing months and temporal leakage."""
import gzip
import unittest

import numpy as np
import pandas as pd

from scripts import experiment_ecobatman as E


class EcobatmanExperimentTest(unittest.TestCase):
    def observations(self, code):
        return E.parse_monthly(gzip.decompress(
            (E.ROOT / 'data/flood/ecobatman_research' / f'{code}.html.gz').read_bytes()), code)

    def test_archive_gaps_are_not_filled(self):
        obs = self.observations('11412')
        self.assertTrue(obs.loc['2000':'2009'].observed_m3s.isna().all())
        self.assertEqual(obs.loc['2011':'2020'].observed_m3s.count(), 120)

    def test_station_identity_is_required(self):
        raw = gzip.decompress((E.ROOT / 'data/flood/ecobatman_research/19063.html.gz').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Station code'):
            E.parse_monthly(raw, '11412')

    def test_partial_month_is_excluded(self):
        dates = pd.date_range('2020-01-01', '2020-03-31')
        frame = pd.DataFrame({'river_discharge': 10.0}, index=dates)
        frame.loc['2020-02-29', 'river_discharge'] = np.nan
        result = E.complete_months(frame)
        self.assertEqual(result.loc['2020-01-01'], 10.0)
        self.assertTrue(np.isnan(result.loc['2020-02-01']))
        self.assertEqual(result.loc['2020-03-01'], 10.0)

    def test_test_observations_do_not_change_calibration(self):
        index = pd.date_range('1997-01-01', '2020-12-01', freq='MS')
        obs = pd.DataFrame({'observed_m3s': 20.0}, index=index)
        daily = pd.DataFrame({'river_discharge': 10.0},
                             index=pd.date_range('1997-01-01', '2020-12-31'))
        test, calibration, _ = E.prepare_pairs(obs, daily)
        changed = obs.copy()
        changed.loc['2011':'2020', 'observed_m3s'] = 10000.0
        altered, calibration2, _ = E.prepare_pairs(changed, daily)
        pd.testing.assert_frame_equal(calibration, calibration2)
        pd.testing.assert_series_equal(test.corrected_m3s, altered.corrected_m3s)
        self.assertTrue((calibration.factor == 2).all())

    def test_sparse_training_does_not_relax_gate(self):
        obs = self.observations('11412')
        daily = E.F.daily_frame(E.F.read_response(
            E.F.DATA / 'history/ishim_ishim.json')['payload'], E.F.POINTS['ishim_ishim'])
        test, calibration, count = E.prepare_pairs(obs, daily)
        self.assertEqual(count, 48)
        self.assertEqual(len(test), 120)
        self.assertTrue(calibration.empty)
        self.assertTrue(test.corrected_m3s.isna().all())


if __name__ == '__main__':
    unittest.main()
