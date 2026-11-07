"""Проверки связи физических дат с планом без подмены фактов прогнозом."""
import unittest
from datetime import date
import pandas as pd
from okno.navigation import navigation_windows, observation_as_of, compare_navigation
from okno.corridor import CorridorConfig


class NavigationTests(unittest.TestCase):
    def forecast(self):
        return {k+'_date': date(2027, 3, d) for k,d in
                [('calendar',4), ('q10',3), ('q50',4), ('q90',5)]}

    def test_shared_shift_and_end(self):
        windows=navigation_windows(self.forecast(), date(2027,3,1), 14, date(2027,3,14), 2)
        self.assertEqual([w.start for w in windows.values()], [5,4,5,6])
        self.assertEqual({w.end for w in windows.values()}, {13})
        with self.assertRaises(ValueError):
            navigation_windows(self.forecast(), date(2027,3,1), 14, date(2027,3,4), 2)

    def test_same_dates_produce_same_plan(self):
        windows=navigation_windows(self.forecast(), date(2027,3,1), 14, date(2027,3,14))
        cfg=CorridorConfig(horizon_days=14)
        results=compare_navigation(cfg,windows)
        pd.testing.assert_frame_equal(results['calendar'][1], results['q50'][1])
        for summary,daily in results.values():
            self.assertAlmostEqual(summary['mass_balance_error_tons'],0,places=5)

    def test_satellite_observation_never_comes_from_future(self):
        frame=pd.DataFrame(dict(point=['p','p','other'],date=['2026-11-01','2026-11-10','2026-11-05'],p_ice=[.1,.9,.7]))
        self.assertAlmostEqual(observation_as_of(frame,'p','2026-11-07').p_ice,.1)
        self.assertIsNone(observation_as_of(frame,'p','2026-10-01'))


if __name__=='__main__':
    unittest.main()
