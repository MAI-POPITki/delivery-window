"""Operational and time-isolation invariants of the scenario replay."""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from okno.yakutsk_replay import (
    ACTUAL, ANNOUNCED, RULES, Scenario, capacity_table, fuse_observation, simulate, simulate_all,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/scenarios"


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = pd.read_csv(INPUTS / "yakutsk_spring_2026_forecasts.csv")
        cls.o = pd.read_csv(INPUTS / "yakutsk_spring_2026_observations.csv")
        cls.results = simulate_all(cls.f, cls.o)

    def test_independent_capacity_arithmetic_and_last_open_day(self):
        actual, planned = capacity_table(), capacity_table(schedule=ANNOUNCED)
        self.assertEqual(actual.base_capacity_t.sum(), 4440)
        self.assertEqual(actual.extra_capacity_t.sum(), 2220)
        self.assertEqual(planned.base_capacity_t.sum(), 4740)
        self.assertEqual(actual.groupby('gross_limit_t').base_capacity_t.sum().to_dict(),
                         {20.: 480., 30.: 1440., 40.: 2520.})
        self.assertEqual(actual.set_index('date').loc['2026-04-15', 'base_capacity_t'], 80.)
        later = capacity_table(replace(Scenario(), demand_date='2026-04-20'))
        self.assertEqual(later[later.date > '2026-04-15'].base_capacity_t.sum(), 0.)

    def test_mass_cost_and_fleet_limits_for_every_method(self):
        for name, result in self.results.items():
            with self.subTest(name=name):
                d, r = result.schedule, result.summary
                self.assertAlmostEqual(d.delivered_t.sum()+r['backup_t'], 5000.)
                self.assertTrue((d.base_t <= d.base_capacity_t + 1e-7).all())
                self.assertTrue((d.extra_t <= d.extra_capacity_t + 1e-7).all())
                self.assertTrue((d.remaining_t >= -1e-7).all())
                self.assertEqual(d.loc[~d.workday, 'delivered_t'].sum(), 0.)
                self.assertAlmostEqual(r['road_rub'], r['delivered_t']*4000.)
                self.assertAlmostEqual(r['extra_rub'], r['extra_t']*1500.)
                self.assertAlmostEqual(r['backup_rub'], r['backup_t']*20000.)
                self.assertAlmostEqual(r['cost_rub'], sum(r[k] for k in
                    ['road_rub','extra_rub','hold_rub','backup_rub']))
                expected_hold = (d.delivered_t * (pd.Timestamp('2026-04-16')-d.date).dt.days * 100/7).sum()
                self.assertAlmostEqual(r['hold_rub'], expected_hold)

    def test_no_extra_fleet_respects_capacity_and_counts_backup_once(self):
        s = replace(Scenario(), extra_trucks=0)
        r = simulate('asap', self.f, self.o, s).summary
        self.assertEqual(r['delivered_t'], 4440)
        self.assertEqual(r['backup_t'], 560)
        self.assertEqual(r['backup_rub'], 560*20000)
        self.assertEqual(r['recipient_wait_t_days'], 560*40)

    def test_adaptive_calendar_uses_actual_today_but_not_future_truth(self):
        # Prior to 27 March, both worlds have identical observations.
        changed = replace(ACTUAL, date30='2026-03-25', date20='2026-04-02')
        for name in RULES:
            with self.subTest(name=name):
                a = self.results[name].schedule
                b = simulate(name, self.f, self.o, actual_schedule=changed).schedule
                pd.testing.assert_frame_equal(a[a.date < '2026-03-25'].reset_index(drop=True),
                                              b[b.date < '2026-03-25'].reset_index(drop=True))
        r = self.results['calendar'].summary
        self.assertAlmostEqual(r['delivered_t'], 5000.)
        self.assertAlmostEqual(r['extra_t'], 560.)

    def test_future_forecasts_and_unpublished_images_do_not_change_past_decisions(self):
        poisoned = self.f.copy()
        future = pd.to_datetime(poisoned.issue_date) >= '2026-03-25'
        poisoned.loc[future, ['q10','q50','q90']] = [150.,151.,152.]
        obs = self.o.copy()
        obs.loc[obs.date >= '2026-03-18', 'p_ice'] = 0.
        for name in ['forecast', 'forecast_m2']:
            original = self.results[name].schedule
            new = simulate(name, poisoned, obs).schedule
            # The first changed image is not usable until 20 March.
            cutoff = '2026-03-20' if name == 'forecast_m2' else '2026-03-25'
            pd.testing.assert_frame_equal(original[original.date < cutoff].reset_index(drop=True),
                                          new[new.date < cutoff].reset_index(drop=True))

    def test_image_availability_and_freshness_are_enforced(self):
        d = self.results['forecast_m2'].decisions
        used = d[d.image_date.notna()]
        self.assertEqual(used.date.min(), pd.Timestamp('2026-03-20'))
        self.assertTrue((used.image_available_date <= used.date).all())
        self.assertTrue(((used.date-used.image_date).dt.days <= 14).all())
        self.assertFalse(d.loc[d.date < '2026-03-20', 'image_date'].notna().any())
        self.assertFalse(d.loc[d.date.between('2026-04-02','2026-04-12'), 'image_date'].notna().any())

    def test_no_truncation_has_exactly_zero_satellite_effect(self):
        q = np.array([232.,242.,252.])
        new, changed = fuse_observation(q, 198., .998)
        np.testing.assert_array_equal(new, q)
        self.assertFalse(changed)
        q2, changed = fuse_observation([190.,200.,210.], 200., .99)
        self.assertTrue(changed)
        self.assertGreater(q2[1], 200.)
        pd.testing.assert_frame_equal(self.results['forecast'].schedule.drop(columns='rule'),
                                      self.results['forecast_m2'].schedule.drop(columns='rule'))

    def test_prepared_weather_and_model_inputs_have_no_future_targets(self):
        f = self.f
        self.assertTrue((pd.to_datetime(f.weather_cutoff_date) <= pd.to_datetime(f.issue_date)-pd.Timedelta(days=5)).all())
        train = pd.read_csv(ROOT/'data/processed/m1_dataset_close.csv', usecols=['hydro_year'])
        self.assertLess(train.hydro_year.max(), 2026)
        self.assertTrue((f.q10 <= f.q50).all() and (f.q50 <= f.q90).all())
        self.assertEqual(f.issue_date.nunique(), 31)

    def test_invalid_configuration_and_ambiguous_inputs_are_rejected(self):
        for values in [dict(Q_t=0),dict(c_hold_week=-1),dict(tare_t=40),dict(Q_t=float('nan')),
                       dict(phase20_days=20,phase30_days=10),dict(n_scenarios=0)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Scenario(**values)
        with self.assertRaises(ValueError):
            simulate('forecast', pd.concat([self.f,self.f.iloc[[0]]]))
        bad = self.o.copy()
        bad.loc[0,'available_date'] = '2025-01-01'
        with self.assertRaises(ValueError):
            simulate('forecast_m2', self.f, bad)


if __name__ == '__main__':
    unittest.main()
