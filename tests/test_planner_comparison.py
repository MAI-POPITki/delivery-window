"""Shared inputs, physical capacity and expected costs in the two M3 plans."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from okno.planner import Params, compare_ml_weekly_plans


def forecast(**changes):
    return dict(issue_day=150, q10=210, q50=210, q90=210, calendar=210) | changes


def params(**changes):
    return replace(Params(Q_t=2, K_t_week=1, K_extra_t_week=0,
        c_road=10, c_extra=5, c_exp=100, c_hold_week=2,
        close_margin_days=0, start_day=150, transit_days=7), **changes)


class WeeklyMlComparisonTests(unittest.TestCase):
    def test_both_plans_keep_capacity_quantity_and_a_common_risk_distribution(self):
        p = params(Q_t=4.5, K_extra_t_week=.5)
        f = dict(issue_day=150, q10=170, q50=185, q90=200, calendar=185)
        weekly, summary = compare_ml_weekly_plans(f, 150, p)
        self.assertEqual(set(weekly.rule), {'ml', 'economic_ml'})
        for rule, plan in weekly.groupby('rule'):
            with self.subTest(rule=rule):
                self.assertTrue(plan.base_t.between(0, p.K_t_week).all())
                self.assertTrue(plan.extra_t.between(0, p.K_extra_t_week).all())
                self.assertLessEqual(plan.shipped_t.sum(), p.Q_t+1e-9)
                self.assertAlmostEqual(plan.expected_delivered_t.sum()+summary.loc[rule,'expected_backup_t'], p.Q_t)
                self.assertAlmostEqual(summary.loc[rule,'expected_cost_rub'],
                    summary.loc[rule,['road_cost_rub','extra_cost_rub','holding_cost_rub','backup_cost_rub']].sum())
        normal = weekly[weekly.rule.eq('ml')].reset_index(drop=True)
        economic = weekly[weekly.rule.eq('economic_ml')].reset_index(drop=True)
        np.testing.assert_array_equal(normal.dispatch_day, economic.dispatch_day)
        np.testing.assert_array_equal(normal.p_closed, economic.p_closed)
        self.assertLessEqual(summary.loc['economic_ml','expected_cost_rub'], summary.loc['ml','expected_cost_rub']+1e-8)

    def test_ordinary_plan_is_uniform_and_cannot_hide_insufficient_capacity(self):
        f = dict(issue_day=150, q10=171, q50=171, q90=171, calendar=171)
        weekly, summary = compare_ml_weekly_plans(f, 150, params(Q_t=5))
        normal = weekly[weekly.rule.eq('ml')]
        np.testing.assert_array_equal(normal.shipped_t.iloc[:3], [1, 1, 1])
        self.assertEqual(normal.shipped_t.iloc[3:].sum(), 0)
        self.assertAlmostEqual(summary.loc['ml','unallocated_t'], 2)
        self.assertAlmostEqual(summary.loc['ml','expected_backup_t'], 2)
        self.assertEqual(summary.loc['ml','backup_cost_rub'], 200)

    def test_high_storage_prices_delay_the_economic_plan_without_changing_the_ml_window(self):
        a, _ = compare_ml_weekly_plans(forecast(), 150, params(c_hold_week=0), end_day=200)
        b, _ = compare_ml_weekly_plans(forecast(), 150, params(c_hold_week=10), end_day=200)
        pd.testing.assert_frame_equal(a[a.rule.eq('ml')], b[b.rule.eq('ml')])
        early = a[a.rule.eq('economic_ml')]
        late = b[b.rule.eq('economic_ml')]
        self.assertGreater((late.dispatch_day*late.shipped_t).sum(),
                           (early.dispatch_day*early.shipped_t).sum())
        self.assertEqual(late.extra_t.sum(), 0)
        self.assertEqual(late.shipped_t.sum(), 2)

    def test_a_cheaper_backup_is_an_explicit_costed_choice(self):
        weekly, summary = compare_ml_weekly_plans(forecast(), 150,
                                                 params(c_road=100, c_exp=20))
        self.assertEqual(weekly[weekly.rule.eq('economic_ml')].shipped_t.sum(), 0)
        self.assertEqual(summary.loc['economic_ml','expected_backup_t'], 2)
        self.assertEqual(summary.loc['economic_ml','expected_cost_rub'], 40)
        self.assertGreater(summary.loc['ml','expected_cost_rub'], 40)

    def test_no_past_dispatch_or_forced_future_opening(self):
        # Closure precedes the future January start: do not move the closure
        # samples to that planned start and make impossible road trips look safe.
        f = dict(issue_day=100, q10=140, q50=140, q90=140, calendar=140)
        weekly, summary = compare_ml_weekly_plans(f, 100, params())
        self.assertTrue(weekly.dispatch_day.ge(150).all())
        self.assertTrue(weekly.p_closed.eq(1).all())
        self.assertEqual(weekly.shipped_t.sum(), 0)
        self.assertTrue(summary.expected_backup_t.eq(2).all())
        late, _ = compare_ml_weekly_plans(forecast(), 190, params())
        self.assertTrue(late.dispatch_day.ge(190).all())

    def test_future_issues_are_rejected_and_actual_targets_never_enter_plans(self):
        f = forecast()
        with self.assertRaises(ValueError):
            compare_ml_weekly_plans(f, 149, params())
        before, scores = compare_ml_weekly_plans(dict(f, actual=190), 150, params())
        after, changed_scores = compare_ml_weekly_plans(dict(f, actual=290), 150, params())
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(scores, changed_scores)

    def test_empty_horizon_retains_the_entire_order_and_its_cost(self):
        weekly, summary = compare_ml_weekly_plans(forecast(), 310, params())
        self.assertTrue(weekly.empty)
        self.assertTrue(summary.expected_backup_t.eq(2).all())
        self.assertTrue(summary.expected_cost_rub.eq(200).all())


if __name__ == '__main__':
    unittest.main()
