"""Economic decisions, independent optima, service and information boundaries."""
from dataclasses import replace
from itertools import product
import unittest

import numpy as np
import pandas as pd

from okno.corridor import CorridorConfig, compare_policies, simulate_corridor, weekly_dispatch_comparison


def config(**kwargs):
    base = CorridorConfig(daily_demand_tons=1, horizon_days=4,
        rail_lead_time_days=0, rail_cap_tons=4, rail_cost_per_ton=8,
        nb_holding_cost_per_ton_day=3, crossing_cap_normal_tons=4,
        crossing_cap_alt_tons=1, crossing_cost_normal=2, crossing_cost_alt=20,
        yak_warehouse_cap_tons=4, yak_holding_cost_per_ton_day=2,
        yak_overflow_cost_per_ton_day=5, shortage_penalty_per_ton_day=100,
        initial_yak_tons=0)
    return replace(base, **kwargs)


class EconomicPlannerTests(unittest.TestCase):
    def run_plan(self, cfg, close=10, opening=10, **kwargs):
        return simulate_corridor(cfg, close, opening, close, opening,
                                 economic=True, return_daily=True, **kwargs)

    def test_just_in_time_cost_wins_without_delivering_faster(self):
        cfg = config(horizon_days=3, daily_demand_tons=10,
                     rail_cap_tons=40, crossing_cap_normal_tons=40,
                     yak_warehouse_cap_tons=40)
        result, daily = self.run_plan(cfg)
        old = simulate_corridor(cfg, 10, 10, 10, 10)
        np.testing.assert_allclose(daily.hub_dispatched_t, [10, 10, 10])
        self.assertEqual(result['total_cost_rub'], 300)
        self.assertEqual(result['days_with_deficit'], old['days_with_deficit'])
        self.assertLess(result['total_cost_rub'], old['total_cost_rub'])
        self.assertEqual(result['economic_plan_count'], 1)

    def test_storage_and_reserve_prices_change_loading(self):
        cfg = config(crossing_cap_normal_tons=2, yak_warehouse_cap_tons=1,
                     yak_holding_cost_per_ton_day=5)
        _, expensive = self.run_plan(cfg, 1, 3)
        _, cheap = self.run_plan(replace(cfg, crossing_cost_alt=3), 1, 3)
        self.assertAlmostEqual(expensive.inv_yak_t.iloc[0], 1)
        self.assertAlmostEqual(cheap.inv_yak_t.iloc[0], 0)
        self.assertLess(expensive.shipped_alt_t.sum(), cheap.shipped_alt_t.sum())

    def test_waiting_can_be_cheaper_but_the_deadline_still_holds(self):
        cfg = config(horizon_days=5, crossing_cap_normal_tons=2,
                     yak_warehouse_cap_tons=1, shortage_penalty_per_ton_day=0)
        low, _ = self.run_plan(cfg, 1, 3)
        high, _ = self.run_plan(replace(cfg, shortage_penalty_per_ton_day=100), 1, 3)
        self.assertGreater(low['shortage_ton_days'], high['shortage_ton_days'])
        self.assertLess(low['tons_alt_transport'], high['tons_alt_transport'])
        self.assertTrue(low['terminal_service_complete'])
        self.assertEqual(low['served_tons'], 5)

    def test_terminal_value_does_not_reward_free_initial_stock(self):
        cfg = config(horizon_days=2, initial_yak_tons=2,
                     yak_holding_cost_per_ton_day=1)
        result, _ = self.run_plan(cfg)
        self.assertEqual(result['total_rail_tons'], 0)
        self.assertEqual(result['cash_cost_rub'], 1)
        self.assertEqual(result['terminal_commitment_rub'], 20)
        self.assertEqual(result['total_cost_rub'], 21)

    def test_zero_prices_do_not_allow_abandoning_feasible_demand(self):
        cfg = config(rail_cost_per_ton=0, crossing_cost_normal=0,
                     crossing_cost_alt=0, nb_holding_cost_per_ton_day=0,
                     yak_holding_cost_per_ton_day=0,
                     shortage_penalty_per_ton_day=0)
        result, _ = self.run_plan(cfg)
        self.assertAlmostEqual(result['served_tons'], 4)
        self.assertTrue(result['terminal_service_complete'])

    def test_impossible_deadline_maximises_service_before_minimising_cost(self):
        cfg = config(horizon_days=3, daily_demand_tons=10, rail_cap_tons=1,
                     shortage_penalty_per_ton_day=0)
        result, daily = self.run_plan(cfg)
        self.assertAlmostEqual(result['served_tons'], 3, places=5)
        self.assertAlmostEqual(result['terminal_backlog_tons'], 27, places=5)
        self.assertFalse(result['terminal_service_complete'])
        self.assertTrue(daily.optimizer_status.eq('optimal_with_unavoidable_shortfall').all())
        np.testing.assert_allclose(daily.forecast_terminal_shortfall_t, 27, atol=1e-6)

    def test_future_forecasts_and_actual_dates_do_not_change_earlier_actions(self):
        cfg = config(horizon_days=12, rail_lead_time_days=2, initial_yak_tons=2)
        forecasts = pd.DataFrame([
            dict(issue_day=0, planned_close_day=6, planned_open_day=9),
            dict(issue_day=4, planned_close_day=5, planned_open_day=11)])
        poisoned = forecasts.copy()
        poisoned.loc[1, ['planned_close_day', 'planned_open_day']] = [5, 50]
        args = dict(economic=True, return_daily=True)
        _, a = simulate_corridor(cfg, 6, 9, 6, 9, forecasts=forecasts, **args)
        _, b = simulate_corridor(cfg, 6, 9, 6, 9, forecasts=poisoned, **args)
        pd.testing.assert_frame_equal(a[a.day < 4], b[b.day < 4])
        _, b = simulate_corridor(cfg, 7, 11, 6, 9, forecasts=forecasts, **args)
        pd.testing.assert_frame_equal(a[a.day < 6], b[b.day < 6])
        self.assertTrue((a.forecast_issue_day <= a.day).all())

    def test_cheaper_overflow_requires_filling_the_primary_warehouse(self):
        cfg = config(horizon_days=3, initial_yak_tons=3,
                     yak_warehouse_cap_tons=1, yak_overflow_capacity_tons=2,
                     yak_holding_cost_per_ton_day=10,
                     yak_overflow_cost_per_ton_day=1)
        result, daily = self.run_plan(cfg)
        np.testing.assert_allclose(daily.inv_yak_t, [2, 1, 0])
        self.assertEqual(result['cost_holding_yak_base_rub'], 20)
        self.assertEqual(result['cost_holding_yak_overflow_rub'], 1)
        self.assertEqual(result['total_cost_rub'], 51)
        self.assertEqual(result['economic_plan_count'], 1)

    def test_matches_independent_exhaustive_optimum(self):
        # Enumerate physical two-day schedules, independently of LP equations.
        cfg = config(horizon_days=2, crossing_cap_normal_tons=2,
                     rail_cap_tons=2, yak_warehouse_cap_tons=1,
                     crossing_cap_alt_tons=1, crossing_cost_alt=6,
                     shortage_penalty_per_ton_day=5)
        best = np.inf
        for sends in product(range(3), repeat=2):
            for crossings in product(range(3), range(2)):
                hub, nb, yak, waiting, cost = 2, 0, 0, 0, 0
                feasible = True
                for day in range(2):
                    if sends[day] > hub:
                        feasible = False; break
                    hub -= sends[day]; nb += sends[day]; waiting += 1
                    if crossings[day] > nb:
                        feasible = False; break
                    nb -= crossings[day]; yak += crossings[day]
                    served = min(yak, waiting)
                    yak -= served; waiting -= served
                    if yak > 1:
                        feasible = False; break
                    cost += sends[day]*8+crossings[day]*(2 if day == 0 else 6)
                    cost += nb*3+yak*2+waiting*5
                if feasible and waiting == 0:
                    best = min(best, cost+hub*8+(hub+nb)*2)
        result, _ = self.run_plan(cfg, 1, 3)
        self.assertAlmostEqual(result['total_cost_rub'], best)

    def test_equal_information_has_equal_economic_results(self):
        forecast = pd.DataFrame([dict(issue_day=0, planned_close_day=1, planned_open_day=3)])
        results = compare_policies(config(), 1, 3, 1, 3, 1, 3, forecast)
        _, calendar = self.run_plan(config(), 1, 3)
        a = calendar.drop(columns='forecast_issue_day')
        b = results['economic_ml'][1].drop(columns='forecast_issue_day')
        pd.testing.assert_frame_equal(a, b)

    def test_weekly_flows_reconcile_without_summing_inventory_or_extra_days(self):
        cfg = config(horizon_days=8, rail_lead_time_days=1, initial_yak_tons=1)
        results = compare_policies(cfg, 2, 5, 2, 5)
        weekly = weekly_dispatch_comparison(results, '2024-02-27')
        self.assertEqual(len(weekly), 4)
        for policy in ('ml', 'economic_ml'):
            with self.subTest(policy=policy):
                summary, daily = results[policy]
                rows = weekly[weekly.policy.eq(policy)].reset_index(drop=True)
                self.assertEqual(rows.days.tolist(), [7, 1])
                self.assertEqual(rows.week_start_date.tolist(), ['2024-02-27', '2024-03-05'])
                self.assertEqual(rows.week_end_date.tolist(), ['2024-03-04', '2024-03-05'])
                for column in ('hub_dispatched_t', 'rail_arrived_nb_t', 'shipped_norm_t',
                               'shipped_alt_t', 'demanded_t', 'served_t'):
                    self.assertAlmostEqual(rows[column].sum(), daily[column].sum())
                self.assertAlmostEqual(rows.shipped_total_t.sum(), summary['tons_normal_transport']+summary['tons_alt_transport'])
                self.assertAlmostEqual(rows.weekly_cost_rub.sum(), summary['cash_cost_rub'])
                for column in ('inv_nb_t', 'inv_yak_t', 'backlog_t', 'in_transit_t', 'hub_remaining_t'):
                    np.testing.assert_allclose(rows[column], daily[column].iloc[[6, 7]])

    def test_synthetic_weekly_comparison_does_not_invent_calendar_dates(self):
        results = compare_policies(config(horizon_days=8), 2, 5, 2, 5)
        weekly = weekly_dispatch_comparison(results, None)
        self.assertTrue(weekly.week_start_date.eq('').all())
        self.assertTrue(weekly.week_end_date.eq('').all())
        self.assertEqual(weekly.day_start.tolist(), [0, 0, 7, 7])
        self.assertEqual(set(results), {'reactive', 'calendar', 'calendar_q20', 'ml', 'economic_calendar', 'economic_ml'})


if __name__ == '__main__':
    unittest.main()
