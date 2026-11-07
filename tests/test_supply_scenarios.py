import unittest
import numpy as np
import pandas as pd
from itertools import product
from okno.corridor import CorridorConfig, simulate_corridor
from okno.supply_scenarios import Closure, SeasonalWindow, ServiceDays, DeclaredCalendar, calendar_from_frame, compare_supply


class SupplyTests(unittest.TestCase):
    def test_external_calendar_rejects_missing_duplicate_and_invalid_values(self):
        for days, values in [([0, 0], [0, 1]), ([0, 2], [0, 1]),
                             ([0, 1], [1, 2]), ([0, 1], [1, np.nan])]:
            with self.assertRaises(ValueError):
                calendar_from_frame(pd.DataFrame({'day': days, 'available': values}), 2)
        calendar = calendar_from_frame(pd.DataFrame({'day': [1, 0], 'available': [0, 1]}), 2)
        np.testing.assert_array_equal(calendar.calendar(2), [True, False])

    def test_optimizer_matches_independent_enumeration_for_all_three_day_calendars(self):
        # Integer transport network: independently enumerate every dispatch and
        # final-leg shipment. Check service first, then complete cost ledger.
        cfg = self.cfg(horizon_days=3, rail_lead_time_days=0, rail_cap_tons=2,
            crossing_cap_normal_tons=2, initial_yak_tons=0, yak_warehouse_cap_tons=3,
            rail_cost_per_ton=3, crossing_cost_normal=5,
            nb_holding_cost_per_ton_day=2, yak_holding_cost_per_ton_day=1,
            shortage_penalty_per_ton_day=7)
        for mask in product([False, True], repeat=3):
            best = (float('inf'), float('inf'))
            for decisions in product(range(3), repeat=6):
                hub, nb, yak, pending, cost = 3, 0, 0, 0, 0
                feasible = True
                for day in range(3):
                    dispatch, shipment = decisions[2*day:2*day+2]
                    if dispatch > hub or shipment > nb+dispatch or (shipment and not mask[day]):
                        feasible = False
                        break
                    hub -= dispatch
                    nb += dispatch-shipment
                    yak += shipment
                    pending += 1
                    served = min(yak, pending)
                    yak -= served
                    pending -= served
                    if yak > 3:
                        feasible = False
                        break
                    cost += dispatch*3+shipment*5+nb*2+yak+pending*7
                if feasible:
                    best = min(best, (pending, cost+hub*3+(hub+nb)*5))
            result = compare_supply(cfg, DeclaredCalendar(mask))['Экономический план'][0]
            self.assertAlmostEqual(result['terminal_backlog_tons'], best[0], places=5)
            self.assertAlmostEqual(result['total_cost_rub'], best[1], places=3)

    def cfg(self, **kw):
        values = dict(horizon_days=14, daily_demand_tons=1,
            rail_lead_time_days=1, rail_cap_tons=10, crossing_cap_normal_tons=10,
            crossing_cap_alt_tons=0, initial_yak_tons=2, yak_warehouse_cap_tons=20)
        values.update(kw)
        return CorridorConfig(**values)

    def test_calendars_have_distinct_boundaries(self):
        np.testing.assert_array_equal(SeasonalWindow(1, 3).calendar(4), [0, 1, 1, 0])
        np.testing.assert_array_equal(Closure(1, 3).calendar(4), [1, 0, 0, 1])
        np.testing.assert_array_equal(ServiceDays((0, 3)).calendar(4), [1, 0, 0, 1])
        with self.assertRaises(ValueError):
            SeasonalWindow(3, 3).calendar(4)

    def test_all_modes_respect_calendar_pipeline_and_mass(self):
        cfg = self.cfg()
        for schedule in [Closure(4, 8), SeasonalWindow(3, 10), ServiceDays((2, 6, 10))]:
            for summary, daily in compare_supply(cfg, schedule).values():
                self.assertAlmostEqual(summary['mass_balance_error_tons'], 0, places=6)
                self.assertTrue((daily.loc[~schedule.calendar(14), 'shipped_norm_t'] == 0).all())
                self.assertEqual(daily.rail_arrived_nb_t.iloc[0], 0)
                np.testing.assert_allclose(daily.rail_arrived_nb_t.iloc[1:], daily.hub_dispatched_t.iloc[:-1], atol=1e-6)
                self.assertLessEqual(daily.inv_yak_t.max(), 20+1e-6)

    def test_closed_window_does_not_hide_unserved_orders(self):
        cfg = self.cfg(initial_yak_tons=0)
        summary, daily = compare_supply(cfg, ServiceDays((0,)))['Экономический план']
        self.assertAlmostEqual(summary['terminal_backlog_tons'], 14)
        self.assertAlmostEqual(summary['served_tons'], 0)
        self.assertGreater(summary['terminal_commitment_rub'], 0)

    def test_window_optimizer_cannot_be_worse_than_feasible_baseline(self):
        results = compare_supply(self.cfg(), SeasonalWindow(3, 10))
        a, b = [s for s, _ in results.values()]
        self.assertLessEqual(b['terminal_backlog_tons'], a['terminal_backlog_tons']+1e-5)
        self.assertLessEqual(b['total_cost_rub'], a['total_cost_rub']+1e-3)

    def test_explicit_calendar_matches_legacy_gap(self):
        cfg = self.cfg()
        old = simulate_corridor(cfg, 4, 8, 4, 8, economic=True)
        new = compare_supply(cfg, Closure(4, 8))['Экономический план'][0]
        self.assertAlmostEqual(old['total_cost_rub'], new['total_cost_rub'], places=3)


if __name__ == '__main__':
    unittest.main()
