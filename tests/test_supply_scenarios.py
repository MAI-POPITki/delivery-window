import unittest
import numpy as np
from okno.corridor import CorridorConfig, simulate_corridor
from okno.supply_scenarios import Closure, SeasonalWindow, ServiceDays, compare_supply


class SupplyTests(unittest.TestCase):
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
