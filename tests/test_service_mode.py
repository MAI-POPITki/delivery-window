import unittest
from dataclasses import replace

from okno.corridor import CorridorConfig, service_frontier, service_plan
from okno.corridor_data import case_inputs

REAL = dict(rail_cost_per_ton=11003., crossing_cost_normal=373., crossing_cost_alt=25000.,
            yak_holding_cost_per_ton_day=22.3, nb_holding_cost_per_ton_day=22.3)


class ServiceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(CorridorConfig(), **REAL)
        cls.c = case_inputs(2024)
        cls.frontier, cls.base = service_frontier(cls.cfg, cls.c)

    def test_waiting_falls_as_lever_rises(self):
        w = self.frontier.sort_values('penalty').shortage_ton_days.to_numpy()
        self.assertTrue((w[1:] <= w[:-1] + 1e-6).all())

    def test_plan_respects_waiting_limit(self):
        for share in (1.0, 0.5):
            plan, lever = service_plan(self.cfg, self.c, self.base['shortage_ton_days'] * share)
            self.assertIsNotNone(plan)
            self.assertLessEqual(plan['shortage_ton_days'], self.base['shortage_ton_days'] * share + 1e-6)
            self.assertEqual(plan['terminal_backlog_tons'], 0)

    def test_waiting_price_is_not_counted_as_cost(self):
        lo = service_frontier(self.cfg, self.c, penalties=(0.,))[0].operating_cost_rub.iloc[0]
        hi = service_frontier(replace(self.cfg, shortage_penalty_per_ton_day=0.), self.c, penalties=(0.,))[0].operating_cost_rub.iloc[0]
        self.assertAlmostEqual(lo, hi)

    def test_impossible_limit_is_reported(self):
        plan, lever = service_plan(self.cfg, self.c, -1.)
        self.assertIsNone(plan)


if __name__ == '__main__':
    unittest.main()
