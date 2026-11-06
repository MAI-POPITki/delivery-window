"""Проверки соблюдения срока заказа и учёта груза, не попавшего в зимник."""

import unittest

from okno.planner import Params
from scripts.nizhneyansk_batagay import compare


class SmallOrderReportTests(unittest.TestCase):
    def setUp(self):
        self.params = Params(Q_t=45, K_t_week=28, K_extra_t_week=18.7,
                             c_road=22447.5, c_extra=4489.5, c_exp=500000, c_hold_week=149.8)
        self.q = {"q10": 245.0, "q50": 251.0, "q90": 260.0, "calendar": 252.7}

    def test_deadline_excludes_incomplete_week_and_both_plans_meet_order(self):
        totals, schedule = compare(self.params, issue=203, q=self.q, deadline_day=212)
        self.assertTrue((schedule.completion_day <= 212).all())
        self.assertEqual(set(schedule.week_day), {203})
        for row in totals.itertuples():
            self.assertAlmostEqual(row.base_t, 28)
            self.assertAlmostEqual(row.extra_t, 17)
            self.assertAlmostEqual(row.expected_shortage_t, 0)
        self.assertAlmostEqual(totals.iloc[0].total_rub, totals.iloc[1].total_rub)

    def test_unscheduled_cargo_is_charged_to_expensive_channel(self):
        self.params.K_t_week = 14
        self.params.K_extra_t_week = 0
        totals, _ = compare(self.params, issue=203, q=self.q, deadline_day=212)
        for row in totals.itertuples():
            self.assertAlmostEqual(row.expected_road_t, 14)
            self.assertAlmostEqual(row.expected_shortage_t, 31)
            self.assertAlmostEqual(row.expensive_rub, 31 * self.params.c_exp)

    def test_no_complete_week_before_deadline_is_reported(self):
        with self.assertRaisesRegex(ValueError, "полностью завершённой недели"):
            compare(self.params, issue=203, q=self.q, deadline_day=209)


if __name__ == "__main__":
    unittest.main()
