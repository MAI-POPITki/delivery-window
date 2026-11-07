import unittest

from okno.m2 import check_against_m1


class CheckAgainstM1Tests(unittest.TestCase):
    q = (230.0, 240.0, 250.0)  # вскрытие: q10, q50, q90 (дни сезона)

    def test_ice_before_forecast_agrees(self):
        self.assertEqual(check_against_m1(*self.q, 220, 0.95)[0], "agree")

    def test_ice_after_q90_conflicts(self):
        self.assertEqual(check_against_m1(*self.q, 255, 0.95)[0], "conflict")

    def test_water_before_q10_conflicts(self):
        self.assertEqual(check_against_m1(*self.q, 225, 0.05)[0], "conflict")

    def test_water_in_autumn_is_not_a_conflict(self):
        self.assertEqual(check_against_m1(*self.q, 60, 0.05)[0], "early")

    def test_uncertain_image_is_not_used(self):
        self.assertEqual(check_against_m1(*self.q, 245, 0.5)[0], "unsure")


if __name__ == "__main__":
    unittest.main()
