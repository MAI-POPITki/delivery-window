"""Подключение региона без скрытого изменения обучения или обещаний проезда."""
import unittest
from okno.regions import REGIONS, points_of, region_of, all_gauges, label
from okno import forecast


class KomiCoverageTests(unittest.TestCase):
    def test_region_reuses_existing_point_without_training_it(self):
        self.assertEqual(points_of('komi'), ['gauge_70378'])
        self.assertEqual(region_of('gauge_70378'), 'komi')
        self.assertNotIn(70378, all_gauges())
        self.assertEqual(label('gauge_70378'), 'Мезень — Макарьб')

    def test_unverified_opening_is_unavailable(self):
        self.assertIsNone(forecast.forecast('gauge_70378',2027,67,'open'))
        self.assertIn('open',REGIONS['komi']['disabled_kinds'])


if __name__=='__main__':
    unittest.main()
