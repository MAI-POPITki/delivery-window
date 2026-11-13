"""A nearby road and a light-vehicle opening cannot advance the selected plan."""
import unittest
from unittest.mock import patch

import pandas as pd

from okno.ice import to_season_day
from okno.journey import planned_opening


def event(road, kind='open', planned='2026-12-01', actual='', publication='2026-10-01', source='https://example.org/order'):
    return dict(road=road, kind=kind, announced_date=planned,
                announced_published=publication, source_announced=source,
                actual_date=actual, actual_published='2026-12-05' if actual else '',
                source_actual=source if actual else '')


class PlannedOpeningTests(unittest.TestCase):
    def opening(self, rows, **kwargs):
        with patch('okno.journey.road_rows', return_value=pd.DataFrame(rows)):
            return planned_opening('gauge_3414', 2027, kwargs.pop('decision', '2026-11-13'), **kwargs)

    def test_multiple_roads_require_selection(self):
        rows=[event('Ранняя', planned='2026-11-20'), event('Выбранная', planned='2026-12-20')]
        self.assertIsNone(self.opening(rows))
        result=self.opening(rows, road='Выбранная')
        self.assertEqual(result[0], to_season_day(pd.Timestamp('2026-12-20'), 2027))
        self.assertIn('Выбранная', result[1])
        self.assertIsNone(self.opening(rows, road='Другая'))

    def test_ten_tonne_opening_is_not_used_for_twenty_tonnes(self):
        rows=[event('Дорога', kind='open10', planned='2026-11-20'), event('Дорога', planned='2026-12-20')]
        self.assertEqual(self.opening(rows)[0], to_season_day(pd.Timestamp('2026-12-20'), 2027))
        self.assertEqual(self.opening(rows, gross_t=10)[0], to_season_day(pd.Timestamp('2026-11-20'), 2027))

    def test_actual_overrides_schedule_only_once_available(self):
        rows=[event('Дорога', actual='2026-12-04')]
        self.assertIn('график', self.opening(rows)[1])
        result=self.opening(rows, decision='2026-12-06')
        self.assertEqual(result[0], to_season_day(pd.Timestamp('2026-12-04'), 2027))
        self.assertIn('факт', result[1])

    def test_future_publication_or_missing_source_is_not_evidence(self):
        self.assertIsNone(self.opening([event('Дорога', publication='2026-11-20')]))
        self.assertIsNone(self.opening([event('Дорога', source='')]))


if __name__ == '__main__':
    unittest.main()
