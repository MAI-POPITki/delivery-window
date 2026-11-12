"""Repeated archive throttling stops requests for the rest of the run."""
import unittest
from unittest.mock import Mock, patch

from scripts import download_data as download


class WeatherRateLimitTests(unittest.TestCase):
    def test_repeated_429_stops_other_points_without_network_calls(self):
        response=Mock(status_code=429, text='rate limit')
        point=dict(id='test', lat=62., lon=129.)
        with patch.object(download, 'RATE_LIMITED', False), \
             patch.object(download, 'FAILED', []), \
             patch.object(download.requests, 'get', return_value=response) as get, \
             patch.object(download.time, 'sleep') as sleep, \
             patch.object(download, 'log'):
            with self.assertRaises(download.RateLimited):
                download._get_weather_chunk(point, '2026-11-01', '2026-11-02')
            self.assertEqual(get.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            download.safe_update(dict(id='next'), None, None, None)
            self.assertEqual(download.FAILED, ['next'])
            self.assertEqual(get.call_count, 3)


if __name__ == '__main__':
    unittest.main()
