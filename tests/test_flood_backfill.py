import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from okno import flood_observations as O


class BackfillTest(unittest.TestCase):
    now=datetime(2026,11,13,tzinfo=timezone.utc)

    def response(self,url,params,timeout):
        end=datetime.fromisoformat(params['date_loc']).replace(tzinfo=timezone.utc)
        levels=[[(end-timedelta(days=i)).strftime('%Y-%m-%d %H:%M'),-i] for i in range(8,0,-1)]
        payload=dict(name='Лена, Якутск',levels=levels,prev_y=[[levels[0][0],9999]])
        response=requests.Response();response.status_code=200
        response._content=json.dumps(payload).encode();response.url=url+'?date='+params['date_loc']
        return response

    def test_weekly_windows_make_full_month_and_preserve_first_seen(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(O.requests,'get',side_effect=self.response) as get:
            result=O.backfill_meteo(directory=tmp,now=self.now)
            self.assertEqual(get.call_count,5)
            self.assertTrue(result['continuous'])
            self.assertEqual(result['days'],31)
            before=O.observations('lena_yakutsk',directory=tmp,now=self.now)
            self.assertFalse(any(r['value_cm']==9999 for r in before))
            # Same data is not newly available at an earlier decision time.
            self.assertEqual(O.observations('lena_yakutsk',directory=tmp,now=self.now-timedelta(seconds=1)),[])
            O.backfill_meteo(directory=tmp,now=self.now+timedelta(hours=1))
            after=O.observations('lena_yakutsk',directory=tmp,now=self.now+timedelta(hours=1))
            for old in before:
                self.assertEqual(next(r for r in after if r['id']==old['id'])['first_seen_at'],old['first_seen_at'])

    def test_failure_preserves_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lena_yakutsk.json';path.write_text('{"records":[]}')
            before=path.read_bytes()
            with patch.object(O.requests,'get',side_effect=requests.ConnectionError('offline')):
                with self.assertRaises(RuntimeError):O.backfill_meteo(directory=tmp,now=self.now)
            self.assertEqual(path.read_bytes(),before)

    def test_unsupported_post_and_unbounded_window_rejected(self):
        for kwargs in [dict(pid='ural_orenburg'),dict(days=10000),dict(days=0)]:
            with self.assertRaises(ValueError):O.backfill_meteo(now=self.now,**kwargs)


if __name__=='__main__':unittest.main()
