import gzip
import hashlib
import json
import unittest
from streamlit.testing.v1 import AppTest
from okno.flood_sites import SITES,EXTRA_SITES
from okno import flood_m1_quantile as M
from okno import flood_m2 as S


class ExpansionTests(unittest.TestCase):
    def test_two_new_regions_have_real_stations_not_copied_models(self):
        raw=gzip.decompress((M.ROOT/'data/flood/sites/posts_geojson.json.gz').read_bytes())
        source=json.loads(raw)['features']
        self.assertEqual(len(SITES),7)
        self.assertEqual(len({s['region'] for s in SITES.values()}),6)
        for pid,site in EXTRA_SITES.items():
            meta=json.loads((M.DATA/f'{pid}.manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(hashlib.sha256(raw).hexdigest(),meta['sha256'])
            self.assertTrue(any(f==meta['station'] for f in source))
            self.assertEqual(meta['station']['properties']['post_code'],site['gauge_code'])
            self.assertNotIn(pid,M.STATIONS)

    def test_new_regions_have_real_verified_satellite_runs(self):
        for pid in EXTRA_SITES:
            runs=S.available_runs(pid);self.assertTrue(runs,pid)
            for path in runs:
                meta=S.load_run(path)
                self.assertEqual(meta['pid'],pid)
                self.assertTrue(meta['observation']['item_id'].startswith('S2'))

    def test_new_region_explains_coverage_and_opens_satellite(self):
        app=AppTest.from_file(str(M.ROOT/'app/dashboard.py'),default_timeout=90)
        app.query_params['module']='flood';app.run()
        app.sidebar.selectbox(key='m1_flood_region').select('Омская область').run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertEqual(len(app.metric),0)
        app.button(key='kurgan_satellite').click().run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertEqual(app.sidebar.selectbox(key='m2_flood_point').value,'irtysh_omsk')
        self.assertEqual(len(app.metric),3)
