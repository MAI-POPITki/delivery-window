"""М2: неизвестное не становится сушей, временная честность, реальные маски и UI."""
from datetime import date
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from okno import flood_m2 as M
from okno.flood import POINTS
from scripts.download_flood_m2 import radiometry
from scripts.validate_flood_m2 import validate


class FloodM2Tests(unittest.TestCase):
    def test_open_water_and_dry_land_have_distinct_masks(self):
        bands={b:np.full((8,8),v) for b,v in {'B03':.10,'B04':.04,'B08':.03,'B11':.02}.items()}
        valid,water=M.classify(bands,np.full((8,8),6))
        self.assertTrue(valid.all());self.assertTrue(water.all())
        bands['B08'][:]=.4
        _,water=M.classify(bands,np.full((8,8),4))
        self.assertFalse(water.any())

    def test_cloud_shadow_snow_nodata_and_zero_denominators_are_unknown(self):
        bands={b:np.full((9,9),.04) for b in ['B03','B04','B08','B11']}
        for code in [0,1,2,3,7,8,9,10,11]:
            valid,water=M.classify(bands,np.full((9,9),code))
            self.assertFalse(valid.any());self.assertFalse(water.any())
        scl=np.full((9,9),4);scl[4,4]=9
        valid,_=M.classify(bands,scl)
        self.assertFalse(valid[4,6]);self.assertTrue(valid[0,0])
        bands['B03'][:]=np.nan
        self.assertFalse(M.classify(bands,np.full((9,9),4))[0].any())
        bands={b:np.zeros((9,9)) for b in bands}
        self.assertFalse(M.classify(bands,np.full((9,9),4))[0].any())

    def test_baseline_requires_repeated_known_observations(self):
        v=np.array([[True,True,False,True]])
        w=np.array([[False,True,False,True]])
        second=(np.array([[True,True,True,True]]),np.array([[False,True,False,False]]))
        np.testing.assert_array_equal(M.baseline([(v,w),second]),[[1,2,0,0]])
        with self.assertRaises(ValueError):M.baseline([(v,w)])

    def test_new_water_area_uses_only_observed_dry_reference(self):
        valid=np.array([[True,True,True,False]])
        water=np.array([[True,True,True,False]])
        mask,stats=M.compare(valid,water,np.array([[1,2,0,1]]))
        np.testing.assert_array_equal(mask,[[3,2,2,0]])
        self.assertAlmostEqual(stats['new_water_km2'],.0004)
        self.assertAlmostEqual(stats['water_unknown_baseline_km2'],.0004)
        self.assertEqual(stats['observed_fraction'],.75)

    def test_unconfirmed_scl_water_is_unknown_not_dry_land(self):
        bands={b:np.full((8,8),v) for b,v in {'B03':.1,'B04':.1,'B08':.4,'B11':.2}.items()}
        valid,water=M.classify(bands,np.full((8,8),6))
        self.assertFalse(valid.any());self.assertFalse(water.any())
        quality,_=M.classify(bands,np.full((8,8),6),abstain=False)
        self.assertTrue(quality.all())

    def test_future_or_reprocessed_baselines_rejected(self):
        def scene(acquired,generated):
            return {'properties':{'datetime':acquired+'T00:00:00Z','s2:generation_time':generated+'T00:00:00Z'}}
        current=scene('2024-04-11','2024-04-11')
        with self.assertRaisesRegex(ValueError,'будущего'):
            M.require_past(current,[scene('2024-04-12','2024-04-12')])
        with self.assertRaisesRegex(ValueError,'переработку'):
            M.require_past(current,[scene('2023-08-01','2025-01-01')])
        with self.assertRaisesRegex(ValueError,'разные дни'):
            M.require_past(current,[scene('2023-08-01','2023-08-01')]*2)

    def test_xml_offsets_are_used_and_missing_new_baseline_offset_rejected(self):
        xml='<root><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>'+''.join(
            f'<BOA_ADD_OFFSET band_id="{i}">-1000</BOA_ADD_OFFSET>' for i in [1,2,3,7,11])+'</root>'
        meta=radiometry(xml,'05.10')
        self.assertEqual(meta['offsets']['B11'],-1000)
        self.assertEqual(meta['quantification'],10000)
        with self.assertRaises(ValueError):
            radiometry('<root><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE></root>','05.10')

    def test_all_five_real_regions_reproduce_masks_from_dn(self):
        for pid in POINTS:
            runs=M.available_runs(pid)
            self.assertTrue(runs,pid)
            for path in runs:
                row=validate(path)
                self.assertGreater(row['observed_fraction'],0)
                self.assertLessEqual(row['water_km2'],row['observed_km2'])
                self.assertLessEqual(row['new_water_km2'],row['comparable_km2'])
                meta=M.load_run(path)
                for scene in [meta['observation'],*meta['references']]:
                    self.assertTrue(all('\\' not in name for name in scene['files']))

    def test_snapshot_selection_never_uses_future_acquisition_or_generation(self):
        meta={'acquired_at':'2024-04-11T07:00:00Z','product_generated_at':'2024-04-12T01:00:00Z'}
        with patch.object(M,'available_runs',return_value=[Path('one.json')]),patch.object(M,'load_run',return_value=meta):
            self.assertIsNone(M.select_run('ural_orsk','2024-04-10'))
            self.assertIsNone(M.select_run('ural_orsk','2024-04-11'))
            self.assertIsNotNone(M.select_run('ural_orsk','2024-04-12'))

    def test_tampered_overlay_is_rejected(self):
        path=M.available_runs('ural_orsk')[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'runs'/path.parent.name
            shutil.copytree(path.parent,folder)
            with (folder/'overlay.png').open('ab') as out:out.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'целостность'):
                M.load_run(folder/'manifest.json')

    def test_ui_all_tabs_and_old_scene_warning(self):
        archived=[p for p in M.available_runs('ural_orsk') if M.load_run(p)['acquired_at'].startswith('2024-')]
        available=patch.object(M,'available_runs',return_value=archived)
        available.start();self.addCleanup(available.stop)
        app=AppTest.from_string('from app.flood_m2_view import render_m2\nrender_m2()',default_timeout=60)
        app.session_state['m2_flood_region']='Оренбургская область'
        app.session_state['m2_flood_point']='ural_orsk'
        for tab in ['Карта воды','Снимки до и после','Как работает М2','Источники']:
            app.session_state['m2_flood_tab']=tab
            app.run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(len(app.metric),3)
            if tab=='Карта воды':self.assertTrue(app.get('plotly_chart'))
        app.sidebar.date_input(key='m2_as_of_ural_orsk').set_value(date(2026,11,10)).run()
        self.assertTrue(any('устарел' in w.value for w in app.warning))

    def test_dashboard_direct_m2_link(self):
        app=AppTest.from_file(str(M.ROOT/'app/dashboard.py'),default_timeout=60)
        app.query_params['module']='flood_m2'
        app.run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any('М2' in title.value for title in app.title))


if __name__=='__main__':unittest.main()
