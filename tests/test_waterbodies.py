import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from okno import waterbodies as W
from scripts.download_waterbody import scale_offset


class WaterbodyTests(unittest.TestCase):
    def test_cloud_does_not_become_retreat(self):
        aoi=np.ones((2,2),bool);bv=aoi.copy();bw=aoi.copy();av=aoi.copy();av[0,0]=False
        mask,s=W.compare_masks(bv,bw,av,av.copy(),aoi,1.)
        self.assertEqual(s['delta_common_km2'],0.)
        self.assertEqual(s['after_observed_water_km2'],3.)
        self.assertEqual(s['before_observed_water_km2'],4.)
        self.assertEqual(s['full_area_delta_bounds_km2'],[-1.,0.])
        self.assertEqual(mask[0,0],0)
        self.assertFalse(s['comparable'])

    def test_known_transition_oracle(self):
        v=np.ones((2,3),bool);bw=np.array([[1,1,0],[0,1,0]],bool)
        av=v.copy();av[0,1]=False;aw=np.array([[1,0,0],[1,0,0]],bool)
        mask,s=W.compare_masks(v,bw,av,aw,v,.5)
        self.assertEqual(s['before_common_km2'],1.)
        self.assertEqual(s['after_common_km2'],1.)
        self.assertEqual(s['gain_km2'],.5);self.assertEqual(s['loss_km2'],.5)
        self.assertEqual(mask.tolist(),[[2,0,1],[4,3,1]])
        self.assertTrue(s['comparable'])

    def test_zero_water_and_no_common_area(self):
        v=np.ones((2,2),bool);z=np.zeros((2,2),bool)
        _,s=W.compare_masks(v,z,v,v,v,1.)
        self.assertIsNone(s['delta_common_percent'])
        self.assertEqual(s['delta_common_km2'],4.)
        _,s=W.compare_masks(z,z,v,v,v,1.)
        self.assertFalse(s['comparable']);self.assertEqual(s['common_fraction'],0.)

    def test_shapes_invalid_water_and_projection_rejected(self):
        v=np.ones((2,2),bool);z=np.zeros_like(v)
        for args in [(v,v,v,np.ones((3,3),bool),v,1.),(z,v,v,v,v,1.),(v,v,v,v,z,1.),(v,v,v,v,v,0.)]:
            with self.assertRaises(ValueError):W.compare_masks(*args)
        with self.assertRaises(ValueError):W.pixel_area_km2(dict(crs='EPSG:4326',transform=[.01,0,0,0,-.01,0]))
        self.assertEqual(W.pixel_area_km2(dict(crs='EPSG:32641',transform=[20,0,0,0,-20,0])),.0004)

    def test_unverified_radiometry_fails_closed(self):
        asset={'raster:bands':[dict(scale=.0001,offset=0)]}
        self.assertEqual(scale_offset(dict(properties={'s2:processing_baseline':'03.01'}),asset),(.0001,0.))
        with self.assertRaises(ValueError):scale_offset(dict(properties={'s2:processing_baseline':'05.09'}),asset)
        with self.assertRaises(ValueError):scale_offset(dict(properties={'s2:processing_baseline':'03.01'}),{'raster:bands':[dict(scale=.0001,offset=-.1)]})

    def test_saved_sample_and_reverse_dates(self):
        meta,mask,s,arrays=W.compare_sample()
        self.assertTrue(s['comparable']);self.assertEqual(len(meta['scenes']),2)
        self.assertTrue(np.isfinite(s['before_common_km2']))
        with self.assertRaises(ValueError):W.compare_sample(1,0)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(W.DATA,Path(tmp)/'sample')
            path=Path(tmp)/'sample'/'masks.npz';path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'целостность'):W.load_sample(path.parent)

    def test_dashboard_deep_link(self):
        app=AppTest.from_file(str(W.M.ROOT/'app/dashboard.py'))
        app.query_params['module']='waterbodies';app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value,'Водоёмы · бета')
        self.assertEqual(len(app.metric),4)
        self.assertTrue(any('Независимых контуров' in t.value for t in app.markdown))


if __name__=='__main__':unittest.main()
