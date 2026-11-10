from datetime import date
import unittest
from streamlit.testing.v1 import AppTest
from app.flood_overview import overview_rows
from okno.map_style import FLOOD_MAP_STYLE


class OverviewTests(unittest.TestCase):
    def test_all_five_points_same_date_and_missing_model_is_unknown(self):
        rows=overview_rows('2023-05-01',7)
        self.assertEqual(len(rows),5)
        self.assertEqual(len({r['pid'] for r in rows}),5)
        kurgan=next(r for r in rows if r['pid']=='tobol_kurgan')
        self.assertIsNone(kurgan['probability'])
        self.assertEqual({r['name'] for r in rows},{'Якутск','Ишим','Курган','Оренбург','Орск'})

    def test_background_has_no_foreign_text_labels(self):
        self.assertTrue(all(l['type']=='raster' for l in FLOOD_MAP_STYLE['layers']))
        self.assertIn('World_Terrain_Base',FLOOD_MAP_STYLE['sources']['background']['tiles'][0])

    def test_buttons_select_point_and_real_example(self):
        app=AppTest.from_string('from app.flood_m1_view import render_m1\nrender_m1()',default_timeout=90).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        for pid in ['lena_yakutsk','ishim_ishim','tobol_kurgan','ural_orenburg','ural_orsk']:
            self.assertTrue(app.button(key='choose_flood_'+pid))
        app.button(key='choose_flood_ishim_ishim').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.selectbox(key='m1_flood_point').value,'ishim_ishim')
        app.button(key='flood_ready_example').click().run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertEqual(app.sidebar.selectbox(key='m1_flood_point').value,'ural_orsk')
        self.assertEqual(app.sidebar.date_input(key='m1_issue_ural_orsk').value,date(2024,4,11))
