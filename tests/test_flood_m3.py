from datetime import date
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from okno.flood_m3 import calendar_for_levels,plan_variants


class FloodM3Tests(unittest.TestCase):
    def inputs(self):
        dates=pd.date_range('2023-05-02',periods=3)
        f=pd.DataFrame(dict(date=dates,lower_cm=[1,2,3],predicted_cm=[2,3,4],upper_cm=[3,4,5]))
        r=pd.DataFrame(dict(date=dates.date,confirmed=True,capacity_t=2.,source='test fixture',published=date(2023,5,1)))
        return r,f

    def test_full_trip_and_threshold_equality(self):
        r,f=self.inputs()
        c=calendar_for_levels(r,f,date(2023,5,1),2,5)
        self.assertEqual(c.policy_allowed.tolist(),[True,False,False])
        r.loc[1,'confirmed']=False
        c=calendar_for_levels(r,f,date(2023,5,1),2,5)
        self.assertFalse(c.confirmed.iloc[0])

    def test_optimizer_conservation_and_uncertainty(self):
        r,f=self.inputs()
        v=plan_variants(r,f,date(2023,5,1),1,4,5,10,100,0)
        self.assertAlmostEqual(v['upper'][1]['road_t'],2)
        self.assertAlmostEqual(v['upper'][1]['backup_t'],3)
        for plan,s in v.values():
            self.assertAlmostEqual(s['road_t']+s['backup_t'],5)
            self.assertLessEqual(s['cost'],s['baseline_cost']+1e-8)
        self.assertGreater(v['upper'][1]['cost'],v['lower'][1]['cost'])

    def test_future_confirmation_and_invalid_forecast_rejected(self):
        r,f=self.inputs();r.loc[0,'published']=date(2023,5,2)
        with self.assertRaises(ValueError):calendar_for_levels(r,f,date(2023,5,1),1,5)
        r,f=self.inputs();f.loc[0,'upper_cm']=np.nan
        with self.assertRaises(ValueError):calendar_for_levels(r,f,date(2023,5,1),1,5)

    def test_ui_no_invented_costs_or_threshold(self):
        app=AppTest.from_string("from app.flood_m1_view import render_m1\nrender_m1()",default_timeout=60)
        app.session_state['m1_flood_tab']='М3: план доставки'
        app.run();self.assertFalse(app.exception)
        self.assertIsNone(app.number_input(key='own_m3_threshold').value)
        app.button(key='own_m3_run').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Введите маршрут' in i.value for i in app.info))
