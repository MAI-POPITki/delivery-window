"""Ограничения, пересечения, ранжирование и передача асимметричных плеч."""
import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest
from okno.routing import assess,rank_routes,fetch_routes
from okno.backhaul import Route,compare,audit_plan
from okno.backhaul_io import example_frames,parse_frames

def sample(i=1,lat=0,hours=2):
    return dict(id=i,distance_km=100,travel_h=hours,coordinates=[[0,lat],[2,lat]],ferries=0,
                source='https://example.org/route',fetched_at='2026-11-08',data_version='test',roads=[])

def restriction(**kwargs):
    r=dict(name='Закрытый участок',min_lon=.9,max_lon=1.1,min_lat=-.1,max_lat=.1,
           kind='closed',value=0,start='2026-03-01 00:00',end='2026-03-03 00:00',
           published='2026-02-28 00:00',source='https://example.org/order')
    r.update(kwargs); return r

def args():
    return dict(departure='2026-03-01 08:00',deadline='2026-03-01 12:00',gross_t=18,restrictions=[])

class RoutingTests(unittest.TestCase):
    def test_longer_route_avoids_actual_segment_crossing(self):
        options=args();options['restrictions']=[restriction()]
        ranked=rank_routes([sample(),sample(2,lat=1,hours=3)],**options)
        self.assertEqual(ranked[0]['route']['id'],2)
        self.assertEqual(ranked[1]['assessment']['blocked'],['Закрытый участок'])

    def test_mass_unknown_future_publication_and_dates(self):
        a=args(); a['restrictions']=[restriction(kind='mass',value=20)]
        self.assertFalse(assess(sample(),**a)['blocked'])
        a['gross_t']=21
        self.assertTrue(assess(sample(),**a)['blocked'])
        a['restrictions']=[restriction(published='2026-03-02 00:00')]
        result=assess(sample(),**a)
        self.assertFalse(result['blocked']);self.assertTrue(result['unknown'])
        a['restrictions']=[restriction(end='2026-03-01 07:00')]
        self.assertFalse(assess(sample(),**a)['matched'])

    def test_scenarios_are_not_probability_and_delay_counts(self):
        a=args();a['restrictions']=[restriction(kind='delay',value=1)]
        result=assess(sample(hours=3),**a)
        self.assertEqual(result['late_scenarios'],2)
        self.assertEqual(result['base_h'],4)
        self.assertNotIn('probability',result)
        self.assertEqual(assess(sample(),**args())['matched'],[])

    def test_missing_route_not_straight_line_fallback(self):
        with patch('okno.routing._get',return_value={'code':'NoRoute'}):
            with self.assertRaises(ValueError): fetch_routes([dict(lat=1,lon=1),dict(lat=2,lon=2)])

    def test_asymmetric_backhaul_distances_and_cost(self):
        orders,cars=parse_frames(*example_frames('2026-03-01'),'2026-03-01')
        route=Route(300,8,0,120,20,return_distance_km=400,return_travel_h=10)
        plans=compare(orders,cars,route)
        self.assertEqual(plans['combined'].summary(orders)['total_km'],700)
        self.assertEqual(plans['separate'].summary(orders)['total_km'],1400)
        for p in plans.values(): audit_plan(p.trips,orders,cars,route)

    def test_unlinked_route_is_not_requested(self):
        import app.route_view as view
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/dashboard.py'),default_timeout=90)
        app.query_params['module']='routes';app.run()
        with patch.object(view,'_routes') as fetch:
            app.run()
            self.assertFalse(app.exception)
            self.assertNotIn('route_build',[b.key for b in app.button])
            fetch.assert_not_called()
            self.assertTrue(any('Построение маршрута отключено' in x.value for x in app.info))


if __name__=='__main__': unittest.main()
