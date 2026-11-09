"""Масса машины, выбор дороги, дата публикации и отсутствие скрытых разрешений."""
import unittest
import pandas as pd
from okno.truck_load import check_load, limit_as_of, route_events


class TruckLoadTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame([
            dict(road='А',hydro_year=2026,kind=kind,actual_date=date,actual_published=pub,
                 source_actual='https://example.org/source',comment='point_id=gauge_1; запись')
            for kind,date,pub in [('open','2025-12-01','2025-12-01'),
                ('limit30','2026-03-01','2026-03-01'),('limit20','2026-04-01','2026-04-02'),
                ('close','2026-05-01','2026-05-01')]])

    def test_mass_includes_vehicle(self):
        result=check_load(12,10,20)
        self.assertEqual(result['gross_t'],22)
        self.assertEqual(result['cargo_limit_t'],8)
        self.assertEqual(result['excess_t'],2)
        self.assertEqual(result['excess_pct'],10)
        self.assertFalse(result['within_mass_limit'])
        self.assertTrue(check_load(12,8,20)['within_mass_limit'])
        self.assertTrue(check_load(21,0,20)['tare_exceeds_limit'])
        for values in [(-1,10,20),(12,float('nan'),20),(12,10,0)]:
            with self.assertRaises(ValueError):
                check_load(*values)

    def test_latest_known_event_and_no_future_publication(self):
        data=self.events()
        self.assertEqual(limit_as_of(data,'А','2026-03-15')['limit_t'],30)
        self.assertEqual(limit_as_of(data,'А','2026-04-01')['status'],'unknown')
        self.assertEqual(limit_as_of(data,'А','2026-04-02')['limit_t'],20)
        self.assertEqual(limit_as_of(data,'А','2026-05-01')['status'],'closed')

    def test_open_missing_season_and_road_do_not_create_permission(self):
        data=self.events()
        for road,date in [('А','2025-12-02'),('А','2026-11-01'),('Б','2026-04-02')]:
            self.assertEqual(limit_as_of(data,road,date)['status'],'unknown')
        self.assertEqual(len(route_events(data,'gauge_1')),4)
        self.assertTrue(route_events(data,'gauge_10').empty)

    def test_conflicting_same_day_limits_are_unknown(self):
        data=self.events()
        row=data.iloc[2].copy();row['kind']='limit30'
        self.assertEqual(limit_as_of(pd.concat([data,pd.DataFrame([row])]),'А','2026-04-02')['status'],'unknown')


if __name__=='__main__':
    unittest.main()
