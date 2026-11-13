"""Проверки рекомендаций без будущих фактов и неподтверждённого процента."""
import unittest
from okno.journey import road_rows
from okno.risk_advice import scenarios
from okno.ice import to_season_day

class RiskAdviceTests(unittest.TestCase):
    def test_same_issue_and_known_registry(self):
        day=to_season_day('2026-04-08',2026)
        q=dict(issue_day=day,q10=220.,q50=230.,q90=240.)
        rows=scenarios(road_rows('yakutsk_crossing',2026),'Якутск — Нижний Бестях',q,
                       '2026-04-08','2026-04-15',2026,21,18)
        self.assertEqual(rows[0]['status'],'closed')
        self.assertEqual(rows[-1]['status'],'limited')
        self.assertTrue(all(x['date'].isoformat()>='2026-04-08' for x in rows))
        self.assertGreaterEqual(rows[0]['risk'],rows[-1]['risk'])
        heavy=scenarios(road_rows('yakutsk_crossing',2026),'Якутск — Нижний Бестях',q,
                        '2026-04-08','2026-04-08',2026,21,25)
        self.assertEqual(heavy[0]['status'],'overweight')
        self.assertIsNone(heavy[0]['risk'])

    def test_unknown_is_not_safe_and_past_date_rejected(self):
        rows=scenarios(road_rows('yakutsk_crossing',2027),None,None,
                       '2026-11-08','2026-11-09',2027,21,18)
        self.assertTrue(all(x['risk'] is None and x['status']=='unknown' for x in rows))
        with self.assertRaises(ValueError):
            scenarios(road_rows('yakutsk_crossing',2027),None,None,'2026-11-08','2026-11-07',2027,21,18)
