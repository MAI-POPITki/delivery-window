import unittest
from dataclasses import replace
from okno.planner import Params, compare_ml_weekly_plans
from okno.decision_support import start_options, start_sensitivity
from okno.bottlenecks import diagnose_corridor
from okno.corridor import CorridorConfig

class Decisions(unittest.TestCase):
    def test_same_start_and_no_future(self):
        f=dict(q10=210,q50=230,q90=250,calendar=230,issue_day=180)
        p=Params()
        table=start_options(f,180,p,194,180)
        _, direct=compare_ml_weekly_plans(f,180,replace(p,start_day=194))
        self.assertAlmostEqual(table.iloc[0].expected_cost_rub,direct.loc['economic_ml'].expected_cost_rub)
        self.assertTrue((table.expected_delivered_t+table.expected_backup_t-p.Q_t).abs().max()<1e-6)
        with self.assertRaises(ValueError): start_options(f,180,p,194,179)
        self.assertEqual(len(start_options(f,180,p,180,180)),1)
        sens=start_sensitivity(f,180,p,180,180,.3)
        self.assertTrue((sens.saving_rub==0).all())
    def test_zero_reserve_not_invented(self):
        cfg=CorridorConfig(crossing_cap_alt_tons=0)
        result=diagnose_corridor(cfg,dict(ac=30,ao=57,cc=36,co=63))
        row=result.iloc[2]
        self.assertEqual(row['Проверяемая_мощность'],0)
        self.assertEqual(row['Сокращение_ожидания_т_сут'],0)

if __name__=='__main__': unittest.main()
