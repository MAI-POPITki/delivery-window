"""Независимые проверки одинаковой работы, ограничений и точного решения малых задач."""
from dataclasses import replace
import unittest
import pandas as pd

from okno.backhaul import Order, Vehicle, Route, candidates, compare, solve, audit_plan, fixed_plan_sensitivity
from okno.backhaul_io import example_frames, parse_frames, schedule_frame, report_payload


def fixture():
    return ([Order('туда','out',5,'тент',0,8,0,0),Order('назад','back',4,'тент',1,8,0,0)],
            [Vehicle('М1','тент',10,8,1,1,0)],Route(100,1,0,8,20,1,0))


def exhaustive(trips, order_ids):
    """Полный перебор покрытий без использования MILP и его матрицы ограничений."""
    best=float('inf')
    def visit(left,chosen,cost):
        nonlocal best
        if not left:
            best=min(best,cost)
            return
        first=sorted(left)[0]
        for t in trips:
            covered=set(t.orders)
            if first not in covered or not covered<=left:
                continue
            if any(t.vehicle==x.vehicle and max(t.start_h,x.start_h)<min(t.finish_h,x.finish_h)-1e-7 for x in chosen):
                continue
            visit(left-covered,chosen+[t],cost+t.cost_rub)
    visit(set(order_ids),[],0)
    return best


class BackhaulTests(unittest.TestCase):
    def test_same_work_four_lengths_become_two(self):
        orders,vehicles,route=fixture()
        plans=compare(orders,vehicles,route)
        a,b=(plans[k].summary(orders) for k in ('separate','combined'))
        self.assertEqual((a['total_km'],a['empty_km'],b['total_km'],b['empty_km']),(400,200,200,0))
        self.assertEqual((a['tons'],b['tons'],a['orders'],b['orders']),(9,9,2,2))
        self.assertEqual((a['cost_rub'],b['cost_rub']),(400,200))
        for p in plans.values():
            self.assertEqual(p.status,'optimal')
            audit_plan(p.trips,orders,vehicles,route)

    def test_waiting_can_cancel_benefit(self):
        orders,vehicles,route=fixture()
        orders=[replace(orders[0],due_h=2),replace(orders[1],ready_h=20,due_h=24)]
        vehicles=[replace(vehicles[0],occupied_rub_h=100)]
        plans=compare(orders,vehicles,replace(route,close_h=24))
        self.assertEqual(plans['combined'].summary(orders)['paired'],0)
        self.assertEqual(plans['combined'].summary(orders)['cost_rub'],plans['separate'].summary(orders)['cost_rub'])

    def test_one_direction_has_no_invented_backhaul(self):
        orders,vehicles,route=fixture()
        plans=compare(orders[:1],vehicles,route)
        self.assertEqual(plans['combined'].summary(orders[:1])['empty_km'],100)
        self.assertEqual(plans['combined'].summary(orders[:1])['paired'],0)

    def test_incompatible_body_uses_separate_vehicles(self):
        orders,vehicles,route=fixture()
        orders[1]=replace(orders[1],body='рефрижератор')
        vehicles.append(replace(vehicles[0],id='М2',body='рефрижератор'))
        result=solve(orders,vehicles,route)
        self.assertEqual(result.status,'optimal')
        self.assertEqual(result.summary(orders)['paired'],0)

    def test_gross_mass_checks_return_and_empty_vehicle(self):
        orders,vehicles,route=fixture()
        self.assertFalse(solve([replace(orders[1],tons=13)], [replace(vehicles[0],capacity_t=15)],route).feasible)
        self.assertFalse(solve(orders,vehicles,replace(route,gross_limit_t=7)).feasible)

    def test_cannot_leave_vehicle_beyond_closure(self):
        orders,vehicles,route=fixture()
        self.assertFalse(solve(orders[:1],vehicles,replace(route,close_h=1.5)).feasible)
        self.assertTrue(solve(orders[:1],vehicles,replace(route,close_h=2)).feasible)

    def test_no_saving_against_infeasible_baseline(self):
        orders,vehicles,route=fixture()
        plans=compare(orders,vehicles,replace(route,close_h=2))
        self.assertEqual(plans['separate'].status,'infeasible')
        self.assertIsNone(plans['separate'].summary(orders))
        self.assertTrue(plans['combined'].feasible)
        self.assertTrue(fixed_plan_sensitivity(plans,orders).empty)

    def test_deliver_by_unloading_and_readiness(self):
        orders,vehicles,route=fixture()
        order=replace(orders[0],loading_h=1,unloading_h=1,due_h=2.5)
        self.assertFalse(solve([order],vehicles,route).feasible)
        order=replace(order,due_h=5,ready_h=1)
        result=solve([order],vehicles,route)
        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.trips[0].start_h,1)
        self.assertLessEqual(result.trips[0].outward_done_h,5)

    def test_milp_matches_independent_enumeration(self):
        orders,vehicles,route=fixture()
        orders.append(Order('ещё','out',2,'тент',2,6,0,0))
        route=replace(route,close_h=6,step_h=2)
        vehicles=[replace(vehicles[0],occupied_rub_h=3,fixed_rub=10)]
        for paired in (False,True):
            trips=candidates(orders,vehicles,route,paired)
            expected=exhaustive(trips,[o.id for o in orders])
            result=solve(orders,vehicles,route,paired=paired)
            self.assertTrue(result.feasible)
            self.assertAlmostEqual(result.summary(orders)['cost_rub'],expected)

    def test_independent_audit_rejects_dropped_duplicated_overlapping_and_wrong_cost(self):
        orders,vehicles,route=fixture()
        good=solve(orders,vehicles,route)
        for bad in ([],good.trips*2,[replace(good.trips[0],movement_rub=0)]):
            with self.assertRaises(ValueError): audit_plan(bad,orders,vehicles,route)
        singles=candidates(orders,vehicles,route,False)
        one=next(t for t in singles if t.outward and t.start_h==0)
        two=next(t for t in singles if t.backward and t.start_h==0)
        with self.assertRaises(ValueError): audit_plan([one,two],orders,vehicles,route)

    def test_validation_and_roundtrip_report(self):
        frames=example_frames('2026-03-01')
        orders,vehicles=parse_frames(*frames,'2026-03-01')
        route=Route(300,8,0,96,20)
        plans=compare(orders,vehicles,route)
        self.assertEqual(plans['combined'].status,'optimal')
        report=report_payload(plans,orders,vehicles,route,'2026-03-01',{'source':'учебный'})
        self.assertEqual(report['results']['combined']['summary']['tons'],14)
        self.assertIn('Возврат в А',schedule_frame(plans['combined'],'2026-03-01'))
        for bad in (replace(route,distance_km=float('nan')),replace(route,step_h=0)):
            with self.assertRaises(ValueError): solve(orders,vehicles,bad)
        with self.assertRaises(ValueError): solve(orders+[orders[0]],vehicles,route)
        with self.assertRaises(ValueError): parse_frames(frames[0].drop(columns=['Источник']),frames[1],'2026-03-01')
        broken=frames[0].copy(); broken.loc[0,'Готовность']='не дата'
        with self.assertRaises(ValueError): parse_frames(broken,frames[1],'2026-03-01')


if __name__=='__main__':
    unittest.main()
