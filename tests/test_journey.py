"""Данные проекта, единая навигация и сквозная передача ограничений в рейсы."""
import datetime as dt
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno.journey import road_rows,evidence,model_risk,assess_project,seasonal_segment,seasonal_route,route_calendar,weekly_access
from okno.backhaul import Route,compare,audit_plan
from okno.backhaul_io import example_frames
from okno.ice import to_season_day

ROOT=Path(__file__).resolve().parents[1]
ROAD='Якутск — Нижний Бестях'


class JourneyDataTests(unittest.TestCase):
    def test_weekly_m3_and_start_advice_respect_same_registry(self):
        from okno.planner import Params,compare_ml_weekly_plans
        from okno.decision_support import start_options
        issue=to_season_day(dt.date(2026,4,8),2026)
        q=dict(issue_day=issue,q10=270.,q50=280.,q90=290.,calendar=280.)
        p=Params(Q_t=10,K_t_week=10,K_extra_t_week=0,start_day=issue,transit_days=.1,close_margin_days=0)
        legal=weekly_access(road_rows('yakutsk_crossing',2026),ROAD,'2026-04-08',2026,18)
        weekly,summary=compare_ml_weekly_plans(q,issue,p,availability=legal)
        forbidden=weekly[weekly.dispatch_day>=issue+7]
        self.assertEqual(forbidden.shipped_t.sum(),0)
        self.assertFalse(forbidden.allowed_by_calendar.any())
        # Даже благоприятный прогноз не разрешает старт после объявленного закрытия.
        options=start_options(q,issue,p,issue+14,issue+7,availability=legal)
        self.assertEqual(options.planned_road_t.sum(),0)
        self.assertTrue((options.expected_backup_t==10).all())

    def test_real_registry_no_future_knowledge(self):
        frame=road_rows('yakutsk_crossing',2026)
        now=evidence(frame,ROAD,'2026-04-08','2026-04-08',18)
        self.assertEqual(now['limit_t'],20)
        self.assertEqual(now['record']['basis'],'Факт')
        self.assertIn('ysia.ru',now['record']['source'])
        early=evidence(frame,ROAD,'2026-03-01','2026-04-09',18)
        self.assertNotEqual(early['limit_t'],20)
        planned=evidence(frame,ROAD,'2026-03-08','2026-04-09',18)
        self.assertEqual(planned['limit_t'],20)
        self.assertEqual(planned['record']['basis'],'Объявленный график')
        closed=evidence(frame,ROAD,'2026-04-08','2026-04-16',18)
        self.assertEqual(closed['status'],'closed')
        self.assertFalse(any(x['publication']>pd.Timestamp('2026-03-01') for x in early['records']))

    def test_calendar_transfers_mass_and_closure_to_full_roundtrip(self):
        from okno.backhaul import Order,Vehicle
        frame=road_rows('yakutsk_crossing',2026)
        calendar,_=route_calendar(frame,ROAD,'2026-04-08','2026-04-08',10,18)
        self.assertEqual(calendar[:7],[20.]*7)
        self.assertIsNone(calendar[7])
        route=Route(10,1,0,240,30,1,0,daily_mass_limits_t=tuple(calendar))
        jobs=[Order('рейс','out',9,'тент',167,190,0,0)]
        cars=[Vehicle('М1','тент',10,10,1,1,0)]
        # Выезд в конце разрешённого дня, но возврат уже после закрытия — недопустимо.
        for p in compare(jobs,cars,route).values(): self.assertFalse(p.feasible)
        jobs=[replace(jobs[0],ready_h=0,due_h=20)]
        plans=compare(jobs,cars,route)
        for p in plans.values():
            self.assertTrue(p.feasible);audit_plan(p.trips,jobs,cars,route)
        self.assertFalse(compare([replace(jobs[0],tons=11)],[replace(cars[0],capacity_t=20)],route)['combined'].feasible)

    def test_unknown_days_remain_closed_and_mass_model_not_extrapolated(self):
        calendar,_=route_calendar(road_rows('yakutsk_crossing',2027),ROAD,'2026-11-08','2026-11-08',5,18)
        self.assertEqual(calendar,[None]*5)
        q=dict(issue_day=200,q10=220.,q50=230.,q90=240.)
        self.assertIsNone(model_risk(q,'2026-04-08','2026-03-01',2026,21,18))
        self.assertIsNone(model_risk(q,'2026-04-08','2026-04-01',2026,21,25))
        q['issue_day']=180
        a=model_risk(q,'2026-04-01','2026-04-01',2026,21,18)
        b=model_risk(q,'2026-04-10','2026-04-01',2026,21,18)
        self.assertLessEqual(a,b)

    def test_real_geometry_is_explicit_and_unrelated_route_not_bound(self):
        segment=seasonal_segment('yakutsk_crossing')
        points=[dict(lon=lon,lat=lat) for lon,lat in (segment['coordinates'][0],segment['coordinates'][-1])]
        route=seasonal_route(points,segment,10,lambda _:self.fail('Для концов участка подъезды не нужны'))
        self.assertGreater(route['distance_km'],5)
        self.assertEqual(route['project_binding']['point_id'],'yakutsk_crossing')
        context=dict(pid='yakutsk_crossing',road=ROAD,hy=2026,issue_date=dt.date(2026,4,8),margin=21,
                     events=road_rows('yakutsk_crossing',2026),close_forecast=None)
        self.assertEqual(assess_project(route,context,'2026-04-08 08:00',18)['status'],'limited')
        unrelated=dict(route);unrelated.pop('project_binding')
        self.assertFalse(assess_project(unrelated,context,'2026-04-08',18)['covered'])


class JourneyUITests(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(ROOT/'app/dashboard.py'),default_timeout=90)
        for key,value in dict(filter_region='yakutia',filter_point='yakutsk_crossing',filter_season=2026,
                              filter_date=dt.date(2026,4,8)).items(): app.session_state[key]=value
        app.query_params['module']='journey'
        return app.run()

    def test_navigation_and_live_project_data_without_demo_orders(self):
        app=self.app()
        self.assertFalse(app.exception)
        self.assertIn('Маршрут и план завоза',[t.label for t in app.tabs])
        self.assertNotIn('Рейтинг риска',[t.label for t in app.tabs])
        self.assertNotIn('Планировщик',[t.label for t in app.tabs])
        self.assertFalse(any(b.key in ('open_backhaul','open_routes') for b in app.button))
        self.assertTrue(any('Есть запись ограничения массы' in x.value for x in app.markdown))
        # Источник — общий реестр, никакой загрузки пользовательского CSV не требуется.
        self.assertTrue(any('ysia.ru' in x.value for x in app.markdown))
        app.sidebar.button(key='show_welcome').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs),0)
        self.assertEqual([b.key for b in app.button],['open_dashboard'])

    def test_selected_seasonal_route_connects_to_empty_orders_and_m3(self):
        app=self.app()
        import app.route_view as view
        with patch.object(view,'_routes',side_effect=ValueError('В обычном графе путь не найден')):
            app.number_input(key='route_seasonal_speed').set_value(10.).run()
            app.button(key='route_build').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key='route_apply').disabled)
            app.button(key='route_apply').click().run()
            self.assertFalse(app.exception)
            self.assertAlmostEqual(app.number_input(key='bh_travel').value,app.number_input(key='bh_return_travel').value)
            self.assertTrue(any('Учебные заказы не подставляются' in x.value for x in app.info))
            self.assertIsNone(app.number_input(key='planner_quantity').value)
            self.assertNotIn('bh_date',[x.key for x in app.date_input])
            self.assertNotIn('bh_open',[x.key for x in app.date_input])
            self.assertTrue(app.number_input(key='bh_travel').disabled)
            # Введённые пользователем заявки становятся источником объёма М3.
            orders,cars=example_frames('2026-04-08')
            orders['Источник']='Проверочные заявки интеграционного теста'
            cars['Источник']='Проверочный парк интеграционного теста'
            app.session_state['journey_order_frame']=orders
            app.session_state['journey_vehicle_frame']=cars
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.number_input(key='planner_quantity').value,8.)
            self.assertTrue(app.number_input(key='planner_quantity').disabled)
            app.button(key='bh_calculate').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Выполнены все заявки (2)' in x.value for x in app.success))
            # Изменённая общая масса делает прежний выбор неактуальным.
            app.number_input(key='journey_gross').set_value(25.).run()
            self.assertFalse(app.exception)
            self.assertNotIn('bh_travel',[x.key for x in app.number_input])


if __name__=='__main__': unittest.main()
