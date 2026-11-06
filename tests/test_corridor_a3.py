"""Автотесты симулятора сквозного коридора (4 звена) и интерактивной вкладки А3 в Streamlit."""

from pathlib import Path
import sys
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okno.corridor import CorridorConfig, compute_prestock_schedule, run_sensitivity_check, simulate_corridor


class CorridorSimulatorTests(unittest.TestCase):
    def test_comparable_volume_and_cross_segment_tradeoff(self):
        cfg = CorridorConfig()
        before, daily_b = simulate_corridor(cfg, 30, 57, 36, 63, use_prestock=False, return_daily=True)
        after, daily_a = simulate_corridor(cfg, 30, 57, 29, 58, use_prestock=True, return_daily=True)

        # Равный объём ж/д потока за квартал (сопоставимые условия А3: 2 250 т)
        self.assertAlmostEqual(before["total_rail_tons"], 2250.0, places=1)
        self.assertAlmostEqual(after["total_rail_tons"], 2250.0, places=1)
        self.assertEqual(len(daily_b), cfg.horizon_days)
        self.assertEqual(len(daily_a), cfg.horizon_days)
        self.assertTrue((daily_a["inv_nb_t"] >= -1e-9).all())
        self.assertTrue((daily_a["inv_yak_t"] >= -1e-9).all())

        # Участок 4а (хранение на ТЛЦ Якутска) явно дорожает, но полная цепочка дешевеет и дефицит = 0
        self.assertGreater(after["cost_holding_yak_rub"], before["cost_holding_yak_rub"])
        self.assertEqual(after["days_with_deficit"], 0)
        self.assertEqual(after["tons_stuck_nb_ton_days"], 0.0)
        self.assertEqual(after["tons_alt_transport"], 0.0)
        self.assertLess(after["total_cost_rub"], before["total_cost_rub"])

    def test_bottleneck_schedule_and_sensitivity_30_pct(self):
        cfg = CorridorConfig()
        sched = compute_prestock_schedule(cfg, planned_close_day=29, planned_open_day=58)
        self.assertGreaterEqual(sched["prestock_duration_days"], 20)
        sens = run_sensitivity_check(cfg)
        self.assertGreaterEqual(len(sens), 16)
        self.assertTrue(sens["effect_positive"].all())

    def test_dashboard_corridor_tab_renders_and_updates_sliders(self):
        app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90).run()
        app.button(key="open_dashboard").click().run()
        self.assertEqual([e.message for e in app.exception], [])

        scen_box = app.selectbox(key="corr_scenario")
        self.assertGreater(len(scen_box.options), 1)
        # Переключаем на реальный исторический сезон и двигаем ползунок тарифа СВП/Ми-8
        scen_box.select(scen_box.options[1]).run()
        self.assertEqual([e.message for e in app.exception], [])

        app.slider(key="corr_calt").set_value(45000.0).run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertTrue(any("Сквозной логистический коридор из 4 звеньев" in s.value for s in app.subheader))

        # Краевой режим ползунков: Q_max <= d не должен вызывать исключений KeyError / ZeroDivisionError
        app.slider(key="corr_d").set_value(50.0).run()
        app.slider(key="corr_qmax").set_value(35.0).run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertTrue(any("не превышает суточное потребление" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
