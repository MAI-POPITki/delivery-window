"""Тесты внешних таблиц раздела 3 docs/VALIDATION_PLAN.md и валидатора okno/external_data.py."""

import tempfile
import unittest
from pathlib import Path

from okno.external_data import (
    extract_point_id,
    hydro_year_for_date,
    is_valid_https_url,
    load_ice_thickness_bulletins,
    load_incidents,
    load_northern_supply_volumes,
    load_road_events,
    load_tariffs,
    validate_all_external,
    validate_road_events,
)
from okno.paths import EXTERNAL


class TestExternalDataHelpers(unittest.TestCase):
    def test_hydro_year_convention(self):
        import datetime as dt

        self.assertEqual(hydro_year_for_date(dt.date(2021, 12, 27)), 2022)
        self.assertEqual(hydro_year_for_date(dt.date(2022, 4, 15)), 2022)
        self.assertEqual(hydro_year_for_date(dt.date(2025, 9, 1)), 2026)
        self.assertEqual(hydro_year_for_date(dt.date(2026, 8, 31)), 2026)

    def test_extract_point_id(self):
        self.assertEqual(extract_point_id("point_id=yakutsk_crossing; 40 т"), "yakutsk_crossing")
        self.assertEqual(extract_point_id("приказ МЧС; point_id=gauge_3414, 30 т"), "gauge_3414")
        self.assertIsNone(extract_point_id("без привязки"))

    def test_valid_https_url(self):
        self.assertTrue(is_valid_https_url("https://ysia.ru/news/123/"))
        self.assertFalse(is_valid_https_url("http://ysia.ru/news/123/"))
        self.assertFalse(is_valid_https_url(""))
        self.assertFalse(is_valid_https_url("ysia.ru/news"))

    def test_future_leak_detected(self):
        """Если дата публикации плана позже плановой даты или факта — проверка должна упасть."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "road_events.csv"
            p.write_text(
                "road,region,hydro_year,kind,announced_date,announced_published,"
                "actual_date,actual_published,source_announced,source_actual,comment\n"
                "Тест,yakutia,2026,close,2026-04-10,2026-04-12,2026-04-11,2026-04-11,"
                "https://ysia.ru/a/,https://ysia.ru/b/,point_id=yakutsk_crossing\n",
                encoding="utf-8",
            )
            checks, _ = validate_road_events(p)
            by_name = {c["проверка"]: c["результат"] for c in checks}
            self.assertEqual(by_name["нет утечки будущего в объявленных датах (правило 1.2)"], "НЕ прошла")

    def test_duplicate_announced_and_actual_source_rejected(self):
        """Одна и та же публикация не может одновременно считаться предварительным планом и пост-фактом."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "road_events.csv"
            p.write_text(
                "road,region,hydro_year,kind,announced_date,announced_published,"
                "actual_date,actual_published,source_announced,source_actual,comment\n"
                "Тест,yakutia,2026,close,2026-04-15,2026-04-14,2026-04-15,2026-04-14,"
                "https://ysia.ru/same/,https://ysia.ru/same/,point_id=yakutsk_crossing\n",
                encoding="utf-8",
            )
            checks, _ = validate_road_events(p)
            by_name = {c["проверка"]: c["результат"] for c in checks}
            self.assertEqual(by_name["нет утечки будущего в объявленных датах (правило 1.2)"], "НЕ прошла")


class TestSection3Datasets(unittest.TestCase):
    def test_all_external_checks_pass(self):
        checks, summary = validate_all_external(EXTERNAL)
        failed = [c for c in checks if c["результат"] != "прошла"]
        self.assertEqual(failed, [], f"Упавшие проверки внешних данных: {failed}")
        self.assertGreaterEqual(summary["road_events"]["total"], 100)
        self.assertGreaterEqual(summary["road_events"]["yakutia"], 60)
        self.assertGreaterEqual(summary["road_events"]["other_regions"], 40)
        self.assertGreaterEqual(summary["road_events"]["road_seasons_open_close"], 30)
        self.assertGreaterEqual(summary["tariffs"]["total"], 15)
        self.assertGreaterEqual(summary["incidents"]["total"], 10)

    def test_loaders_return_typed_frames(self):
        ev = load_road_events(with_point_id=True)
        self.assertIn("point_id", ev.columns)
        self.assertTrue((ev["point_id"].notna() & (ev["point_id"] != "")).all())

        tf = load_tariffs()
        self.assertTrue((tf["price_rub_per_t"] > 0).all())

        inc = load_incidents()
        self.assertGreaterEqual(len(inc), 10)

        vol = load_northern_supply_volumes()
        self.assertGreaterEqual(len(vol), 5)

        bul = load_ice_thickness_bulletins()
        self.assertGreaterEqual(len(bul), 8)


if __name__ == "__main__":
    unittest.main()
