"""Сохранённые внешние данные + граничные случаи, без сетевых запросов."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from okno import flood as F
from app.flood_view import make_map


class FloodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pid = "lena_yakutsk"
        cls.point = F.POINTS[cls.pid]
        cls.history = F.read_response(F.DATA / "history" / f"{cls.pid}.json")
        cls.live = F.read_response(F.DATA / "latest" / f"{cls.pid}.json")
        cls.now = max(datetime.fromisoformat(F.read_response(F.DATA / "latest" / f"{pid}.json")["fetched_at"])
                      for pid in F.POINTS)

    def test_all_real_points_have_complete_data(self):
        rows, details = F.load_points(self.now)
        self.assertEqual(len(details), 5)
        self.assertEqual(rows.region.nunique(), 4)
        for result in details.values():
            self.assertGreaterEqual(result["years"], 20)
            self.assertGreaterEqual(result["members"], 30)
            self.assertEqual(len(result["curve"]), 7)
            self.assertTrue(0 <= result["fraction"] <= 1)

    def test_stale_future_timestamp_and_naive_timestamp_rejected(self):
        for offset in (timedelta(hours=49), timedelta(hours=-1)):
            with self.assertRaises(ValueError):
                F.summarize(self.history, self.live, self.point, self.now + offset)
        bad = deepcopy(self.live)
        bad["fetched_at"] = self.now.replace(tzinfo=None).isoformat()
        with self.assertRaises(ValueError):
            F.summarize(self.history, bad, self.point, self.now)

    def test_missing_member_day_not_filled(self):
        bad = deepcopy(self.live)
        for key in bad["payload"]["daily"]:
            if key.startswith("river_discharge_member"):
                bad["payload"]["daily"][key][-1] = None
        with self.assertRaises(ValueError):
            F.summarize(self.history, bad, self.point, self.now)

    def test_union_of_scenarios_not_max_daily_probability(self):
        bad = deepcopy(self.live)
        q, _ = F.threshold(F.daily_frame(self.history["payload"], self.point))
        members = [k for k in bad["payload"]["daily"] if k.startswith("river_discharge_member")]
        for i, key in enumerate(members):
            bad["payload"]["daily"][key] = [0.] * 8
            bad["payload"]["daily"][key][1 + i % 7] = q * 2
        result = F.summarize(self.history, bad, self.point, self.now)
        self.assertEqual(result["fraction"], 1.)

    def test_wrong_cell_units_dates_and_negative_values_rejected(self):
        for field in ("cell", "nan_cell", "units", "duplicate", "negative", "infinite", "missing_date", "time_in_date"):
            bad = deepcopy(self.live["payload"])
            if field == "cell":
                bad["latitude"] += 1
            elif field == "nan_cell":
                bad["latitude"] = float("nan")
            elif field == "units":
                bad["daily_units"]["river_discharge"] = "cm"
            elif field == "duplicate":
                bad["daily"]["time"][1] = bad["daily"]["time"][0]
            elif field == "missing_date":
                bad["daily"]["time"][0] = None
            elif field == "time_in_date":
                bad["daily"]["time"] = [v + "T12:00:00" for v in bad["daily"]["time"]]
            else:
                bad["daily"]["river_discharge"][0] = -1 if field == "negative" else float("inf")
            with self.subTest(field=field), self.assertRaises(ValueError):
                F.daily_frame(bad, self.point)

    def test_threshold_cannot_use_future_years(self):
        frame = F.daily_frame(self.history["payload"], self.point)
        before = F.threshold(frame, before_year=2019)
        frame.loc[frame.index.year >= 2019, "river_discharge"] = 1e12
        self.assertEqual(F.threshold(frame, before_year=2019), before)
        self.assertGreater(F.threshold(frame)[0], before[0])

    def test_too_short_history_rejected(self):
        frame = F.daily_frame(self.history["payload"], self.point)
        with self.assertRaises(ValueError):
            F.threshold(frame[frame.index.year > 2010])

    def test_checksum_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            F.save_response(path, self.live["payload"], self.live["url"], self.live["fetched_at"])
            data = json.loads(path.read_text())
            data["payload"]["latitude"] = 0
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                F.read_response(path)

    def test_missing_and_stale_data_remain_on_map_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, details = F.load_points(self.now, Path(directory))
        self.assertFalse(details)
        self.assertTrue(rows.fraction.isna().all())
        fig = make_map(rows)
        self.assertEqual(sum(len(trace.lat) for trace in fig.data), 5)
        self.assertEqual(fig.data[0].marker.color, "#9e9e9e")
        rows, _ = F.load_points(self.now + timedelta(days=5))
        self.assertTrue(rows.fraction.isna().all())

    def test_map_contains_all_region_points(self):
        rows, _ = F.load_points(self.now)
        fig = make_map(rows)
        self.assertEqual(sum(len(trace.lat) for trace in fig.data), 5)
        ids = {item[0] for trace in fig.data for item in trace.customdata}
        self.assertEqual(ids, set(F.POINTS))

    def test_map_fits_region_and_keeps_labels_readable(self):
        rows, _ = F.load_points(self.now)
        all_map = make_map(rows)
        regional_map = make_map(rows[rows.region == "Оренбургская область"], "ural_orsk")
        self.assertGreater(regional_map.layout.map.zoom, all_map.layout.map.zoom)
        self.assertEqual(regional_map.data[0].textfont.color, "#172033")
        self.assertIn(23, regional_map.data[0].marker.size)

    def test_archive_cache_invalidates_when_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps(self.history), encoding="utf-8")
            stat = path.stat()
            F._history_reference.cache_clear()
            F._history_reference(path, self.pid, stat.st_mtime_ns, stat.st_size, 2026)
            F._history_reference(path, self.pid, stat.st_mtime_ns, stat.st_size, 2026)
            self.assertEqual(F._history_reference.cache_info().hits, 1)
            path.write_text('{"payload": {}}', encoding="utf-8")
            changed = path.stat()
            with self.assertRaises(KeyError):
                F._history_reference(path, self.pid, changed.st_mtime_ns, changed.st_size, 2026)

    def test_streamlit_renders_map_and_region_filter(self):
        with patch("app.flood_view.load_points", side_effect=lambda **kw: F.load_points(self.now, **kw)):
            at = AppTest.from_string("from app.flood_view import render_flood\nrender_flood()")
            at.session_state['flood_decision'] = self.now.date()
            at.run(timeout=30)
            self.assertFalse(at.exception)
            self.assertEqual(len(at.get("plotly_chart")), 1)
            self.assertEqual([t.label for t in at.tabs], ['Карта путей','Прогноз','Риски','План','Данные'])
            at.sidebar.selectbox(key='flood_region').select("Оренбургская область").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.sidebar.selectbox(key='flood_point').options), 2)

    def test_dashboard_switches_between_flood_and_winter_modes(self):
        at = AppTest.from_file(str(F.ROOT / "app" / "dashboard.py"), default_timeout=90)
        at.session_state["dashboard_open"] = True
        at.session_state["service_mode"] = "Паводки"
        at.run()
        self.assertFalse(at.exception)
        self.assertIn("Паводки", at.title[0].value)
        self.assertEqual(len(at.sidebar.selectbox[0].options), 5)
        at.sidebar.radio(key="service_mode").set_value("Зимники и переправы").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.tabs[0].label, "Карта путей")

    def test_direct_link_opens_flood_and_can_return_to_welcome(self):
        at = AppTest.from_file(str(F.ROOT / "app" / "dashboard.py"), default_timeout=90)
        at.query_params["module"] = "flood"
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Паводки")
        at.sidebar.button(key="flood_welcome").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Окно завоза")
        self.assertNotIn("module", at.query_params)


if __name__ == "__main__":
    unittest.main()
