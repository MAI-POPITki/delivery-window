"""Source identity, missing dates, causal inputs, outages and regional coverage."""
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from okno import hydro_ingest as ingest, official_hydro as H, snow
from okno.hydro_refresh import due
from okno import hydro_refresh as refresh
from okno.regions import REGIONS, points_of

NOW = datetime(2026, 11, 13, tzinfo=timezone.utc)
POINT = dict(point_id="gauge_3414", region="yakutia", station_code=3414,
             river="Яна", site="Верхоянск", scope="post", lat=67.34, lon=133.38)
STATION = dict(code=3414, name="г.Верхоянск", geoobject="Яна", latitude=67.57,
               longitude=133.4, level_now=206, last_l_dt="2026-11-09 22:00:00")


def measured(value=206, metric="water_level_cm", observed="2026-11-09 22:00:00", fetched=NOW):
    return H.observation(POINT, metric, value, observed, fetched,
                         "https://meteo-dv.ru/hydro_dfo/", "a"*64)


class OfficialParsersTests(unittest.TestCase):
    def test_registry_contains_every_point_of_all_seven_regions(self):
        points = H.registry()
        self.assertEqual({p["region"] for p in points.values()}, set(REGIONS))
        self.assertEqual(set(points), {p for key in REGIONS for p in points_of(key)})
        self.assertEqual(points["yakutsk_crossing"]["scope"], "reference_post")
        self.assertTrue(all(p["station_code"] is None for p in points.values() if p["scope"] == "satellite"))

    def test_code_alone_or_geographic_proximity_cannot_match_a_post(self):
        points = {POINT["point_id"]: POINT}
        self.assertIsNone(H.match_station({**STATION, "geoobject": "Лена"}, points))
        self.assertIsNone(H.match_station({**STATION, "name": "Другой пост"}, points))
        self.assertIsNone(H.match_station({**STATION, "code": 9999}, points))
        self.assertEqual(H.match_station(STATION, points), POINT)

    def test_negative_and_zero_levels_survive_missing_value_filter(self):
        self.assertEqual(H.number(-157, "water_level_cm"), -157)
        self.assertEqual(H.number(0, "water_level_cm"), 0)
        for value in (None, "", "—", 999.999, float("nan"), float("inf")):
            self.assertIsNone(H.number(value, "water_level_cm"))
        self.assertIsNone(H.number(-1, "ice_cm"))
        self.assertEqual(H.number(0, "snow_on_ice_cm"), 0)

    def test_list_uses_measurement_date_not_query_date_and_flags_coordinate_difference(self):
        rows, matches = ingest.parse_station_list([STATION], NOW, "https://meteo-dv.ru/", "a"*64,
                                                  {POINT["point_id"]: POINT})
        self.assertEqual(len(matches), 1)
        self.assertEqual(rows[0]["observed_at"], "2026-11-09T22:00:00+00:00")
        self.assertIn("Координаты", rows[0]["note"])
        self.assertEqual(rows[0]["available_at"], NOW.isoformat())

    def test_graph_never_imports_relabelled_previous_year_or_another_instrument(self):
        data = dict(name="Яна, Верхоянск", levels=[["2026-11-09 22:00", 206]],
                    prev_y=[["2026-11-09 22:00", 999]], agk=[["2026-11-09 22:00", 888]],
                    falls=[["2026-11-09 22:00", 777]])
        rows = ingest.parse_graph(data, POINT, NOW, "https://meteo-dv.ru/", "a"*64)
        self.assertEqual([r["value"] for r in rows], [206])
        data["name"] = "Лена, Якутск"
        with self.assertRaises(ValueError):
            ingest.parse_graph(data, POINT, NOW, "https://meteo-dv.ru/", "a"*64)

    def test_modal_does_not_turn_display_date_into_ice_measurement_date(self):
        data = dict(station_name="Верхоянск", geo_object="Яна", level_last=206,
                    level_last_date="2026-11-09 22:00", level_current_date="2026-11-12 23:00",
                    date_utc="2026-11-12 23:00", ice_level_last=27)
        rows = ingest.parse_modal(data, POINT, NOW, "https://meteo-dv.ru/", "a"*64)
        ice = next(r for r in rows if r["metric"] == "ice_cm")
        self.assertIsNone(ice["observed_at"])
        self.assertEqual(ice["quality"], "undated")
        self.assertIsNone(H.latest_measurement(POINT["point_id"], "ice_cm", NOW,
                                             snapshot={"observations": rows}))

    def test_missing_dates_and_future_observations_cannot_become_current_measurements(self):
        self.assertIsNone(measured(observed="2026-11-14"))
        row = measured(observed=None)
        self.assertIsNone(H.latest_measurement(POINT["point_id"], "water_level_cm", NOW,
                                             snapshot={"observations": [row]}))

    def test_bulletin_range_remains_text_without_a_synthetic_point_value(self):
        text = "На Яне у Верхоянска толщина льда составляет от 80 до 140 см."
        rows = ingest.bulletin("Якутия", text, NOW, "https://ykuthydromet.ru/reki/", "a"*64)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scope"], "river_reach")
        self.assertEqual(rows[0]["text"], text)
        self.assertNotIn("value", rows[0])
        self.assertNotIn("ice_cm", rows[0])

    def test_yakutia_calendar_dates_and_snow_context_stay_a_bulletin(self):
        html = '<div class="post" data-year="2026" data-month="2" data-day="11">' \
               '<div class="content">По измерениям от 10 февраля на Лене 90–124 см. ' \
               'Снег на суше до 74 см.</div></div>'
        rows = ingest.parse_yakutia(html, "a"*64)
        self.assertEqual(rows[0]["published_at"], "2026-02-11T23:59:59+00:00")
        self.assertEqual(rows[0]["scope"], "river_reach")
        self.assertNotIn("snow_on_ice_cm", rows[0])
        self.assertEqual(rows[0]["region"], "yakutia")

    def test_sea_ice_is_not_added_to_river_gauge_context(self):
        self.assertEqual(ingest.bulletin("Ледовая обстановка в Охотском море", "Магаданская область", NOW,
                                        "https://www.meteorf.gov.ru/press/news/1/", "a"*64), [])
        self.assertFalse(ingest.river_product("Ледовая обстановка в восточном секторе Арктики"))
        self.assertEqual(ingest._regions("Сахалинская область", "реки Сахалинской области"), [])

    def test_article_publication_is_later_than_the_month_it_describes(self):
        html = '<div class="inside layout"><h1>Обзор в марте 2026 года Архангельской области</h1>' \
               '<p>10/04/2026</p><p>На Пинеге 20 марта толщина льда 40–57 см.</p>' \
               '<p>См. также: посторонний текст</p></div>'
        rows = ingest.parse_article(html, "https://www.sevmeteo.ru/weather/gidro/19018/", "a"*64)
        self.assertEqual(rows[0]["published_at"], "2026-04-10T23:59:59+00:00")
        self.assertNotIn("посторонний", rows[0]["text"])
        self.assertEqual(rows[0]["region"], "arkhangelsk")

    def test_discovery_accepts_new_official_articles_and_rejects_other_hosts(self):
        html = '<a href="/news/123/">Гидрологическая обстановка в Иркутской области</a>' \
               '<a href="https://example.com/999/">Гидрологическая обстановка</a>'
        self.assertEqual(ingest.discover(html, "https://www.rcpod.ru/news/"),
                         [("https://www.rcpod.ru/news/123/", None)])
        rss = '<rss><channel><item><title>Гидрологическая обстановка в Якутии</title>' \
              '<link>https://www.meteorf.gov.ru/press/news/123/</link>' \
              '<pubDate>Thu, 12 Nov 2026 12:00:00 +0300</pubDate></item></channel></rss>'
        found = ingest.discover(rss, "https://www.meteorf.gov.ru/press/news/rss/")
        self.assertEqual(found[0][1].hour, 12)


class AvailabilityAndRetentionTests(unittest.TestCase):
    def test_the_same_timestamp_in_different_source_formats_is_not_a_new_observation(self):
        a = measured(observed="2026-11-09 22:00")
        b = measured(observed="2026-11-09T22:00:00Z")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(len(ingest.merge_records([a], [b])), 1)

    def test_retrieving_an_undated_card_again_keeps_its_latest_capture_and_original_availability(self):
        old = measured(value=27, metric="ice_cm", observed=None, fetched="2026-11-10T10:00:00Z")
        new = measured(value=27, metric="ice_cm", observed=None, fetched="2026-11-12T10:00:00Z")
        rows = ingest.merge_records([old], [new])
        self.assertEqual(rows[0]["available_at"], old["available_at"])
        self.assertEqual(rows[0]["fetched_at"], new["fetched_at"])

    def test_later_retrieval_preserves_first_seen_but_correction_has_its_own_availability(self):
        early = measured(fetched="2026-11-10T10:00:00Z")
        same = measured(fetched="2026-11-12T10:00:00Z")
        rows = ingest.merge_records([early], [same])
        self.assertEqual(rows[0]["available_at"], early["available_at"])
        correction = measured(value=250, fetched="2026-11-12T10:00:00Z")
        rows = ingest.merge_records(rows, [correction])
        before = H.latest_measurement(POINT["point_id"], "water_level_cm", date(2026,11,11),
                                      snapshot={"observations": rows})
        after = H.latest_measurement(POINT["point_id"], "water_level_cm", NOW,
                                     snapshot={"observations": rows})
        self.assertEqual(before["value"], 206)
        self.assertEqual(after["value"], 250)

    def test_stale_and_reference_post_records_are_not_forecast_inputs(self):
        row = measured()
        self.assertIsNone(H.latest_measurement(POINT["point_id"], "water_level_cm", "2026-12-01",
                                             snapshot={"observations": [row]}))
        reference = {**row, "scope": "reference_post"}
        self.assertIsNone(H.latest_measurement(POINT["point_id"], "water_level_cm", NOW,
                                             snapshot={"observations": [reference]}))

    def test_source_outage_keeps_all_previous_records_and_last_success(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            old = dict(schema_version=1, last_success_at="2026-11-10T10:00:00+00:00",
                       observations=[measured()], bulletins=[], documents=[])
            (directory/"snapshot.json").write_text(json.dumps(old))
            with patch.object(ingest, "_get", side_effect=requests.ConnectionError("offline")):
                result = ingest.update(directory=directory, now=NOW)
            saved = json.loads((directory/"snapshot.json").read_text())
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(saved["observations"], old["observations"])
            self.assertEqual(saved["last_success_at"], old["last_success_at"])
            self.assertFalse((directory/".update.lock").exists())

    def test_another_worker_cannot_overwrite_an_active_update(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with ingest.update_lock(directory) as first:
                self.assertTrue(first)
                result = ingest.update(directory=directory, now=NOW)
                self.assertEqual(result["status"], "already_running")
                self.assertFalse((directory/"snapshot.json").exists())

    def test_redirects_cannot_send_requests_outside_official_hosts(self):
        reply = Mock(status_code=302, headers={"Location": "https://example.com/private"})
        with tempfile.TemporaryDirectory() as name, patch.object(requests, "get", return_value=reply) as call:
            with self.assertRaises(ValueError):
                ingest._get("https://meteo-dv.ru/hydro_dfo/", name)
            self.assertEqual(call.call_count, 1)

    def test_refresh_interval_and_missing_status(self):
        self.assertTrue(due({}, NOW))
        self.assertFalse(due({"checked_at": (NOW-timedelta(hours=5)).isoformat()}, NOW))
        self.assertTrue(due({"checked_at": (NOW-timedelta(hours=6)).isoformat()}, NOW))

    def test_application_starts_one_background_collector_without_blocking(self):
        called = threading.Event()
        with patch.dict(os.environ, {"OKNO_HYDRO_AUTO_REFRESH": "1"}), \
             patch.object(refresh, "due", return_value=True), \
             patch.object(ingest, "update", side_effect=lambda: called.set()):
            try:
                worker = refresh.start_auto_update()
                self.assertIs(worker, refresh.start_auto_update())
                self.assertTrue(called.wait(3))
                self.assertTrue(worker.daemon)
            finally:
                refresh._stop.set()
                if refresh._worker:
                    refresh._worker.join(timeout=3)
                refresh._worker = None

    def test_snow_features_only_accept_dated_available_measurements_at_same_post(self):
        issue = int((date(2026,11,12)-date(2026,9,1)).days)
        row = measured(value=22, metric="snow_on_ice_cm", observed="2026-11-11T08:00:00Z",
                       fetched="2026-11-12T10:00:00Z")
        arrays = (np.array([]), np.array([]), np.array([]), np.array([]))
        with patch.object(snow, "_season", return_value=arrays), \
             patch.object(H, "load_snapshot", return_value={"observations": [row]}):
            self.assertEqual(snow.features(POINT["point_id"], 2027, issue)["snow_on_ice_cm"], 22)
            self.assertEqual(snow.features(POINT["point_id"], 2027, issue-1)["snow_on_ice_observed"], 0)
            self.assertEqual(snow.features("gauge_3219", 2027, issue)["snow_on_ice_observed"], 0)


class SavedOfficialDataTests(unittest.TestCase):
    def test_snapshot_preserves_all_regions_and_has_verifiable_originals(self):
        snapshot = H.load_snapshot()
        self.assertEqual(snapshot.get("schema_version"), 1)
        rows = H.coverage(snapshot=snapshot)
        self.assertEqual({r["region"] for r in rows}, set(REGIONS))
        self.assertTrue(all(r["observations"] or r["bulletins"] for r in rows), rows)
        for digest in {r["raw_sha256"] for r in snapshot.get("observations", [])+snapshot.get("bulletins", [])}:
            raw = gzip.decompress((H.DIRECTORY/"raw"/f"{digest}.gz").read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        known = set(H.registry())
        self.assertTrue(all(r["point_id"] in known for r in snapshot["observations"]))

    def test_dashboard_shows_coverage_and_data_for_every_supported_region(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {"OKNO_HYDRO_AUTO_REFRESH": "0"}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app/dashboard.py"), default_timeout=90)
            app.session_state["dashboard_open"] = True
            app.run()
            for region in REGIONS:
                app.sidebar.selectbox(key="filter_region").select(region).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                coverage = next(d.value for d in app.dataframe if "Точек в проекте" in d.value.columns)
                self.assertEqual(len(coverage), 7)
                self.assertTrue(any("Официальные наблюдения" in e.label for e in app.expander))


if __name__ == "__main__":
    unittest.main()
