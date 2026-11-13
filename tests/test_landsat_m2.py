import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from okno import landsat as L, m2_landsat as M
from okno import m2_observations as O

NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)
ITEM = {"item_id": "LC09_L2SP_120017_20231001_02_T1", "date": "2023-10-01",
        "observed_at": "2023-10-01T02:00:00Z", "published_at": "2023-10-03T00:00:00Z", "platform": "landsat-9"}


def stack(ice=False):
    values = [25000, 30000, 28000, 25000, 11000, 9500] if ice else [9000, 10000, 9000, 8000, 7500, 7500]
    result = np.zeros((8, L.SIZE, L.SIZE), dtype=np.uint16)
    result[:6] = np.asarray(values, dtype=np.uint16)[:, None, None]
    return result


class OpticalPhysicsTests(unittest.TestCase):
    def test_usgs_scale_and_offset_are_both_applied(self):
        data = stack()
        self.assertAlmostEqual(float(L.reflectance(data)[1, 0, 0]), 0.075, places=5)

    def test_snow_is_preserved_but_cloud_shadow_cirrus_and_fill_are_not(self):
        data = stack(ice=True)
        data[6] = 1 << 5
        self.assertTrue(L.clear_pixels(data).all())
        for bit in range(5):
            data[6] = 1 << bit
            self.assertFalse(L.clear_pixels(data).any(), bit)

    def test_selected_band_saturation_and_terrain_occlusion_are_rejected(self):
        data = stack()
        for bit in [1, 2, 3, 4, 5, 6, 11]:
            data[7] = 1 << bit
            self.assertFalse(L.clear_pixels(data).any(), bit)
        data[7] = 1  # saturation of the unused coastal band
        self.assertTrue(L.clear_pixels(data).all())

    def test_nodata_or_clouds_do_not_become_a_water_prediction(self):
        mask = np.ones((L.SIZE, L.SIZE), dtype=bool)
        for data in [np.zeros_like(stack()), stack()]:
            if data.any():
                data[6] = 8
            feat, quality = M.features(data, mask)
            self.assertIsNone(feat)
            self.assertEqual(quality["quality"], "cloud_or_nodata")

    def test_physical_optical_features_distinguish_bright_ice_and_dark_water(self):
        mask = np.ones((L.SIZE, L.SIZE), dtype=bool)
        ice, _ = M.features(stack(True), mask)
        water, _ = M.features(stack(), mask)
        self.assertGreater(ice["nir08_p50"], water["nir08_p50"])
        self.assertGreater(ice["ndsi_p50"], 0.6)
        self.assertGreater(water["ndwi_p50"], ice["ndwi_p50"])
        self.assertEqual(set(ice), set(M.FEATURES))
        self.assertNotIn("date", M.FEATURES)
        self.assertNotIn("point", M.FEATURES)

    def test_low_fraction_or_too_few_river_pixels_abstain(self):
        data = stack()
        mask = np.ones((L.SIZE, L.SIZE), dtype=bool)
        data[6, :150] = 8
        self.assertIsNone(M.features(data, mask)[0])
        mask[:] = False
        mask[:2, :2] = True
        self.assertIsNone(M.features(stack(), mask)[0])

    def test_mask_needs_repeated_clear_summer_water(self):
        data = stack(True)
        data[:6, 50:150, 120:170] = stack()[:6, 50:150, 120:170]
        seed = np.zeros((300, 300), dtype=bool)
        seed[:, 210:270] = True
        self.assertIsNone(M.build_mask([data], seed))
        mask = M.build_mask([data, data.copy(), data.copy()], seed)
        self.assertIsNotNone(mask)
        self.assertTrue(mask[100, 140])
        self.assertFalse(mask[20, 20])

    def test_transition_buffer_and_missing_reference_do_not_invent_labels(self):
        dates = {2024: {"freezeup": pd.Timestamp("2023-10-15"), "breakup": pd.Timestamp("2024-05-15")}}
        self.assertEqual(M.reference_label("2024-02-01", dates), ("ice", 2024))
        self.assertEqual(M.reference_label("2024-05-10", dates)[0], "transition")
        self.assertEqual(M.reference_label("2024-07-01", dates)[0], "water")
        self.assertEqual(M.reference_label("2026-02-01", dates)[0], "unknown")
        dates[2024]["freezeup"] = pd.Timestamp("2023-08-15")
        self.assertEqual(M.reference_label("2024-02-01", dates)[0], "unknown")


class DownloadTests(unittest.TestCase):
    def response(self, data):
        stream = io.BytesIO()
        np.save(stream, data, allow_pickle=False)
        response = Mock(status_code=200, content=stream.getvalue(), headers={})
        response.raise_for_status.return_value = None
        return response

    def test_download_preserves_all_bands_and_qa_without_rendering_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Mock()
            data = stack(True)
            data[6] = 32
            client.request.return_value = self.response(data)
            path = L.download("yakutsk_crossing", ITEM, session=client, directory=tmp, now=NOW)
            self.assertTrue(np.array_equal(L.load_chip(path), data))
            params = client.request.call_args.kwargs["params"]
            self.assertEqual([v for k, v in params if k == "assets"], list(L.BANDS))
            self.assertIn(("unscale", "false"), params)
            self.assertIn(("resampling", "nearest"), params)
            L.download("yakutsk_crossing", ITEM, session=client, directory=tmp, now=NOW)
            self.assertEqual(client.request.call_count, 1)
            row = L.index_row("yakutsk_crossing", ITEM, path, now=NOW)
            self.assertEqual(row["available_at"], NOW.isoformat())

    def test_wrong_grid_or_dtype_never_enters_feature_extraction(self):
        for data in [np.zeros((3, 200, 200), dtype=np.uint16), stack().astype(np.float32)]:
            with tempfile.TemporaryDirectory() as tmp:
                client = Mock()
                client.request.return_value = self.response(data)
                with self.assertRaises(ValueError):
                    L.download("yakutsk_crossing", ITEM, session=client, directory=tmp, now=NOW)
                self.assertFalse(L.chip_path("yakutsk_crossing", ITEM["item_id"], tmp).exists())

    def test_collection1_level1_and_path_traversal_are_rejected(self):
        for item in ["../../model", "LC08_L1TP_120017_20231001_02_T1", "LC08_L2SP_120017_20231001_01_T1"]:
            with self.assertRaises(ValueError):
                L.chip_path("yakutsk_crossing", item)

    def test_provider_failure_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = {"points": {"yakutsk_crossing": {"item_id": ITEM["item_id"], "status": "ready", "date": ITEM["date"]}}}
            (root/"snapshot.json").write_text(json.dumps(previous))
            with patch.object(L, "points", return_value={"yakutsk_crossing": L.points()["yakutsk_crossing"]}), \
                 patch.object(L, "search", side_effect=requests.Timeout):
                result = L.update(directory=root, now=NOW)
            self.assertEqual(result["points"]["yakutsk_crossing"]["item_id"], ITEM["item_id"])
            self.assertEqual(result["points"]["yakutsk_crossing"]["status"], "source_unavailable")

    def test_cloudy_image_is_not_given_a_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = stack()
            data[6] = 8
            stream = io.BytesIO()
            np.savez_compressed(stream, bands=data)
            path = L.chip_path("yakutsk_crossing", ITEM["item_id"], root)
            path.parent.mkdir(parents=True)
            path.write_bytes(stream.getvalue())
            row = {**ITEM, "point": "yakutsk_crossing", "available_at": NOW.isoformat()}
            fake = Mock()
            meta = {"enabled": True, "trained_at": NOW.isoformat(), "model_sha256": hashlib.sha256(M.MODEL.read_bytes()).hexdigest()}
            with patch.object(M, "metadata", return_value=meta), patch.object(L, "load_index", return_value=pd.DataFrame([row])), \
                 patch.object(M, "load_masks", return_value={"yakutsk_crossing": np.ones((200, 200), dtype=bool)}), \
                 patch.object(M, "load_model", return_value=fake):
                result = M.predict_index(directory=root, now=NOW)
            fake.predict_proba.assert_not_called()
            self.assertTrue(pd.isna(result.p_ice.iloc[0]))


class AvailabilityTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            {"point": "p", "item_id": "s1", "observed_at": "2026-11-10T00:00:00Z", "available_at": "2026-11-10T00:00:00Z", "p_ice": 0.95, "source": "Sentinel-1"},
            {"point": "p", "item_id": "l8", "observed_at": "2026-11-11T00:00:00Z", "available_at": "2026-11-13T00:00:00Z", "p_ice": 0.97, "source": "Landsat 8"}])

    def test_newly_downloaded_old_scene_does_not_leak_into_historical_decision(self):
        out = O.select_latest(self.frame(), as_of=date(2026, 11, 12), now=NOW)
        self.assertEqual(out.item_id.iloc[0], "s1")
        self.assertEqual(O.select_latest(self.frame(), now=NOW).item_id.iloc[0], "l8")

    def test_confident_conflicting_sensors_abstain(self):
        data = self.frame()
        data.loc[1, "p_ice"] = 0.02
        out = O.select_latest(data, now=NOW)
        self.assertTrue(out.conflict.iloc[0])
        self.assertTrue(pd.isna(out.p_ice.iloc[0]))

    def test_stale_and_future_observations_are_not_used(self):
        data = self.frame()
        self.assertTrue(O.select_latest(data, as_of=date(2018, 1, 1), now=NOW).empty)
        self.assertTrue(O.select_latest(data, now=NOW+timedelta(days=30), max_age=14).empty)
        data.loc[1, "observed_at"] = "2027-01-01T00:00:00Z"
        self.assertEqual(O.select_latest(data, as_of=date(2027, 1, 1), now=NOW).item_id.iloc[0], "s1")

    def test_one_background_worker_starts_without_blocking_ui(self):
        from okno import landsat_refresh as refresh
        called = threading.Event()
        with patch.dict("os.environ", {"OKNO_LANDSAT_AUTO_REFRESH": "1"}), patch.object(L, "due", return_value=True), \
             patch.object(L, "update", side_effect=called.set):
            try:
                refresh._stop.set()  # рабочий поток, оставленный тестом дашборда, не должен подменять новый
                if refresh._worker:
                    refresh._worker.join(timeout=3)
                refresh._worker = None
                first = refresh.start_auto_update()
                self.assertIs(first, refresh.start_auto_update())
                self.assertTrue(called.wait(3))
                self.assertTrue(first.daemon)
            finally:
                refresh._stop.set()
                if refresh._worker:
                    refresh._worker.join(timeout=3)
                refresh._worker = None


class SavedModelTests(unittest.TestCase):
    def test_real_model_and_reference_masks_match_validation_manifest(self):
        meta = M.metadata()
        self.assertTrue(meta["enabled"])
        self.assertEqual(hashlib.sha256(M.MODEL.read_bytes()).hexdigest(), meta["model_sha256"])
        self.assertEqual(hashlib.sha256(M.MASKS.read_bytes()).hexdigest(), meta["mask_sha256"])
        self.assertEqual(set(M.load_masks()), set(L.points()))
        oof = pd.read_csv(Path(__file__).resolve().parents[1]/"reports"/"m2_landsat_oof.csv")
        self.assertTrue((oof.trained_through_season < oof.hydro_year).all())
        self.assertGreater(oof.hydro_year.min(), max(meta["mask_calibration_years"]))
        self.assertEqual(len(oof), meta["validation"]["n"])
        self.assertEqual(set(oof.platform), {"landsat-8", "landsat-9"})
        self.assertFalse(oof.duplicated(["point", "item_id"]).any())


class DashboardMapTests(unittest.TestCase):
    def test_combined_map_and_historical_cutoff(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        with patch.dict("os.environ", {"OKNO_LANDSAT_AUTO_REFRESH": "0", "OKNO_GAOFEN_AUTO_REFRESH": "0",
                                      "OKNO_HYDRO_AUTO_REFRESH": "0"}):
            app = AppTest.from_file(str(root/"app/dashboard.py"), default_timeout=120)
            app.session_state["dashboard_open"] = True
            app.session_state["filter_season"] = 2027
            app.session_state["filter_date"] = date(2026, 11, 13)
            app.session_state["layer_s1"] = True
            app.run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            spec = json.loads(app.get("plotly_chart")[0].proto.spec)
            self.assertTrue(any("Landsat" in str(t.get("hovertext", "")) for t in spec["data"]))

            app.sidebar.selectbox(key="filter_season").select(2001).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            spec = json.loads(app.get("plotly_chart")[0].proto.spec)
            self.assertFalse(any("Landsat" in str(t.get("hovertext", "")) for t in spec["data"]))

            app.sidebar.selectbox(key="filter_season").select(2026).run()
            app.sidebar.date_input(key="filter_date").set_value(date(2026, 3, 16)).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            spec = json.loads(app.get("plotly_chart")[0].proto.spec)
            self.assertFalse(any("Landsat" in str(t.get("hovertext", "")) for t in spec["data"]))
            self.assertTrue(any("Sentinel-1" in str(t.get("hovertext", "")) for t in spec["data"]))


if __name__ == "__main__":
    unittest.main()
