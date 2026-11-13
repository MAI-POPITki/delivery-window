"""Provider failures, real footprints, and strict separation from M2 inference."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

from okno import gaofen as G

NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
POINT = {"name": "Test", "lat": 62.0, "lon": 129.8, "bbox": [129.7, 61.9, 129.9, 62.1]}
GEOMETRY = {"type": "Polygon", "coordinates": [[[129, 61], [131, 61], [131, 63], [129, 63], [129, 61]]]}


def source_row(identifier="1", day="2026-11-09"):
    return {"searchId": identifier, "satelliteName": "GF3", "productLevelName": "L1A", "sensorName": "FSII",
            "productName": f"GF3_MDJ_FSII_053957_E128.7_N62.6_{day.replace('-', '')}_L1A_HHHV_L10007496592",
            "imagingStartTime": day+" 08:57:20", "geomSearchGeoJson": json.dumps(GEOMETRY),
            "browseFileLocation": f"//metadata_135/DDS_IMAGES/browse/GF3/FSII/L1A/{day.replace('-', '')}/GF3_{identifier}_browse.png"}


def png():
    out = io.BytesIO()
    Image.new("L", (256, 256), 72).save(out, format="PNG")
    return out.getvalue()


class Response:
    def __init__(self, rows=(), *, content=None, content_type="image/png", count=None, code=0):
        self.status_code = 200
        self.rows = rows
        self.count = len(rows) if count is None else count
        self.code = code
        self.content = png() if content is None else content
        self.headers = {"Content-Type": content_type}

    def json(self):
        return {"code": self.code, "data": {"recordCount": self.count, "pageData": self.rows}}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, _size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def session(rows=()):
    client = Mock()
    client.post.return_value = Response(rows)
    client.get.return_value = Response()
    return client


class FootprintTests(unittest.TestCase):
    def test_entire_aoi_required(self):
        self.assertTrue(G.covers_aoi(GEOMETRY, POINT["bbox"]))
        self.assertFalse(G.covers_aoi(GEOMETRY, [130.9, 62, 131.1, 62.1]))

    def test_holes_concave_and_invalid_footprints_rejected(self):
        hole = copy.deepcopy(GEOMETRY)
        hole["coordinates"].append(hole["coordinates"][0])
        self.assertFalse(G.covers_aoi(hole, POINT["bbox"]))
        concave = {"type": "Polygon", "coordinates": [[[129, 61], [131, 61], [129.8, 62], [131, 63], [129, 63], [129, 61]]]}
        self.assertFalse(G.covers_aoi(concave, POINT["bbox"]))
        invalid = copy.deepcopy(GEOMETRY)
        invalid["coordinates"][0][1][0] = float("nan")
        self.assertFalse(G.covers_aoi(invalid, POINT["bbox"]))

    def test_optical_wrong_provider_future_and_missing_footprint_rejected(self):
        for key, value in [("satelliteName", "GF1"), ("productLevelName", "L2"),
                           ("imagingStartTime", "2027-01-01 10:00:00"), ("geomSearchGeoJson", "{}")]:
            row = source_row()
            row[key] = value
            self.assertIsNone(G.normalize_scene(row, POINT, NOW), key)

    def test_nas_path_never_becomes_another_network_host(self):
        for path in ["https://other.test/image.png", "//other.test/a.png",
                     "//metadata_135/DDS_IMAGES/browse/GF3/../secret.png"]:
            with self.assertRaises(ValueError):
                G.browse_url(path)

    def test_radar_browse_is_never_a_calibrated_model_input(self):
        scene = G.normalize_scene(source_row(), POINT, NOW)
        self.assertEqual(scene["polarization_code"], "HHHV")
        self.assertFalse(scene["ml_ready"])
        self.assertEqual(scene["usage"], "visual_review_only")
        self.assertEqual(scene["time_zone"], "unspecified_by_catalogue")


class ProviderTests(unittest.TestCase):
    def test_partial_newer_scene_does_not_replace_complete_older_scene(self):
        newer = source_row("2", "2026-11-10")
        newer["geomSearchGeoJson"] = json.dumps({"type": "Polygon", "coordinates": [[[129.8, 61], [131, 61], [131, 63], [129.8, 63], [129.8, 61]]]})
        client = session([newer, source_row()])
        scenes, count = G.search_point(POINT, now=NOW, session=client)
        self.assertEqual(count, 2)
        self.assertEqual([r["id"] for r in scenes], ["1"])
        query = client.post.call_args.kwargs["json"]["filterObject"]
        self.assertIn("129.7 62.1", query["geometryQuery"])
        self.assertEqual(query["satelliteId"], ["530"])

    def test_incomplete_or_excessive_catalogue_does_not_claim_latest_scene(self):
        for count in [101, 301]:
            client = session([source_row()])
            client.post.return_value.count = count
            with self.assertRaises(ValueError):
                G.search_point(POINT, now=NOW, session=client)

    def test_error_payload_rejected_even_when_http_succeeds(self):
        client = session()
        client.post.return_value.code = 401
        with self.assertRaises(ValueError):
            G.search_point(POINT, now=NOW, session=client)

    def test_download_validates_png_and_skips_duplicate_network_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = session()
            scene = G.normalize_scene(source_row(), POINT, NOW)
            preview = G.download_preview(scene, directory=root, session=client, now=NOW)
            self.assertEqual(preview["sha256"], hashlib.sha256(png()).hexdigest())
            self.assertEqual((preview["width"], preview["height"]), (256, 256))
            scene["preview"] = preview
            repeated = G.download_preview(scene, directory=root, session=client, now=NOW+timedelta(hours=6))
            self.assertEqual(client.get.call_count, 1)
            self.assertEqual(repeated["downloaded_at"], preview["downloaded_at"])
            self.assertNotEqual(repeated["validated_at"], preview["validated_at"])

    def test_html_and_oversized_download_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = session()
            scene = G.normalize_scene(source_row(), POINT, NOW)
            client.get.return_value = Response(content=b"<html>login</html>", content_type="text/html")
            with self.assertRaises(ValueError):
                G.download_preview(scene, directory=Path(tmp), session=client, now=NOW)
            client.get.return_value = Response()
            with patch.object(G, "MAX_IMAGE_BYTES", 8), self.assertRaises(ValueError):
                G.download_preview(scene, directory=Path(tmp), session=client, now=NOW)
            self.assertFalse(G.preview_path(scene, tmp).exists())

    def test_corrupted_cache_is_restored_from_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = G.normalize_scene(source_row(), POINT, NOW)
            path = G.preview_path(scene, tmp)
            path.parent.mkdir()
            path.write_bytes(b"broken cache")
            client = session()
            G.download_preview(scene, directory=Path(tmp), session=client, now=NOW)
            self.assertEqual(path.read_bytes(), png())
            client.get.assert_called_once()


class UpdateTests(unittest.TestCase):
    def _update(self, root, client, **kwargs):
        with patch.object(G, "points", return_value={"yakutsk_crossing": POINT}):
            return G.update(directory=root, session=client, now=kwargs.pop("now", NOW), **kwargs)

    def test_source_failure_preserves_last_good_scene_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = self._update(root, session([source_row()]))
            initial_scene = copy.deepcopy(initial["points"]["yakutsk_crossing"]["scene"])
            broken = session()
            broken.post.side_effect = requests.Timeout("outage")
            updated = self._update(root, broken, now=NOW+timedelta(days=1))
            row = updated["points"]["yakutsk_crossing"]
            self.assertEqual(row["scene"], initial_scene)
            self.assertEqual(row["status"], "source_unavailable")
            self.assertEqual(G.preview_path(initial_scene, root).read_bytes(), png())
            self.assertFalse(updated["ml_ready"])
            self.assertEqual(G.load_snapshot(root), updated)

    def test_failed_new_image_never_replaces_old_good_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._update(root, session([source_row()]))
            client = session([source_row("2", "2026-11-10")])
            client.get.return_value = Response(content=b"not a PNG")
            updated = self._update(root, client)
            self.assertEqual(updated["points"]["yakutsk_crossing"]["scene"]["id"], "1")
            self.assertEqual(updated["points"]["yakutsk_crossing"]["status"], "source_unavailable")

    def test_missing_cache_triggers_refresh_despite_fresh_catalogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._update(root, session([source_row()]))
            self.assertFalse(G.due(result, directory=root, now=NOW+timedelta(hours=1)))
            self.assertTrue(G.due(result, directory=root, now=NOW+timedelta(hours=6)))
            G.preview_path(result["points"]["yakutsk_crossing"]["scene"], root).unlink()
            self.assertTrue(G.due(result, directory=root, now=NOW+timedelta(hours=1)))

    def test_points_use_actual_sentinel_coordinates_and_reject_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.csv"
            path.write_text("point,lat,lon\nyakutsk_crossing,62.01,129.81\nyakutsk_crossing,nan,5\n")
            points = G.points(path)
            self.assertEqual(points["yakutsk_crossing"]["coordinate_source"], "sentinel_1_index")
            self.assertEqual(points["yakutsk_crossing"]["lon"], 129.81)
            self.assertEqual(len(points), 7)

    def test_offline_mode_disables_background_network(self):
        from okno.gaofen_refresh import start_auto_update
        with patch.dict("os.environ", {"OKNO_GAOFEN_AUTO_REFRESH": "0"}):
            self.assertIsNone(start_auto_update())


if __name__ == "__main__":
    unittest.main()
