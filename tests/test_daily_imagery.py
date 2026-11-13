from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from PIL import Image
import requests

from okno import daily_imagery as D

NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)
POINT = "yakutsk_crossing"


def png(color=(100, 140, 180, 255), size=(D.SIZE, D.SIZE)):
    buffer = io.BytesIO()
    Image.new("RGBA", size, color).save(buffer, "PNG")
    return buffer.getvalue()


def capabilities(end="2026-11-13"):
    layers = "".join(f'<Layer><o:Identifier>{s["layer"]}</o:Identifier>'
                     f'<Dimension><o:Identifier>Time</o:Identifier><Value>2026-11-01/{end}/P1D</Value></Dimension></Layer>'
                     for s in D.SOURCES.values())
    return ('<Capabilities xmlns="http://www.opengis.net/wmts/1.0" '
            'xmlns:o="http://www.opengis.net/ows/1.1"><Contents>'+layers+'</Contents></Capabilities>').encode()


class Response:
    def __init__(self, content=None, mime="image/png"):
        self.content = png() if content is None else content
        self.headers = {"Content-Type": mime}

    def raise_for_status(self):
        pass

    def iter_content(self, _size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class Session:
    def __init__(self, catalog=None):
        self.catalog = capabilities() if catalog is None else catalog
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.catalog, "application/xml") if url == D.CAPABILITIES else Response()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class CatalogAndImageTests(unittest.TestCase):
    def test_exact_daily_dates_including_catalogue_gaps(self):
        catalog = D.parse_catalog(capabilities())
        catalog["modis_terra"]["dates"] = ["2026-11-01/2026-11-10/P1D", "2026-11-12"]
        self.assertTrue(D.published(catalog, "modis_terra", "2026-11-12"))
        self.assertFalse(D.published(catalog, "modis_terra", "2026-11-11"))
        self.assertFalse(D.published(catalog, "modis_terra", "2026-11-14"))
        with self.assertRaises(ValueError):
            D.parse_catalog(b"<Capabilities/>")

    def test_wms_uses_explicit_date_and_longitude_latitude_axis_order(self):
        url = D.image_url(POINT, "modis_terra", "2026-11-12")
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.hostname, "gibs.earthdata.nasa.gov")
        self.assertEqual(query["VERSION"], ["1.1.1"])
        self.assertEqual(query["TIME"], ["2026-11-12"])
        bbox = list(map(float, query["BBOX"][0].split(",")))
        self.assertGreater(bbox[0], 120)  # longitude, not latitude
        self.assertLess(bbox[1], 70)
        self.assertEqual(query["LAYERS"], [D.SOURCES["modis_terra"]["layer"]])

    def test_opaque_black_and_transparent_fill_are_not_displayable_images(self):
        for color in [(0, 0, 0, 255), (150, 150, 150, 0)]:
            result = D.inspect_image(png(color))
            self.assertEqual(result["status"], "no_data")
            self.assertEqual(result["nonempty_fraction"], 0)

    def test_white_cloud_like_image_is_not_claimed_cloud_free(self):
        result = D.inspect_image(png((255, 255, 255, 255)))
        self.assertEqual(result["status"], "image")
        self.assertEqual(result["cloud_quality"], "not_assessed")
        self.assertNotIn("p_ice", result)

    def test_partial_fill_is_reported(self):
        image = Image.new("RGBA", (D.SIZE, D.SIZE), (0, 0, 0, 255))
        image.paste((120, 130, 140, 255), (0, 0, D.SIZE//2, D.SIZE))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        result = D.inspect_image(buffer.getvalue())
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["nonempty_fraction"], .5)

    def test_wrong_size_and_bad_payloads_are_rejected(self):
        for content in [png(size=(32, 32)), b"<ServiceException>bad layer</ServiceException>",
                        b"x"*(D.MAX_IMAGE_BYTES+1)]:
            with self.assertRaises((ValueError, OSError)):
                D.inspect_image(content)

    def test_download_is_explicitly_browse_only_and_receipt_is_not_backdated(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = D.fetch_image(POINT, "viirs_noaa21", "2026-11-10", session=Session(), directory=tmp, now=NOW)
            self.assertEqual(row["available_at"], NOW.isoformat())
            self.assertEqual(row["date"], "2026-11-10")
            self.assertEqual(row["usage"], "visual_review_only")
            self.assertFalse(row["ml_ready"])
            self.assertEqual(row["cloud_quality"], "not_assessed")
            self.assertTrue(D.preview_path(POINT, "viirs_noaa21", row["date"], tmp).exists())
            with self.assertRaises(ValueError):
                D.fetch_image(POINT, "viirs_noaa21", "2026-11-14", session=Session(), directory=tmp, now=NOW)
            for key,source,day in [("../x", "modis_terra", "2026-11-10"), (POINT, "x", "2026-11-10"),
                                   (POINT, "modis_terra", "../x")]:
                with self.assertRaises(ValueError):
                    D.preview_path(key, source, day, tmp)


class RefreshTests(unittest.TestCase):
    def only_point(self):
        return {POINT: D.points()[POINT]}

    def test_missing_cache_triggers_restore_before_six_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = D.fetch_image(POINT, "modis_terra", "2026-11-12", session=Session(), directory=tmp, now=NOW)
            Path(tmp,"snapshot.json").write_text(json.dumps({"schema_version": 1,"checked_at":NOW.isoformat(),"records":[row]}))
            self.assertFalse(D.due(tmp, NOW))
            D.preview_path(POINT,"modis_terra",row["date"],tmp).unlink()
            self.assertTrue(D.due(tmp, NOW))

    def test_catalogue_does_not_substitute_nearest_day(self):
        point = self.only_point()
        with tempfile.TemporaryDirectory() as tmp, patch.object(D, "points", return_value=point), \
             patch.object(D.requests, "Session", return_value=Session(capabilities("2026-11-12"))), \
             patch.object(D, "fetch_image") as fetch:
            result = D.update(days=1, directory=tmp, now=NOW)
            fetch.assert_not_called()
            self.assertTrue(all(r["date"] == "2026-11-13" and r["status"] == "not_published" for r in result["records"]))

    def test_provider_failure_keeps_previously_downloaded_picture(self):
        point = self.only_point()
        with tempfile.TemporaryDirectory() as tmp:
            row = D.fetch_image(POINT,"modis_terra","2026-11-13",session=Session(),directory=tmp,now=NOW)
            Path(tmp,"snapshot.json").write_text(json.dumps({"schema_version":1,"checked_at":NOW.isoformat(),"records":[row]}))
            with patch.object(D,"points",return_value=point), patch.object(D.requests,"Session",return_value=Session()), \
                 patch.object(D,"fetch_image",side_effect=requests.Timeout):
                result = D.update(days=1,directory=tmp,now=NOW+timedelta(hours=6))
            saved = next(r for r in result["records"] if r["source"] == "modis_terra")
            self.assertEqual(saved,row)
            self.assertTrue(result["errors"])
            self.assertTrue(D.preview_path(POINT,"modis_terra",row["date"],tmp).exists())

    def test_refreshing_same_mosaic_preserves_its_first_availability(self):
        point = self.only_point()
        with tempfile.TemporaryDirectory() as tmp, patch.object(D,"points",return_value=point), \
             patch.object(D.requests,"Session",return_value=Session()):
            first = D.update(days=1,directory=tmp,now=NOW)
            second = D.update(days=1,directory=tmp,now=NOW+timedelta(hours=6))
            for before,after in zip(first["records"],second["records"]):
                self.assertEqual(after["available_at"],before["available_at"])
                self.assertEqual(after["first_received_at"],before["first_received_at"])
                self.assertNotEqual(after["fetched_at"],before["fetched_at"])

    def test_background_collector_is_singleton_and_can_be_stopped(self):
        from okno import daily_imagery_refresh as refresh
        called = threading.Event()
        with patch.dict("os.environ", {"OKNO_DAILY_IMAGERY_AUTO_REFRESH":"1"}), \
             patch.object(D,"due",return_value=True), patch.object(D,"update",side_effect=called.set):
            try:
                first = refresh.start_auto_update()
                self.assertIs(first,refresh.start_auto_update())
                self.assertTrue(called.wait(3))
                self.assertTrue(first.daemon)
            finally:
                refresh._stop.set()
                if refresh._worker:
                    refresh._worker.join(timeout=3)
                refresh._worker = None


class GalleryTests(unittest.TestCase):
    def test_same_date_comparison_and_empty_satellite_do_not_swap_in_old_frame(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as tmp:
            rows = [D.fetch_image(POINT,source,"2026-11-12",session=Session(),directory=tmp,now=NOW)
                    for source in ["modis_terra","viirs_noaa21"]]
            rows.append({"point":POINT,"source":"modis_aqua","date":"2026-11-12","status":"no_data"})
            Path(tmp,"snapshot.json").write_text(json.dumps({"schema_version":1,"checked_at":NOW.isoformat(),"records":rows}))
            with patch.object(D,"DIRECTORY",Path(tmp)):
                app=AppTest.from_string('from app.daily_imagery_view import render_daily_imagery\nrender_daily_imagery()',default_timeout=30).run()
                self.assertFalse(app.exception,[e.message for e in app.exception])
                self.assertEqual(app.selectbox(key="daily_imagery_date").value,"2026-11-12")
                self.assertEqual(len(app.get("download_button")),2)
                app.selectbox(key="daily_modis_terra").select("modis_aqua").run()
                self.assertFalse(app.exception,[e.message for e in app.exception])
                self.assertEqual(app.selectbox(key="daily_imagery_date").value,"2026-11-12")
                self.assertEqual(len(app.get("download_button")),1)
                self.assertTrue(any("нет изображения" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
