"""Тесты ежедневного автообновления ледовой фенологии ЯУГМС."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd

from okno import hydro_ingest, hydro_refresh, ykut_phenology

SAMPLE_HTML = """
<div class=post data-year=2025 data-month=11 data-day=12>
  <div class=content>
    У г. Якутск 11 ноября установился ледостав.
  </div>
</div>
<div class=post data-year=2026 data-month=05 data-day=18>
  <div class=content>
    У г. Якутск вчера начался ледоход.
  </div>
</div>
"""


class YkutPhenologyTests(unittest.TestCase):
    def test_due_checks_missing_and_24h_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ykut.csv"
            now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
            self.assertTrue(ykut_phenology.due(p, now=now))
            p.write_text("point_id\n", encoding="utf-8")
            ts_fresh = (now - timedelta(hours=23)).timestamp()
            os.utime(p, (ts_fresh, ts_fresh))
            self.assertFalse(ykut_phenology.due(p, now=now))
            ts_old = (now - timedelta(hours=25)).timestamp()
            os.utime(p, (ts_old, ts_old))
            self.assertTrue(ykut_phenology.due(p, now=now))

    def test_update_writes_atomically_and_guards_against_shrink(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_file = Path(tmp) / "reki.html"
            html_file.write_text(SAMPLE_HTML, encoding="utf-8")
            out_csv = Path(tmp) / "ykut_reki_phenology.csv"

            df, posts = ykut_phenology.update(cached=html_file, out=out_csv)
            self.assertEqual(len(posts), 2)
            self.assertEqual(len(df), 2)
            self.assertTrue(out_csv.exists())

            # Пустой или неполный ответ не должен затирать сохранённый файл
            empty_html = Path(tmp) / "empty.html"
            empty_html.write_text("<html></html>", encoding="utf-8")
            with self.assertRaises(ValueError):
                ykut_phenology.update(cached=empty_html, out=out_csv)
            self.assertEqual(len(pd.read_csv(out_csv)), 2)

            shrunk_html = Path(tmp) / "shrunk.html"
            shrunk_html.write_text(
                "<div class=post data-year=2025 data-month=11 data-day=12>"
                "<div class=content>У г. Якутск 11 ноября установился ледостав.</div></div>",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ykut_phenology.update(cached=shrunk_html, out=out_csv)
            self.assertEqual(len(pd.read_csv(out_csv)), 2)

    def test_hydro_refresh_triggers_phenology_update(self):
        called = threading.Event()
        with patch.dict(os.environ, {"OKNO_HYDRO_AUTO_REFRESH": "1"}), \
             patch.object(hydro_refresh, "due", return_value=False), \
             patch.object(hydro_ingest, "update"), \
             patch.object(ykut_phenology, "due", return_value=True), \
             patch.object(ykut_phenology, "update", side_effect=lambda: called.set()):
            try:
                hydro_refresh._stop.set()
                if hydro_refresh._worker:
                    hydro_refresh._worker.join(timeout=3)
                hydro_refresh._worker = None
                worker = hydro_refresh.start_auto_update()
                self.assertTrue(called.wait(3))
                self.assertTrue(worker.daemon)
            finally:
                hydro_refresh._stop.set()
                if hydro_refresh._worker:
                    hydro_refresh._worker.join(timeout=3)
                hydro_refresh._worker = None


if __name__ == "__main__":
    unittest.main()
