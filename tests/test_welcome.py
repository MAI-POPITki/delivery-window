"""Проверки первого экрана, скачивания PDF и переходов в реальный дашборд.

Запуск: python -m unittest discover -s tests -v
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.welcome import PROJECT_PDF

DASHBOARD = ROOT / "app" / "dashboard.py"
UPLOADED_PDF_SHA256 = "5096e22f179e33731a4a4eddef9bc538c662cd8a97e358139f7750f3aa71ffb2"  # полное описание v6: сверенные цифры и интеграция ZXC
DASHBOARD_TABS = ["Карта путей", "Карточка пути", "Сквозной коридор (А3)", "Риски и рекомендации",
                  "Планировщик", "До / после", "О данных"]



class WelcomeTests(unittest.TestCase):
    def new_app(self):
        return AppTest.from_file(str(DASHBOARD), default_timeout=90).run()

    def assert_welcome(self, app):
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual([title.value for title in app.title], ["Окно завоза"])
        self.assertEqual(app.button(key="open_dashboard").label, "Открыть дашборд")
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(len(app.sidebar.selectbox), 0)

    def test_welcome_is_first_screen_without_loading_forecasts(self):
        with patch("okno.forecast.data_version", side_effect=AssertionError("Данные загружены до входа")) as data_version:
            app = self.new_app()
        self.assert_welcome(app)
        self.assertEqual(len(app.button), 1)
        self.assertEqual(len(app.download_button), 1)
        data_version.assert_not_called()

    def test_download_serves_the_original_uploaded_pdf(self):
        storage = MemoryMediaFileStorage("/mock/media")
        with patch("streamlit.testing.v1.app_test.MemoryMediaFileStorage", return_value=storage):
            app = self.new_app()
        self.assert_welcome(app)
        download = app.download_button(key="download_project")
        media = storage.get_file(Path(download.proto.url).name)
        self.assertEqual(media.mimetype, "application/pdf")
        self.assertEqual(media.filename, "okno-zavoza-project.pdf")
        self.assertTrue(media.content.startswith(b"%PDF-"))
        self.assertEqual(hashlib.sha256(media.content).hexdigest(), UPLOADED_PDF_SHA256)
        self.assertEqual(media.content, PROJECT_PDF.read_bytes())
        self.assertTrue(download.proto.ignore_rerun)

    def test_download_keeps_the_welcome_screen(self):
        app = self.new_app()
        app.download_button(key="download_project").click().run()
        self.assert_welcome(app)
        self.assertEqual(len(app.download_button), 1)

    def test_dashboard_navigation_filters_and_return(self):
        app = self.new_app()
        app.button(key="open_dashboard").click().run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual([tab.label for tab in app.tabs], DASHBOARD_TABS)
        self.assertEqual([widget.label for widget in app.sidebar.selectbox], ["Регион", "Сезон", "Путь"])

        # Обычное изменение фильтра не возвращает пользователя на первый экран.
        horizon = next(widget for widget in app.sidebar.slider if widget.label == "Горизонт N, дней")
        horizon.set_value(21).run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual([tab.label for tab in app.tabs], DASHBOARD_TABS)
        self.assertIn("21 дн.", app.subheader[0].value)

        app.sidebar.button(key="show_welcome").click().run()
        self.assert_welcome(app)
        self.assertEqual(len(app.download_button), 1)

        app.button(key="open_dashboard").click().run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual([tab.label for tab in app.tabs], DASHBOARD_TABS)
        self.assert_welcome(self.new_app())  # Новая сессия снова начинается со вступления.

    def test_missing_document_does_not_block_dashboard_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.welcome.PROJECT_PDF", Path(directory) / "missing.pdf"):
                app = self.new_app()
        self.assert_welcome(app)
        self.assertEqual(len(app.download_button), 0)
        self.assertEqual(len(app.warning), 1)

    def test_document_path_is_independent_of_working_directory(self):
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                app = self.new_app()
            finally:
                os.chdir(previous_directory)
        self.assert_welcome(app)
        self.assertEqual(len(app.download_button), 1)


if __name__ == "__main__":
    unittest.main()
