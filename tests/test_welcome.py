"""Проверки первого экрана, скачивания PDF и переходов в реальный дашборд.

Запуск: python -m unittest discover -s tests -v
"""

import hashlib
import datetime as dt
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
UPLOADED_PDF_SHA256 = "2610c9cc8bd94d6fea79223a3b5c131035f4288f98345056e1b1df2d51553760"
DASHBOARD_TABS = ["Карта путей", "Карточка пути", "Планировщик",
                  "Рейтинг риска", "До / после", "О данных"]


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

    def test_planner_accepts_small_order_and_fractional_parameters(self):
        app = self.new_app()
        app.button(key="open_dashboard").click().run()
        values = {
            "Осталось вывезти на дату решения, т": 45.0,
            "Своя техника, т/нед": 28.0,
            "Доп. техника максимум, т/нед": 18.7,
            "Зимник, ₽/т": 22447.50,
            "Надбавка за доп. технику, ₽/т": 4489.50,
            "Дорогой канал (авиа/СВП), ₽/т": 500000.0,
            "Хранение, ₽/т·нед": 149.80,
        }
        for widget in app.number_input:
            widget.set_value(values[widget.label])
        app.run()
        self.assertEqual([e.message for e in app.exception], [])
        for widget in app.number_input:
            self.assertAlmostEqual(widget.value, values[widget.label])
            self.assertLessEqual(widget.min, widget.value)
            self.assertGreaterEqual(widget.max, widget.value)
        self.assertTrue(any("Рекомендация: своей техникой" in item.value for item in app.markdown))

    def test_small_order_example_loads_all_inputs_and_keeps_filters_working(self):
        app = self.new_app()
        app.button(key="open_dashboard").click().run()
        app.sidebar.button(key="load_nizhneyansk_batagay").click().run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(app.sidebar.selectbox(key="region").value, "yakutia")
        self.assertEqual(app.sidebar.selectbox(key="hydro_year").value, 2024)
        self.assertEqual(app.sidebar.date_input(key="decision_date").value, dt.date(2024, 3, 22))
        self.assertEqual(app.sidebar.selectbox(key="point_id").value, "gauge_3414")
        self.assertFalse(app.sidebar.checkbox(key="weather_close_enabled_v2").value)
        self.assertEqual(app.number_input(key="Q_t").value, 45.0)
        self.assertEqual(app.number_input(key="K_t_week").value, 28.0)
        self.assertEqual(app.number_input(key="K_extra_t_week").value, 18.7)
        self.assertEqual(app.number_input(key="c_road").value, 22447.50)
        self.assertEqual(app.number_input(key="c_extra").value, 4489.50)
        self.assertEqual(app.number_input(key="c_exp").value, 500000.0)
        self.assertEqual(app.number_input(key="c_hold_week").value, 149.80)
        self.assertTrue(any("не ограничивает отправки 31 марта" in item.value for item in app.info))

        app.sidebar.selectbox(key="hydro_year").select(2023).run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(app.sidebar.date_input(key="decision_date").value, dt.date(2023, 3, 1))
        app.sidebar.selectbox(key="region").select("yanao").run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(app.sidebar.selectbox(key="point_id").value, "gauge_11558")

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
