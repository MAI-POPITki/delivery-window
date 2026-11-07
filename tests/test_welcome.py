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
UPLOADED_PDF_SHA256 = "f3ec3a496c8df257080519da8a886eccbc02ee60294b0ef51ef28dabec0d6cb8"  # паспорт v3 после ревью А3 (одна версия с docs/project.pdf)
DASHBOARD_TABS = ["Карта путей", "Карточка пути", "Сквозной коридор (А3)", "Планировщик",
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
        planner = next(tab for tab in app.tabs if tab.label == "Планировщик")
        self.assertEqual({w.label for w in planner.number_input}, set(values))
        for widget in planner.number_input:
            widget.set_value(values[widget.label])
        app.run()
        self.assertEqual([e.message for e in app.exception], [])
        planner = next(tab for tab in app.tabs if tab.label == "Планировщик")
        for widget in planner.number_input:
            self.assertAlmostEqual(widget.value, values[widget.label])
            self.assertLessEqual(widget.min, widget.value)
            self.assertGreaterEqual(widget.max, widget.value)
        self.assertTrue(any("Рекомендация: своей техникой" in item.value for item in app.markdown))
        charts = planner.get('plotly_chart')
        self.assertEqual([json.loads(chart.proto.spec)['layout']['title']['text'] for chart in charts],
                         ['Обычный ML: отправки по неделям', 'Экономический ML: отправки по неделям'])
        before = next(d.value for d in planner.dataframe if 'Обычный ML: своя, т' in d.value.columns)
        self.assertAlmostEqual((before['Обычный ML: своя, т']+before['Обычный ML: доп., т']).sum(), 45.)
        self.assertAlmostEqual((before['Экономический ML: своя, т']+before['Экономический ML: доп., т']).sum(), 45.)
        # Same order and forecast; a backup rate below the road rate must affect
        # the economic chart, while the ordinary window schedule stays intact.
        app.number_input(key='planner_backup_rate').set_value(10000.).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        planner = next(tab for tab in app.tabs if tab.label == 'Планировщик')
        after = next(d.value for d in planner.dataframe if 'Обычный ML: своя, т' in d.value.columns)
        self.assertEqual(after['Обычный ML: своя, т'].tolist(), before['Обычный ML: своя, т'].tolist())
        self.assertEqual((after['Экономический ML: своя, т']+after['Экономический ML: доп., т']).sum(), 0.)
        summary = next(d.value for d in planner.dataframe if 'Ожидается дорогим каналом, т' in d.value.columns)
        self.assertEqual(summary.loc['Экономический ML','Ожидается дорогим каналом, т'], 45.)
        self.assertEqual(summary.loc['Экономический ML','Ожидаемые расходы, ₽'], 450000.)

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
