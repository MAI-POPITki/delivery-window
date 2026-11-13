"""Обновление работающего Streamlit-процесса с модулями предыдущей версии."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportReloadTests(unittest.TestCase):
    def test_dashboard_refreshes_flood_dependencies(self):
        script = textwrap.dedent("""
            import importlib
            from streamlit.testing.v1 import AppTest
            flood = importlib.import_module('okno.flood')
            view = importlib.import_module('app.flood_view')
            def stale(*args, **kwargs):
                raise RuntimeError('Stale flood module')
            flood.load_points = stale
            view.load_points = stale
            flood._loaded_mtime_ns = 0
            view._loaded_mtime_ns = 0
            app = AppTest.from_file('app/dashboard.py', default_timeout=90)
            app.query_params['module'] = 'flood'
            app.run()
            assert not app.exception, [error.message for error in app.exception]
            assert app.title[0].value == 'Паводки'
            assert flood.load_points is not stale
            assert view.load_points is flood.load_points
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dashboard_refreshes_dependencies_before_importing_new_weather_module(self):
        # Отдельный процесс: не повреждаем sys.modules других тестов.
        script = textwrap.dedent("""
            import importlib
            from pathlib import Path
            import sys
            from streamlit.testing.v1 import AppTest

            paths = importlib.import_module('okno.paths')
            forecast = importlib.import_module('okno.forecast')

            # До обновления WEATHER_FORECASTS и самого weather_forecast не было.
            del paths.WEATHER_FORECASTS
            sys.modules.pop('okno.weather_forecast')
            import okno
            del okno.weather_forecast

            def stale_forecast(*args, **kwargs):
                raise RuntimeError('В памяти осталась старая функция прогноза')

            forecast.forecast = stale_forecast
            forecast._loaded_mtime_ns = 0

            app = AppTest.from_file('app/dashboard.py', default_timeout=90)
            app.session_state['dashboard_open'] = True
            app.run()
            assert not app.exception, [error.message for error in app.exception]
            assert paths.WEATHER_FORECASTS == Path.cwd() / 'data' / 'weather_forecasts'
            assert forecast.WF is importlib.import_module('okno.weather_forecast')
            prediction = forecast.forecast('yakutsk_crossing', 2027, 65, 'close', use_weather=True)
            assert prediction['weather_model_applied']
            assert not app.sidebar.checkbox(key='weather_close_enabled_v3').value  # выключено по умолчанию
            app.sidebar.checkbox(key='weather_close_enabled_v3').check().run()
            assert not app.exception, [error.message for error in app.exception]
            print('OK: старые зависимости обновлены до импорта нового модуля')
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
