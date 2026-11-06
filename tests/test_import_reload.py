"""Обновление работающего Streamlit-процесса с модулями предыдущей версии."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportReloadTests(unittest.TestCase):
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
            app.sidebar.checkbox(key='weather_close_enabled').check().run()
            assert not app.exception, [error.message for error in app.exception]
            print('OK: старые зависимости обновлены до импорта нового модуля')
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
