"""Временная честность архива, обучение и включение погодной модели закрытия."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okno import features as F, forecast as FC, weather_forecast as WF
from okno.ice import to_season_day
from scripts import train_weather_close as training


def hourly_payload(issue="2023-04-01"):
    hours = pd.date_range(pd.Timestamp(issue) + pd.Timedelta(days=1), periods=24 * WF.HORIZON, freq="h")
    return {"timezone": "GMT", "utc_offset_seconds": 0, "hourly": {
        "time": hours.strftime("%Y-%m-%dT%H:%M").tolist(),
        **{f"temperature_2m_previous_day{i}": [float(i)] * len(hours) for i in range(2, 8)},
    }}


class WeatherArchiveTests(unittest.TestCase):
    def setUp(self):
        self.rows = WF.from_hourly(hourly_payload(), "yakutsk_crossing", ["2023-04-01"])

    def test_each_future_day_uses_an_older_forecast_and_has_a_publication_margin(self):
        self.assertEqual(self.rows.temperature_2m_mean.tolist(), [2, 3, 4, 5, 6, 7])
        self.assertEqual(self.rows.offset_days.tolist(), [2, 3, 4, 5, 6, 7])
        self.assertTrue((self.rows.source_date == pd.Timestamp("2023-03-31")).all())
        self.assertTrue((self.rows.available_at < pd.Timestamp("2023-04-02", tz="UTC")).all())
        self.assertTrue(self.rows.model.eq("gfs_seamless").all())

    def test_future_or_late_source_data_are_rejected(self):
        for field, value in (("available_at", pd.Timestamp("2023-04-02T06:00Z")),
                             ("source_date", pd.Timestamp("2023-04-02")),
                             ("offset_days", 1), ("source", "ERA5 observations")):
            with self.subTest(field=field):
                bad = self.rows.copy()
                bad.loc[0, field] = value
                with self.assertRaises(ValueError):
                    WF.validate_snapshots(bad)

    def test_incomplete_day_is_not_filled_with_future_observations(self):
        payload = hourly_payload()
        payload["hourly"]["temperature_2m_previous_day2"][0] = None
        rows = WF.from_hourly(payload, "yakutsk_crossing", ["2023-04-01"])
        self.assertEqual(len(rows), 5)
        values = WF.weather_features(rows, "yakutsk_crossing", 2023,
                                     int(to_season_day("2023-04-01", 2023)), np.zeros(365))
        self.assertEqual(values["weather_days"], 0)
        self.assertTrue(np.isnan(values["weather_mean_6d"]))

    def test_wrong_timezone_or_duplicate_hours_are_rejected(self):
        payload = hourly_payload()
        payload["timezone"] = "Asia/Yakutsk"
        with self.assertRaises(ValueError):
            WF.from_hourly(payload, "yakutsk_crossing", ["2023-04-01"])
        payload = hourly_payload()
        payload["hourly"]["time"][1] = payload["hourly"]["time"][0]
        with self.assertRaises(ValueError):
            WF.from_hourly(payload, "yakutsk_crossing", ["2023-04-01"])

    def test_snapshot_must_match_the_exact_point_and_decision_date(self):
        day = int(to_season_day("2023-04-01", 2023))
        features = WF.weather_features(self.rows, "yakutsk_crossing", 2023, day, np.ones(365))
        self.assertEqual(features["weather_days"], 6)
        self.assertEqual(features["weather_mean_6d"], 4.5)
        self.assertEqual(features["weather_anomaly_6d"], 3.5)
        self.assertEqual(features["weather_tdd_6d"], 27)
        for point, issue_day in (("gauge_3414", day), ("yakutsk_crossing", day - 1)):
            self.assertEqual(WF.weather_features(self.rows, point, 2023, issue_day, np.ones(365))["weather_days"], 0)

    def test_duplicate_days_are_rejected(self):
        with self.assertRaises(ValueError):
            WF.validate_snapshots(pd.concat([self.rows, self.rows.iloc[:1]], ignore_index=True))

    def test_live_forecast_waits_until_its_declared_publication_time(self):
        day = int(to_season_day("2023-04-01", 2023))
        early = WF.weather_features(self.rows, "yakutsk_crossing", 2023, day, np.zeros(365),
                                    available_before=pd.Timestamp("2023-04-01T03:00Z"))
        ready = WF.weather_features(self.rows, "yakutsk_crossing", 2023, day, np.zeros(365),
                                    available_before=pd.Timestamp("2023-04-01T07:00Z"))
        self.assertEqual(early["weather_days"], 0)
        self.assertEqual(ready["weather_days"], 6)

    def test_future_observations_do_not_change_features_at_the_issue_date(self):
        day = 210
        weather = pd.DataFrame([np.linspace(-30, 10, 365)], index=[2023])
        snow = pd.DataFrame([np.ones(365)], index=[2023])
        first = F.season_features(weather, snow, 2023, day, np.zeros(365), 1.8, 60, 20)
        weather.iloc[0, day + 1:] = 50
        snow.iloc[0, day + 1:] = 1000
        second = F.season_features(weather, snow, 2023, day, np.zeros(365), 1.8, 60, 20)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], equal_nan=True, err_msg=key)


class WeatherTrainingTests(unittest.TestCase):
    def test_rolling_training_and_calibration_never_use_future_seasons(self):
        rows = []
        for year in range(2021, 2025):
            rows.append({**{name: 0.0 for name in WF.MODEL_FEATURES},
                         "point_id": "yakutsk_crossing", "hydro_year": year, "issue_day": 220,
                         "weather_days": 6, "lead_days": 30, "target_day": 245 + year - 2021,
                         "cal_target": 245})
        frame = pd.DataFrame(rows)

        class DummyModel:
            def __init__(self, train):
                self.offset = float((train.target_day - train.cal_target).mean())

            def predict(self, data):
                return np.tile([self.offset - 1, self.offset, self.offset + 1], (len(data), 1))

        with patch.object(training, "fit", side_effect=lambda train, features: DummyModel(train)):
            before = training.rolling_backtest(frame)
            changed = frame.copy()
            changed.loc[changed.hydro_year.eq(2024), "target_day"] = 999
            after = training.rolling_backtest(changed)
        self.assertTrue((before.train_through < before.hydro_year).all())
        self.assertTrue(before.loc[before.hydro_year.eq(2022), "calibration_n"].eq(0).all())
        self.assertTrue(before.loc[before.hydro_year.eq(2023), "calibration_n"].gt(0).all())
        columns = ["q10", "q50", "q90", "correction_days"]
        # Даже прогноз 2024 года не должен измениться от подмены его неизвестного факта.
        pd.testing.assert_frame_equal(before[columns], after[columns])


class WeatherPredictionTests(unittest.TestCase):
    def test_missing_model_and_missing_forecast_use_the_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(WF, "MODEL_PATH", Path(directory) / "missing.cbm"):
                result = WF.predict_close("yakutsk_crossing", 2027, 65, {}, np.zeros(365))
                self.assertNotIn("quantiles", result)
            with patch.object(WF, "WEATHER_FORECASTS", Path(directory)):
                result = WF.predict_close("yakutsk_crossing", 2027, 65, {}, np.zeros(365))
                self.assertNotIn("quantiles", result)

    def test_model_cannot_be_applied_to_its_training_seasons(self):
        result = WF.predict_close("yakutsk_crossing", 2024, 220, {}, np.zeros(365))
        self.assertNotIn("quantiles", result)
        self.assertIn("обучающие сезоны", result["message"])

    def test_live_pilot_uses_six_forecast_days(self):
        day = int(to_season_day("2026-11-05", 2027))
        result = FC.forecast("yakutsk_crossing", 2027, day, "close", use_weather=True)
        self.assertTrue(result["weather_model_applied"])
        self.assertEqual(result["forecast_days_used"], 6)
        self.assertLessEqual(result["q10"], result["q50"])
        self.assertLessEqual(result["q50"], result["q90"])

    def test_opening_and_unsupported_points_keep_their_original_predictions(self):
        for kind in ("open", "close"):
            with self.subTest(kind=kind):
                ordinary = FC.forecast("gauge_3414", 2027, 65, kind)
                experiment = FC.forecast("gauge_3414", 2027, 65, kind, use_weather=True)
                self.assertFalse(experiment["weather_model_applied"])
                for name in ("q10", "q50", "q90", "physics"):
                    self.assertEqual(ordinary[name], experiment[name])


class WeatherDashboardTests(unittest.TestCase):
    def test_toggle_is_optional_and_changes_only_the_supported_closing_model(self):
        app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90).run()
        app.button(key="open_dashboard").click().run()
        self.assertEqual([error.message for error in app.exception], [])
        # эксперимент выключен по умолчанию (выигрыш 1,3%, на 7 и 14 днях — без улучшения); включаем вручную
        self.assertFalse(app.sidebar.checkbox(key="weather_close_enabled_v3").value)
        app.sidebar.checkbox(key="weather_close_enabled_v3").check().run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertTrue(any("нет обучающего архива" in caption.value for caption in app.caption))
        app.sidebar.selectbox[2].select("yakutsk_crossing").run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertTrue(any("Экспериментальный диапазон" in item.value for item in app.markdown))
        self.assertTrue(any("архивом GFS" in caption.value for caption in app.caption))
        app.sidebar.checkbox(key="weather_close_enabled_v3").uncheck().run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertFalse(any("Экспериментальный диапазон" in item.value for item in app.markdown))
        app.run()
        self.assertFalse(app.sidebar.checkbox(key="weather_close_enabled_v3").value)
        app.sidebar.checkbox(key="weather_close_enabled_v3").check().run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertTrue(any("Экспериментальный диапазон" in item.value for item in app.markdown))
        self.assertFalse(app.get("deck_gl_json_chart"))  # Схемы путей не возвращаются.


if __name__ == "__main__":
    unittest.main()
