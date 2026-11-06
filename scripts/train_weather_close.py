"""Эксперимент закрытия: обучение на архиве GFS, проверка на следующих сезонах.

python scripts/train_weather_close.py
Существующие модели М1 и исторические отчёты не перезаписываются.
"""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from okno import m1
from okno.features import FEATURES
from okno.paths import MODELS, PROCESSED, REPORTS
from okno.report import md_table
from okno.weather_forecast import (API_URL, EXTRA_FEATURES, HORIZON, METADATA_PATH,
                                   MODEL_FEATURES, MODEL_PATH, training_rows)


def fit(train, features):
    model = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
                              thread_count=2, **m1.CB_PARAMS)
    model.fit(train[features], train.target_day - train.cal_target)
    return model


def rolling_backtest(frame):
    """Ни обучение, ни поправка интервала не видят текущий или будущие сезоны."""
    observed = frame[frame.weather_days.eq(HORIZON)]
    parts, scores = [], {"baseline": [], "weather": []}
    for year in sorted(observed.hydro_year.unique()):
        train = frame[frame.hydro_year < year]
        test = observed[observed.hydro_year.eq(year)].copy()
        seen = set(train.loc[train.weather_days.eq(HORIZON), "point_id"])
        test = test[test.point_id.isin(seen)]
        if test.empty:
            continue
        print(f"Проверка сезона {year}: обучение до {year - 1}, {len(test)} дат решения", flush=True)
        for mode, features in (("baseline", FEATURES["close"]), ("weather", MODEL_FEATURES)):
            model = fit(train, features)
            q = np.sort(np.asarray(model.predict(test[features])), axis=1) + test.cal_target.to_numpy()[:, None]
            raw_score = np.maximum(q[:, 0] - test.target_day.to_numpy(), test.target_day.to_numpy() - q[:, 2])
            past_scores = pd.Series(scores[mode], dtype=float)
            correction = max(0.0, m1.conformal_correction(past_scores)) if len(past_scores) else 0.0
            rows = test[["point_id", "hydro_year", "issue_day", "lead_days", "target_day", "cal_target"]].copy()
            rows["mode"] = mode
            rows["train_through"] = int(train.hydro_year.max())
            rows["calibration_n"] = len(past_scores)
            rows["correction_days"] = correction
            rows["raw_score"] = raw_score
            q[:, 0] -= correction
            q[:, 2] += correction
            q = np.maximum(q, test.issue_day.to_numpy()[:, None])
            rows[["q10", "q50", "q90"]] = q
            parts.append(rows)
            scores[mode].extend(raw_score.tolist())
    if not parts:
        raise ValueError("Нужны минимум два сезона с архивом для проверки на следующем сезоне")
    return pd.concat(parts, ignore_index=True)


def summary(backtest):
    rows = []
    for name, (low, high) in {**m1.LEAD_BUCKETS, "все": (0, 400)}.items():
        group = backtest[backtest.lead_days.between(low, high)]
        if group.empty:
            continue
        baseline = group[group["mode"].eq("baseline")]
        weather = group[group["mode"].eq("weather")]
        base_error = float((baseline.q50 - baseline.target_day).abs().mean())
        error = float((weather.q50 - weather.target_day).abs().mean())
        calibrated = weather[weather.calibration_n.gt(0)]
        coverage = ((calibrated.q10 <= calibrated.target_day) & (calibrated.q90 >= calibrated.target_day)).mean()
        rows.append({"горизонт_сут": name, "дат_решения": len(weather), "сезонов": weather.hydro_year.nunique(),
                     "MAE_календарь": (weather.cal_target - weather.target_day).abs().mean(),
                     "MAE_М1_без_погоды": base_error, "MAE_М1_с_погодой": error,
                     "выигрыш_к_М1_%": 100 * (1 - error / base_error) if base_error else np.nan,
                     "покрытие_после_калибровки": coverage,
                     "ширина_интервала_сут": (calibrated.q90 - calibrated.q10).mean()})
    return pd.DataFrame(rows).round(3)


def main():
    base_path = PROCESSED / "m1_dataset_close.csv"
    frame = training_rows(pd.read_csv(base_path))
    forecasts = frame[frame.weather_days.eq(HORIZON)]
    counts = forecasts.groupby("point_id").hydro_year.nunique()
    supported = counts[counts >= 3].index.tolist()
    if not supported:
        raise ValueError("Для эксперимента нужны хотя бы три сезона прогнозов с известными датами закрытия")
    backtest = rolling_backtest(frame)
    metrics = summary(backtest)
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    backtest.to_csv(REPORTS / "weather_close_backtest.csv", index=False)
    metrics.to_csv(REPORTS / "weather_close_metrics.csv", index=False)
    model = fit(frame, MODEL_FEATURES)
    temporary = MODEL_PATH.with_suffix(".cbm.tmp")
    model.save_model(str(temporary))
    temporary.replace(MODEL_PATH)
    correction = max(0.0, m1.conformal_correction(backtest.loc[backtest["mode"].eq("weather"), "raw_score"]))
    metadata = {"features": MODEL_FEATURES, "point_ids": supported,
                "trained_through_season": int(frame.hydro_year.max()), "correction_days": correction,
                "forecast_days": HORIZON, "experimental": True, "default_enabled": True,
                "source": API_URL, "baseline_data_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
                "archive_seasons": sorted(int(year) for year in forecasts.hydro_year.unique()),
                "validation_seasons": sorted(int(year) for year in backtest.hydro_year.unique())}
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["point_id", "hydro_year", "issue_day", *EXTRA_FEATURES]
    forecasts[columns].to_csv(REPORTS / "weather_close_features.csv", index=False)
    lines = ["# Закрытие с архивными метеопрогнозами: эксперимент", "",
             "Пилот: переправа Якутск — Нижний Бестях, цель — спутниковое начало вскрытия. "
             "Замеры гидропостов заканчиваются до архива GFS, поэтому вывод на зимник на Батагай не переносится.", "",
             "Источник: [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api), GFS. "
             "Для будущего дня D+k взят прогноз с offset k+1: исходная дата D−1, "
             "с консервативным запасом 6 часов после конца исходных суток на публикацию. "
             "Используются только полные 24 прогнозных часа. Подтверждённый горизонт — 6 суток, "
             "а не 14–28 суток будущей погоды; за 14/28 дней проверяется прогноз события.", "",
             f"Архив с целями: сезоны {metadata['archive_seasons']}. "
             f"Проверка: сезоны {metadata['validation_seasons']}.", "",
             "Каждый сезон проверяется моделью, обученной только на более ранних сезонах. "
             "Обычная М1 и М1 с погодой обучены на одинаковых строках с одинаковыми параметрами; "
             "сравнение выполняется на одних датах. Калибровка использует ошибки только прошлых проверочных сезонов. "
             "Первый проверочный сезон ещё без калибровки и не включён в оценку покрытия.", "",
             md_table(metrics), "",
             "Положительный выигрыш — снижение MAE относительно М1 без погоды, отрицательный — ухудшение. "
             "Мало независимых сезонов: обещание улучшения на 5–10% не подтверждается одной такой проверкой. "
             "Покрытие интервала оценивается по таблице: это экспериментальный диапазон, "
             "его нельзя считать проверенным 80%-интервалом. "
             "В интерфейсе ветки kirill эксперимент включён по умолчанию и может быть выключен переключателем. "
             "Исходные модели и их отчёты сохранены. "
             "При отсутствии полного прогноза, неподдерживаемой точке или дате из обучения применяется обычная М1.", "",
             "Воспроизведение: `python scripts/download_weather_forecasts.py --start-year 2021 --end-year 2024`, "
             "затем `python scripts/train_weather_close.py`. Обновление текущих прогнозов: "
             "`python scripts/download_weather_forecasts.py --live`.", ""]
    (REPORTS / "weather_close.md").write_text("\n".join(lines), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print("Сохранены экспериментальная модель и reports/weather_close.md", flush=True)


if __name__ == "__main__":
    main()
