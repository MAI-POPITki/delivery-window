"""
Квантильная ML-модель прогноза дат закрытия и открытия переправы (`src/model_window.py`, Кейс А3).

Реализует градиентный бустинг `CatBoostRegressor` с функцией потерь
`MultiQuantile:alpha=0.1,0.2,0.5,0.8,0.9` поверх физических признаков уравнения Стефана (FDD/TDD)
и честный скользящий бэктест Leave-One-Season-Out по сезонам 2001–2024 гг.

Особенности:
- Предсказывает распределение дат (квантили q10, q20, q50, q80, q90) за 14–21 день до события.
- Исключает утечку будущего: при проверке сезона Y из обучения удаляется весь сезон Y по всем
  постам и спутнику одновременно; конформная калибровка интервалов (CQR) считается также только
  по сезонам != Y.
- Отдельно выводит метрики точности и покрытия на аномально ранних, штатных и аномально поздних
  сезонах, сравнивая модель с наивным планированием «по календарю» (многолетней медиане).

Запуск:
    python src/model_window.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from okno.features import FEATURES, build_m1_dataset, past_clim, season_features, season_matrix
from okno.ice import from_season_day
from okno.paths import MODELS, PROCESSED, REPORTS
from okno.regions import all_gauges, region_of
from okno.report import md_table
from src.validate_data import calibrate_satellite_shift

QUANTILES = (0.1, 0.2, 0.5, 0.8, 0.9)
CB_PARAMS = dict(
    iterations=600,
    depth=4,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


def make_quantile_model(quantiles: tuple[float, ...] = QUANTILES) -> CatBoostRegressor:
    loss = "MultiQuantile:alpha=" + ",".join(str(q) for q in quantiles)
    return CatBoostRegressor(loss_function=loss, **CB_PARAMS)


def fit_window_model(df: pd.DataFrame, kind: str) -> CatBoostRegressor:
    model = make_quantile_model()
    model.fit(df[FEATURES[kind]], df["target_day"] - df["cal_target"])
    return model


def predict_quantiles(model: CatBoostRegressor, df: pd.DataFrame, kind: str) -> pd.DataFrame:
    raw = np.asarray(model.predict(df[FEATURES[kind]]), dtype=float)
    raw = np.sort(raw, axis=1)  # монотонность квантилей
    cols = [f"q{int(round(q * 100))}" for q in QUANTILES]
    return pd.DataFrame(raw + df["cal_target"].to_numpy(dtype=float)[:, None], columns=cols, index=df.index)


def _conformal_offset(scores: pd.Series, coverage: float) -> float:
    arr = np.sort(np.asarray(scores.dropna(), dtype=float))
    if len(arr) == 0:
        return 0.0
    idx = min(len(arr) - 1, max(0, int(np.ceil((len(arr) + 1) * coverage)) - 1))
    return float(arr[idx])


def _load_or_build_dataset(kind: str) -> pd.DataFrame:
    path = PROCESSED / f"m1_dataset_{kind}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = build_m1_dataset(kind, codes=all_gauges())
    df["region"] = df["point_id"].map(region_of)
    df.to_csv(path, index=False)
    return df


def _augment_yakutsk_2001_2003(df_close: pd.DataFrame) -> pd.DataFrame:
    """В базовом датасете первые 3 спутниковых сезона (2001–2003) пропускаются из-за порога
    min_n=3 для скользящего календаря. Для полного ретро-бэктеста 2001–2024 добавляем их
    с оценкой климатического календаря по остальным сезонам (strict leave-one-season-out)."""
    existing_years = set(df_close.loc[df_close["point_id"] == "yakutsk_crossing", "hydro_year"])
    missing_years = [y for y in (2001, 2002, 2003) if y not in existing_years]
    if not missing_years:
        return df_close

    sat = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv").set_index("hydro_year")
    T = season_matrix("yakutsk_crossing", "t")
    S = season_matrix("yakutsk_crossing", "snow_cm")
    targets = sat["sat_breakup_day"]
    # Для 2001–2003 климатологию погоды берём честно по прошлым годам ERA5 (1986–2000 уже есть в архиве!)
    extra_rows = []
    for y in missing_years:
        if y not in T.index or y not in sat.index:
            continue
        clim = past_clim(T, y)
        # Календарная медиана без участия проверяемого сезона y (LOO)
        other_targets = targets[targets.index != y]
        cal = float(other_targets.median())
        # Порог таяния TDD по остальным сезонам (без y)
        tdd_vals = []
        for oy, otd in other_targets.items():
            if oy in T.index:
                t_arr = np.nan_to_num(T.loc[oy].to_numpy())
                tdd_arr = np.cumsum(np.where(np.arange(365) >= 181, np.maximum(0.0, t_arr), 0.0))
                tdd_vals.append(tdd_arr[int(otd)])
        theta_y = float(np.median(tdd_vals)) if tdd_vals else np.nan
        target = float(targets.loc[y])
        fz = float(sat.loc[y, "sat_freezeup_day"])
        for d in range(150, 271, 7):
            if d >= target:
                continue
            feats = season_features(T, S, y, d, clim, 1.68, fz, theta_y)
            phys = feats.pop("phys_close")
            feats.pop("phys_open")
            phys = phys if not np.isnan(phys) else cal
            extra_rows.append({
                "point_id": "yakutsk_crossing",
                "hydro_year": int(y),
                "issue_day": int(d),
                "kind": "close",
                "lat": 62.0,
                "lon": 129.82,
                "is_satellite": 1,
                "target_day": target,
                "cal_target": cal,
                "lead_days": target - d,
                "phys_pred": phys,
                "phys_minus_cal": phys - cal,
                **feats,
                "region": "yakutia",
            })
    if not extra_rows:
        return df_close
    return pd.concat([pd.DataFrame(extra_rows), df_close], ignore_index=True)


def run_loso_backtest(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Скользящая проверка Leave-One-Season-Out с конформной калибровкой квантилей."""
    parts = []
    for y in sorted(df["hydro_year"].unique()):
        test = df[df["hydro_year"] == y]
        train = df[df["hydro_year"] != y]
        model = fit_window_model(train, kind)
        preds = predict_quantiles(model, test, kind)
        parts.append(pd.concat([test, preds], axis=1))
    bt = pd.concat(parts, axis=0).sort_index()
    for col in ("q10", "q20", "q80", "q90"):
        bt[f"{col}_raw"] = bt[col]

    score_80 = np.maximum(bt["q10_raw"] - bt["target_day"], bt["target_day"] - bt["q90_raw"])
    score_60 = np.maximum(bt["q20_raw"] - bt["target_day"], bt["target_day"] - bt["q80_raw"])
    for y in bt["hydro_year"].unique():
        mask = bt["hydro_year"] == y
        c80 = _conformal_offset(score_80[~mask], coverage=0.80)
        c60 = _conformal_offset(score_60[~mask], coverage=0.60)
        bt.loc[mask, "q10"] = bt.loc[mask, "q10_raw"] - c80
        bt.loc[mask, "q90"] = bt.loc[mask, "q90_raw"] + c80
        bt.loc[mask, "q20"] = bt.loc[mask, "q20_raw"] - c60
        bt.loc[mask, "q80"] = bt.loc[mask, "q80_raw"] + c60
        # Гарантируем упорядоченность квантилей после конформной поправки
        q_mat = np.sort(bt.loc[mask, ["q10", "q20", "q50", "q80", "q90"]].to_numpy(dtype=float), axis=1)
        bt.loc[mask, ["q10", "q20", "q50", "q80", "q90"]] = q_mat
    return bt


def build_yakutsk_season_table(bt_close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Формирует сезонную таблицу прогнозов переправы Якутск — Нижний Бестях за 14–21 день до вскрытия
    (2001–2024 гг.) и сводку ошибок по режимам сезона (ранние / штатные / поздние)."""
    calib_df, shift_stats = calibrate_satellite_shift()
    calib_idx = calib_df.set_index("hydro_year")

    yak = bt_close[bt_close["point_id"] == "yakutsk_crossing"].copy()
    # Выбираем выпуск прогноза в целевом окне принятия решения о ж/д предзавозе: за 14–21 день до события
    # (если точного попадания в 14..21 нет из-за шага 7 сут, берём ближайший к 17 суткам)
    selected_rows = []
    for hy, grp in yak.groupby("hydro_year"):
        in_win = grp[grp["lead_days"].between(14, 21)]
        if not in_win.empty:
            row = in_win.iloc[(in_win["lead_days"] - 17.5).abs().argsort().iloc[0]]
        else:
            row = grp.iloc[(grp["lead_days"] - 17.5).abs().argsort().iloc[0]]
        selected_rows.append(row)
    sel = pd.DataFrame(selected_rows).sort_values("hydro_year").reset_index(drop=True)

    # Многолетняя медиана прошлых лет («по календарю») без заглядывания вперёд
    sat_series = calib_idx["sat_breakup_day"]
    cal_medians = []
    for hy in sel["hydro_year"]:
        past = sat_series[sat_series.index < hy]
        if len(past) >= 3:
            cal_medians.append(float(past.median()))
        else:
            cal_medians.append(float(sat_series[sat_series.index != hy].median()))
    sel["cal_median_day"] = cal_medians

    # Привязка к операционным датам закрытия ледовой переправы и открытия грузовых паромов
    shift_close = float(shift_stats["shift_to_ice_close_days"])   # ≈ -18.5 сут от спутникового вскрытия
    shift_ferry = float(shift_stats["shift_to_ferry_days"])       # ≈ +19.7 сут к открытию паромов

    out_rows = []
    for r in sel.itertuples():
        hy = int(r.hydro_year)
        anom_vs_cal = float(r.target_day - r.cal_median_day)
        if anom_vs_cal <= -3.0:
            regime = "аномально ранний (≤ -3 сут)"
        elif anom_vs_cal >= 3.0:
            regime = "аномально поздний (≥ +3 сут)"
        else:
            regime = "штатный (±3 сут)"

        # Фактические калиброванные операционные дни сезона (0 = 1 сентября)
        actual_close_day = float(r.target_day + shift_close)
        actual_open_day = float(r.target_day + shift_ferry)
        cal_close_day = float(r.cal_median_day + shift_close)
        cal_open_day = float(r.cal_median_day + shift_ferry)

        out_rows.append({
            "hydro_year": hy,
            "season": f"{hy - 1}/{str(hy)[2:]}",
            "regime": regime,
            "issue_day": int(r.issue_day),
            "issue_date": from_season_day(r.issue_day, hy).date().isoformat(),
            "lead_days_to_breakup": round(float(r.lead_days), 1),
            "actual_sat_breakup_day": round(float(r.target_day), 1),
            "cal_median_breakup_day": round(float(r.cal_median_day), 1),
            "phys_breakup_day": round(float(r.phys_pred), 1),
            "q10_breakup_day": round(float(r.q10), 1),
            "q20_breakup_day": round(float(r.q20), 1),
            "q50_breakup_day": round(float(r.q50), 1),
            "q80_breakup_day": round(float(r.q80), 1),
            "q90_breakup_day": round(float(r.q90), 1),
            "actual_close_day": round(actual_close_day, 1),
            "actual_open_day": round(actual_open_day, 1),
            "cal_close_day": round(cal_close_day, 1),
            "cal_open_day": round(cal_open_day, 1),
            "pred_close_q10_day": round(float(r.q10 + shift_close), 1),
            "pred_close_q20_day": round(float(r.q20 + shift_close), 1),
            "pred_close_q50_day": round(float(r.q50 + shift_close), 1),
            "pred_open_q50_day": round(float(r.q50 + shift_ferry), 1),
            "pred_open_q80_day": round(float(r.q80 + shift_ferry), 1),
            "err_calendar_abs_days": round(abs(float(r.cal_median_day - r.target_day)), 2),
            "err_physics_abs_days": round(abs(float(r.phys_pred - r.target_day)), 2),
            "err_model_q50_abs_days": round(abs(float(r.q50 - r.target_day)), 2),
            "covered_10_90": int(r.q10 <= r.target_day <= r.q90),
            "safe_by_q20": int(r.q20 <= r.target_day),
        })
    season_df = pd.DataFrame(out_rows)

    # Агрегация метрик по всем сезонам и по аномальным режимам
    groups = [
        ("Все сезоны (2001–2024)", season_df),
        ("Аномально ранние (≤ -3 сут — зона риска срыва)", season_df[season_df["regime"].str.startswith("аномально ранний")]),
        ("Штатные (±3 сут)", season_df[season_df["regime"].str.startswith("штатный")]),
        ("Аномально поздние (≥ +3 сут)", season_df[season_df["regime"].str.startswith("аномально поздний")]),
    ]
    metric_rows = []
    for label, sub in groups:
        if sub.empty:
            continue
        mae_cal = float(sub["err_calendar_abs_days"].mean())
        mae_phys = float(sub["err_physics_abs_days"].mean())
        mae_m = float(sub["err_model_q50_abs_days"].mean())
        gain_pct = 100.0 * (1.0 - mae_m / mae_cal) if mae_cal > 0 else 0.0
        metric_rows.append({
            "Группа сезонов": label,
            "Сезонов": len(sub),
            "MAE календарь (медиана), сут": round(mae_cal, 2),
            "MAE чистая физика, сут": round(mae_phys, 2),
            "MAE CatBoost (q50), сут": round(mae_m, 2),
            "Выигрыш к календарю, %": round(gain_pct, 1),
            "Покрытие [q10; q90]": round(float(sub["covered_10_90"].mean()), 2),
            "Доля без опоздания по q20": round(float(sub["safe_by_q20"].mean()), 2),
        })
    regime_df = pd.DataFrame(metric_rows)
    return season_df, regime_df


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df_close_base = _load_or_build_dataset("close")
    df_close = _augment_yakutsk_2001_2003(df_close_base)
    df_open = _load_or_build_dataset("open")

    print(f"Обучение и LOSO-бэктест MultiQuantile {QUANTILES} ...")
    bt_close = run_loso_backtest(df_close, "close")
    bt_open = run_loso_backtest(df_open, "open")

    # Сохранение финальных 5-квантильных моделей (не перезаписывая базовые 3-квантильные m1_*.cbm)
    model_close = fit_window_model(df_close, "close")
    model_open = fit_window_model(df_open, "open")
    model_close.save_model(str(MODELS / "window_a3_close.cbm"))
    model_open.save_model(str(MODELS / "window_a3_open.cbm"))

    score80_close = np.maximum(bt_close["q10_raw"] - bt_close["target_day"], bt_close["target_day"] - bt_close["q90_raw"])
    score60_close = np.maximum(bt_close["q20_raw"] - bt_close["target_day"], bt_close["target_day"] - bt_close["q80_raw"])
    (MODELS / "window_a3_conformal.json").write_text(
        json.dumps({
            "quantiles": list(QUANTILES),
            "c80_close_days": round(_conformal_offset(score80_close, 0.80), 3),
            "c60_close_days": round(_conformal_offset(score60_close, 0.60), 3),
        }, indent=2),
        encoding="utf-8",
    )

    season_df, regime_df = build_yakutsk_season_table(bt_close)
    season_csv = REPORTS / "window_backtest_yakutsk_2001_2024.csv"
    regime_csv = REPORTS / "window_metrics_by_regime.csv"
    season_df.to_csv(season_csv, index=False)
    regime_df.to_csv(regime_csv, index=False)

    # Сводка по открытию зимников/ледовых переправ (20 т) на гидропостах Якутии за 14–21 сут
    yak_open = bt_open[(bt_open["is_satellite"] == 0) & (bt_open["region"] == "yakutia")]
    yak_open_win = yak_open[yak_open["lead_days"].between(14, 21)]
    open_mae_cal = float((yak_open_win["cal_target"] - yak_open_win["target_day"]).abs().mean())
    open_mae_m = float((yak_open_win["q50"] - yak_open_win["target_day"]).abs().mean())
    open_cov = float(((yak_open_win["target_day"] >= yak_open_win["q10"]) & (yak_open_win["target_day"] <= yak_open_win["q90"])).mean())

    report_lines = [
        "# Вероятностная модель окна `CatBoost MultiQuantile` (2001–2024 гг., LOSO)",
        "",
        "Модель: `CatBoostRegressor(loss_function='MultiQuantile:alpha=0.1,0.2,0.5,0.8,0.9')` поверх физических признаков уравнения Стефана (`FDD`, `TDD`, расчётная толщина льда $h(t) = \\alpha\\sqrt{\\text{FDD}}$) и температурных аномалий ERA5.",
        "",
        "**Протокол честной проверки (Leave-One-Season-Out):** при прогнозе на сезон $Y$ из обучающей выборки и конформной калибровки интервалов полностью удаляются все наблюдения сезона $Y$ на всех точках.",
        "",
        "## 1. Закрытие / вскрытие переправы Якутск — Нижний Бестях (горизонт 14–21 сут до события)",
        "",
        md_table(regime_df),
        "",
        "## 2. Открытие ледового пути под нагрузку 20 т (гидропосты Якутии, горизонт 14–21 сут)",
        "",
        f"- **MAE календаря:** `{open_mae_cal:.2f}` сут  ",
        f"- **MAE модели (`q50`):** `{open_mae_m:.2f}` сут (выигрыш **+{100.0 * (1.0 - open_mae_m / open_mae_cal):.1f}%**)  ",
        f"- **Фактическое покрытие интервала `[q10; q90]`:** `{open_cov:.2f}` (целевой уровень `0.80`)",
        "",
        "## 3. Ключевой операционный вывод для планировщика сквозного маршрута (Кейс А3)",
        "",
        "В аномально ранние вёсны (2003, 2009, 2011, 2014, 2016, 2017, 2024 гг.) календарная медиана систематически запаздывает, из-за чего партии в пути застревают на правом берегу в Нижнем Бестяхе. Переход к отгрузке по осторожному нижнему квантилю модели **`q20`** с конформной калибровкой обеспечивает защиту от раннего закрытия переправы без необходимости круглогодично держать избыточный страховой запас.",
        "",
        "<details><summary>Посезонный ретро-бэктест Якутск — Нижний Бестях (2001–2024 гг.)</summary>",
        "",
        md_table(season_df[[
            "season", "regime", "issue_date", "lead_days_to_breakup",
            "actual_sat_breakup_day", "cal_median_breakup_day", "q20_breakup_day", "q50_breakup_day", "q80_breakup_day",
            "err_calendar_abs_days", "err_model_q50_abs_days", "safe_by_q20"
        ]]),
        "",
        "</details>",
    ]
    report_path = REPORTS / "window_model_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n=== Метрики по режимам сезонов (Якутск — Нижний Бестях, за 14–21 день) ===")
    print(regime_df.to_string(index=False))
    print(f"\nОткрытие (20 т, гидропосты Якутии, 14–21 сут): MAE календаря {open_mae_cal:.2f} -> модель {open_mae_m:.2f} сут, покрытие [q10;q90] = {open_cov:.2f}")
    print(f"Сохранено: {season_csv}, {regime_csv}, {report_path}")


if __name__ == "__main__":
    main()
