"""
Модуль авто-проверки данных и калибровки систематического сдвига спутника (Кейс А3).

Запуск:
    python src/validate_data.py

Что делает:
1. Проверяет полноту и физическую корректность рядов ERA5 (отсутствие пропусков, дублей,
   контроль климатической нормы января/июля у переправы Якутск — Нижний Бестях).
2. Переводит спутниковый ряд вскрытия и ледостава Лены (Qiu et al., 2026, 2001–2024)
   в межгодовые аномалии Δt_sat и калибрует систематический сдвиг (2–3 недели между
   началом поверхностного таяния/шуги на снимке и фактическим ледоходом / открытием паромов
   по сводкам ГУ МЧС и Минтранса РС(Я)), сохраняя `data/processed/yakutsk_sat_calibrated.csv`.
3. Сверяет калиброванный спутник с замерами гидропостов Росгидромета на общих годах
   2001–2012 (контрольный порог MAE ≤ 11 суток).
4. Проверяет сходимость формулы Стефана с реальными замерами толщины льда на гидропостах.
5. Формирует сводный протокол `docs/validation_report.md` со статусами `PASS / FAIL`.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from okno.calib import calibrate_all
from okno.ice import from_season_day, to_season_day
from okno.paths import DOCS, PROCESSED, REPORTS, ROSSTAT_PRICES, WEATHER
from okno.report import md_table


def _record(checks: list[dict], check_id: str, title: str, passed: bool, metric: str, threshold: str, source: str):
    status = "PASS" if passed else "FAIL"
    checks.append({
        "Код": check_id,
        "Проверка": title,
        "Статус": status,
        "Факт": metric,
        "Критерий": threshold,
        "Источник": source,
    })
    print(f"[{status}] {check_id} | {title}: {metric} (критерий: {threshold})")


def calibrate_satellite_shift() -> tuple[pd.DataFrame, dict]:
    """Переводит спутниковые даты (Qiu et al., 2026) в аномалии Δt_sat и привязывает
    их к абсолютным датам закрытия ледовой переправы, ледохода и открытия паромов у Якутска."""
    sat = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv").sort_values("hydro_year").reset_index(drop=True)
    mchs_path = ROSSTAT_PRICES / "yakutsk_mchs_crossing_dates.csv"
    if not mchs_path.exists():
        from scripts.download_data import save_rosstat_and_tariffs
        save_rosstat_and_tariffs()
    mchs = pd.read_csv(mchs_path)
    for col in ("ice_road_close_date", "lena_breakup_yakutsk_date", "ferry_open_date"):
        day_col = col.replace("_date", "_day")
        mchs[day_col] = [to_season_day(d, int(hy)) for d, hy in zip(mchs[col], mchs["hydro_year"])]

    mean_sat_breakup = float(sat["sat_breakup_day"].mean())
    mean_sat_freezeup = float(sat["sat_freezeup_day"].mean())
    sat["sat_breakup_anom_days"] = (sat["sat_breakup_day"] - mean_sat_breakup).round(2)
    sat["sat_freezeup_anom_days"] = (sat["sat_freezeup_day"] - mean_sat_freezeup).round(2)

    merged = sat.merge(mchs, on="hydro_year", how="inner")
    # Систематические сдвиги относительно спутниковой даты вскрытия (поверхностное таяние / шуга)
    shift_to_breakup = float((merged["lena_breakup_yakutsk_day"] - merged["sat_breakup_day"]).mean())
    shift_to_ferry = float((merged["ferry_open_day"] - merged["sat_breakup_day"]).mean())
    shift_to_ice_close = float((merged["ice_road_close_day"] - merged["sat_breakup_day"]).mean())

    # MAE после устранения систематического сдвига на референсных сезонах МЧС
    mae_mchs_breakup = float(
        ((merged["sat_breakup_day"] + shift_to_breakup) - merged["lena_breakup_yakutsk_day"]).abs().mean()
    )

    # Привязка многолетнего среднего + межгодовой аномалии спутника Δt_sat
    anchor_close = float(merged["ice_road_close_day"].mean() - merged["sat_breakup_anom_days"].mean())
    anchor_breakup = float(merged["lena_breakup_yakutsk_day"].mean() - merged["sat_breakup_anom_days"].mean())
    anchor_ferry = float(merged["ferry_open_day"].mean() - merged["sat_breakup_anom_days"].mean())

    out = sat[["point_id", "hydro_year", "sat_freezeup_day", "sat_freezeup_date",
               "sat_breakup_day", "sat_breakup_date", "sat_breakup_anom_days", "sat_freezeup_anom_days"]].copy()
    out["calib_ice_close_day"] = (anchor_close + out["sat_breakup_anom_days"]).round(1)
    out["calib_breakup_day"] = (anchor_breakup + out["sat_breakup_anom_days"]).round(1)
    out["calib_ferry_open_day"] = (anchor_ferry + out["sat_breakup_anom_days"]).round(1)

    for col in ("calib_ice_close", "calib_breakup", "calib_ferry_open"):
        out[f"{col}_date"] = [
            from_season_day(int(round(d)), int(hy)).date().isoformat()
            for d, hy in zip(out[f"{col}_day"], out["hydro_year"])
        ]
    # Фактические даты МЧС подставляются напрямую там, где опубликованы приказы, с сохранением калиброванного ряда
    mchs_map = mchs.set_index("hydro_year")
    out["mchs_ice_close_date"] = out["hydro_year"].map(mchs_map["ice_road_close_date"])
    out["mchs_breakup_date"] = out["hydro_year"].map(mchs_map["lena_breakup_yakutsk_date"])
    out["mchs_ferry_open_date"] = out["hydro_year"].map(mchs_map["ferry_open_date"])
    actual_close_day = out["hydro_year"].map(mchs_map["ice_road_close_day"])
    actual_ferry_day = out["hydro_year"].map(mchs_map["ferry_open_day"])
    out["spring_rasputitsa_days"] = np.where(
        actual_ferry_day.notna() & actual_close_day.notna(),
        (actual_ferry_day - actual_close_day).round(1),
        (out["calib_ferry_open_day"] - out["calib_ice_close_day"]).round(1),
    )
    out["source_type"] = np.where(
        out["mchs_breakup_date"].notna(),
        "спутник + факт МЧС/Минтранс РС(Я)",
        "спутник (аномалия Δt_sat) + калибровка сдвига",
    )
    out["source"] = "Qiu et al. 2026 (doi:10.5281/zenodo.17054619) + сводки ГУ МЧС / Минтранса РС(Я)"
    out.to_csv(PROCESSED / "yakutsk_sat_calibrated.csv", index=False)

    row_2024 = merged[merged["hydro_year"] == 2024].iloc[0]
    stats = {
        "n_sat_years": len(sat),
        "n_mchs_years": len(merged),
        "mean_sat_breakup_day": round(mean_sat_breakup, 1),
        "shift_to_breakup_days": round(shift_to_breakup, 1),
        "shift_to_ferry_days": round(shift_to_ferry, 1),
        "shift_to_ice_close_days": round(shift_to_ice_close, 1),
        "mae_mchs_breakup_days": round(mae_mchs_breakup, 2),
        "sat_2024_date": str(row_2024["sat_breakup_date"]),
        "mchs_2024_breakup_date": str(row_2024["lena_breakup_yakutsk_date"]),
        "mchs_2024_ferry_date": str(row_2024["ferry_open_date"]),
        "shift_2024_days": int(round(row_2024["lena_breakup_yakutsk_day"] - row_2024["sat_breakup_day"])),
    }
    return out, stats


def run_validation() -> tuple[pd.DataFrame, dict]:
    checks: list[dict] = []

    # 1. Проверка полноты метеорядов ERA5 по всем точкам и гидропостам
    weather_files = sorted(WEATHER.glob("*.csv")) + sorted((WEATHER / "gauges").glob("*.csv"))
    today = pd.Timestamp(dt.date.today())
    total_gaps = 0
    total_dups = 0
    total_nans = 0
    total_bad_temp = 0
    for wf in weather_files:
        w = pd.read_csv(wf, parse_dates=["date"])
        total_dups += int(w["date"].duplicated().sum())
        full_idx = pd.date_range(w["date"].min(), w["date"].max(), freq="D")
        total_gaps += int(len(full_idx) - w["date"].nunique())
        total_nans += int(w["temperature_2m_mean"].isna().sum())
        total_bad_temp += int(((w["temperature_2m_mean"] < -65) | (w["temperature_2m_mean"] > 40) | (w["date"] > today)).sum())

    yak_w = pd.read_csv(WEATHER / "yakutsk_crossing.csv", parse_dates=["date"])
    _record(
        checks,
        "V1_ERA5_COMPLETE",
        "Полнота суточных рядов ERA5 (переправа, райцентры, гидропосты)",
        total_gaps == 0 and total_dups == 0 and total_nans == 0 and total_bad_temp == 0 and len(yak_w) >= 17000,
        f"точек {len(weather_files)}, переправа {len(yak_w)} сут ({yak_w['date'].min():%Y-%m-%d}…{yak_w['date'].max():%Y-%m-%d}), дыр {total_gaps}, дублей {total_dups}",
        "0 пропусков, 0 дублей, T ∈ [-65; +40] °C",
        "ERA5 (Open-Meteo Archive API)",
    )

    # 2. Контроль климатической нормы января и июля у Якутска
    jan_all = float(yak_w[yak_w["date"].dt.month == 1]["temperature_2m_mean"].mean())
    yak_norm = yak_w[(yak_w["date"].dt.year >= 1991) & (yak_w["date"].dt.year <= 2020)]
    jan_91_20 = float(yak_norm[yak_norm["date"].dt.month == 1]["temperature_2m_mean"].mean())
    jul_91_20 = float(yak_norm[yak_norm["date"].dt.month == 7]["temperature_2m_mean"].mean())
    _record(
        checks,
        "V2_CLIMATE_NORM",
        "Сверка температуры ERA5 у Якутска с климатической нормой Росгидромета",
        abs(jan_all + 38.0) <= 3.0 and abs(jul_91_20 - 19.5) <= 3.0,
        f"январь {jan_all:.1f} °C (1979–2026) / {jan_91_20:.1f} °C (1991–2020), июль +{jul_91_20:.1f} °C",
        "январь -38 ± 3 °C, июль +19.5 ± 3 °C",
        "Климатическая норма ВМО / Росгидромет",
    )

    # 3. Проверка архива гидропостов Росгидромета (Arctic Data Center, 1970–2012)
    seasons = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    gauges = pd.read_csv(PROCESSED / "gauges.csv")
    thick = pd.read_csv(PROCESSED / "hydro_ice_thickness.csv")
    yak_codes = set(gauges.loc[gauges["yakutia"], "gauge_code"])
    yak_thick = thick[thick["gauge_code"].isin(yak_codes)]
    bad_share = float((~seasons["valid"]).mean())
    max_clean_cm = float(yak_thick["ice_cm"].max())
    _record(
        checks,
        "V3_HYDROPOSTS_QC",
        "Качество замеров толщины льда и дат фаз на гидропостах Росгидромета",
        bad_share < 0.05 and max_clean_cm <= 400.0 and len(yak_codes) >= 13,
        f"постов {len(gauges)} (Якутия: {len(yak_codes)}), сезонов {len(seasons)}, брак {bad_share:.1%}, макс. лёд {max_clean_cm:.0f} см",
        "брак < 5%, толщина после фильтра выбросов ≤ 400 см",
        "Shiklomanov et al., doi:10.18739/A22Z12Q02",
    )

    # 4. Калибровка систематического сдвига спутника (2–3 недели к ледоходу у Якутска)
    calib_df, shift_stats = calibrate_satellite_shift()
    shift_ok = 10.0 <= shift_stats["shift_to_breakup_days"] <= 25.0 and shift_stats["mae_mchs_breakup_days"] <= 4.0
    _record(
        checks,
        "V4_SAT_SHIFT_CALIB",
        "Калибровка систематического сдвига спутника к ледоходу у Якутска (МЧС)",
        shift_ok,
        f"сдвиг к ледоходу +{shift_stats['shift_to_breakup_days']} сут (2024: {shift_stats['sat_2024_date']} → {shift_stats['mchs_2024_breakup_date']}, +{shift_stats['shift_2024_days']} сут), MAE после калибровки {shift_stats['mae_mchs_breakup_days']} сут",
        "сдвиг +10…+25 сут (2–3 нед.), остаточный MAE ≤ 4 сут",
        "Qiu et al. 2026 + ГУ МЧС / Минтранс РС(Я)",
    )

    # 5. Сверка аномалий калиброванного спутника с гидропостами на общих годах 2001–2012
    sat = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv")
    ref_botoma = seasons[(seasons["gauge_code"] == 3210) & seasons["valid"]][
        ["hydro_year", "freezeup_day", "melt_start_day", "drift_start_day"]
    ]
    overlap = sat.merge(ref_botoma, on="hydro_year")
    sat_anom = overlap["sat_breakup_day"] - overlap["sat_breakup_day"].mean()
    gauge_anom = overlap["drift_start_day"] - overlap["drift_start_day"].mean()
    mae_overlap = float((sat_anom - gauge_anom).abs().mean())

    # Также средняя MAE аномалий по всем валидным постам бассейна Лены в Якутии (2001–2012)
    lena_yak_codes = gauges[(gauges["yakutia"]) & (gauges["basin"] == "Lena")]["gauge_code"]
    lena_maes = []
    for gcode in lena_yak_codes:
        g_sub = seasons[(seasons["gauge_code"] == gcode) & seasons["valid"]][["hydro_year", "drift_start_day"]].dropna()
        m_sub = sat.merge(g_sub, on="hydro_year")
        if len(m_sub) >= 6:
            a_s = m_sub["sat_breakup_day"] - m_sub["sat_breakup_day"].mean()
            a_g = m_sub["drift_start_day"] - m_sub["drift_start_day"].mean()
            lena_maes.append(float((a_s - a_g).abs().mean()))
    mean_lena_mae = float(np.mean(lena_maes)) if lena_maes else mae_overlap

    _record(
        checks,
        "V5_OVERLAP_2001_2012",
        "Сверка калиброванного спутника с гидропостами на общих годах (2001–2012)",
        mae_overlap <= 11.0 and mean_lena_mae <= 11.0,
        f"MAE аномалий (п. Ботома, n={len(overlap)} лет): {mae_overlap:.1f} сут; среднее по постам бассейна Лены: {mean_lena_mae:.1f} сут",
        "MAE ≤ 11.0 сут (паспортная точность набора ESSD)",
        "Сверка двух независимых источников",
    )

    # 6. Физическая сходимость формулы Стефана с замерами толщины льда
    pooled, per_gauge, _ = calibrate_all()
    stefan_ok = pooled is not None and (1.2 <= pooled["alpha"] <= 2.8) and (pooled["r2"] >= 0.5)
    _record(
        checks,
        "V6_STEFAN_PHYSICS",
        "Сходимость физической модели нарастания льда (закон Стефана) с замерами",
        stefan_ok,
        f"α = {pooled['alpha']:.2f} см/(°C·сут)^0.5, R² = {pooled['r2']:.2f}, MAE = {pooled['mae_cm']:.1f} см (n={pooled['n']} замеров, {len(per_gauge)} постов)",
        "α ∈ [1.2; 2.8], R² ≥ 0.50",
        "Физическая верификация ERA5 × Росгидромет",
    )

    # 7. Наличие и маркировка тарифного справочника сквозного коридора А3
    tariffs_path = ROSSTAT_PRICES / "yakutsk_tariffs_and_prices.csv"
    tariffs_ok = tariffs_path.exists()
    t_df = pd.read_csv(tariffs_path) if tariffs_ok else pd.DataFrame()
    has_sources = tariffs_ok and t_df["source_type"].notna().all() and len(t_df) >= 10
    _record(
        checks,
        "V7_TARIFFS_TAGGED",
        "Полнота и маркировка источников тарифов сквозного маршрута и цен в распутицу",
        has_sources,
        f"{len(t_df)} параметров в data/rosstat_prices/yakutsk_tariffs_and_prices.csv, немаркированных: {0 if has_sources else 'есть'}",
        "все параметры имеют явный source_type и источник",
        "Госкомцен РС(Я) / Саха(Якутия)стат / допущения ±30%",
    )

    summary_df = pd.DataFrame(checks)
    meta = {
        "shift_stats": shift_stats,
        "calib_df": calib_df,
        "pooled_stefan": pooled,
        "per_gauge_stefan": per_gauge,
        "mae_overlap_botoma": round(mae_overlap, 2),
        "mae_overlap_lena": round(mean_lena_mae, 2),
    }
    return summary_df, meta


def write_validation_report(summary_df: pd.DataFrame, meta: dict) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(REPORTS / "validation_summary_a3.csv", index=False)

    n_pass = int((summary_df["Статус"] == "PASS").sum())
    n_total = len(summary_df)
    overall = "PASS" if n_pass == n_total else "FAIL"
    st = meta["shift_stats"]
    calib_preview = meta["calib_df"].tail(6)[
        ["hydro_year", "sat_breakup_date", "sat_breakup_anom_days",
         "mchs_ice_close_date", "mchs_breakup_date", "mchs_ferry_open_date", "spring_rasputitsa_days"]
    ].rename(columns={
        "hydro_year": "Сезон",
        "sat_breakup_date": "Спутник (таяние/шуга)",
        "sat_breakup_anom_days": "Δt_sat, сут",
        "mchs_ice_close_date": "Закрытие льда (Минтранс)",
        "mchs_breakup_date": "Ледоход у Якутска (МЧС)",
        "mchs_ferry_open_date": "Открытие паромов",
        "spring_rasputitsa_days": "Окно распутицы, сут",
    })

    lines = [
        "# Протокол валидации данных и калибровки сдвига (`PASS / FAIL`)",
        "",
        f"**Дата проверки:** {dt.date.today():%d.%m.%Y}  ",
        f"**Итоговый статус:** **{overall}** (`{n_pass}/{n_total}` проверок успешно)  ",
        "**Скрипт генерации:** `python src/validate_data.py` (детальная пофайловая сводка гидропостов — в `reports/data_checks.md`).",
        "",
        "---",
        "",
        "## 1. Сводная таблица автопроверок",
        "",
        md_table(summary_df),
        "",
        "---",
        "",
        "## 2. Калибровка систематического сдвига спутниковых дат (Qiu et al., 2026)",
        "",
        "При верификации спутникового набора *Daily River Ice Concentration / Phenology for the Lena River* (сетка 3 км, 2001–2024 гг.) обнаружен физически объяснимый систематический сдвиг:",
        "",
        f"- Радиометр/оптический индекс фиксирует **появление воды поверх льда, закраин и шуги** в среднем на день сезона **{st['mean_sat_breakup_day']}** (около 4 мая; в 2024 г. — `{st['sat_2024_date']}`).",
        f"- Фактическое **активное вскрытие и ледоход у Якутска** по сводкам ГУ МЧС России по РС(Я) наступает в среднем на **+{st['shift_to_breakup_days']} суток позже** (в 2024 г. — `{st['mchs_2024_breakup_date']}`, сдвиг `+{st['shift_2024_days']}` сут), а регулярная **грузовая паромная переправа** открывается на **+{st['shift_to_ferry_days']} суток позже** спутниковой метки (`{st['mchs_2024_ferry_date']}`).",
        f"- Официальное **закрытие ледовой переправы** для грузового транспорта вводится приказом Минтранса РС(Я) в среднем за **{abs(st['shift_to_ice_close_days']):.1f} суток до** спутниковой метки вскрытия.",
        "",
        "**Решение в модели:** спутниковый ряд переводится в безразмерные межгодовые аномалии $\\Delta t_{sat, y} = t_{sat, y} - \\overline{t}_{sat}$ и привязывается к референсным операционным фазам переправы (`data/processed/yakutsk_sat_calibrated.csv`). После устранения константы сдвига средняя абсолютная ошибка между калиброванным спутником и фактом МЧС составляет **"
        f"{st['mae_mchs_breakup_days']} суток**, а на 12 пересекающихся годах (2001–2012) с гидропостом Росгидромета Ботома (`3210`) — **{meta['mae_overlap_botoma']} суток** (при паспортном допуске датасета $\\le 11$ суток).",
        "",
        "### Сверка последних сезонов (2019–2024 гг.)",
        "",
        md_table(calib_preview),
        "",
        "---",
        "",
        "## 3. Маркировка синтетики и допущений",
        "",
        "1. **Измеренные и архивные ряды (не синтетика):** суточная погода ERA5 (`data/weather/`), замеры толщины и фаз льда Росгидромета (`data/processed/hydro_ice_*.csv`), спутниковая фенология Лены (`data/lena_ice_phenology_points.csv`), радарные вырезки Sentinel-1 (`data/s1/`) и даты приказов МЧС/Минтранса РС(Я).",
        "2. **Тарифные и экономические параметры (`data/rosstat_prices/yakutsk_tariffs_and_prices.csv`):** опираются на порядки публичных тарифов Госкомцен РС(Я), ЖДЯ и АК «Полярные авиалинии», явно помечены как допущения симулятора и проходят обязательный стресс-тест на чувствительность $\\pm 30\\%$ в `src/corridor_simulator.py`.",
    ]

    report_path = DOCS / "validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    summary_df, meta = run_validation()
    out_path = write_validation_report(summary_df, meta)
    failed = int((summary_df["Статус"] != "PASS").sum())
    print(f"\nСформирован отчёт: {out_path} (PASS: {len(summary_df) - failed}/{len(summary_df)})")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
