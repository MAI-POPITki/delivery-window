"""
Проверки валидности данных → reports/data_checks.md и reports/data_checks.csv.

    python scripts/validate_data.py

Правило проекта: каждая цифра либо измерена, либо сверена с независимым источником,
либо помечена как допущение. Скрипт печатает таблицу «проверка — прошла / не прошла».
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.calib import calibrate_all
from okno.paths import DATA, PROCESSED, REPORTS, WEATHER
from okno.report import md_table

CHECKS = []


def check(name, passed, detail, kind="автопроверка"):
    CHECKS.append({"проверка": name, "тип": kind, "результат": "прошла" if passed else "НЕ прошла",
                   "детали": detail})
    print(f"[{'OK ' if passed else 'FAIL'}] {name}: {detail}")


def weather_checks():
    files = sorted(WEATHER.glob("*.csv")) + sorted((WEATHER / "gauges").glob("*.csv"))
    today = pd.Timestamp(dt.date.today())
    for f in files:
        w = pd.read_csv(f, parse_dates=["date"])
        dups = int(w["date"].duplicated().sum())
        full = pd.date_range(w["date"].min(), w["date"].max(), freq="D")
        gaps = len(full) - w["date"].nunique()
        nan_t = int(w["temperature_2m_mean"].isna().sum())
        future = int((w["date"] > today).sum())
        bad_range = int(((w["temperature_2m_mean"] < -65) | (w["temperature_2m_mean"] > 40)).sum())
        ok = dups == 0 and gaps == 0 and nan_t == 0 and future == 0 and bad_range == 0
        check(f"погода {f.stem}: ряд полный", ok,
              f"{w['date'].min():%Y-%m-%d}…{w['date'].max():%Y-%m-%d}, дублей {dups}, дыр {gaps}, "
              f"пустых T {nan_t}, дат из будущего {future}, T вне [-65; 40] °C: {bad_range}")

    # климатическая норма Якутска: январь ≈ -38…-39 °C, июль ≈ +19…+20 °C (норма 1991–2020)
    w = pd.read_csv(WEATHER / "yakutsk_crossing.csv", parse_dates=["date"])
    w = w[(w.date.dt.year >= 1991) & (w.date.dt.year <= 2020)]
    jan = w[w.date.dt.month == 1]["temperature_2m_mean"].mean()
    jul = w[w.date.dt.month == 7]["temperature_2m_mean"].mean()
    check("ERA5 у Якутска совпадает с климатической нормой", abs(jan + 38.5) < 3 and abs(jul - 19.5) < 3,
          f"январь {jan:.1f} °C (норма ≈ -38.5), июль {jul:.1f} °C (норма ≈ +19.5), 1991–2020",
          "сверка с независимым источником")


def hydro_checks():
    s = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    g = pd.read_csv(PROCESSED / "gauges.csv")
    yak = g.loc[g.yakutia, "gauge_code"]
    sy = s[s.gauge_code.isin(yak)]
    phase_known=s[['first_ice_day','freezeup_day','melt_start_day','drift_start_day','ice_end_day']].notna().any(axis=1)
    bad = s[phase_known & ~s.valid]
    check("гидропосты: даты ледовых явлений в правильном порядке и в правдоподобных месяцах",
          len(bad) / phase_known.sum() < 0.05,
          f"сезонов с известными фазами {phase_known.sum()}, с флагами {len(bad)} ({len(bad) / phase_known.sum():.1%}) — исключаются из обучения; "
          f"по Якутии {(~sy.valid).sum()} из {len(sy)}")
    dur = (sy.drift_start_day - sy.freezeup_day)[sy.valid].dropna()
    check("длительность ледостава в Якутии в пределах литературы (180–250 сут)",
          dur.between(170, 260).mean() > 0.9,
          f"медиана {dur.median():.0f} сут, 5–95%: {dur.quantile(.05):.0f}–{dur.quantile(.95):.0f}")
    th = pd.read_csv(PROCESSED / "hydro_ice_thickness.csv")
    thy = th[th.gauge_code.isin(yak)]
    spikes = thy[thy.spike]
    big = int((thy.ice_cm > 400).sum())
    check("толщина льда без одиночных выбросов и в пределах 0–400 см", big == 0,
          f"замеров по Якутии {len(thy)}; отсеяно одиночных выбросов {len(spikes)} "
          f"(сырые значения: {', '.join(str(int(v)) for v in spikes.ice_cm_raw)} см); "
          f"после очистки максимум {thy.ice_cm.max():.0f} см (наледи на горных реках, напр. Алгий); "
          f"«промерзание до дна»: {int(thy.frozen_to_bottom.sum())}")
    from okno.northeast import STATIONS
    new=th[th.gauge_code.isin(STATIONS)]
    after=new[new.date>='2013-01-01']
    check('Новые посты: данные после 2012 и неповторяющиеся даты',
          len(after)>0 and not new.duplicated(['gauge_code','date']).any(),
          f'{new.gauge_code.nunique()} постов, {len(after)} строк после 2012, последняя дата {after.date.max()}; '
          'противоречивые повторные даты в сыром файле исключены')
    new_phases=s[s.gauge_code.isin(STATIONS)]
    check('Недостающие фазы PANGAEA не подменены датами замеров',
          new_phases[['first_ice_day','freezeup_day','melt_start_day','drift_start_day','ice_end_day']].isna().all().all(),
          'Доступны оценки достижения пороговой толщины; начала таяния и ледохода в источнике нет')
    return s, g


def snow_checks():
    from okno.regions import all_gauges
    expected={f'gauge_{c}' for c in all_gauges()}|{'yakutsk_crossing'}
    missing=[];bad=[];total=0
    for pid in sorted(expected):
        path=DATA/'snow_depth'/f'{pid}.csv'
        if not path.exists():
            missing.append(pid)
            continue
        frame=pd.read_csv(path,parse_dates=['date'])
        total+=len(frame)
        dates=frame.date
        gaps=len(pd.date_range(dates.min(),dates.max()))-dates.nunique()
        if (dates.duplicated().any() or not dates.is_monotonic_increasing or gaps or
            frame.snow_depth_cm.isna().mean()>.05 or (frame.snow_depth_cm.dropna()<0).any() or
            dates.gt(pd.Timestamp(dt.date.today())).any()):
            bad.append(pid)
    check('Снег на суше: полные суточные ряды без отрицательной толщины',not missing and not bad,
          f'точек {len(expected)-len(missing)}/{len(expected)}, строк {total}; '
          f'отсутствуют: {", ".join(missing) or "нет"}; ошибки: {", ".join(bad) or "нет"}. '
          'ERA5-Land, сантиметры, UTC; в прогнозе номинальный лаг пять суток')
    measured=pd.read_csv(PROCESSED/'hydro_snow_on_ice.csv')
    check('Снег на льду: измерения отделены от модельного снега на суше',
          not measured.duplicated(['gauge_code','date']).any() and measured.snow_on_ice_cm.ge(0).all(),
          f'{len(measured)} замеров на {measured.gauge_code.nunique()} постах, '
          f'последняя дата {measured.date.max()}; в прогнозе только свой сезон и возраст не более 30 суток')


def stefan_check():
    pooled, per, pairs = calibrate_all()
    if pooled is None:
        check("закон Стефана против замеров толщины", False, "нет погоды для гидропостов", "физическая проверка")
        return None, per
    ok = 1.2 <= pooled["alpha"] <= 2.8 and pooled["r2"] > 0.5
    check("закон Стефана (ERA5) против замеров толщины льда", ok,
          f"alpha = {pooled['alpha']:.2f} см/(°C·сут)^0.5 (литература 1.4–2.7), "
          f"MAE {pooled['mae_cm']:.1f} см, R² {pooled['r2']:.2f}, замеров {pooled['n']}, "
          f"постов {len(per)}", "физическая проверка")
    per.to_csv(REPORTS / "stefan_calibration.csv", index=False)
    return pooled, per


def satellite_checks(seasons, gauges):
    sat = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv")
    yrs = sat["hydro_year"]
    check("спутник: ряд 2001–2024 без пропусков", set(range(2001, 2025)) <= set(yrs)
          and sat[["sat_breakup_day", "sat_freezeup_day"]].notna().all().all(),
          f"сезонов {len(sat)}, пустых дат {int(sat[['sat_breakup_day', 'sat_freezeup_day']].isna().sum().sum())}")

    # Сверка со вторым независимым источником: ближайший к переправе гидропост — Ботома (3210, ~60 км).
    # Реки разные, поэтому сравниваем (а) систематический сдвиг и (б) согласованность аномалий по годам.
    ref = seasons[(seasons.gauge_code == 3210) & seasons.valid][
        ["hydro_year", "freezeup_day", "melt_start_day", "drift_start_day"]]
    m = sat.merge(ref, on="hydro_year")
    rows = []
    for sat_col, g_col, label in [("sat_breakup_day", "drift_start_day", "вскрытие ↔ начало ледохода"),
                                  ("sat_breakup_day", "melt_start_day", "вскрытие ↔ начало таяния"),
                                  ("sat_freezeup_day", "freezeup_day", "ледостав ↔ ледостав")]:
        d = m[[sat_col, g_col]].dropna()
        bias = float((d[sat_col] - d[g_col]).mean())
        mad = float((d[sat_col] - d[g_col]).abs().mean())
        a, b = d[sat_col] - d[sat_col].mean(), d[g_col] - d[g_col].mean()
        r = float(np.corrcoef(a, b)[0, 1]) if len(d) > 2 else np.nan
        mae_anom = float((a - b).abs().mean())
        rows.append({"пара": label, "лет": len(d), "сдвиг_сут": round(bias, 1), "MAE_абс_сут": round(mad, 1),
                     "MAE_аномалий_сут": round(mae_anom, 1), "корреляция_аномалий": round(r, 2)})
    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS / "satellite_vs_gauge.csv", index=False)
    br = comp.iloc[0]
    check("спутник (Лена у Якутска) против гидропоста Ботома, общие годы 2001–2012",
          br["MAE_аномалий_сут"] <= 11 * 1.5,
          f"вскрытие: сдвиг {br['сдвиг_сут']:+.0f} сут к ледоходу, MAE аномалий {br['MAE_аномалий_сут']} сут "
          f"(ошибка набора по статье ≈ 11 сут), корреляция аномалий {br['корреляция_аномалий']}; "
          "абсолютные даты не используем, только отклонения года",
          "сверка с независимым источником")
    return comp


def write_report(comp, pooled, per):
    REPORTS.mkdir(exist_ok=True)
    df = pd.DataFrame(CHECKS)
    df.to_csv(REPORTS / "data_checks.csv", index=False)
    n_ok = (df["результат"] == "прошла").sum()
    lines = [
        "# Отчёт о проверках данных",
        "",
        f"Сформирован скриптом `scripts/validate_data.py`, {dt.date.today():%d.%m.%Y}. "
        f"Прошло проверок: **{n_ok} из {len(df)}**.",
        "",
        "Правило: каждая цифра в итогах либо измерена, либо сверена с независимым источником, "
        "либо помечена как допущение и проверена на чувствительность.",
        "",
        "## Сводка",
        "",
        md_table(df),
        "",
        "## Закон Стефана против замеров толщины по постам",
        "",
        (md_table(per.round(3)) if per is not None and len(per) else "нет данных"),
        "",
        "## Спутник против гидропоста (Ботома, 3210)",
        "",
        md_table(comp),
        "",
        "Вывод: спутниковые даты используем только как отклонение года от среднего; абсолютные даты "
        "переправы калибруем по гидропостам и известным датам из приказов и новостей.",
        "",
        "## Что синтетическое",
        "",
        "Только стоимость и объёмы завоза (модуль планировщика М3). Они не участвуют в оценке точности М1. "
        "Каждое допущение проверяется изменением на ±30% (`reports/sensitivity.csv`).",
    ]
    (REPORTS / "data_checks.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nотчёт: {REPORTS / 'data_checks.md'}  ({n_ok}/{len(df)} прошли)")


def main():
    REPORTS.mkdir(exist_ok=True)
    weather_checks()
    seasons, gauges = hydro_checks()
    snow_checks()
    pooled, per = stefan_check()
    comp = satellite_checks(seasons, gauges)
    write_report(comp, pooled, per)
    from src.validate_data import main as run_a3_validation
    if run_a3_validation() or any(row['результат']=='НЕ прошла' for row in CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
