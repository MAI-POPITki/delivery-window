"""
Проверка: даёт ли переобучение М1 (закрытие) на свежих официальных датах 2021–2026 выигрыш.

    python scripts/m1_fresh_official_retrain.py

Цель М1 закрытия на гидропостах — начало разрушения ледостава (melt_start_day). В свежих
официальных сводках (data/external/hydro_post2012_phenology.csv) есть дата вскрытия / ледохода.
Переводим её в цель М1 сдвигом «ледоход − начало таяния»: медиана по сезонам того же поста
из архива 1970–2012 (все они раньше проверяемых лет, будущее не используется).
Переправа Якутск не берётся: её цель — спутниковая первая вода, ряд Qiu уже есть до 2024 г.,
а связь «официальный ледоход ↔ первая вода» не одинакова по годам.

Сравнение на свежих сезонах (они не входили в обучение текущей модели):
  A — текущая М1 (обучена на архиве) против календаря;
  B — М1, переобученная на архиве + свежих сезонах других лет (leave-one-season-out).
И на архиве: не ухудшило ли добавление свежих строк прежний бэктест.
Выход: reports/m1_fresh_official_retrain.md, .csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m1
from okno.features import build_m1_dataset, point_rows
from okno.ice import to_season_day
from okno.paths import DATA, PROCESSED, REPORTS
from okno.regions import all_gauges, region_of
from okno.report import md_table

KIND = "close"
SKIP_SOURCES = ("ogirk.ru/2026/04/07",)  # статья от 07.04.2026 — прогноз сроков вскрытия, а не факт


def fresh_targets():
    ph = pd.read_csv(DATA / "external" / "hydro_post2012_phenology.csv")
    ph = ph[ph.point_id.str.startswith("gauge_")]
    ph = ph[~ph.source_breakup.astype(str).str.contains("|".join(SKIP_SOURCES), regex=True)]
    seasons = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    gauges = pd.read_csv(PROCESSED / "gauges.csv")
    out = {}
    for pid, g in ph.groupby("point_id"):
        code = int(pid.split("_")[1])
        s = seasons[(seasons.gauge_code == code) & seasons.valid]
        lag = (s.drift_start_day - s.melt_start_day).dropna()
        if len(lag) < 10 or s.melt_start_day.notna().sum() < 10:
            continue
        shift = float(lag.median())
        hist = pd.DataFrame({"target_day": s.melt_start_day.to_numpy(), "freezeup_day": s.freezeup_day.to_numpy()},
                            index=s.hydro_year.to_numpy())
        new = pd.DataFrame({"target_day": [to_season_day(pd.Timestamp(d), y) - shift for d, y in zip(g.breakup_date, g.hydro_year)],
                            "freezeup_day": [to_season_day(pd.Timestamp(d), y) for d, y in zip(g.freezeup_date, g.hydro_year)]},
                           index=g.hydro_year.to_numpy())
        new = new[~new.index.isin(hist.index)]
        meta = gauges[gauges.gauge_code == code].iloc[0]
        out[pid] = dict(hist=hist, new=new, shift=shift, n_lag=len(lag), lat=meta.lat, lon=meta.lon)
    return out


def score(df, pred_col="q50"):
    rows = []
    for lead, (lo, hi) in {**m1.LEAD_BUCKETS, "все": (0, 400)}.items():
        s = df[df.lead_days.between(lo, hi)]
        if s.empty:
            continue
        rows.append({"за_сколько_дней": lead, "n": len(s), "сезонов": s.hydro_year.nunique(),
                     "MAE_календарь": round((s.cal_target - s.target_day).abs().mean(), 2),
                     "MAE_М1": round((s[pred_col] - s.target_day).abs().mean(), 2)})
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    base = build_m1_dataset(KIND, codes=all_gauges())
    targets = fresh_targets()
    parts = []
    for pid, t in targets.items():
        rows = pd.DataFrame(point_rows(pid, {"lat": t["lat"], "lon": t["lon"]},
                                       pd.concat([t["hist"], t["new"]]).sort_index(), {}, KIND))
        if not rows.empty:
            parts.append(rows[rows.hydro_year.isin(t["new"].index)])
    fresh = pd.concat(parts, ignore_index=True)
    for d in (base, fresh):
        d["region"] = d.point_id.map(region_of)
    print(f"архив: {len(base)} строк; свежие: {len(fresh)} строк, {fresh.groupby(['point_id','hydro_year']).ngroups} пост-сезонов")

    # A: текущая М1 (обучена только на архиве)
    model = m1.fit(base, KIND)
    a = pd.concat([fresh, m1.predict(model, fresh, KIND)], axis=1)
    # B: переобучение на архиве + свежих, leave-one-season-out
    both = pd.concat([base, fresh], ignore_index=True)
    bt_both = m1.backtest(both, KIND)
    b = bt_both.iloc[len(base):]
    bt_base = m1.backtest(base, KIND)
    arch_old = score(bt_base); arch_new = score(bt_both.iloc[:len(base)])

    sa, sb = score(a), score(b)
    tab = sa.rename(columns={"MAE_М1": "MAE_М1_текущая"}).merge(sb[["за_сколько_дней", "MAE_М1"]].rename(
        columns={"MAE_М1": "MAE_М1_переобученная"}), on="за_сколько_дней")
    arch = arch_old.rename(columns={"MAE_М1": "MAE_М1_до"}).merge(
        arch_new[["за_сколько_дней", "MAE_М1"]].rename(columns={"MAE_М1": "MAE_М1_после"}), on="за_сколько_дней")
    per = (a.assign(err_cal=(a.cal_target - a.target_day).abs(), err_cur=(a.q50 - a.target_day).abs(),
                    err_new=(b.q50.to_numpy() - b.target_day.to_numpy()).__abs__())
           .groupby(["point_id", "hydro_year"])[["err_cal", "err_cur", "err_new"]].mean().round(1).reset_index())
    per.to_csv(REPORTS / "m1_fresh_official_retrain.csv", index=False)

    shifts = pd.DataFrame([{"пост": k, "сдвиг ледоход − начало таяния, сут": v["shift"], "сезонов в архиве": v["n_lag"],
                            "свежих сезонов": len(v["new"])} for k, v in targets.items()])
    lines = ["# М1 закрытия: проверка и переобучение на свежих официальных датах 2021–2026", "",
             "Источник свежих дат — `data/external/hydro_post2012_phenology.csv` (обзоры УГМС/ЦГМС, МЧС). "
             "Дата вскрытия переведена в цель М1 (начало разрушения ледостава) сдвигом по архиву того же поста. "
             "Строка Иркутского УГМС из статьи от 07.04.2026 исключена: это прогноз сроков, а не факт. "
             "Календарь — средняя по 10 прошлым сезонам (нужно ≥3), поэтому после разрыва в рядах "
             "оцениваются только сезоны, где календарь уже определён.", "",
             "## Сдвиги", "", md_table(shifts), "",
             "## Свежие сезоны: текущая М1 и переобученная", "", md_table(tab), "",
             "## Архив 1970–2018: не ухудшило ли переобучение", "", md_table(arch), "",
             "## По пост-сезонам (средняя ошибка, сут)", "", md_table(per), ""]
    (REPORTS / "m1_fresh_official_retrain.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
