"""
Эксперимент: добавить в обучение прогноза закрытия (М1) спутниковые сезоны по всему бассейну Лены.

    python scripts/lena_points.py --weather   # точки, даты, погода
    python scripts/lena_experiment.py

Две модели с одинаковыми параметрами, проверка «сезон целиком вне обучения»:
  база   — как сейчас: гидропосты 5 регионов + переправа Якутск;
  +Лена  — то же плюс ~50 точек бассейна Лены (Qiu et al. 2026, 2001–2024).
Сравнение — на одних и тех же строках (гидропосты и переправа Якутск), за ~14 дней до события.
Достоверность — бутстрэп по сезонам разницы ошибок (2000 пересборок).

Выход: reports/lena_experiment.md, reports/lena_experiment.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m1
from okno.features import build_m1_dataset
from okno.paths import REPORTS
from okno.regions import REGIONS, all_gauges, region_of
from okno.report import md_table

LEADS = {"≈7 дней": (4, 10), "≈14 дней": (11, 17), "≈28 дней": (25, 31), "все": (0, 400)}


def group(pid):
    if pid == "yakutsk_crossing":
        return "переправа Якутск (спутник)"
    if pid.startswith("lena_"):
        return "точки бассейна Лены (спутник)"
    return f"{REGIONS[region_of(pid)]['name']} (гидропосты)"


def boot_diff(d, n=2000, seed=0):
    """90%-интервал разницы MAE (+Лена − база) по сезонам; < 0 — +Лена точнее."""
    rng = np.random.default_rng(seed)
    s = d.groupby("hydro_year").agg(eb=("err_base", "sum"), ex=("err_ext", "sum"), n=("err_base", "size"))
    yrs = s.index.to_numpy()
    res = []
    for _ in range(n):
        k = rng.choice(yrs, len(yrs))
        res.append((s.loc[k, "ex"].sum() - s.loc[k, "eb"].sum()) / s.loc[k, "n"].sum())
    return np.percentile(res, [5, 95])


def main():
    base = build_m1_dataset("close", codes=all_gauges())
    ext = build_m1_dataset("close", codes=all_gauges(), lena=True)
    n_lena = ext.point_id.str.startswith("lena_").sum()
    print(f"строк: база {len(base)}, +Лена {len(ext)} (точек Лены {ext.point_id[ext.point_id.str.startswith('lena_')].nunique()}, "
          f"строк {n_lena})", flush=True)
    bt_b = m1.backtest(base, "close")
    bt_x = m1.backtest(ext, "close")
    key = ["point_id", "hydro_year", "issue_day"]
    m = bt_b[key + ["target_day", "cal_target", "q50", "q10", "q90", "lead_days"]].merge(
        bt_x[key + ["q50", "q10", "q90"]], on=key, suffixes=("_b", "_x"))
    m["err_base"] = (m.q50_b - m.target_day).abs()
    m["err_ext"] = (m.q50_x - m.target_day).abs()
    m["err_cal"] = (m.cal_target - m.target_day).abs()
    m["grp"] = m.point_id.map(group)
    rows = []
    for g, d in list(m.groupby("grp")) + [("все гидропосты", m[~m.point_id.isin(["yakutsk_crossing"])])]:
        for lead, (lo, hi) in LEADS.items():
            s = d[d.lead_days.between(lo, hi)]
            if len(s) < 10:
                continue
            lo_ci, hi_ci = boot_diff(s)
            rows.append({"группа": g, "за сколько": lead, "прогнозов": len(s), "сезонов": s.hydro_year.nunique(),
                         "MAE календарь": round(s.err_cal.mean(), 2), "MAE база": round(s.err_base.mean(), 2),
                         "MAE +Лена": round(s.err_ext.mean(), 2),
                         "разница, сут": round(s.err_ext.mean() - s.err_base.mean(), 2),
                         "90%-интервал разницы": f"{lo_ci:+.2f} … {hi_ci:+.2f}",
                         "покрытие база": round(((s.target_day >= s.q10_b) & (s.target_day <= s.q90_b)).mean(), 2),
                         "покрытие +Лена": round(((s.target_day >= s.q10_x) & (s.target_day <= s.q90_x)).mean(), 2)})
    res = pd.DataFrame(rows)
    # сами точки Лены — только в модели +Лена; сравнение с календарём
    lx = bt_x[bt_x.point_id.str.startswith("lena_")]
    lena_rows = []
    for lead, (lo, hi) in LEADS.items():
        s = lx[lx.lead_days.between(lo, hi)]
        if len(s):
            ec, em = (s.cal_target - s.target_day).abs().mean(), (s.q50 - s.target_day).abs().mean()
            lena_rows.append({"за сколько": lead, "прогнозов": len(s), "сезонов": s.hydro_year.nunique(),
                              "MAE календарь": round(ec, 2), "MAE М1": round(em, 2),
                              "выигрыш к календарю, %": round(100 * (1 - em / ec), 1),
                              "покрытие 10–90": round(((s.target_day >= s.q10) & (s.target_day <= s.q90)).mean(), 2)})
    res.to_csv(REPORTS / "lena_experiment.csv", index=False)
    lines = ["# Эксперимент: спутниковые сезоны бассейна Лены в обучении прогноза закрытия", "",
             f"База — как сейчас (гидропосты 5 регионов + переправа Якутск). «+Лена» — плюс {n_lena} строк по "
             f"{lx.point_id.nunique()} точкам бассейна Лены (Qiu et al. 2026, вскрытие 2001–2024). Сравнение на одних и тех же "
             "прогнозах; «разница» < 0 — с Леной точнее. Интервал — бутстрэп по сезонам: если он целиком ниже нуля, "
             "улучшение не случайно.", "", "## Влияние на текущие точки", "", md_table(res), "",
             "## Сами точки бассейна Лены (только в модели «+Лена»)", "", md_table(pd.DataFrame(lena_rows)), ""]
    (REPORTS / "lena_experiment.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))
    print(pd.DataFrame(lena_rows).to_string(index=False))


if __name__ == "__main__":
    main()
