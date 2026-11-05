"""
Достоверность выигрыша М1: бутстрэп по сезонам.

    python scripts/significance.py

Для каждого региона и события берём прогнозы за ~14 дней до события и 2000 раз пересобираем
выборку сезонов с повторением. Получаем 90%-ный интервал выигрыша к календарю и долю пересборок,
где выигрыша нет. Сезон берётся целиком, потому что прогнозы одного сезона связаны между собой.

Выход: reports/m1_significance.csv, reports/m1_significance.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.paths import REPORTS
from okno.regions import REGIONS
from okno.report import md_table

N_BOOT, SEED, LEAD = 2000, 0, (11, 17)


def verdict(lo, hi, n_seasons):
    if n_seasons < 8:
        return "мало данных"
    if lo > 0:
        return "надёжно точнее календаря"
    if hi < 0:
        return "надёжно хуже календаря"
    return "не отличается от календаря" if lo < -5 or hi < 10 else "скорее точнее, не доказано"


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for kind, name in (("open", "открытие"), ("close", "закрытие")):
        bt = pd.read_csv(REPORTS / f"m1_backtest_{kind}.csv")
        bt = bt[bt.lead_days.between(*LEAD)].copy()
        bt["grp"] = np.where(bt.is_satellite == 1, "переправа Якутск (спутник)",
                             bt["region"].map(lambda r: REGIONS.get(r, {}).get("name", r)))
        for g, d in bt.groupby("grp"):
            d = d.assign(ec=(d.cal_target - d.target_day).abs(), em=(d.q50 - d.target_day).abs())
            s = d.groupby("hydro_year")[["ec", "em"]].sum()
            n = d.groupby("hydro_year").size()
            yrs = s.index.to_numpy()
            boot = []
            for _ in range(N_BOOT):
                k = rng.choice(yrs, len(yrs))
                boot.append(100 * (1 - s.loc[k, "em"].sum() / s.loc[k, "ec"].sum()))
            lo, hi = np.percentile(boot, [5, 95])
            gain = 100 * (1 - d.em.mean() / d.ec.mean())
            rows.append({"событие": name, "группа": g, "сезонов": len(yrs),
                         "календарь, сут": round(d.ec.mean(), 1), "М1, сут": round(d.em.mean(), 1),
                         "выигрыш, сут": round(d.ec.mean() - d.em.mean(), 1), "выигрыш, %": round(gain, 1),
                         "90%-интервал, %": f"{lo:.0f} … {hi:.0f}",
                         "P(выигрыша нет)": round(float(np.mean(np.array(boot) <= 0)), 2),
                         "вывод": verdict(lo, hi, len(yrs))})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "m1_significance.csv", index=False)
    lines = ["# Достоверность выигрыша М1", "",
             f"Бутстрэп по сезонам ({N_BOOT} пересборок), прогнозы за {LEAD[0]}–{LEAD[1]} дней до события. "
             "«90%-интервал» — в каких пределах лежит выигрыш к календарю с вероятностью 90%; если интервал "
             "целиком выше нуля, выигрыш не случаен.", "", md_table(df), ""]
    (REPORTS / "m1_significance.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
