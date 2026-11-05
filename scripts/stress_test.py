"""
Стресс-тест экономических допущений — строже, чем ±30% по одному.

    python scripts/stress_test.py

1. Совместный разброс: 100 случайных комбинаций, где ВСЕ допущения одновременно
   умножаются на случайный множитель от 0,5 до 1,5. Доля комбинаций, где «после» дешевле «до».
2. Точка перелома: насколько должна быть дешевле цена дорогого канала (самое уязвимое
   допущение), чтобы вывод «после дешевле, чем до» перестал выполняться.

Выход: reports/stress_test.csv, reports/stress_breakeven.csv, reports/stress_test.md
"""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from before_after import EXAMPLES, run_example
from okno.paths import REPORTS
from okno.planner import SENSITIVE, Params
from okno.report import md_table

N_DRAWS, LOW, HIGH, SEED = 100, 0.5, 1.5, 7
RULES = ("calendar", "forecast", "asap")


def costs(bt, pid, p):
    r = run_example(bt, pid, p, rules=RULES)
    return {rule: r[r.rule == rule].cost_rub_per_t.mean() for rule in RULES} | {
        f"недовоз_{rule}": r[r.rule == rule].undelivered_t.mean() for rule in RULES}


def joint(bt, base):
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(N_DRAWS):
        mult = {k: float(rng.uniform(LOW, HIGH)) for k in SENSITIVE}
        p = Params(**{**asdict(base), **{k: getattr(base, k) * m for k, m in mult.items()}})
        for key, ex in EXAMPLES.items():
            c = costs(bt, ex["point_id"], p)
            rows.append({"draw": i, "пример": key, **{f"x_{k}": round(v, 3) for k, v in mult.items()},
                         "₽/т до": c["calendar"], "₽/т после": c["forecast"], "₽/т контроль": c["asap"],
                         "экономия_%": 100 * (1 - c["forecast"] / c["calendar"]),
                         "после лучше": c["forecast"] <= c["calendar"],
                         "после лучше контроля": c["forecast"] <= c["asap"]})
    return pd.DataFrame(rows)


def breakeven(bt, base):
    rows = []
    for key, ex in EXAMPLES.items():
        for f in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
            p = base.scaled("c_exp", f)
            c = costs(bt, ex["point_id"], p)
            rows.append({"пример": key, "множитель_цены_дорогого_канала": f, "c_exp_₽_т": round(p.c_exp),
                         "отношение_к_зимнику": round(p.c_exp / p.c_road, 1),
                         "экономия_%": round(100 * (1 - c["forecast"] / c["calendar"]), 1)})
    return pd.DataFrame(rows)


def main():
    bt = pd.read_csv(REPORTS / "m1_backtest_close.csv")
    base = Params()
    j = joint(bt, base)
    j.to_csv(REPORTS / "stress_test.csv", index=False)
    b = breakeven(bt, base)
    b.to_csv(REPORTS / "stress_breakeven.csv", index=False)
    lines = ["# Стресс-тест экономических допущений", "",
             f"Все {len(SENSITIVE)} допущений одновременно умножаются на случайный множитель от {LOW} до {HIGH} "
             f"({N_DRAWS} комбинаций на каждый пример). Это строже, чем проверка ±30% по одному допущению.", ""]
    summ = []
    for key, ex in EXAMPLES.items():
        g = j[j["пример"] == key]
        summ.append({"пример": ex["name"], "комбинаций": len(g),
                     "после дешевле до, %": round(100 * g["после лучше"].mean()),
                     "после дешевле контроля, %": round(100 * g["после лучше контроля"].mean()),
                     "экономия: медиана, %": round(g["экономия_%"].median(), 1),
                     "экономия: худшие 5%, %": round(g["экономия_%"].quantile(0.05), 1)})
        bb = b[(b["пример"] == key) & (b["отношение_к_зимнику"] >= 1)]  # «дорогой» дешевле зимника — бессмысленный режим
        neg = bb[bb["экономия_%"] <= 0]
        small = bb[bb["экономия_%"] < 1]
        if len(neg):
            note = f"вывод переворачивается при отношении цен ≤ {neg['отношение_к_зимнику'].max()}"
        elif len(small):
            note = (f"не переворачивается, но экономия < 1%, когда дорогой канал дороже зимника "
                    f"в ≤ {small['отношение_к_зимнику'].max()} раза (в базе — в 6 раз)")
        else:
            note = "экономия ≥ 1% во всём диапазоне"
        summ[-1]["цена дорогого канала"] = note
    s = pd.DataFrame(summ)
    lines += [md_table(s), "", "## Точка перелома: цена дорогого канала", "",
              "Экономия «после» относительно «до» при снижении цены дорогого канала (остальное — как в базе).", "",
              md_table(b), ""]
    (REPORTS / "stress_test.md").write_text("\n".join(lines), encoding="utf-8")
    print(s.to_string(index=False))
    print("готово:", REPORTS / "stress_test.md")


if __name__ == "__main__":
    main()
