"""
Независимая проверка М1 паводков (прогноз уровня воды на гидропосту) по сохранённым прогнозам.

    python scripts/flood_value_check.py      # → reports/flood_value.md

Пересчитывается из сырых прогнозов reports/flood_m1_predictions.csv.gz (scripts/train_flood_m1.py), а не из сводки:
1. Нет утечки: прогноз выпущен до целевой даты; «сохранить уровень» равен замеру на дату выпуска.
2. Ошибка за 7 суток против «уровень не изменится», отдельно половодье (апрель–июль) и остальное время.
3. Интервал по годам — бутстрэп по годам проверки (5–95%).
4. Покрытие 80%-диапазона.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.paths import REPORTS
from okno.report import md_table

NAMES = {"lena_yakutsk": "Лена — Якутск", "ishim_ishim": "Ишим — Ишим", "ural_orenburg": "Урал — Оренбург",
         "ural_orsk": "Урал — Орск"}


def main():
    d = pd.read_csv(REPORTS / "flood_m1_predictions.csv.gz", parse_dates=["issue", "target_date"])
    d["year"], d["spring"] = d.target_date.dt.year, d.target_date.dt.month.isin([4, 5, 6, 7])
    no_leak = bool((d.issue < d.target_date).all() and ((d.target_date - d.issue).dt.days == d.lead).all())
    obs = d[d.lead == 1][["pid", "target_date", "target"]].rename(columns={"target_date": "issue", "target": "obs"})
    m = d.merge(obs, on=["pid", "issue"]).dropna(subset=["obs"])
    persist_ok = bool(np.allclose(m.persistence, m.obs))
    rng = np.random.default_rng(0)
    rows = []
    for (pid, spring), g in d[d.lead == 7].groupby(["pid", "spring"]):
        em, ep = (g.m1 - g.target).abs(), (g.persistence - g.target).abs()
        ys, bs = g.year.unique(), []
        for _ in range(2000):
            gg = pd.concat([g[g.year == v] for v in rng.choice(ys, len(ys))])
            bs.append(100 * (1 - (gg.m1 - gg.target).abs().mean() / (gg.persistence - gg.target).abs().mean()))
        lo, hi = np.percentile(bs, [5, 95])
        rows.append({"пост": NAMES.get(pid, pid), "период": "половодье (апр–июл)" if spring else "остальное время",
                     "прогнозов": len(g), "лет": len(ys), "ошибка М1, см": round(em.mean(), 1),
                     "ошибка «уровень не изменится», см": round(ep.mean(), 1),
                     "М1 точнее, %": round(100 * (1 - em.mean() / ep.mean())), "интервал 5–95%": f"{lo:+.0f} … {hi:+.0f}",
                     "доказано": "да" if lo > 0 else "нет",
                     "покрытие 80%-диапазона": f"{((g.target >= g.lower) & (g.target <= g.upper)).mean():.0%}"})
    res = pd.DataFrame(rows)
    lines = ["# М1 паводков v1: независимая проверка пользы", "",
             "Эта проверка относится к v1. Новая квантильная v2 и сравнение на одинаковых датах:",
             "[flood_m1_quantile.md](flood_m1_quantile.md).", "",
             "Пересчёт из сырых прогнозов `reports/flood_m1_predictions.csv.gz`; метод — в шапке "
             "`scripts/flood_value_check.py`. Прогноз уровня воды на 7 суток; соперник — «уровень не изменится».", "",
             f"- Прогноз всегда выпущен до целевой даты: **{'да' if no_leak else 'НЕТ'}**.",
             f"- «Уровень не изменится» равен замеру на дату выпуска: **{'да' if persist_ok else 'НЕТ'}**.", "",
             md_table(res), "",
             "**Вывод.** На Лене у Якутска и на Ишиме прогноз уровня за неделю надёжно точнее простого правила, сильнее всего "
             "в половодье (у Якутска ошибка вдвое меньше). На Урале (Оренбург, Орск) преимущество не доказано. 80%-диапазон "
             "в половодье попадает реже обещанного — им нельзя пользоваться как гарантией. Связь «уровень → затопление дороги "
             "или склада» не проверена: это прогноз уровня, а не прогноз затопления. Внешний прогноз расхода GloFAS "
             "наводнение 2024 г. в Орске не поймал (docs/FLOOD_METHOD.md).", ""]
    (REPORTS / "flood_value.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
