"""
Стоит ли дообучать М1 закрытия на официальных датах начала таяния ЯУГМС 2022–2026.

    python scripts/eval_m1_close_ykut.py

Методика (по ветке dmitrii, scripts/eval_m1_fresh_retrain.py, перенесено на код main):
  - цель — начало разрушения ледостава (melt_start), прямо названное в бюллетене; ледоход не пересчитывается;
  - «прежняя» модель обучена только на сезонах < 2022 — так устроена рабочая М1 (свежих целей в ней нет);
  - «дообученная» — walk-forward: сезон y прогнозирует модель, обученная строго на сезонах < y;
  - сезон 2026 отдельно: в финальное обучение он не входит (контроль переправы Якутск весной 2026);
  - архив: leave-one-season-out на сезонах ≤ 2021 тем же набором, чтобы видеть, не испортило ли
    добавление свежих сезонов прежнее качество.
Выход: reports/m1_close_ykut.md, reports/m1_close_ykut_oos.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m1
from okno.features import build_m1_dataset
from okno.paths import REPORTS
from okno.regions import all_gauges
from okno.report import md_table


def with_pred(train, test):
    model = m1.fit(train, "close")
    out = pd.concat([test.copy(), m1.predict(model, test, "close")], axis=1)
    return out


def score(df, group, label):
    rows = []
    for lead, (lo, hi) in {**m1.LEAD_BUCKETS, "все": (0, 400)}.items():
        s = df[df.lead_days.between(lo, hi)]
        if s.empty:
            continue
        ec, em = (s.cal_target - s.target_day).abs().mean(), (s.q50 - s.target_day).abs().mean()
        rows.append({"группа": group, "модель": label, "за_сколько_дней": lead, "n": len(s),
                     "сезонов": s.hydro_year.nunique(), "постов": s.point_id.nunique(),
                     "MAE_календарь": round(ec, 2), "MAE_М1": round(em, 2),
                     "выигрыш_к_календарю_%": round(100 * (1 - em / ec), 1)})
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = build_m1_dataset("close", codes=all_gauges())
    test = df[(df.is_satellite == 0) & (df.hydro_year >= 2022)].copy()
    print(f"строк: {len(df)}; свежих (гидропосты ≥ 2022): {len(test)}, пост-сезонов {test.groupby(['point_id','hydro_year']).ngroups}")

    base = with_pred(df[df.hydro_year < 2022], test); base["модель"] = "прежняя"
    parts = []
    for y in sorted(test.hydro_year.unique()):
        p = with_pred(df[df.hydro_year < y], test[test.hydro_year == y]); parts.append(p)
    wf = pd.concat(parts); wf["модель"] = "дообученная"

    rows = []
    rows += score(base, "2022–2026, гидропосты", "прежняя (обучена < 2022)")
    rows += score(wf, "2022–2026, гидропосты", "дообученная (walk-forward < y)")
    rows += score(base[base.hydro_year == 2026], "2026 (вне обучения)", "прежняя (обучена < 2022)")
    rows += score(wf[wf.hydro_year == 2026], "2026 (вне обучения)", "дообученная (обучена ≤ 2025)")
    tbl = pd.DataFrame(rows)

    # архив: не испортило ли добавление свежих сезонов прежние сезоны (leave-one-season-out на ≤ 2021)
    old = df[df.hydro_year < 2022]
    bt_old = m1.backtest(old, "close")
    bt_new = m1.backtest(df[df.hydro_year < 2026], "close")
    bt_new = bt_new[bt_new.hydro_year < 2022]
    arch = pd.DataFrame(score(bt_old, "архив ≤ 2021", "без свежих сезонов") + score(bt_new, "архив ≤ 2021", "со свежими сезонами"))

    pd.concat([base, wf]).to_csv(REPORTS / "m1_close_ykut_oos.csv", index=False)
    lines = ["# М1 закрытия: дообучение на официальных датах начала таяния ЯУГМС 2022–2026", "",
             "Данные: `data/external/ykut_reki_phenology.csv` (бюллетени ФГБУ «Якутское УГМС», дословные цитаты; "
             "собрано в ветке dmitrii). Цель — начало разрушения ледостава, прямо названное в бюллетене. "
             "Проверка walk-forward: сезон прогнозирует модель, обученная только на более ранних. "
             "Сезон 2026 в финальную модель не входит.", "",
             "## Свежие сезоны", "", md_table(tbl), "",
             "## Архив: качество на прежних сезонах", "", md_table(arch), "",
             "Пересчёт: `python scripts/eval_m1_close_ykut.py`."]
    (REPORTS / "m1_close_ykut.md").write_text("\n".join(lines), encoding="utf-8")
    print(tbl.to_string(index=False)); print(arch.to_string(index=False))


if __name__ == "__main__":
    main()
