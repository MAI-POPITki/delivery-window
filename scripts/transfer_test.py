"""
Проверка переносимости на другие регионы.

    python scripts/transfer_test.py      (нужна погода: download_data.py --gauges-other)

Модель М1 обучается ТОЛЬКО на якутских постах и применяется к постам других регионов
(ЯНАО, Красноярский край, Иркутская и Архангельская обл., Коми), которых она не видела.
Два сценария нового региона:
  A — есть гидропост с замерами толщины (коэффициент Стефана подбирается локально);
  B — замеров толщины нет (берём коэффициент, подобранный по Якутии).
В обоих сценариях нужна только история дат ледовых явлений для «календаря».

Выход: reports/transfer_stefan.csv, reports/transfer_m1.csv, reports/transfer.md
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m1
from okno.calib import calibrate_all, fit_alpha
from okno.features import build_m1_dataset, load_alpha
from okno.paths import PROCESSED, REPORTS, WEATHER
from okno.report import md_table

REGION_RU = {"Yamalo-Nenetskiy Okrug": "ЯНАО", "Krasnoyarskiy Krai": "Красноярский край",
             "Irkutskaya Obl": "Иркутская обл.", "Arkhangelskaya Obl": "Архангельская обл.",
             "Respublika Komi": "Коми", "Respublika Sakha (Yakutiya)": "Якутия (обучение)"}


def stefan_transfer(gauges, other):
    alpha_yak = load_alpha()
    _, per, pairs = calibrate_all(other)
    rows = []
    for g, p in pairs.groupby("gauge_code"):
        x, y = np.sqrt(p.fdd.clip(lower=0)), p.ice_cm
        own = fit_alpha(p)
        pred_yak = alpha_yak * x
        ss_tot = ((y - y.mean()) ** 2).sum()
        rows.append({"gauge_code": g, "регион": REGION_RU.get(gauges.loc[g, "region"], gauges.loc[g, "region"]),
                     "пост": gauges.loc[g, "name"], "замеров": own["n"],
                     "alpha_свой": round(own["alpha"], 2), "R2_свой": round(own["r2"], 2),
                     "MAE_см_свой": round(own["mae_cm"], 1),
                     "R2_alpha_Якутии": round(1 - ((y - pred_yak) ** 2).sum() / ss_tot, 2),
                     "MAE_см_alpha_Якутии": round(float((y - pred_yak).abs().mean()), 1)})
    return pd.DataFrame(rows)


def m1_transfer(gauges, other):
    out = []
    for kind in ("open", "close"):
        # обучение строго на Якутии (по умолчанию build_m1_dataset берёт якутские посты + спутник)
        train = build_m1_dataset(kind)
        model = m1.fit(train, kind)
        bt = m1.backtest(train, kind)  # поправка диапазона — тоже только по якутским сезонам
        score = np.maximum(bt["q10_raw"] - bt["target_day"], bt["target_day"] - bt["q90_raw"])
        corr = m1.conformal_correction(score)
        scenarios = [("A: есть замеры толщины", True)] + ([("B: замеров толщины нет", False)] if kind == "open" else [])
        for scen, local in scenarios:
            df = build_m1_dataset(kind, codes=other, include_sat=False, local_alpha=local)
            if df.empty:
                continue
            q = m1.predict(model, df, kind)
            q["q10"] -= corr
            q["q90"] += corr
            df = pd.concat([df, q], axis=1)
            df["регион"] = df.point_id.str.replace("gauge_", "").astype(int).map(
                lambda c: REGION_RU.get(gauges.loc[c, "region"], gauges.loc[c, "region"]))
            for (reg), g in list(df.groupby("регион")) + [("все регионы", df)]:
                s = g[g.lead_days.between(11, 17)]
                if s.empty:
                    continue
                e_cal = (s.cal_target - s.target_day).abs().mean()
                e_m1 = (s.q50 - s.target_day).abs().mean()
                out.append({"событие": "открытие (лёд держит 20 т)" if kind == "open" else "закрытие (начало таяния)",
                            "сценарий": scen, "регион": reg, "постов": g.point_id.nunique(),
                            "сезонов": g.hydro_year.nunique(),
                            "MAE_календарь": round(e_cal, 1),
                            "MAE_физика": round((s.phys_pred - s.target_day).abs().mean(), 1),
                            "MAE_М1": round(e_m1, 1), "выигрыш_%": round(100 * (1 - e_m1 / e_cal), 1),
                            "покрытие_10_90": round(((s.target_day >= s.q10) & (s.target_day <= s.q90)).mean(), 2)})
    return pd.DataFrame(out)


def main():
    gauges = pd.read_csv(PROCESSED / "gauges.csv").set_index("gauge_code")
    other = [c for c in gauges.index[~gauges.yakutia]
             if (WEATHER / "gauges" / f"gauge_{c}.csv").exists()]
    print(f"постов вне Якутии с погодой: {len(other)}")
    st = stefan_transfer(gauges, other)
    st.to_csv(REPORTS / "transfer_stefan.csv", index=False)
    print(st.to_string(index=False))
    mt = m1_transfer(gauges, other)
    mt.to_csv(REPORTS / "transfer_m1.csv", index=False)
    print(mt.to_string(index=False))
    lines = ["# Проверка переносимости на другие регионы", "",
             "М1 обучена только на Якутии и применена к постам регионов, которых она не видела. "
             "Ошибки — средние, в сутках, за 14 дней до события. «Календарь» — средняя дата за прошлые "
             "10 сезонов на этом же посту.", "",
             "## Модель М1 в чужих регионах", "", md_table(mt), "",
             "## Физика льда (закон Стефана) в чужих регионах", "",
             "«свой» — коэффициент подобран по местным замерам; «alpha Якутии» — взят из Якутии без подгонки.", "",
             md_table(st), ""]
    (REPORTS / "transfer.md").write_text("\n".join(lines), encoding="utf-8")
    print("готово:", REPORTS / "transfer.md")


if __name__ == "__main__":
    main()
