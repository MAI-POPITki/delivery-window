"""
Проверка М1 на новых постах PANGAEA.925460 (Якутия — бассейн Колымы, Магаданская обл., Чукотка), сезоны 2013–2018.

    python scripts/new_gauges_test.py      # → reports/new_gauges.md, .csv

Почему это честная проверка. Прежняя М1 (models/m1_open.cbm) обучена на постах Arctic Data Center до 2012 г. и этих
постов не видела. Обновлённая М1 (models/m1_open_updated*.cbm) для сезонов 2010–2018 берётся в референсной версии,
обученной только на сезонах до 2009 г. Цель — дата, когда лёд по замерам достиг толщины для 20 т (оценка между
замерами, не приказ об открытии).

Соперник — календарь (средняя прошлых сезонов поста) с той же поправкой «событие ещё не наступило», что и у М1:
прогноз не может быть раньше даты выпуска. Без этой поправки календарь проигрывает на коротких горизонтах только
потому, что М1 знает, что дорога ещё не открылась. Интервал — бутстрэп по сезонам (5–95%).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import okno.forecast as FC
import okno.snow as SN
from okno.northeast import STATIONS
from okno.paths import REPORTS
from okno.report import md_table

LEADS = (7, 14, 28)
MODES = {"old": "прежняя М1", "updated": "обновлённая М1 (без снега)"}


def run():
    orig = SN.predict
    no_update = lambda *a, **k: {}  # noqa: E731 — прежняя М1: обновлённая модель не подставляется
    rows = []
    for code, (name, region, _) in STATIONS.items():
        pid = f"gauge_{code}"
        try:
            h = FC.history(pid)
        except Exception:  # noqa: BLE001
            continue
        for y, t in h.open_day.dropna().items():
            if y < 2013:
                continue
            for lead in LEADS:
                for mode in MODES:
                    SN.predict = no_update if mode == "old" else orig
                    try:
                        f = FC.forecast(pid, int(y), int(t - lead), "open")
                    finally:
                        SN.predict = orig
                    if f:
                        rows.append(dict(point_id=pid, name=name, region=region, hydro_year=int(y), lead=lead,
                                         mode=mode, target=t, issue=t - lead, q10=f["q10"], q50=f["q50"],
                                         q90=f["q90"], calendar=f["calendar"]))
    return pd.DataFrame(rows)


def main():
    d = run()
    d["calendar_clip"] = np.maximum(d.calendar, d.issue)
    d.to_csv(REPORTS / "new_gauges.csv", index=False)
    rng = np.random.default_rng(0)
    out = []
    for lead in LEADS:
        for mode, label in MODES.items():
            g = d[(d.lead == lead) & (d["mode"] == mode)]
            em, ec = (g.q50 - g.target).abs(), (g.calendar_clip - g.target).abs()
            ys, bs = g.hydro_year.unique(), []
            for _ in range(2000):
                gg = pd.concat([g[g.hydro_year == v] for v in rng.choice(ys, len(ys))])
                bs.append(100 * (1 - (gg.q50 - gg.target).abs().mean() / (gg.calendar_clip - gg.target).abs().mean()))
            lo, hi = np.percentile(bs, [5, 95])
            out.append({"за сколько дней": lead, "модель": label, "прогнозов": len(g), "постов": g.point_id.nunique(),
                        "сезонов": len(ys), "MAE календарь, сут": round(ec.mean(), 1), "MAE М1, сут": round(em.mean(), 1),
                        "выигрыш к календарю, %": round(100 * (1 - em.mean() / ec.mean())),
                        "интервал 5–95%": f"{lo:+.0f} … {hi:+.0f}",
                        "покрытие 10–90%": round(((g.target >= g.q10) & (g.target <= g.q90)).mean(), 2)})
    res = pd.DataFrame(out)
    reg = (d[d["mode"] == "updated"].groupby(["region", "lead"])
           .apply(lambda g: pd.Series({"прогнозов": len(g),
                                       "выигрыш к календарю, %": round(100 * (1 - (g.q50 - g.target).abs().mean()
                                                                              / (g.calendar_clip - g.target).abs().mean()))}),
                  include_groups=False).reset_index().rename(columns={"region": "регион", "lead": "за сколько дней"}))
    lines = ["# М1 на новых постах после 2012 г. (PANGAEA.925460)", "",
             "19 постов Якутии (бассейн Колымы), Магаданской обл. и Чукотки, сезоны 2013–2018 — вне обучения обеих моделей. "
             "Цель — дата, когда лёд по замерам достиг толщины для 20 т. Соперник — календарь поста с той же поправкой "
             "«событие ещё не наступило». Метод — в шапке `scripts/new_gauges_test.py`.", "",
             md_table(res), "", "**Обновлённая М1 по регионам:**", "", md_table(reg), "",
             "**Как читать.** За 28 сут обе модели точнее календаря примерно на треть. За 14 сут прежняя М1 на новых постах "
             "почти не лучше календаря, обновлённая (обучена с прошлыми сезонами этих постов) — на четверть точнее. "
             "За 7 сут обе не лучше календаря с поправкой. Поэтому на новых постах сервис использует обновлённую М1, "
             "на прежних — прежнюю (на закрытии прежних постов обновлённая хуже календаря, см. reports/snow_model.md). "
             "Толщина снега на этих постах ошибку не снижает. Шесть сезонов — мало; живая проверка — журнал прогнозов.", ""]
    (REPORTS / "new_gauges.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))
    print(reg.to_string(index=False))


if __name__ == "__main__":
    main()
