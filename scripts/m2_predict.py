"""
Ежедневное применение М2: P(лёд) для новых снимков и автоматические даты событий.

    python scripts/m2_predict.py

1. Для строк data/s1/index.csv без прогноза считает признаки по маске русла и P(лёд)
   (модель models/m2_ice.cbm, маски data/s1/river_masks.npz) → дописывает data/s1/predictions.csv.
2. По ряду снимков каждой точки находит смену состояния: вскрытие (лёд → вода, март–июль) и
   ледостав (вода → лёд, сентябрь–декабрь). Дата — середина между двумя снимками, окно — сами снимки.
   → data/s1/events_auto.csv. Проверка на 2016–2021: ошибка вскрытия ≈ 9 сут, ледостава ≈ 14 сут
   (на уровне ошибки самих меток и интервала между снимками).
3. Вскрытие у переправы Якутск — автоматический факт для журнала прогнозов
   (data/forecast_log/facts_auto.csv; ручные факты в facts.csv имеют приоритет).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m2
from okno.paths import DATA

PRED = m2.S1 / "predictions.csv"
EVENTS = m2.S1 / "events_auto.csv"
FACTS_AUTO = DATA / "forecast_log" / "facts_auto.csv"


def update_predictions() -> pd.DataFrame:
    idx = pd.read_csv(m2.S1 / "index.csv")
    pred = pd.read_csv(PRED) if PRED.exists() else pd.DataFrame(columns=["item_id", "point"])
    new = idx[~idx.item_id.isin(pred.item_id)]
    if len(new):
        masks, model = m2.load_masks(), m2.load_model()
        feat = m2.features_table(new, masks).dropna(subset=m2.FEATURES)
        if len(feat):
            feat["p_ice"] = model.predict_proba(feat[m2.FEATURES])[:, 1]
            feat["p_ice_source"] = "финальная модель"
            cols = [c for c in pred.columns if c in feat.columns] or list(feat.columns)
            pred = pd.concat([pred, feat[cols]], ignore_index=True)
            pred.round(4).to_csv(PRED, index=False)
        print(f"новых снимков: {len(new)}, классифицировано: {len(feat)}")
    return pred


def detect_events(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.dropna(subset=["p_ice"]).copy()
    p["date"] = pd.to_datetime(p["date"])
    p["hy"] = p.date.dt.year + (p.date.dt.month >= 9).astype(int)
    rules = (("breakup", [3, 4, 5, 6, 7], lambda a, b: a >= 0.5 > b),
             ("freezeup", [9, 10, 11, 12], lambda a, b: a < 0.5 <= b))
    rows = []
    for (pt, hy), g in p.groupby(["point", "hy"]):
        g = g.sort_values("date")
        for kind, months, flip in rules:
            s = g[g.date.dt.month.isin(months)]
            for i in range(1, len(s)):
                a, b = s.p_ice.iloc[i - 1], s.p_ice.iloc[i]
                if flip(a, b):
                    d0, d1 = s.date.iloc[i - 1], s.date.iloc[i]
                    rows.append({"point": pt, "hydro_year": hy, "kind": kind,
                                 "date_est": (d0 + (d1 - d0) / 2).date(), "between_from": d0.date(),
                                 "between_to": d1.date(), "window_days": (d1 - d0).days,
                                 "source": "Sentinel-1, М2"})
                    break
    ev = pd.DataFrame(rows)
    ev.to_csv(EVENTS, index=False)
    return ev


def auto_facts(ev: pd.DataFrame):
    y = ev[(ev.point == "yakutsk_crossing") & (ev.kind == "breakup")]
    f = pd.DataFrame({"point_id": "yakutsk_crossing", "hydro_year": y.hydro_year, "kind": "close",
                      "date": y.date_est, "source": "авто: Sentinel-1 (М2)",
                      "comment": [f"между снимками {a} и {b}" for a, b in zip(y.between_from, y.between_to)]})
    FACTS_AUTO.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(FACTS_AUTO, index=False)


def main():
    pred = update_predictions()
    ev = detect_events(pred)
    auto_facts(ev)
    last = pred.assign(date=pd.to_datetime(pred.date)).sort_values("date").groupby("point").tail(1)
    print(last[["point", "date", "p_ice"]].to_string(index=False))
    print(f"событий найдено: {len(ev)}")


if __name__ == "__main__":
    main()
