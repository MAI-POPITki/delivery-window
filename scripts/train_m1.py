"""
Обучение и бэктест М1.

    python scripts/train_m1.py

Выход:
    reports/m1_metrics.csv / .md        — MAE (сут) за 7/14/28 дней до события: календарь, физика, М1;
                                          доля фактов в диапазоне 10–90% (должна быть ≈ 0.8)
    reports/m1_backtest_{open,close}.csv — прогнозы по каждой строке (только на невиданных сезонах)
    models/m1_{open,close}.cbm           — модели, обученные на всех сезонах
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from okno import m1
from okno.features import build_m1_dataset
from okno.northeast import STATIONS
from okno.paths import MODELS, PROCESSED, REPORTS
from okno.regions import all_gauges, all_training_gauges, region_of
from okno.report import md_table

_NE_REGIONS = {f"gauge_{c}": reg for c, (_, reg, _) in STATIONS.items()}


def train_lena_model() -> pd.DataFrame:
    """Отдельная модель закрытия для точек бассейна Лены (притоки, спутниковые даты Qiu 2001–2024).
    Обучается на гидропостах + переправе + точках Лены, применяется ТОЛЬКО к точкам Лены: эксперимент
    (reports/lena_experiment.md) показал, что старым точкам такие данные не помогают."""
    df = build_m1_dataset("close", codes=all_gauges(), lena=True)
    df = df[df["hydro_year"] < 2026].copy()
    if not df.point_id.str.startswith("lena_").any():
        return pd.DataFrame()
    bt = m1.backtest(df, "close")
    lena = bt[bt.point_id.str.startswith("lena_")].copy()
    lena["region"] = "lena_basin"
    rows = []
    for lead, (lo, hi) in {**m1.LEAD_BUCKETS, "все": (0, 400)}.items():
        s = lena[lena.lead_days.between(lo, hi)]
        if s.empty:
            continue
        e_cal, e_m1 = (s.cal_target - s.target_day).abs().mean(), (s.q50 - s.target_day).abs().mean()
        rows.append({"kind": "close", "группа": "бассейн Лены (спутник, отдельная модель)", "за_сколько_дней": lead,
                     "n": len(s), "сезонов": s.hydro_year.nunique(), "MAE_календарь": round(e_cal, 1),
                     "MAE_физика": round((s.phys_pred - s.target_day).abs().mean(), 1), "MAE_М1": round(e_m1, 1),
                     "выигрыш_к_календарю_%": round(100 * (1 - e_m1 / e_cal), 1),
                     "покрытие_10_90": round(((s.target_day >= s.q10) & (s.target_day <= s.q90)).mean(), 2),
                     "ширина_10_90_сут": round((s.q90 - s.q10).mean(), 1)})
    model = m1.fit(df, "close")
    model.save_model(str(MODELS / "m1_close_lena.cbm"))
    score = np.maximum(bt["q10_raw"] - bt["target_day"], bt["target_day"] - bt["q90_raw"])
    (MODELS / "m1_close_lena_conformal.json").write_text(
        json.dumps({"correction_days": m1.conformal_correction(score), "coverage": m1.TARGET_COVERAGE}))
    print(f"модель для бассейна Лены: точек {lena.point_id.nunique()}, строк {len(lena)}")
    return pd.DataFrame(rows)


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    all_metrics = []
    for kind in ("open", "close"):
        # одна общая модель на все подтверждённые регионы (для open — ещё и посты PANGAEA Магадана и Чукотки)
        codes = all_training_gauges() if kind == "open" else all_gauges()
        df = build_m1_dataset(kind, codes=codes)
        if kind == "close":
            # Сезон 2026 остаётся вне обучения М1 (строгий OOS-контроль переправы Якутск и сводок ЯУГМС 2026).
            df = df[df["hydro_year"] < 2026].copy()
        df["region"] = df.point_id.map(lambda p: region_of(p) or _NE_REGIONS.get(p))
        df.to_csv(PROCESSED / f"m1_dataset_{kind}.csv", index=False)
        print(f"{kind}: {len(df)} строк, точек {df.point_id.nunique()}, сезонов {df.hydro_year.nunique()}")
        bt = m1.backtest(df, kind)
        bt.to_csv(REPORTS / f"m1_backtest_{kind}.csv", index=False)
        met = m1.metrics(bt)
        all_metrics.append(met)
        print(met.to_string(index=False))
        model = m1.fit(df, kind)
        model.save_model(str(MODELS / f"m1_{kind}.cbm"))
        score = np.maximum(bt["q10_raw"] - bt["target_day"], bt["target_day"] - bt["q90_raw"])
        (MODELS / f"m1_{kind}_conformal.json").write_text(
            json.dumps({"correction_days": m1.conformal_correction(score), "coverage": m1.TARGET_COVERAGE}))
        imp = pd.Series(model.get_feature_importance(), index=model.feature_names_).sort_values(ascending=False)
        imp.round(1).to_csv(REPORTS / f"m1_importance_{kind}.csv", header=["importance"])
    all_metrics.append(train_lena_model())
    met = pd.concat(all_metrics, ignore_index=True)
    met.to_csv(REPORTS / "m1_metrics.csv", index=False)
    names = {"open": "Открытие: лёд держит 20 т (гидропосты, замеры толщины)",
             "close": "Закрытие: начало разрушения ледостава"}
    lines = ["# М1: точность прогноза окна (бэктест leave-one-season-out)", "",
             "MAE — средняя ошибка в сутках. «Календарь» — средняя дата за прошлые 10 сезонов "
             "(так планируют сейчас). «Физика» — закон Стефана / сумма оттепельных градусо-дней "
             "с климатологией вместо будущей погоды. Покрытие 10–90% должно быть около 0.8.", ""]
    for kind in ("open", "close"):
        lines += [f"## {names[kind]}", "", md_table(met[met.kind == kind].drop(columns="kind")), ""]
    (REPORTS / "m1_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("готово:", REPORTS / "m1_metrics.md")


if __name__ == "__main__":
    main()
