"""
Обучение и проверка М2 (лёд/вода по Sentinel-1) и её вклад в прогноз вскрытия.

    python scripts/s1_chips.py        # вырезки (если их ещё нет)
    python scripts/train_m2.py

1. Маски русла, признаки обратного рассеяния по руслу.
2. Проверка «год вне обучения»: для каждого гидрологического года модель учится на остальных;
   метрики — точность и F1 по классам вода/лёд. Для сравнения — логистическая регрессия.
3. Финальная модель → models/m2_ice.cbm; P(лёд) для всех снимков → data/s1/predictions.csv
   (для размеченных — вне обучения, для 2025–2026 — финальной моделью).
4. Вклад в М1: прогноз вскрытия у переправы Якутск (бэктест М1) с отсечением сценариев по последнему
   снимку не старше 14 дней — MAE и покрытие с М2 и без.

Выход: reports/m2_metrics.md, reports/m2_metrics.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m2
from okno.ice import to_season_day
from okno.paths import REPORTS
from okno.report import md_table

LABELS = {"water": 0, "ice": 1}


def loyo(feat: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lab = feat[feat.label.isin(LABELS)].dropna(subset=m2.FEATURES).copy()
    lab["y"] = lab.label.map(LABELS)
    years = [y for y, n in lab.groupby("hydro_year").size().items() if n >= 20]
    oof = pd.Series(np.nan, index=feat.index)
    oof_lr = pd.Series(np.nan, index=feat.index)
    rows = []
    for y in years:
        tr, te = lab[lab.hydro_year != y], lab[lab.hydro_year == y]
        cb = m2.make_model().fit(tr[m2.FEATURES], tr.y)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[m2.FEATURES], tr.y)
        oof.loc[te.index] = cb.predict_proba(te[m2.FEATURES])[:, 1]
        oof_lr.loc[te.index] = lr.predict_proba(te[m2.FEATURES])[:, 1]
        # переходные снимки этого года — той же моделью (она их не видела)
        tz = feat[(feat.hydro_year == y) & (feat.label == "transition")].dropna(subset=m2.FEATURES)
        if len(tz):
            oof.loc[tz.index] = cb.predict_proba(tz[m2.FEATURES])[:, 1]
        rows.append({"год": y, "снимков": len(te)})
    return lab, oof, oof_lr, years


def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    pred = (p >= 0.5).astype(int)
    return {"точность": round(accuracy_score(y, pred), 3), "F1 вода": round(f1_score(y, pred, pos_label=0), 3),
            "F1 лёд": round(f1_score(y, pred, pos_label=1), 3)}


def m1_with_m2(pred: pd.DataFrame) -> pd.DataFrame:
    """Бэктест прогноза вскрытия у переправы Якутск с отсечением по снимкам."""
    bt = pd.read_csv(REPORTS / "m1_backtest_close.csv")
    bt = bt[bt.is_satellite == 1]
    obs = pred[(pred.point == "yakutsk_crossing") & pred.p_ice.notna()].copy()
    obs["date"] = pd.to_datetime(obs.date)
    obs["hy"] = obs.date.dt.year + (obs.date.dt.month >= 9).astype(int)
    obs["day"] = [to_season_day(d, hy) for d, hy in zip(obs.date, obs.hy)]
    rows = []
    for r in bt[bt.hydro_year.isin(obs.hy.unique())].itertuples():
        o = obs[(obs.hy == r.hydro_year) & (obs.day <= r.issue_day) & (obs.day >= r.issue_day - 14)]
        q = (r.q10, r.q50, r.q90)
        used = 0
        if len(o):
            last = o.sort_values("day").iloc[-1]
            q2 = m2.apply_observation(*q, last.day, last.p_ice)
            used = int(q2 != q)
            q = q2
        rows.append({"hydro_year": r.hydro_year, "lead": r.lead_days, "target": r.target_day, "cal": r.cal_target,
                     "q10_0": r.q10, "q50_0": r.q50, "q90_0": r.q90, "q10": q[0], "q50": q[1], "q90": q[2], "used": used})
    d = pd.DataFrame(rows)
    out = []
    for name, (lo, hi) in {"≈7 дней": (4, 10), "≈14 дней": (11, 17), "≈28 дней": (25, 31), "все": (0, 400)}.items():
        s = d[d.lead.between(lo, hi)]
        if s.empty:
            continue
        out.append({"за сколько": name, "прогнозов": len(s), "сезонов": s.hydro_year.nunique(),
                    "со снимком": int(s.used.sum()),
                    "MAE календарь": round((s.cal - s.target).abs().mean(), 2),
                    "MAE М1": round((s.q50_0 - s.target).abs().mean(), 2),
                    "MAE М1+М2": round((s.q50 - s.target).abs().mean(), 2),
                    "ширина 10–90 М1": round((s.q90_0 - s.q10_0).mean(), 1),
                    "ширина 10–90 М1+М2": round((s.q90 - s.q10).mean(), 1),
                    "покрытие М1": round(((s.target >= s.q10_0) & (s.target <= s.q90_0)).mean(), 2),
                    "покрытие М1+М2": round(((s.target >= s.q10) & (s.target <= s.q90)).mean(), 2)})
    return pd.DataFrame(out)


def main():
    idx = pd.read_csv(m2.S1 / "index.csv")
    masks = m2.build_masks(idx)
    print("маски русла, пикселей:", {k: int(v.sum()) for k, v in masks.items()})
    feat = m2.features_table(idx, masks)
    lab, oof, oof_lr, years = loyo(feat)
    y = lab.y
    overall = [{"модель": "CatBoost (М2)", **scores(y, oof.loc[lab.index])},
               {"модель": "логистическая регрессия", **scores(y, oof_lr.loc[lab.index])}]
    per_year = [{"год": yr, "снимков": int((lab.hydro_year == yr).sum()),
                 **scores(lab[lab.hydro_year == yr].y, oof.loc[lab[lab.hydro_year == yr].index])} for yr in years]
    per_point = [{"точка": pt, "снимков": int((lab.point == pt).sum()),
                  **scores(lab[lab.point == pt].y, oof.loc[lab[lab.point == pt].index])} for pt in sorted(lab.point.unique())]
    tr = feat[feat.label == "transition"].index
    trans_p = oof.loc[tr].dropna()

    # финальная модель на всех размеченных
    model = m2.make_model().fit(lab[m2.FEATURES], lab.y)
    model.save_model(str(m2.MODEL))
    feat["p_ice"] = oof
    unk = feat.p_ice.isna() & feat[m2.FEATURES].notna().all(axis=1)
    feat.loc[unk, "p_ice"] = model.predict_proba(feat.loc[unk, m2.FEATURES])[:, 1]
    feat["p_ice_source"] = np.where(unk, "финальная модель", "вне обучения")
    pred = feat[["point", "date", "item_id", "label", "hydro_year", "p_ice", "p_ice_source"] + m2.FEATURES]
    pred.round(4).to_csv(m2.S1 / "predictions.csv", index=False)

    fusion = m1_with_m2(pred)
    pd.DataFrame(overall).to_csv(REPORTS / "m2_metrics.csv", index=False)
    lines = ["# М2: лёд или вода по снимкам Sentinel-1", "",
             f"Проверка «год вне обучения» ({len(years)} лет: {min(years)}–{max(years)}), размеченных снимков "
             f"{len(lab)} (вода {int((y == 0).sum())}, лёд {int((y == 1).sum())}). Переходные (±7 сут от событий) "
             "в обучении не участвуют. Метки — по спутниковым датам Qiu et al., их собственная ошибка ≈ 11 сут.", "",
             "## Качество классификации", "", md_table(pd.DataFrame(overall)), "",
             "### По годам", "", md_table(pd.DataFrame(per_year)), "",
             "### По точкам Лены", "", md_table(pd.DataFrame(per_point)), "",
             f"Переходные снимки (ледоход, шуга): медиана P(лёд) = {trans_p.median():.2f}, "
             f"доля «уверенных» (P<0,2 или P>0,8) — {((trans_p < 0.2) | (trans_p > 0.8)).mean():.0%}.", "",
             "## Вклад в прогноз вскрытия у переправы Якутск", "",
             "Бэктест М1 по сезонам, где есть снимки. Если последний снимок не старше 14 дней уверенно показывает "
             "лёд (P ≥ 0,8), сценарии вскрытия раньше даты снимка отбрасываются; если воду (P ≤ 0,2) — позже.", "",
             md_table(fusion) if len(fusion) else "Нет пересечения снимков с бэктестом.", ""]
    (REPORTS / "m2_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(overall).to_string(index=False))
    print(pd.DataFrame(per_year).to_string(index=False))
    print(fusion.to_string(index=False))


if __name__ == "__main__":
    main()
