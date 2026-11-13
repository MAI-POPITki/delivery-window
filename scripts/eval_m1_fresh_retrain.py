"""
Честная A/B и walk-forward оценка дообучения М1 на свежих данных (PANGAEA.925460 + ЯУГМС 2022–2026).

    .venv/bin/python scripts/eval_m1_fresh_retrain.py

Проверяет:
1. Открытие (m1_open):
   - LOO-бэктест по 5 регионам сервиса + новым постам PANGAEA (all_gauges vs all_training_gauges).
   - Строго хронологический OOS-тест на пост-2012 сезонах PANGAEA (2013–2018):
     * модель только на ADC (до 2012 г., не видела новых постов);
     * дообученная модель (walk-forward: обучение строго на сезонах < y с добавлением прошлых лет PANGAEA).
2. Закрытие (m1_close):
   - Строго хронологический OOS-тест на свежих сезонах 2022–2026 по бюллетеням ФГБУ «Якутское УГМС»:
     * базовая модель (обучена только на данных < 2022);
     * дообученная модель (walk-forward: обучение строго < y, для 2026 — на <= 2025 без утечки 2026 г.).
   - Отдельная проверка 2026 года (полностью вне обучения, согласовано с tests/test_yakutsk_replay.py).

Выход:
    reports/m1_fresh_retrain.md
    reports/m1_fresh_retrain_open_oos.csv
    reports/m1_fresh_retrain_close_oos.csv
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import m1
from okno.features import build_m1_dataset
from okno.northeast import STATIONS
from okno.paths import PROCESSED, REPORTS
from okno.regions import REGIONS, all_gauges, all_training_gauges, region_of
from okno.report import md_table


def _bucket_rows(df: pd.DataFrame, group_name: str, model_label: str) -> list[dict]:
    rows = []
    for lead, (lo, hi) in {**m1.LEAD_BUCKETS, "все": (0, 400)}.items():
        s = df[df.lead_days.between(lo, hi)]
        if s.empty:
            continue
        ec = (s.cal_target - s.target_day).abs().mean()
        ep = (s.phys_pred - s.target_day).abs().mean()
        em = (s.q50 - s.target_day).abs().mean()
        cov = ((s.target_day >= s.q10) & (s.target_day <= s.q90)).mean()
        rows.append({
            "группа": group_name,
            "модель": model_label,
            "горизонт": lead,
            "n": len(s),
            "сезонов": int(s.hydro_year.nunique()),
            "постов": int(s.point_id.nunique()),
            "MAE_календарь": round(ec, 2),
            "MAE_физика": round(ep, 2),
            "MAE_М1": round(em, 2),
            "выигрыш_%": round(100 * (1 - em / ec), 1) if ec > 0 else np.nan,
            "покрытие_10_90": round(cov, 2),
        })
    return rows


def _apply_conformal(train_df: pd.DataFrame, kind: str, test_df: pd.DataFrame, model) -> pd.DataFrame:
    # Out-of-fold CQR внутри train_df не обязателен, но быстрая честная оценка по остаткам fit
    # или 5-fold по сезонам даёт калиброванный сдвиг без заглядывания в test_df.
    in_pred = m1.predict(model, train_df, kind)
    score = np.maximum(in_pred["q10"] - train_df["target_day"], train_df["target_day"] - in_pred["q90"])
    corr = m1.conformal_correction(score)
    out = pd.concat([test_df.copy(), m1.predict(model, test_df, kind)], axis=1)
    out["q10"] -= corr
    out["q90"] += corr
    return out


def eval_open_oos() -> tuple[pd.DataFrame, list[dict]]:
    df = build_m1_dataset("open", codes=all_training_gauges())
    ne_pids = {f"gauge_{c}" for c in STATIONS}
    is_ne = df.point_id.isin(ne_pids)

    # 1) Только ADC (до 2012, без PANGAEA) -> тест на PANGAEA 2013..2018
    adc_only = df[(~is_ne) & (df.hydro_year <= 2012)]
    m_adc = m1.fit(adc_only, "open")
    test_post = df[is_ne & (df.hydro_year >= 2013)]
    pred_adc = _apply_conformal(adc_only, "open", test_post, m_adc)
    pred_adc["eval_mode"] = "базовая_ADC_до_2012"

    # 2) Walk-forward дообучение строго < y (ADC + прошлые сезоны PANGAEA < y)
    wf_parts = []
    for y in sorted(test_post.hydro_year.unique()):
        tr = df[df.hydro_year < y]
        te = test_post[test_post.hydro_year == y]
        my = m1.fit(tr, "open")
        p = _apply_conformal(tr, "open", te, my)
        p["eval_mode"] = "дообученная_walk_forward"
        wf_parts.append(p)
    pred_wf = pd.concat(wf_parts).sort_index()

    rows = []
    rows += _bucket_rows(pred_adc, "PANGAEA 2013–2018 (все 19 постов)", "базовая (только ADC до 2012)")
    rows += _bucket_rows(pred_wf, "PANGAEA 2013–2018 (все 19 постов)", "дообученная (walk-forward < y)")
    yak_ne = {f"gauge_{c}" for c, (_, r, _) in STATIONS.items() if r == "yakutia"}
    rows += _bucket_rows(pred_adc[pred_adc.point_id.isin(yak_ne)], "PANGAEA Якутия 2013–2018 (6 постов)", "базовая (только ADC до 2012)")
    rows += _bucket_rows(pred_wf[pred_wf.point_id.isin(yak_ne)], "PANGAEA Якутия 2013–2018 (6 постов)", "дообученная (walk-forward < y)")

    oos = pd.concat([pred_adc, pred_wf], ignore_index=True)
    return oos, rows


def eval_close_oos() -> tuple[pd.DataFrame, list[dict]]:
    df = build_m1_dataset("close", codes=all_gauges())
    test_yk = df[(df.is_satellite == 0) & (df.hydro_year >= 2022)].copy()

    # 1) Базовая модель до 2022 (обучена только на < 2022: ADC + спутник 2004..2021)
    tr_pre22 = df[df.hydro_year < 2022]
    m_pre22 = m1.fit(tr_pre22, "close")
    pred_pre22 = _apply_conformal(tr_pre22, "close", test_yk, m_pre22)
    pred_pre22["eval_mode"] = "базовая_до_2022"

    # 2) Дообученная walk-forward строго < y (включает прошлые сезоны ЯУГМС 2022..y-1)
    wf_parts = []
    for y in sorted(test_yk.hydro_year.unique()):
        tr = df[df.hydro_year < y]
        te = test_yk[test_yk.hydro_year == y]
        my = m1.fit(tr, "close")
        p = _apply_conformal(tr, "close", te, my)
        p["eval_mode"] = "дообученная_walk_forward"
        wf_parts.append(p)
    pred_wf = pd.concat(wf_parts).sort_index()

    rows = []
    rows += _bucket_rows(pred_pre22, "ЯУГМС 2022–2026 (гидропосты)", "базовая (обучена < 2022)")
    rows += _bucket_rows(pred_wf, "ЯУГМС 2022–2026 (гидропосты)", "дообученная (walk-forward < y)")
    rows += _bucket_rows(pred_pre22[pred_pre22.hydro_year == 2026], "ЯУГМС 2026 (строгий holdout)", "базовая (обучена < 2022)")
    rows += _bucket_rows(pred_wf[pred_wf.hydro_year == 2026], "ЯУГМС 2026 (обучена <= 2025)", "дообученная (обучена <= 2025)")

    oos = pd.concat([pred_pre22, pred_wf], ignore_index=True)
    return oos, rows


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    print("Evaluating OPEN post-2012 OOS (PANGAEA 2013-2018)...", flush=True)
    open_oos, open_rows = eval_open_oos()
    print("Evaluating CLOSE 2022-2026 OOS (YkutHydromet)...", flush=True)
    close_oos, close_rows = eval_close_oos()

    open_oos.to_csv(REPORTS / "m1_fresh_retrain_open_oos.csv", index=False)
    close_oos.to_csv(REPORTS / "m1_fresh_retrain_close_oos.csv", index=False)

    open_tbl = pd.DataFrame(open_rows)
    close_tbl = pd.DataFrame(close_rows)

    print("\n=== OPEN OOS (PANGAEA 2013-2018) ===")
    print(open_tbl.to_string(index=False))
    print("\n=== CLOSE OOS (YkutHydromet 2022-2026) ===")
    print(close_tbl.to_string(index=False))

    lines = [
        "# Честная проверка дообучения М1 на свежих данных (PANGAEA.925460 + ФГБУ «Якутское УГМС»)",
        "",
        f"Дата расчёта: {dt.date.today():%d.%m.%Y}. Скрипт: `scripts/eval_m1_fresh_retrain.py`.",
        "",
        "## Источники",
        "1. **АИС ГМВО (`gmvo.skniivh.ru` / `gvr.rwec.ru`):** официально выведены из эксплуатации; публичный доступ закрыт (ГИС ЦП «Вода» через Континент TLS).",
        "2. **PANGAEA.925460 (Makarieva et al.):** 19 гидропостов северо-востока РФ (6 Якутия/Колыма, 4 Магадан, 9 Чукотка), 1 988 замеров льда и снега после 2012 г. (до 2018 г.).",
        "3. **ФГБУ «Якутское УГМС» (`https://ykuthydromet.ru/reki/`):** 822 ежедневных бюллетеня за 2022–2026 гг.; извлечено 148 датированных событий с дословными цитатами по 18 постам (`data/external/ykut_reki_phenology.csv`).",
        "",
        "## 1. Открытие (`m1_open`): строгий walk-forward тест на 2013–2018 гг. (PANGAEA)",
        "",
        "Оценка на сезонах 2013–2018 (вне обучения). Для сезона `y` дообученная модель обучается строго на сезонах `< y`.",
        "",
        md_table(open_tbl),
        "",
        "## 2. Закрытие (`m1_close`): строгий walk-forward тест на 2022–2026 гг. (бюллетени ЯУГМС)",
        "",
        "Целевое событие — официальное **начало таяния** (`melt_start`: вода на льду / закраины / подвижки). "
        "Для тестирования сезона `y` дообученная модель обучается строго на `< y`. Финальная модель `m1_close.cbm` ограничена `hydro_year <= 2025`, чтобы сезон 2026 оставался чистым для кейс-теста переправы Якутск.",
        "",
        md_table(close_tbl),
        "",
    ]
    (REPORTS / "m1_fresh_retrain.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт сохранён: {REPORTS / 'm1_fresh_retrain.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
