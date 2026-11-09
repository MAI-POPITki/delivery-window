"""
Журнал прогнозов: честная проверка «вперёд», а не на прошлом.

    python scripts/forecast_journal.py          # записать сегодняшние прогнозы и сверить с фактами

1. Записывает в data/forecast_log/forecasts.csv прогноз открытия и закрытия на сегодня для всех путей
   всех регионов текущего сезона (повторный запуск в тот же день перезаписывает строки этого дня).
2. Сверяет журнал с фактами из data/forecast_log/facts.csv — их вносит команда по мере наступления
   событий (приказы об открытии/закрытии, новости, замеры). Колонки:
       point_id, hydro_year, kind (open|close), date (ГГГГ-ММ-ДД), source, comment
   kind=open — дата, когда лёд выдержал 20 т (или дата приказа об открытии, если сверяем операционную цель);
   kind=close — начало разрушения ледостава (или дата приказа о закрытии).
3. Пишет reports/forecast_journal.md: сколько прогнозов записано, свежие прогнозы, точность на фактах.
"""

import datetime as dt
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import forecast as FC
from okno.ice import hydro_year, to_season_day
from okno.paths import DATA, MODELS, REPORTS
from okno.regions import REGIONS, label, region_of
from okno.report import md_table

LOG = DATA / "forecast_log"
FORECASTS = LOG / "forecasts.csv"
FACTS = LOG / "facts.csv"
FACT_COLS = ["point_id", "hydro_year", "kind", "date", "source", "comment"]
BUCKETS = {"≈7 дней": (4, 10), "≈14 дней": (11, 17), "≈28 дней": (25, 31), "все": (0, 400)}


def model_tag(kind,filename=None):
    p = MODELS / (Path(filename).name if filename else f"m1_{kind}.cbm")
    return hashlib.sha1(p.read_bytes()).hexdigest()[:8] if p.exists() else ""


def log_today(today: dt.date):
    hy = int(hydro_year(pd.Series([pd.Timestamp(today)])).iloc[0])
    d = int(to_season_day(today, hy))
    rows = []
    from okno.regions import points_of
    for key, r in REGIONS.items():
        for pid in points_of(key):
            if not FC._weather_path(pid).exists():
                continue
            for kind in ("open", "close"):
                try:
                    q = FC.forecast(pid, hy, d, kind)  # без снежной поправки: она не дала выигрыша (reports/snow_model.md)
                except Exception as e:  # noqa: BLE001
                    print(f"  {pid} {kind}: ошибка {e}")
                    continue
                if not q:
                    continue
                rows.append({"run_date": today.isoformat(), "point_id": pid, "region": key, "hydro_year": hy,
                             "kind": kind, "issue_date": str(FC.from_season_day(q["issue_day"], hy).date()),
                             "q10_date": str(q["q10_date"]), "q50_date": str(q["q50_date"]),
                             "q90_date": str(q["q90_date"]), "calendar_date": str(q["calendar_date"]),
                             "physics_date": str(q["physics_date"]),
                             "forecast_days_used": q.get("forecast_days_used", 0),
                             'model_file':q.get('prediction_model',''),'snow_applied':q.get('snow_model_applied',False),
                             "model": model_tag(kind,q.get('prediction_model')),
                             # точные дни сезона (дробные): по ним дашборд рисует карту без пересчёта
                             **{f"{k}_day": round(float(q[k]), 3) for k in ("q10", "q50", "q90", "calendar", "physics")}})
    new = pd.DataFrame(rows)
    LOG.mkdir(parents=True, exist_ok=True)
    if FORECASTS.exists():
        old = pd.read_csv(FORECASTS, dtype=str)
        old = old[old.run_date != today.isoformat()]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(FORECASTS, index=False)
    if not FACTS.exists():
        pd.DataFrame(columns=FACT_COLS).to_csv(FACTS, index=False)
    print(f"записано прогнозов за {today}: {len(rows)}; всего в журнале: {len(new)}")
    return new


def verify(log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    facts = pd.read_csv(FACTS, dtype=str) if FACTS.exists() else pd.DataFrame(columns=FACT_COLS)
    auto = LOG / "facts_auto.csv"  # даты со спутника (М2, scripts/m2_predict.py); ручные факты важнее
    if auto.exists():
        a = pd.read_csv(auto, dtype=str)
        key = ["point_id", "hydro_year", "kind"]
        a = a[~a.set_index(key).index.isin(facts.set_index(key).index)]
        facts = pd.concat([facts, a], ignore_index=True)
    if facts.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = log.merge(facts[["point_id", "hydro_year", "kind", "date"]], on=["point_id", "hydro_year", "kind"])
    if m.empty:
        return pd.DataFrame(), m
    fact = pd.to_datetime(m["date"])
    issue = pd.to_datetime(m["issue_date"])
    m = m[issue < fact].copy()  # честно: только прогнозы, выпущенные до события
    fact, issue = pd.to_datetime(m["date"]), pd.to_datetime(m["issue_date"])
    m["lead_days"] = (fact - issue).dt.days
    for c in ("q10", "q50", "q90", "calendar"):
        m[f"{c}_err"] = (pd.to_datetime(m[f"{c}_date"]) - fact).dt.days
    rows = []
    for kind, g in m.groupby("kind"):
        for name, (lo, hi) in BUCKETS.items():
            s = g[g.lead_days.between(lo, hi)]
            if s.empty:
                continue
            mae_m1, mae_cal = s.q50_err.abs().mean(), s.calendar_err.abs().mean()
            rows.append({"событие": "открытие" if kind == "open" else "закрытие", "за сколько": name,
                         "прогнозов": len(s), "путей": s.point_id.nunique(),
                         "MAE календарь": round(mae_cal, 1), "MAE М1": round(mae_m1, 1),
                         "выигрыш, %": round(100 * (1 - mae_m1 / mae_cal), 1) if mae_cal else np.nan,
                         "покрытие 10–90": round(((s.q10_err <= 0) & (s.q90_err >= 0)).mean(), 2)})
    return pd.DataFrame(rows), m


def report(log: pd.DataFrame, met: pd.DataFrame):
    last = log[log.run_date == log.run_date.max()].copy()
    last["путь"] = last.point_id.map(label)
    last["регион"] = last.point_id.map(lambda p: REGIONS[region_of(p)]["name"])
    show = last.pivot_table(index=["регион", "путь"], columns="kind", values="q50_date", aggfunc="first").reset_index()
    show = show.rename(columns={"open": "лёд держит 20 т (медиана)", "close": "начало таяния (медиана)"})
    lines = ["# Журнал прогнозов", "",
             "Каждый день автообновление записывает прогноз на сегодня по всем путям "
             "(`data/forecast_log/forecasts.csv`). Факты команда вносит в `data/forecast_log/facts.csv`; "
             "сверка учитывает только прогнозы, выпущенные **до** события.", "",
             f"Записей в журнале: **{len(log)}**, дней: **{log.run_date.nunique()}** "
             f"(с {log.run_date.min()} по {log.run_date.max()}).", "",
             "## Точность на фактах", ""]
    lines += [md_table(met), ""] if len(met) else [
        "Фактов пока нет — сверка начнётся, когда наступят события сезона и их даты внесут в `facts.csv` "
        "(открытие — ноябрь–январь, закрытие — апрель–май).", ""]
    lines += [f"## Последние прогнозы ({log.run_date.max()})", "", md_table(show), ""]
    (REPORTS / "forecast_journal.md").write_text("\n".join(lines), encoding="utf-8")
    print("готово:", REPORTS / "forecast_journal.md")


def main():
    log = log_today(dt.date.today())
    met, _ = verify(log)
    report(log, met)


if __name__ == "__main__":
    main()
