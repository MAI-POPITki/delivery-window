"""
R1 (VALIDATION_PLAN, H2): предсказывает ли М1 ОФИЦИАЛЬНЫЕ даты открытия и закрытия зимников лучше простых правил.

    python scripts/official_dates_test.py      # → reports/official_dates.md, .csv

Данные: data/external/road_events.csv — официальные даты из СМИ со ссылками (5 регионов, 2018–2026),
у каждой строки point_id ближайшей точки прогноза. Цель — фактическая дата (open / close).

Честность:
* Прогноз выпускается в фиксированные даты (открытие: 1 и 15 ноября, 1 декабря; закрытие: 1 и 15 марта,
  1 апреля) и только если событие ещё не наступило.
* М1 по гидропостам обучена на сезонах до 2012 г. — все сезоны 2018–2026 для неё новые.
  Переправа Якутск (закрытие) обучена на спутнике до 2024 г. → в проверку берём только 2025–2026.
* М1 прогнозирует физическую дату (лёд держит 20 т; начало таяния/вскрытие), а не приказ. Перевод в
  официальную дату — сдвиг, оценённый ТОЛЬКО по прошлым сезонам (среднее «официальная − М1» по всем
  дорогам того же вида события за предыдущие годы). Без прошлых лет прогноз не делается.

Соперники (все — только по прошлому):
* «как в прошлом году» — дата той же дороги и вида события в предыдущий сезон;
* «календарь дороги» — средняя дата той же дороги за все прошлые сезоны;
* «объявленный план» — announced_date, если он опубликован не позже даты выпуска прогноза;
* «календарь поста + сдвиг» — климатическая дата точки прогноза с тем же сдвигом (контроль: что даёт
  именно погода текущего сезона, а не перевод физики в приказ).
Сравнение — на одних и тех же строках (где есть оба прогноза); 90%-интервал — бутстрэп по сезонам.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import forecast as FC
from okno.ice import to_season_day
from okno.paths import DATA, REPORTS
from okno.report import md_table

ISSUE = {"open": {"1 ноября": 61, "15 ноября": 75, "1 декабря": 91},
         "close": {"1 марта": 181, "15 марта": 195, "1 апреля": 212}}


def load_events():
    e = pd.read_csv(DATA / "external" / "road_events.csv")
    e["point_id"] = e.comment.str.extract(r"point_id=([a-z_0-9]+)")
    e = e[e.kind.isin(["open", "close"]) & e.actual_date.notna() & e.point_id.notna()].copy()
    e["actual_day"] = [to_season_day(d, y) for d, y in zip(e.actual_date, e.hydro_year)]
    e["ann_day"] = [to_season_day(d, y) if isinstance(d, str) else np.nan for d, y in zip(e.announced_date, e.hydro_year)]
    e["ann_pub_day"] = [to_season_day(d, y) if isinstance(d, str) else np.nan
                        for d, y in zip(e.announced_published, e.hydro_year)]
    # одна строка на дорогу × сезон × вид (если дубли — первая)
    return e.drop_duplicates(["road", "hydro_year", "kind"]).reset_index(drop=True)


def m1_rows(e):
    rows = []
    for r in e.itertuples():
        if r.point_id == "yakutsk_crossing" and r.kind == "close" and r.hydro_year <= 2024:
            continue  # эти сезоны были в обучении
        for name, d in ISSUE[r.kind].items():
            if d >= r.actual_day:
                continue
            f = FC.forecast(r.point_id, int(r.hydro_year), d, r.kind)
            if not f or "error" in f or not np.isfinite(f.get("q50", np.nan)):
                continue
            rows.append({"road": r.road, "region": r.region, "point_id": r.point_id, "hydro_year": r.hydro_year,
                         "kind": r.kind, "issue": name, "issue_day": d, "actual": r.actual_day,
                         "m1": float(f["q50"]), "m1_q10": float(f["q10"]), "m1_q90": float(f["q90"]),
                         "post_cal": float(f["calendar"]),
                         "announced": r.ann_day if (np.isfinite(r.ann_pub_day) and r.ann_pub_day <= d) else np.nan})
    return pd.DataFrame(rows)


def add_baselines(df, e):
    """Сдвиги и календари дорог — только по прошлым сезонам."""
    out = []
    for r in df.itertuples():
        past = df[(df.kind == r.kind) & (df.issue == r.issue) & (df.hydro_year < r.hydro_year)]
        shift_m1 = (past.actual - past.m1).mean() if len(past) >= 3 else np.nan
        shift_cal = (past.actual - past.post_cal).mean() if len(past) >= 3 else np.nan
        road_past = e[(e.road == r.road) & (e.kind == r.kind) & (e.hydro_year < r.hydro_year)]
        last = road_past[road_past.hydro_year == r.hydro_year - 1].actual_day
        out.append({"m1_adj": r.m1 + shift_m1, "postcal_adj": r.post_cal + shift_cal,
                    "last_year": float(last.iloc[0]) if len(last) else np.nan,
                    "road_cal": road_past.actual_day.mean() if len(road_past) else np.nan,
                    "n_past_shift": len(past)})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def boot(d, a, b, n=2000, seed=0):
    """90%-интервал разницы MAE (b − a) по сезонам; > 0 — a точнее."""
    rng = np.random.default_rng(seed)
    g = d.assign(ea=(d[a] - d.actual).abs(), eb=(d[b] - d.actual).abs()).groupby("hydro_year")[["ea", "eb"]].agg(["sum", "count"])
    yrs = g.index.to_numpy()
    res = []
    for _ in range(n):
        k = rng.choice(yrs, len(yrs))
        s = g.loc[k]
        res.append(s[("eb", "sum")].sum() / s[("eb", "count")].sum() - s[("ea", "sum")].sum() / s[("ea", "count")].sum())
    return np.percentile(res, [5, 95])


def compare(df, base, label):
    rows = []
    for kind in ("open", "close"):
        d = df[(df.kind == kind) & df.m1_adj.notna() & df[base].notna()]
        if len(d) < 8:
            continue
        lo, hi = boot(d, "m1_adj", base)
        e_m1 = (d.m1_adj - d.actual).abs().mean()
        e_b = (d[base] - d.actual).abs().mean()
        rows.append({"событие": "открытие" if kind == "open" else "закрытие", "соперник": label,
                     "прогнозов": len(d), "сезонов": d.hydro_year.nunique(), "дорог": d.road.nunique(),
                     "MAE соперник, сут": round(e_b, 1), "MAE М1+сдвиг, сут": round(e_m1, 1),
                     "выигрыш М1, сут": round(e_b - e_m1, 1), "выигрыш, %": round(100 * (1 - e_m1 / e_b), 1),
                     "90%-интервал выигрыша, сут": f"{lo:+.1f} … {hi:+.1f}",
                     "промахов > 7 сут: соперник / М1": f"{((d[base] - d.actual).abs() > 7).mean():.0%} / "
                                                         f"{((d.m1_adj - d.actual).abs() > 7).mean():.0%}"})
    return rows


def main():
    e = load_events()
    df = add_baselines(m1_rows(e), e)
    df.to_csv(REPORTS / "official_dates.csv", index=False)
    rows = []
    for base, label in [("last_year", "как в прошлом году"), ("road_cal", "календарь дороги (прошлые сезоны)"),
                        ("announced", "объявленный план"), ("postcal_adj", "календарь поста + сдвиг (контроль)")]:
        rows += compare(df, base, label)
    res = pd.DataFrame(rows)
    # аномалии: предсказывает ли М1 «раньше/позже обычного» (без сдвигов)
    an = []
    for kind in ("open", "close"):
        d = df[(df.kind == kind) & df.road_cal.notna()].copy()
        if len(d) >= 8:
            x, y = d.m1 - d.post_cal, d.actual - d.road_cal
            same = (np.sign(x) == np.sign(y)).mean()
            an.append({"событие": "открытие" if kind == "open" else "закрытие", "прогнозов": len(d),
                       "корреляция аномалий (Спирмен)": round(x.rank().corr(y.rank()), 2),
                       "доля угаданного знака «раньше/позже»": f"{same:.0%}"})
    an = pd.DataFrame(an)
    lines = ["# R1: официальные даты зимников и переправ — М1 против простых правил", "",
             f"Событий с фактической датой и точкой прогноза: {len(e)}; прогнозов (событие × дата выпуска): "
             f"{len(df)}; с рассчитанным сдвигом: {df.m1_adj.notna().sum()}. Данные — "
             "[road_events.csv](../data/external/road_events.csv) (официальные даты из СМИ со ссылками). "
             "Метод и правила честности — в шапке `scripts/official_dates_test.py`.", "",
             "Выигрыш > 0 — М1 точнее. Если 90%-интервал целиком выше нуля — выигрыш не случаен.", "",
             md_table(res) if len(res) else "_недостаточно данных_", "",
             "## Угадывает ли М1 «раньше или позже обычного»", "",
             "Без сдвигов: аномалия М1 (медиана − климатическая дата поста) против аномалии факта "
             "(официальная дата − средняя дата этой дороги).", "",
             md_table(an) if len(an) else "_недостаточно данных_", ""]
    (REPORTS / "official_dates.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))
    print(an.to_string(index=False))


if __name__ == "__main__":
    main()
