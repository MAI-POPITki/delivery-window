"""
Проверка и сводный отчёт по внешним данным раздела 3 docs/VALIDATION_PLAN.md:
  - data/external/road_events.csv (п. 3.1)
  - data/external/tariffs.csv (п. 3.2)
  - data/external/incidents.csv (п. 3.3)
  - data/external/northern_supply_volumes.csv и ice_thickness_bulletins.csv (п. 3.4)

Запуск:
    python scripts/validate_external_data.py

Создаёт:
    reports/external_data_checks.csv
    reports/external_data_summary.md
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from okno.external_data import (
    load_ice_thickness_bulletins,
    load_incidents,
    load_northern_supply_volumes,
    load_road_events,
    load_tariffs,
    validate_all_external,
)
from okno.paths import EXTERNAL, REPORTS
from okno.regions import REGIONS, label
from okno.report import md_table


def _build_region_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg_key, meta in REGIONS.items():
        sub = events[events["region"] == reg_key]
        oc = sub[sub["kind"].isin(["open", "close"])]
        rows.append({
            "регион": f"{meta['name']} ({reg_key})",
            "всего событий": len(sub),
            "open": int((sub["kind"] == "open").sum()),
            "close": int((sub["kind"] == "close").sum()),
            "limit30 / limit20": int(sub["kind"].isin(["limit30", "limit20"]).sum()),
            "объявлено + факт": int((sub["announced_date"] != "").sum()),
            "дорог": int(sub["road"].nunique()) if len(sub) else 0,
            "пар дорога×сезон (H2)": int(oc[["road", "hydro_year"]].drop_duplicates().shape[0]) if len(oc) else 0,
            "точек модели": int(sub["point_id"].nunique()) if len(sub) else 0,
        })
    return pd.DataFrame(rows)


def _build_tariff_table(tariffs: pd.DataFrame) -> pd.DataFrame:
    zimnik = tariffs[tariffs["mode"] == "зимник"]["price_rub_per_t"]
    z_med = float(zimnik.median()) if len(zimnik) else float("nan")
    rows = []
    for mode, grp in tariffs.groupby("mode", sort=False):
        med = float(grp["price_rub_per_t"].median())
        ratio = med / z_med if z_med and z_med > 0 else float("nan")
        rows.append({
            "канал (mode)": mode,
            "контрактов / тарифов": len(grp),
            "мин, ₽/т": round(float(grp["price_rub_per_t"].min()), 1),
            "медиана, ₽/т": round(med, 1),
            "макс, ₽/т": round(float(grp["price_rub_per_t"].max()), 1),
            "медиана к зимнику, ×": round(ratio, 2) if pd.notna(ratio) else "—",
        })
    return pd.DataFrame(rows)


def _build_points_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (reg_key, pid), grp in events.groupby(["region", "point_id"]):
        rows.append({
            "регион": REGIONS.get(reg_key, {}).get("name", reg_key),
            "point_id": pid,
            "привязка в модели": label(pid) if pid else "—",
            "событий": len(grp),
            "дорог": int(grp["road"].nunique()),
            "сезоны": f"{grp['hydro_year'].min()}–{grp['hydro_year'].max()}",
        })
    return pd.DataFrame(rows).sort_values(["событий", "регион"], ascending=[False, True]).reset_index(drop=True)


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    checks, summary = validate_all_external(EXTERNAL)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(REPORTS / "external_data_checks.csv", index=False)

    for c in checks:
        flag = "OK " if c["результат"] == "прошла" else "FAIL"
        print(f"[{flag}] {c['таблица']} :: {c['проверка']} — {c['детали']}")

    n_ok = int((checks_df["результат"] == "прошла").sum())
    n_all = len(checks_df)

    events = load_road_events(EXTERNAL / "road_events.csv", with_point_id=True)
    tariffs = load_tariffs(EXTERNAL / "tariffs.csv")
    incidents = load_incidents(EXTERNAL / "incidents.csv")
    volumes = load_northern_supply_volumes(EXTERNAL / "northern_supply_volumes.csv")
    bulletins = load_ice_thickness_bulletins(EXTERNAL / "ice_thickness_bulletins.csv")

    reg_tbl = _build_region_table(events)
    pts_tbl = _build_points_table(events) if len(events) else pd.DataFrame()
    tar_tbl = _build_tariff_table(tariffs) if len(tariffs) else pd.DataFrame()
    inc_tbl = (
        incidents.groupby("what")
        .agg(кейсов=("road", "count"), годы=("year", lambda s: f"{s.min()}–{s.max()}"))
        .reset_index()
        .rename(columns={"what": "тип срыва (what)"})
        if len(incidents)
        else pd.DataFrame()
    )

    lines = [
        "# Сводка внешних данных (раздел 3 `docs/VALIDATION_PLAN.md`)",
        "",
        f"Сформировано скриптом `scripts/validate_external_data.py`, {dt.date.today():%d.%m.%Y}. "
        f"Автопроверок пройдено: **{n_ok} из {n_all}**.",
        "",
        "Каждая строка во всех таблицах `data/external/*.csv` содержит прямую `https://`-ссылку на публичный "
        "источник (приказы и сводки Минтранса РС(Я), ГКУ «Управтодор РС(Я)», ГУ МЧС России, КрУДор, «Ямалавтодор», "
        "Дорожная служба Иркутской области, «Архангельскавтодор», ЕИС Закупки / региональные тарифные комитеты).",
        "",
        "## 1. Итоги автопроверок",
        "",
        md_table(checks_df),
        "",
        "## 2. Официальные даты зимников и переправ (`3.1 road_events.csv`)",
        "",
        md_table(reg_tbl),
        "",
        "### Покрытие точек модели (`okno/regions.py`)",
        "",
        md_table(pts_tbl) if len(pts_tbl) else "нет данных",
        "",
        "## 3. Реальные закупки и тарифы (`3.2 tariffs.csv`)",
        "",
        md_table(tar_tbl) if len(tar_tbl) else "нет данных",
        "",
        "## 4. Задокументированные случаи срывов (`3.3 incidents.csv`)",
        "",
        md_table(inc_tbl) if len(inc_tbl) else "нет данных",
        "",
        "## 5. Дополнительные ряды (`3.4`)",
        "",
        f"- **Графики снижения допустимой массы (40 → 30 → 20 т):** в `road_events.csv` зафиксировано "
        f"**{int(events['kind'].isin(['limit30', 'limit20']).sum())}** официальных этапов снижения тоннажа (`limit30`, `limit20`).",
        f"- **Объёмы северного завоза (план / факт):** `data/external/northern_supply_volumes.csv` — **{len(volumes)}** записей.",
        f"- **Сводки толщины льда УГМС / МЧС после 2015 г.:** `data/external/ice_thickness_bulletins.csv` — **{len(bulletins)}** замеров.",
        "",
    ]
    out_md = REPORTS / "external_data_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nСводка записана в {out_md} ({n_ok}/{n_all} проверок пройдено)")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
