"""
Загрузка и строгая валидация внешних таблиц раздела 3 docs/VALIDATION_PLAN.md:
  - 3.1 data/external/road_events.csv — официальные даты открытия/закрытия/ограничений зимников и переправ
  - 3.2 data/external/tariffs.csv — реальные закупки и утверждённые тарифы северного завоза
  - 3.3 data/external/incidents.csv — задокументированные случаи срывов и аномалий завоза
  - 3.4 data/external/northern_supply_volumes.csv и data/external/ice_thickness_bulletins.csv

Все проверки соблюдают правила раздела 1 VALIDATION_PLAN.md:
  - обязательная ссылка https:// на публичный источник в каждой строке;
  - отсутствие знания будущего в объявленных планах (announced_published <= announced_date и <= actual_published);
  - соответствие hydro_year гидрологическому сезону (сентябрь Y-1 … август Y);
  - привязка point_id=... в comment к реальным точкам из okno/regions.py для расчётов R1–R5.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from okno.paths import EXTERNAL
from okno.regions import REGIONS, points_of, region_of

ROAD_EVENTS_COLS = (
    "road",
    "region",
    "hydro_year",
    "kind",
    "announced_date",
    "announced_published",
    "actual_date",
    "actual_published",
    "source_announced",
    "source_actual",
    "comment",
)
VALID_REGIONS = frozenset(REGIONS.keys())
VALID_ROAD_KINDS = frozenset({"open", "close", "limit30", "limit20"})

TARIFFS_COLS = (
    "contract_id",
    "year",
    "route",
    "mode",
    "price_rub_per_t",
    "deadline",
    "penalty",
    "url",
)
VALID_TARIFF_MODES = frozenset({"зимник", "авиа", "река", "ж/д", "СВП"})

INCIDENTS_COLS = (
    "year",
    "road",
    "what",
    "tonnage",
    "extra_cost",
    "source",
)
VALID_INCIDENT_TYPES = frozenset(
    {"ранняя распутица", "позднее открытие", "недовоз", "переход на авиа"}
)

VOLUMES_COLS = (
    "year",
    "region",
    "scope",
    "plan_thousand_t",
    "fact_thousand_t",
    "unit_note",
    "source",
    "published_date",
)

ICE_BULLETINS_COLS = (
    "obs_date",
    "hydro_year",
    "river",
    "location",
    "point_id",
    "ice_cm",
    "norm_cm",
    "tonnage_limit_t",
    "source",
    "published_date",
    "comment",
)

_POINT_ID_RE = re.compile(r"(?:^|[;\s,])point_id=([a-z0-9_]+)")


def hydro_year_for_date(d: dt.date) -> int:
    """Гидрологический год сезона сентябрь (Y-1) … август Y."""
    return d.year + 1 if d.month >= 9 else d.year


def extract_point_id(comment: str | None) -> str | None:
    """Извлекает тег point_id=<id> из поля comment."""
    if not isinstance(comment, str) or not comment:
        return None
    m = _POINT_ID_RE.search(comment)
    return m.group(1) if m else None


def is_valid_https_url(value: str | None) -> bool:
    """Проверяет, что строка — корректный https:// URL с непустым доменом."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s.startswith("https://") or any(ch.isspace() for ch in s):
        return False
    parsed = urlparse(s)
    return parsed.scheme == "https" and bool(parsed.netloc) and "." in parsed.netloc


def _parse_iso_date(value: str | None) -> dt.date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return dt.date.fromisoformat(value.strip())


def _read_header(path: Path) -> tuple[str, ...]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
    return tuple(header) if header is not None else ()


def load_road_events(path: Path | None = None, *, with_point_id: bool = False) -> pd.DataFrame:
    """Читает data/external/road_events.csv с сохранением пустых строк как ''."""
    p = Path(path) if path is not None else EXTERNAL / "road_events.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if len(df):
        df["hydro_year"] = df["hydro_year"].astype(int)
    if with_point_id:
        df = df.copy()
        df["point_id"] = df["comment"].map(extract_point_id)
    return df


def load_tariffs(path: Path | None = None) -> pd.DataFrame:
    """Читает data/external/tariffs.csv."""
    p = Path(path) if path is not None else EXTERNAL / "tariffs.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if len(df):
        df["year"] = df["year"].astype(int)
        df["price_rub_per_t"] = df["price_rub_per_t"].astype(float)
    return df


def load_incidents(path: Path | None = None) -> pd.DataFrame:
    """Читает data/external/incidents.csv."""
    p = Path(path) if path is not None else EXTERNAL / "incidents.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if len(df):
        df["year"] = df["year"].astype(int)
    return df


def load_northern_supply_volumes(path: Path | None = None) -> pd.DataFrame:
    """Читает data/external/northern_supply_volumes.csv (п. 3.4)."""
    p = Path(path) if path is not None else EXTERNAL / "northern_supply_volumes.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if len(df):
        df["year"] = df["year"].astype(int)
    return df


def load_ice_thickness_bulletins(path: Path | None = None) -> pd.DataFrame:
    """Читает data/external/ice_thickness_bulletins.csv (п. 3.4)."""
    p = Path(path) if path is not None else EXTERNAL / "ice_thickness_bulletins.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if len(df):
        df["hydro_year"] = df["hydro_year"].astype(int)
        df["ice_cm"] = df["ice_cm"].astype(float)
    return df


def validate_road_events(path: Path | None = None) -> tuple[list[dict], dict]:
    """Строгая проверка 3.1 road_events.csv. Возвращает (checks, stats)."""
    p = Path(path) if path is not None else EXTERNAL / "road_events.csv"
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"таблица": "road_events.csv", "проверка": name,
                       "результат": "прошла" if ok else "НЕ прошла", "детали": detail})

    if not p.exists():
        add("файл существует", False, f"не найден {p}")
        return checks, {}

    header = _read_header(p)
    add("схема колонок 3.1", header == ROAD_EVENTS_COLS,
        f"колонок {len(header)}: {', '.join(header)}")
    if header != ROAD_EVENTS_COLS:
        return checks, {}

    df = load_road_events(p, with_point_id=True)
    n_total = len(df)
    n_yak = int((df["region"] == "yakutia").sum()) if n_total else 0
    n_other = n_total - n_yak

    add("объём выборки ≥ 100 событий (цель VALIDATION_PLAN)", n_total >= 100,
        f"всего {n_total} событий (Якутия: {n_yak}, другие 4 региона: {n_other})")
    add("Якутия ≥ 60 событий (роль Б)", n_yak >= 60, f"Якутия: {n_yak} событий")
    add("другие 4 региона ≥ 40 событий (роль В)", n_other >= 40,
        f"вне Якутии: {n_other} событий по регионам "
        f"{dict(df[df['region'] != 'yakutia']['region'].value_counts()) if n_other else {}}")

    if n_total == 0:
        return checks, {"total": 0}

    bad_regions = sorted(set(df["region"]) - VALID_REGIONS)
    add("все регионы входят в okno/regions.py", not bad_regions,
        f"регионы: {sorted(set(df['region']))}" + (f", лишние: {bad_regions}" if bad_regions else ""))

    covered_regions = set(df["region"]) & VALID_REGIONS
    add("покрыты все 5 регионов сервиса", covered_regions == VALID_REGIONS,
        f"покрыто {len(covered_regions)} из {len(VALID_REGIONS)}: {sorted(covered_regions)}")

    bad_kinds = sorted(set(df["kind"]) - VALID_ROAD_KINDS)
    add("допустимые значения kind (open|close|limit30|limit20)", not bad_kinds,
        f"распределение kind: {dict(df['kind'].value_counts())}")

    # Даты, ссылки, hydro_year и отсутствие знания будущего
    date_errors: list[str] = []
    url_errors: list[str] = []
    hy_errors: list[str] = []
    leak_errors: list[str] = []
    point_errors: list[str] = []

    all_valid_points = {pt for reg in VALID_REGIONS for pt in points_of(reg)}

    for idx, row in df.iterrows():
        line_no = int(idx) + 2
        if not row["road"].strip():
            date_errors.append(f"строка {line_no}: пустое название дороги")

        try:
            act_d = _parse_iso_date(row["actual_date"])
            act_pub = _parse_iso_date(row["actual_published"])
            ann_d = _parse_iso_date(row["announced_date"])
            ann_pub = _parse_iso_date(row["announced_published"])
        except ValueError as exc:
            date_errors.append(f"строка {line_no}: неверный формат ISO-даты ({exc})")
            continue

        if act_d is None:
            date_errors.append(f"строка {line_no}: отсутствует actual_date")
        else:
            expected_hy = hydro_year_for_date(act_d)
            if int(row["hydro_year"]) != expected_hy:
                hy_errors.append(
                    f"строка {line_no} ({row['road']}): actual_date={act_d} → ожидался hydro_year={expected_hy}, указан {row['hydro_year']}"
                )
            if row["kind"] in ("limit30", "limit20") and act_d.month not in (3, 4, 5):
                hy_errors.append(
                    f"строка {line_no} ({row['road']}): {row['kind']} должен отражать весеннее снижение массы (март–май), указана дата {act_d}"
                )
            # Приказ/сводка о закрытии с полуночи следующего дня может выйти накануне (-1…-3 сут)
            if act_pub is not None and (act_d - act_pub).days > 3:
                leak_errors.append(
                    f"строка {line_no}: actual_published={act_pub} раньше actual_date={act_d} более чем на 3 сут"
                )


        if not is_valid_https_url(row["source_actual"]):
            url_errors.append(f"строка {line_no}: невалидный source_actual='{row['source_actual']}'")

        has_any_ann = bool(row["announced_date"] or row["announced_published"] or row["source_announced"])
        if has_any_ann:
            if ann_d is None or ann_pub is None or not is_valid_https_url(row["source_announced"]):
                leak_errors.append(
                    f"строка {line_no}: неполная тройка announced_date / announced_published / source_announced"
                )
            else:
                if ann_pub > ann_d:
                    leak_errors.append(
                        f"строка {line_no}: announced_published={ann_pub} позже announced_date={ann_d}"
                    )
                if act_pub is not None and ann_pub > act_pub:
                    leak_errors.append(
                        f"строка {line_no}: announced_published={ann_pub} позже actual_published={act_pub}"
                    )
                if row["source_announced"].strip() == row["source_actual"].strip():
                    leak_errors.append(
                        f"строка {line_no}: source_announced совпадает с source_actual (нужны независимые публикации плана и факта)"
                    )

        pid = row["point_id"]
        if not pid:
            point_errors.append(f"строка {line_no}: нет point_id=... в comment")
        elif pid not in all_valid_points:
            point_errors.append(f"строка {line_no}: неизвестный point_id={pid}")
        elif region_of(pid) != row["region"]:
            point_errors.append(
                f"строка {line_no}: point_id={pid} относится к '{region_of(pid)}', а не '{row['region']}'"
            )

    add("формат ISO-дат и заполненность фактов", not date_errors,
        "все фактические даты и даты публикаций заполнены" if not date_errors else "; ".join(date_errors[:3]))
    add("все ссылки https:// валидны", not url_errors,
        f"проверено {n_total} ссылок source_actual" if not url_errors else "; ".join(url_errors[:3]))
    add("соответствие hydro_year сезону Сен..Авг", not hy_errors,
        "все строки согласованы с гидрологическим годом" if not hy_errors else "; ".join(hy_errors[:3]))
    add("нет утечки будущего в объявленных датах (правило 1.2)", not leak_errors,
        f"пар «объявлено + факт»: {int((df['announced_date'] != '').sum())}" if not leak_errors else "; ".join(leak_errors[:3]))
    add("привязка point_id к точкам okno/regions.py", not point_errors,
        f"уникальных точек модели: {df['point_id'].nunique()}" if not point_errors else "; ".join(point_errors[:3]))

    dups = int(df.duplicated(subset=["road", "region", "hydro_year", "kind", "actual_date"]).sum())
    add("отсутствие дублей событий", dups == 0, f"дублей: {dups}")

    open_close = df[df["kind"].isin(["open", "close"])]
    n_h2_pairs = int(open_close[["road", "region", "hydro_year", "kind"]].drop_duplicates().shape[0])
    n_road_seasons = int(open_close[["road", "region", "hydro_year"]].drop_duplicates().shape[0])
    n_roads = int(df["road"].nunique())
    n_ann_pairs = int((df["announced_date"] != "").sum())
    add("порог H2: ≥ 30 официальных событий open/close («дорога × сезон»)", n_road_seasons >= 30,
        f"уникальных пар «дорога × сезон» (open/close): {n_road_seasons}, всего записей open/close: {n_h2_pairs}, дорог: {n_roads}")
    add("порог H2a: наличие пар «объявленный план + факт»", n_ann_pairs >= 10,
        f"подтверждённых пар «объявлено → факт»: {n_ann_pairs}")

    stats = {
        "total": n_total,
        "yakutia": n_yak,
        "other_regions": n_other,
        "roads": n_roads,
        "road_seasons_open_close": n_road_seasons,
        "open_close_events": n_h2_pairs,
        "announced_pairs": n_ann_pairs,
        "by_region": df["region"].value_counts().to_dict(),
        "by_kind": df["kind"].value_counts().to_dict(),
    }
    return checks, stats


def validate_tariffs(path: Path | None = None) -> tuple[list[dict], dict]:
    """Строгая проверка 3.2 tariffs.csv."""
    p = Path(path) if path is not None else EXTERNAL / "tariffs.csv"
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"таблица": "tariffs.csv", "проверка": name,
                       "результат": "прошла" if ok else "НЕ прошла", "детали": detail})

    if not p.exists():
        add("файл существует", False, f"не найден {p}")
        return checks, {}

    header = _read_header(p)
    add("схема колонок 3.2", header == TARIFFS_COLS, f"колонки: {', '.join(header)}")
    if header != TARIFFS_COLS:
        return checks, {}

    df = load_tariffs(p)
    n = len(df)
    add("объём выборки ≥ 15 контрактов/тарифов (роль В)", n >= 15, f"строк: {n}")
    if n == 0:
        return checks, {"total": 0}

    bad_modes = sorted(set(df["mode"]) - VALID_TARIFF_MODES)
    add("допустимые виды транспорта (зимник|авиа|река|ж/д|СВП)", not bad_modes,
        f"распределение mode: {dict(df['mode'].value_counts())}")

    pos_prices = bool((df["price_rub_per_t"] > 0).all())
    add("цены price_rub_per_t > 0", pos_prices,
        f"диапазон {df['price_rub_per_t'].min():,.0f} … {df['price_rub_per_t'].max():,.0f} ₽/т")

    bad_urls = [r["url"] for _, r in df.iterrows() if not is_valid_https_url(r["url"])]
    add("все ссылки url начинаются с https://", not bad_urls,
        f"валидных ссылок: {n - len(bad_urls)} из {n}")

    dups = int(df["contract_id"].duplicated().sum())
    add("уникальность contract_id", dups == 0, f"дублей contract_id: {dups}")

    med_by_mode = df.groupby("mode")["price_rub_per_t"].median().round(1).to_dict()
    stats = {
        "total": n,
        "by_mode": df["mode"].value_counts().to_dict(),
        "median_rub_per_t": med_by_mode,
    }
    return checks, stats


def validate_incidents(path: Path | None = None) -> tuple[list[dict], dict]:
    """Строгая проверка 3.3 incidents.csv."""
    p = Path(path) if path is not None else EXTERNAL / "incidents.csv"
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"таблица": "incidents.csv", "проверка": name,
                       "результат": "прошла" if ok else "НЕ прошла", "детали": detail})

    if not p.exists():
        add("файл существует", False, f"не найден {p}")
        return checks, {}

    header = _read_header(p)
    add("схема колонок 3.3", header == INCIDENTS_COLS, f"колонки: {', '.join(header)}")
    if header != INCIDENTS_COLS:
        return checks, {}

    df = load_incidents(p)
    n = len(df)
    add("случаи срывов собраны (≥ 10 кейсов для R4/H5)", n >= 10, f"кейсов: {n}")
    if n == 0:
        return checks, {"total": 0}

    bad_what = sorted(set(df["what"]) - VALID_INCIDENT_TYPES)
    add("допустимые типы срывов what", not bad_what,
        f"распределение what: {dict(df['what'].value_counts())}")

    bad_urls = [r["source"] for _, r in df.iterrows() if not is_valid_https_url(r["source"])]
    add("все ссылки source начинаются с https://", not bad_urls,
        f"валидных ссылок: {n - len(bad_urls)} из {n}")

    non_empty_roads = bool(df["road"].str.strip().astype(bool).all())
    add("указана дорога/направление в каждом кейсе", non_empty_roads,
        f"уникальных направлений: {df['road'].nunique()}")

    stats = {
        "total": n,
        "by_what": df["what"].value_counts().to_dict(),
    }
    return checks, stats


def validate_supporting_tables(external_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Проверка дополнительных таблиц раздела 3.4 (объёмы северного завоза и сводки толщины льда >2015)."""
    base = Path(external_dir) if external_dir is not None else EXTERNAL
    checks: list[dict] = []
    stats: dict = {}

    def add(table: str, name: str, ok: bool, detail: str) -> None:
        checks.append({"таблица": table, "проверка": name,
                       "результат": "прошла" if ok else "НЕ прошла", "детали": detail})

    vol_path = base / "northern_supply_volumes.csv"
    if not vol_path.exists():
        add("northern_supply_volumes.csv", "файл существует (п. 3.4)", False, f"не найден {vol_path}")
    else:
        v_hdr = _read_header(vol_path)
        add("northern_supply_volumes.csv", "схема колонок п. 3.4 (объёмы)", v_hdr == VOLUMES_COLS,
            f"колонки: {', '.join(v_hdr)}")
        if v_hdr == VOLUMES_COLS:
            vdf = load_northern_supply_volumes(vol_path)
            bad_u = [r["source"] for _, r in vdf.iterrows() if not is_valid_https_url(r["source"])]
            add("northern_supply_volumes.csv", "объёмы завоза со ссылками https://",
                len(vdf) >= 5 and not bad_u, f"строк: {len(vdf)}, невалидных ссылок: {len(bad_u)}")
            stats["volumes_rows"] = len(vdf)

    ice_path = base / "ice_thickness_bulletins.csv"
    if not ice_path.exists():
        add("ice_thickness_bulletins.csv", "файл существует (п. 3.4)", False, f"не найден {ice_path}")
    else:
        i_hdr = _read_header(ice_path)
        add("ice_thickness_bulletins.csv", "схема колонок п. 3.4 (толщина льда)", i_hdr == ICE_BULLETINS_COLS,
            f"колонки: {', '.join(i_hdr)}")
        if i_hdr == ICE_BULLETINS_COLS:
            idf = load_ice_thickness_bulletins(ice_path)
            bad_u = [r["source"] for _, r in idf.iterrows() if not is_valid_https_url(r["source"])]
            post_2015 = all(_parse_iso_date(d).year > 2015 for d in idf["obs_date"]) if len(idf) else False
            valid_cm = bool(((idf["ice_cm"] > 0) & (idf["ice_cm"] <= 300)).all()) if len(idf) else False
            add("ice_thickness_bulletins.csv", "замеры толщины льда после 2015 г. со ссылками https://",
                len(idf) >= 8 and not bad_u and post_2015 and valid_cm,
                f"замеров: {len(idf)}, все после 2015: {post_2015}, в пределах (0; 300] см: {valid_cm}")
            stats["ice_bulletins_rows"] = len(idf)

    return checks, stats


def validate_all_external(external_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Запускает полный набор проверок по разделу 3 VALIDATION_PLAN.md."""
    base = Path(external_dir) if external_dir is not None else EXTERNAL
    c1, s1 = validate_road_events(base / "road_events.csv")
    c2, s2 = validate_tariffs(base / "tariffs.csv")
    c3, s3 = validate_incidents(base / "incidents.csv")
    c4, s4 = validate_supporting_tables(base)
    all_checks = c1 + c2 + c3 + c4
    summary = {
        "road_events": s1,
        "tariffs": s2,
        "incidents": s3,
        "supporting": s4,
    }
    return all_checks, summary
