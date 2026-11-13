"""
Сборка единого проверенного набора данных из сырых файлов.

    python scripts/build_dataset.py

Вход:  data/raw/hydro_ice/*.csv, data/lena_ice_phenology_points.csv
Выход: data/processed/
    gauges.csv               — гидропосты (координаты, регион, флаг «Якутия»)
    hydro_ice_thickness.csv  — замеры толщины льда, длинный формат
    hydro_ice_seasons.csv    — один сезон поста = одна строка: даты явлений, дни сезона,
                               даты достижения нормативной толщины по замерам, флаги качества
    yakutsk_sat_seasons.csv  — спутниковые даты Лены у переправы Якутск — Нижний Бестях
У каждой строки — source_type и source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import hydro, northeast
from okno.ice import required_thickness_cm, to_season_day
from okno.paths import DATA, EXTERNAL, PROCESSED

YKUT_SOURCE = "ФГБУ «Якутское УГМС», бюллетени (https://ykuthydromet.ru/reki/)"

LOADS_T = [10, 20, 30]  # нагрузки, для которых считаем дату «лёд держит»


def open_date_from_measurements(th: pd.DataFrame, h_star: float):
    """Первая дата, когда измеренная толщина достигла h* (линейная интерполяция между замерами).
    Возвращает None, если до h* лёд в замерах не дорос или замеров недостаточно."""
    s = th.dropna(subset=["ice_cm"]).sort_values("date")
    if len(s) < 3:
        return None
    d = pd.to_datetime(s["date"]).to_numpy()
    h = s["ice_cm"].to_numpy()
    above = np.where(h >= h_star)[0]
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return None  # первый же замер выше порога — момент достижения не наблюдали
    # линейная интерполяция между замерами i-1 и i
    frac = (h_star - h[i - 1]) / (h[i] - h[i - 1]) if h[i] != h[i - 1] else 1.0
    return pd.Timestamp(d[i - 1] + (d[i] - d[i - 1]) * frac).normalize()


def quality_flags(row) -> list[str]:
    f = []
    order = ["first_ice", "freezeup", "melt_start", "drift_start", "ice_end"]
    days = [row.get(f"{e}_day") for e in order]
    present = [(e, d) for e, d in zip(order, days) if pd.notna(d)]
    for (e1, d1), (e2, d2) in zip(present, present[1:]):
        if d2 < d1:
            f.append(f"порядок:{e2}<{e1}")
    fz, dr = row.get("freezeup_day"), row.get("drift_start_day")
    if pd.notna(fz) and not (0 <= fz <= 120):  # 1 сен — 30 дек
        f.append("ледостав_вне_сен-дек")
    if pd.notna(dr) and not (200 <= dr <= 300):  # ~20 мар — 28 июн
        f.append("ледоход_вне_мар-июн")
    if pd.notna(fz) and pd.notna(dr) and not (120 <= dr - fz <= 280):
        f.append("длительность_ледостава_вне_120-280")
    for e, d in present:  # дата вообще не из этого сезона (артефакты вида 1900-01-03, год с опечаткой)
        if not (-30 <= d <= 330):
            f.append(f"дата_вне_сезона:{e}")
    for e in order:
        raw = row.get(f"{e}_raw")
        if isinstance(raw, str) and raw.strip() and pd.isna(row.get(f"{e}_day")):
            f.append(f"нечитаемая_дата:{e}")
    return f


def flag_spikes(thick: pd.DataFrame) -> pd.DataFrame:
    """Одиночные выбросы толщины: замер на ≥80 см и в ≥1.8 раза выше медианы двух соседей
    в том же сезоне (типичная опечатка — лишний ноль: 130 → 1300). Сырое значение
    сохраняем в ice_cm_raw, в ice_cm ставим пусто. Стабильно толстый лёд (наледи на горных
    реках, ~300 см много замеров подряд) выбросом не считается."""
    thick = thick.sort_values(["gauge_code", "hydro_year", "date"]).reset_index(drop=True)
    if "ice_cm_raw" in thick.columns:
        thick["ice_cm_raw"] = thick["ice_cm_raw"].fillna(thick["ice_cm"])
    else:
        thick["ice_cm_raw"] = thick["ice_cm"]
    g = thick.groupby(["gauge_code", "hydro_year"])["ice_cm"]
    prev_, next_ = g.shift(1), g.shift(-1)
    neigh = pd.concat([prev_, next_], axis=1).median(axis=1)
    spike = (thick["ice_cm"] - neigh >= 80) & (thick["ice_cm"] >= 1.8 * neigh)
    thick["spike"] = spike.fillna(False)
    thick.loc[thick["spike"], "ice_cm"] = np.nan
    return thick


def _load_ykut_events() -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    """События 2022–2027 гг. из ежедневных бюллетеней ФГБУ «Якутское УГМС» для числовых гидропостов."""
    p = EXTERNAL / "ykut_reki_phenology.csv"
    if not p.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(p, dtype={"gauge_code": str})
    df = df[df["gauge_code"].notna() & (df["gauge_code"].str.strip() != "")].copy()
    if df.empty:
        return pd.DataFrame(), set()
    ev_map = {"freezeup": "freezeup", "melt_start": "melt_start", "breakup": "drift_start"}
    df = df[df["event_type"].isin(ev_map)].copy()
    df["gauge_code"] = df["gauge_code"].astype(int)
    df["hydro_year"] = df["hydro_year"].astype(int)
    df["event"] = df["event_type"].map(ev_map)
    df["date"] = df["event_date"]
    df["raw"] = ""
    keys = set(zip(df["gauge_code"], df["hydro_year"]))
    return df[["gauge_code", "hydro_year", "event", "date", "raw"]], keys


def build_hydro():
    st, thick, events = hydro.read_all()
    st["yakutia"] = st["region"].str.contains("Sakha")
    st["source_type"] = "замер"
    st["source"] = hydro.SOURCE

    if (northeast.DIRECTORY / "ice_snow.tsv").exists():
        extra_st, extra_th = northeast.load_observations()
        st = pd.concat([st, extra_st], ignore_index=True)
        thick = pd.concat([thick, extra_th], ignore_index=True)
    thick = flag_spikes(thick)

    ykut_ev, ykut_keys = _load_ykut_events()
    if not ykut_ev.empty:
        events = pd.concat([events, ykut_ev], ignore_index=True)
        events = events.drop_duplicates(["gauge_code", "hydro_year", "event"], keep="first")

    # сезонная таблица
    ev = events.copy()
    ev["day"] = [to_season_day(d, hy) for d, hy in zip(ev["date"], ev["hydro_year"])]
    wide_date = ev.pivot_table(index=["gauge_code", "hydro_year"], columns="event", values="date", aggfunc="first")
    wide_day = ev.pivot_table(index=["gauge_code", "hydro_year"], columns="event", values="day", aggfunc="first")
    wide_raw = ev.pivot_table(index=["gauge_code", "hydro_year"], columns="event", values="raw", aggfunc="first")
    seasons = pd.concat([
        wide_date.add_suffix("_date"), wide_day.add_suffix("_day"), wide_raw.add_suffix("_raw"),
    ], axis=1).reset_index()
    # PANGAEA содержит толщину льда и снега, но не даты фаз; добавляем измеренные сезоны без домысливания фаз
    seasons = seasons.merge(thick[["gauge_code", "hydro_year"]].drop_duplicates(), how="outer")

    # даты достижения нормативной толщины — по реальным замерам
    max_h = thick.groupby(["gauge_code", "hydro_year"])["ice_cm"].max().rename("max_ice_cm")
    seasons = seasons.merge(max_h.reset_index(), how="left")
    for load in LOADS_T:
        h_star = required_thickness_cm(load)
        res = []
        for (g, hy), grp in thick.groupby(["gauge_code", "hydro_year"]):
            d = open_date_from_measurements(grp, h_star)
            res.append({"gauge_code": g, "hydro_year": hy,
                        f"open_{load}t_date": d.date().isoformat() if d is not None else None,
                        f"open_{load}t_day": to_season_day(d, hy) if d is not None else np.nan})
        seasons = seasons.merge(pd.DataFrame(res), how="left")

    seasons["flags"] = [";".join(quality_flags(r)) for r in seasons.to_dict("records")]
    seasons["valid"] = seasons["flags"] == ""
    seasons = seasons.drop(columns=[c for c in seasons.columns if c.endswith("_raw")])
    seasons = seasons.sort_values(["gauge_code", "hydro_year"]).reset_index(drop=True)

    is_ykut = pd.Series([k in ykut_keys for k in zip(seasons["gauge_code"], seasons["hydro_year"])], index=seasons.index)
    is_ne = seasons["gauge_code"].isin(northeast.STATIONS) & ~is_ykut
    seasons["source_type"] = np.where(is_ykut, "бюллетень ЯУГМС", np.where(is_ne, "оценка по толщине", "замер"))
    st_src = st.set_index("gauge_code")["source"]
    seasons["source"] = np.where(is_ykut, YKUT_SOURCE, seasons["gauge_code"].map(st_src))
    return st, thick, seasons


def build_sat():
    sat = pd.read_csv(DATA / "lena_ice_phenology_points.csv")
    sat = sat[sat["point_id"] == "yakutsk_crossing"]
    w = sat.pivot_table(index="hydro_year", columns="variable", values="value").reset_index()
    # DOHY: 1 = 1 октября; наш день сезона: 0 = 1 сентября → +29 (сентябрь 30 дней, DOHY с 1)
    out = pd.DataFrame({"hydro_year": w["hydro_year"]})
    for v in ("freezeup", "breakup"):
        out[f"sat_{v}_day"] = w[v] + 29
        out[f"sat_{v}_date"] = [
            (pd.Timestamp(hy - 1, 9, 1) + pd.Timedelta(days=d)).date().isoformat() if pd.notna(d) else None
            for hy, d in zip(out["hydro_year"], out[f"sat_{v}_day"])
        ]
    out["sat_ice_duration_days"] = w.get("ice_duration")
    spread = sat.pivot_table(index="hydro_year", columns="variable", values="spread").reset_index()
    out["sat_breakup_spread"] = spread["breakup"].to_numpy()
    out["sat_freezeup_spread"] = spread["freezeup"].to_numpy()
    out.insert(0, "point_id", "yakutsk_crossing")
    out["source_type"] = "спутник"
    out["source"] = "Qiu et al. 2026, ESSD, doi:10.5281/zenodo.17054619"
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    st, thick, seasons = build_hydro()
    st.to_csv(PROCESSED / "gauges.csv", index=False)
    thick.to_csv(PROCESSED / "hydro_ice_thickness.csv", index=False)
    seasons.to_csv(PROCESSED / "hydro_ice_seasons.csv", index=False)
    if "snow_on_ice_cm" in thick.columns:
        snow = thick[thick["snow_on_ice_cm"].notna()].copy()
        if not snow.empty:
            snow[["gauge_code", "hydro_year", "date", "snow_on_ice_cm", "source"]].to_csv(
                PROCESSED / "hydro_snow_on_ice.csv", index=False
            )
    sat = build_sat()
    sat.to_csv(PROCESSED / "yakutsk_sat_seasons.csv", index=False)
    print(f"gauges: {len(st)} (Якутия: {st['yakutia'].sum()})")
    print(f"thickness rows: {len(thick)}; seasons: {len(seasons)}, valid: {seasons['valid'].sum()}")
    print(f"by source_type:\n{seasons['source_type'].value_counts().to_string()}")
    print(f"yakutsk satellite seasons: {len(sat)}")


if __name__ == "__main__":
    main()
