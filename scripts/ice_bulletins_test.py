"""
Проверка физики льда на свежих данных: расчётная толщина (закон Стефана по погоде, без замеров)
против бюллетеней Якутского УГМС / ГИМС за 2021–2026 гг. (data/external/ice_thickness_bulletins.csv).

    python scripts/ice_bulletins_test.py     # → reports/ice_bulletins.md, .csv

Замеры гидропостов в обучении заканчиваются в 2012 г., коэффициент α подобран по ним — бюллетени 2021–2026
модель не видела. Бюллетень даёт диапазон по участку реки; сравниваем с серединой и проверяем попадание
в диапазон. Соперник — та же формула, но с температурой «как в среднем за 10 прошлых зим» (что известно без
погоды текущего сезона): так видно, что даёт именно учёт текущей зимы.

Надёжность источника — data/external/source_audit.csv (ручная сверка ссылок 07.11.2026): «по реке» — значение
есть на странице для этой реки; «по группе рек или области» — на странице только общий диапазон; «не подтверждено»
и «противоречит» — значения на странице нет или оно другое. Строки без записи в аудите — исходные 14 бюллетеней
Якутского УГМС/ГИМС 2021–2026. Главный вывод делается только по надёжным строкам.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.features import load_alpha
from okno.ice import freezeup_from_weather, load_weather, season_weather, stefan_thickness, to_season_day
from okno.paths import DATA, REPORTS
from okno.report import md_table


def climatology(w, hy, years=10):
    past = [season_weather(w, y).t.to_numpy()[:365] for y in range(hy - years, hy) if len(season_weather(w, y)) >= 365]
    return np.nanmean(np.vstack(past), axis=0) if past else None


def main():
    b = pd.read_csv(DATA / "external" / "ice_thickness_bulletins.csv")
    ap = DATA / "external" / "source_audit.csv"
    if ap.exists():
        a = pd.read_csv(ap, dtype=str)
        a = a[a.file == "ice_thickness_bulletins.csv"][["obs_date", "point_id", "verdict"]]
        b = b.merge(a, on=["obs_date", "point_id"], how="left")
    else:
        b["verdict"] = np.nan
    b["verdict"] = b["verdict"].fillna("исходные (Якутское УГМС)")
    alpha = load_alpha()
    rows = []
    for r in b.itertuples():
        hy = int(r.hydro_year)
        w = load_weather(r.point_id, live=True)
        s = season_weather(w, hy)
        d = int(to_season_day(r.obs_date, hy))
        fz = freezeup_from_weather(s)
        clim = climatology(w, hy)
        if fz is None or d >= len(s) or clim is None:
            continue
        h_now = float(stefan_thickness(s.t.to_numpy(), alpha, int(fz))[d])
        cs = pd.DataFrame({"t": clim, "season_day": np.arange(len(clim))})
        fz_c = freezeup_from_weather(cs)
        h_clim = float(stefan_thickness(clim, alpha, int(fz_c))[d])
        m = re.search(r"(\d+)\s*[–-]\s*(\d+)\s*см", str(r.comment))
        lo, hi = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
        rows.append({"дата": r.obs_date, "река": r.river, "точка": r.point_id, "бюллетень, см": r.ice_cm,
                     "диапазон": f"{lo:.0f}–{hi:.0f}" if m else "—", "расчёт по погоде, см": round(h_now, 1),
                     "расчёт по климату, см": round(h_clim, 1),
                     "в диапазоне (погода)": (lo <= h_now <= hi) if m else np.nan,
                     "в диапазоне (климат)": (lo <= h_clim <= hi) if m else np.nan,
                     "источник подтверждён": r.verdict, "источник": r.source})
    x = pd.DataFrame(rows)
    x.to_csv(REPORTS / "ice_bulletins.csv", index=False)
    def summarize(g, name):
        e_now = (g["расчёт по погоде, см"] - g["бюллетень, см"]).abs()
        e_clim = (g["расчёт по климату, см"] - g["бюллетень, см"]).abs()
        rng, rng_c = g["в диапазоне (погода)"].dropna(), g["в диапазоне (климат)"].dropna()
        return [{"строки": name, "способ": "по погоде текущей зимы (сервис)", "замеров": len(g),
                 "MAE к середине диапазона, см": round(e_now.mean(), 1), "в диапазоне бюллетеня": f"{int(rng.sum())} из {len(rng)}"},
                {"строки": name, "способ": "по климату прошлых 10 зим", "замеров": len(g),
                 "MAE к середине диапазона, см": round(e_clim.mean(), 1), "в диапазоне бюллетеня": f"{int(rng_c.sum())} из {len(rng_c)}"}]
    v = x["источник подтверждён"]
    reliable = v.isin(["исходные (Якутское УГМС)", "по реке"])
    summary = pd.DataFrame(
        summarize(x[reliable], "надёжные: исходные + подтверждённые по реке")
        + summarize(x[v == "по группе рек или области"], "только общий диапазон по группе рек / области")
        + summarize(x[v.isin(["не подтверждено", "противоречит"])], "не подтверждены источником (не для выводов)")
        + summarize(x, "все строки"))
    lines = ["# Толщина льда по погоде против бюллетеней УГМС и МЧС", "",
             f"α = {alpha:.2f} подобран по замерам гидропостов до 2012 г.; бюллетени 2021–2026 модель не видела. "
             "Бюллетень — диапазон по участку реки, сравнение с серединой диапазона грубое: важнее попадание "
             "в диапазон. Вывод — по надёжным строкам; строки с неподтверждённым источником показаны отдельно "
             "(сверка ссылок — [source_audit.csv](../data/external/source_audit.csv)).", "", md_table(summary), "", md_table(x.drop(columns=["источник"])), "",
             "Источники — колонка `source` в [ice_thickness_bulletins.csv](../data/external/ice_thickness_bulletins.csv)."]
    (REPORTS / "ice_bulletins.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
