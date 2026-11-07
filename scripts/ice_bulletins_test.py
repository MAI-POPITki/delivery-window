"""
Проверка физики льда на свежих данных: расчётная толщина (закон Стефана по погоде, без замеров)
против бюллетеней Якутского УГМС / ГИМС за 2021–2026 гг. (data/external/ice_thickness_bulletins.csv).

    python scripts/ice_bulletins_test.py     # → reports/ice_bulletins.md, .csv

Замеры гидропостов в обучении заканчиваются в 2012 г., коэффициент α подобран по ним — бюллетени 2021–2026
модель не видела. Бюллетень даёт диапазон по участку реки; сравниваем с серединой и проверяем попадание
в диапазон. Соперник — та же формула, но с температурой «как в среднем за 10 прошлых зим» (что известно без
погоды текущего сезона): так видно, что даёт именно учёт текущей зимы.
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
                     "в диапазоне (климат)": (lo <= h_clim <= hi) if m else np.nan, "источник": r.source})
    x = pd.DataFrame(rows)
    x.to_csv(REPORTS / "ice_bulletins.csv", index=False)
    e_now = (x["расчёт по погоде, см"] - x["бюллетень, см"]).abs()
    e_clim = (x["расчёт по климату, см"] - x["бюллетень, см"]).abs()
    rng = x["в диапазоне (погода)"].dropna()
    rng_c = x["в диапазоне (климат)"].dropna()
    summary = pd.DataFrame([
        {"способ": "закон Стефана по погоде текущей зимы (сервис)", "замеров": len(x),
         "MAE к середине диапазона, см": round(e_now.mean(), 1), "в диапазоне бюллетеня": f"{int(rng.sum())} из {len(rng)}"},
        {"способ": "та же формула по климату прошлых 10 зим", "замеров": len(x),
         "MAE к середине диапазона, см": round(e_clim.mean(), 1), "в диапазоне бюллетеня": f"{int(rng_c.sum())} из {len(rng_c)}"}])
    lines = ["# Толщина льда по погоде против бюллетеней УГМС 2021–2026", "",
             f"α = {alpha:.2f} подобран по замерам гидропостов до 2012 г.; бюллетени 2021–2026 модель не видела. "
             "Бюллетень — диапазон по участку реки, сравнение с серединой диапазона грубое: важнее попадание "
             "в диапазон.", "", md_table(summary), "", md_table(x.drop(columns=["источник"])), "",
             "Источники — колонка `source` в [ice_thickness_bulletins.csv](../data/external/ice_thickness_bulletins.csv)."]
    (REPORTS / "ice_bulletins.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
