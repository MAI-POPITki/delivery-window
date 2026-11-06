"""
Точки по всему руслу Лены со спутниковыми датами льда (Qiu et al. 2026, 2001–2024) — дополнительные
современные сезоны для прогноза закрытия (М1).

    python scripts/lena_points.py            # точки + даты
    python scripts/lena_points.py --weather  # плюс погода ERA5 в точках (Open-Meteo, с 1995 года)

Точки выбираются из пикселей набора (сетка 3 км, пиксель с данными = русло): идём с юга на север и
берём пиксель, если он дальше MIN_KM от уже выбранных и от переправы Якутск (она уже есть в модели).
В точке — медиана дат 3×3 пикселя, как и для переправы.

Выход: data/lena_points.csv, data/processed/lena_sat_seasons.csv, data/weather/lena/<id>.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from okno.paths import DATA, PROCESSED, RAW

MIN_KM = 220
YAKUTSK = (62.0, 129.82)
SOURCE = "Qiu et al. 2026, ESSD, doi:10.5281/zenodo.17054619"


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def river_pixels():
    import rasterio
    from pyproj import Transformer
    tifs = sorted((RAW / "lena_phenology").glob("breakup_DOHY_*.tif"))
    with rasterio.open(tifs[-1]) as src:
        a = src.read(1).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        valid = np.isfinite(a)
        # пиксель — русло, если данные есть хотя бы в половине лет (отсекаем случайные пиксели)
        cnt = np.zeros_like(a)
        for t in tifs:
            with rasterio.open(t) as s2:
                b = s2.read(1).astype(float)
                if s2.nodata is not None:
                    b[b == s2.nodata] = np.nan
                cnt += np.isfinite(b)
        valid &= cnt >= len(tifs) / 2
        rr, cc = np.where(valid)
        xs, ys = rasterio.transform.xy(src.transform, rr, cc)
        lo, la = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    return pd.DataFrame({"lat": la, "lon": lo})


def pick_points(px: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for r in px.sort_values("lat").itertuples():
        p = (r.lat, r.lon)
        if km(p, YAKUTSK) < MIN_KM / 2:
            continue
        if all(km(p, q) >= MIN_KM for q in picked):
            picked.append(p)
    df = pd.DataFrame(picked, columns=["lat", "lon"]).round(4)
    df.insert(0, "point_id", [f"lena_{i:02d}" for i in range(len(df))])
    return df


def sample_dates(points: pd.DataFrame) -> pd.DataFrame:
    from download_data import sample_tif
    rows = []
    for p in points.itertuples():
        for kind in ("freezeup", "breakup"):
            for tif in sorted((RAW / "lena_phenology").glob(f"{kind}_DOHY_*.tif")):
                hy = int(tif.stem.rsplit("_", 1)[1])
                v, n, _ = sample_tif(tif, p.lat, p.lon, radius_px=1)
                rows.append({"point_id": p.point_id, "hydro_year": hy, "kind": kind, "dohy": v})
    d = pd.DataFrame(rows).pivot_table(index=["point_id", "hydro_year"], columns="kind", values="dohy").reset_index()
    # DOHY: 1 = 1 октября; день сезона: 0 = 1 сентября → +29 (как для переправы Якутск)
    out = pd.DataFrame({"point_id": d.point_id, "hydro_year": d.hydro_year,
                        "sat_breakup_day": d.get("breakup") + 29, "sat_freezeup_day": d.get("freezeup") + 29})
    out["source_type"] = "спутник"
    out["source"] = SOURCE
    return out


def download_weather(points: pd.DataFrame):
    import download_data as D
    out = DATA / "weather" / "lena"
    out.mkdir(parents=True, exist_ok=True)
    for p in points.itertuples():
        D.safe_update({"id": p.point_id, "lat": p.lat, "lon": p.lon}, out / f"{p.point_id}.csv", "1995-01-01",
                      D.default_end())
    if D.FAILED:
        print("не скачано:", ", ".join(D.FAILED))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", action="store_true")
    a = ap.parse_args()
    px = river_pixels()
    pts = pick_points(px)
    pts.to_csv(DATA / "lena_points.csv", index=False)
    sat = sample_dates(pts)
    sat.to_csv(PROCESSED / "lena_sat_seasons.csv", index=False)
    print(f"пикселей русла: {len(px)}, точек: {len(pts)} (широта {pts.lat.min():.1f}–{pts.lat.max():.1f})")
    print(f"сезонов с датой вскрытия: {sat.sat_breakup_day.notna().sum()}")
    if a.weather:
        download_weather(pts)


if __name__ == "__main__":
    main()
