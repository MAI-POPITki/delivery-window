"""
Набор данных для М2: вырезки радарных снимков Sentinel-1 над Леной с автоматическими метками.

    pip install pystac-client planetary-computer      (плюс rasterio, pyproj из requirements-data.txt)
    python scripts/s1_chips.py                        # все точки, все годы (~20–40 мин)
    python scripts/s1_chips.py --points yakutsk_crossing --years 2019 2020   # быстрый пробный прогон

Откуда снимки: Microsoft Planetary Computer, коллекция sentinel-1-rtc — Sentinel-1 GRD, приведённые
к карте (UTM, 10 м). Аккаунт не нужен; из облачного файла читается только окно вокруг точки,
сцена целиком (1–2 ГБ) не скачивается.

Покрытие над Якутией: 2017–2021 — снимок примерно раз в 12 дней; 2022–2024 — снимков нет
(отказ Sentinel-1B); с 2025 — снова есть (Sentinel-1C).

Метки: спутниковые даты ледостава и вскрытия Лены (Qiu et al. 2026, ESSD), 2001–2024, сетка 3 км,
берутся в той же точке. Для даты снимка d в гидрологическом году hy:
    ice        — ледостав(hy) + 7 сут ≤ d ≤ вскрытие(hy) − 7 сут
    water      — после вскрытия(hy) + 7 сут и до ледостава(hy+1) − 7 сут
    transition — в пределах ±7 сут от ледостава или вскрытия (ледоход, шуга, неточность меток)
    unknown    — нет меток (сезоны после 2024) — для проверки на живых данных
Точность самих меток ≈ 11 сут к гидропостам (по статье авторов), поэтому «переход» лучше
исключать из обучения или учить как отдельный класс.

Выход:
    data/s1/chips/<точка>/<дата>_<id>.npz   — массивы vv_db, vh_db (300×300, float16), в git не идут
    data/s1/index.csv                       — по строке на вырезку: точка, дата, орбита, метка,
                                              дни до/после событий, статистики обратного рассеяния
    reports/s1_preview.png                  — примеры вырезок «лёд / вода» для проверки глазами
"""

import argparse
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.paths import DATA, RAW, REPORTS
from okno.s1_points import LENA_POINTS

OUT = DATA / "s1"
CHIP_HALF_M = 1500  # вырезка 3 × 3 км
MARGIN = 7          # сут вокруг событий — «переход»
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

def snap_to_river(lat, lon, radius_px=8):
    """Ближайший пиксель набора Qiu с данными (то есть русло Лены) в радиусе radius_px."""
    import rasterio
    from pyproj import Transformer
    tif = sorted((RAW / "lena_phenology").glob("breakup_DOHY_*.tif"))[-1]
    with rasterio.open(tif) as src:
        to = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        x, y = to.transform(lon, lat)
        r0, c0 = src.index(x, y)
        win = rasterio.windows.Window(c0 - radius_px, r0 - radius_px, 2 * radius_px + 1, 2 * radius_px + 1)
        a = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        rr, cc = np.where(np.isfinite(a))
        if not len(rr):
            return None
        k = np.argmin((rr - radius_px) ** 2 + (cc - radius_px) ** 2)
        xs, ys = src.xy(r0 - radius_px + rr[k], c0 - radius_px + cc[k])
        lo, la = back.transform(xs, ys)
        return float(la), float(lo)


def ice_dates(lat, lon):
    """Даты ледостава и вскрытия по годам в точке (набор Qiu, медиана 3×3 пикселя)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_data import dohy_to_date, sample_tif
    out = {}
    for kind in ("freezeup", "breakup"):
        for tif in sorted((RAW / "lena_phenology").glob(f"{kind}_DOHY_*.tif")):
            hy = int(tif.stem.rsplit("_", 1)[1])
            v, n, _ = sample_tif(tif, lat, lon, radius_px=1)
            if v is not None:
                out.setdefault(hy, {})[kind] = pd.Timestamp(dohy_to_date(hy, v))
    return out


def label(d: pd.Timestamp, dates: dict):
    hy = d.year + (1 if d.month >= 9 else 0)
    cur, nxt = dates.get(hy, {}), dates.get(hy + 1, {})
    fz, br = cur.get("freezeup"), cur.get("breakup")
    nfz = nxt.get("freezeup")
    m = pd.Timedelta(days=MARGIN)
    info = {"hydro_year": hy, "freezeup": fz.date() if fz is not None else None,
            "breakup": br.date() if br is not None else None,
            "days_from_freezeup": (d - fz).days if fz is not None else None,
            "days_to_breakup": (br - d).days if br is not None else None}
    if fz is None or br is None:
        return "unknown", info
    if abs((d - fz).days) <= MARGIN or abs((d - br).days) <= MARGIN:
        return "transition", info
    if fz + m < d < br - m:
        return "ice", info
    if d > br + m and (nfz is None or d < nfz - m):
        return "water" if nfz is not None or d.month < 9 else "unknown", info
    if d < fz - m:  # осень до ледостава
        return "water", info
    return "unknown", info


def read_chip(item, lat, lon):
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
    out = {}
    for pol in ("vv", "vh"):
        with rasterio.open(item.assets[pol].href) as src:
            x, y = transform("EPSG:4326", src.crs, [lon], [lat])
            w = from_bounds(x[0] - CHIP_HALF_M, y[0] - CHIP_HALF_M, x[0] + CHIP_HALF_M, y[0] + CHIP_HALF_M,
                            src.transform)
            a = src.read(1, window=w, boundless=True, fill_value=0).astype("float32")
        a[a <= 0] = np.nan
        out[pol] = 10 * np.log10(a)  # дБ
    return out


def process_point(key, ranges, cat, workers=6):
    name, lat0, lon0 = LENA_POINTS[key]
    snapped = snap_to_river(lat0, lon0)
    if snapped is None:
        print(f"{key}: русло рядом не найдено, пропускаю")
        return []
    lat, lon = snapped
    dates = ice_dates(lat, lon)
    items = []
    for rng in ranges:
        items += list(cat.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                 datetime=rng).items())
    print(f"{key} ({name}): русло {lat:.3f}, {lon:.3f}; снимков {len(items)}")
    (OUT / "chips" / key).mkdir(parents=True, exist_ok=True)

    def one(item):
        d = pd.Timestamp(item.datetime).tz_localize(None).normalize()
        path = OUT / "chips" / key / f"{d.date()}_{item.id[-12:]}.npz"
        if not path.exists():
            ch = read_chip(item, lat, lon)
            np.savez_compressed(path, vv_db=ch["vv"].astype("float16"), vh_db=ch["vh"].astype("float16"))
        else:
            z = np.load(path)
            ch = {"vv": z["vv_db"].astype("float32"), "vh": z["vh_db"].astype("float32")}
        lab, info = label(d, dates)
        return {"point": key, "point_name": name, "lat": round(lat, 4), "lon": round(lon, 4), "date": d.date(),
                "item_id": item.id, "orbit": item.properties.get("sat:orbit_state"),
                "platform": item.properties.get("platform"), "label": lab, **info,
                "vv_median_db": float(np.nanmedian(ch["vv"])), "vh_median_db": float(np.nanmedian(ch["vh"])),
                "vv_std_db": float(np.nanstd(ch["vv"])), "valid_share": float(np.isfinite(ch["vv"]).mean()),
                "file": str(path.relative_to(DATA.parent)).replace("\\", "/")}

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, it) for it in items]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as e:  # noqa: BLE001 — один битый снимок не останавливает остальные
                print(f"  {key}: снимок пропущен ({type(e).__name__}: {str(e)[:80]})")
    return rows


def preview(idx: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = pd.concat([idx[idx.label == lab].sample(min(4, (idx.label == lab).sum()), random_state=1)
                      for lab in ("water", "ice", "transition") if (idx.label == lab).any()])
    fig, axes = plt.subplots(len(pick) // 4 + (len(pick) % 4 > 0), 4, figsize=(12, 3.2 * (len(pick) // 4 + 1)))
    for ax, (_, r) in zip(np.ravel(axes), pick.iterrows()):
        z = np.load(DATA.parent / r.file)
        ax.imshow(z["vv_db"].astype(float), cmap="gray", vmin=-25, vmax=0)
        ax.set_title(f"{r.label} · {r.point} · {r.date}", fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(pick):]:
        ax.axis("off")
    fig.suptitle("Sentinel-1 VV, дБ: примеры вырезок 3×3 км с автоматическими метками", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "s1_preview.png", dpi=90)


def main():
    import planetary_computer as pc
    import pystac_client
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*", default=list(LENA_POINTS))
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2015, dt.date.today().year + 1)))
    ap.add_argument("--days", type=int, default=None,
                    help="только снимки за последние N дней (ежедневное обновление в GitHub Actions)")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in a.points:
        if a.days:
            ranges = [f"{dt.date.today() - dt.timedelta(days=a.days)}/{dt.date.today()}"]
        else:
            ranges = [f"{y}-01-01/{y}-12-31" for y in a.years]
        rows += process_point(key, ranges, cat)
    new = pd.DataFrame(rows)
    idx_path = OUT / "index.csv"
    if idx_path.exists() and len(new):
        old = pd.read_csv(idx_path)
        new = pd.concat([old[~old.item_id.isin(new.item_id) | ~old.point.isin(new.point)], new])
    new = new.sort_values(["point", "date"]).reset_index(drop=True)
    new.to_csv(idx_path, index=False)
    print(new.groupby(["point", "label"]).size().unstack(fill_value=0).to_string())
    if len(new) and not a.no_preview:
        preview(new)
    print("готово:", idx_path)


if __name__ == "__main__":
    main()
