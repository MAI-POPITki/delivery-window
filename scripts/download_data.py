"""
Загрузка открытых данных для проекта «Окно завоза».

Запуск (на компьютере с обычным интернетом):
    pip install requests pandas rasterio pyproj
    python download_data.py            # всё
    python download_data.py --weather  # только погода
    python download_data.py --ice      # только спутниковые даты льда Лены
    python download_data.py --extract-only  # только пересчитать CSV из уже скачанных TIFF

Что скачивается:
1. Погода ERA5 (через Open-Meteo, без ключа): суточная температура и осадки
   с 1979 года для точек переправ и центров арктических районов Якутии.
2. Набор Qiu et al. 2026 (ESSD, doi:10.5281/zenodo.17054619):
   спутниковые даты ледостава и вскрытия Лены 2001–2024 (GeoTIFF),
   из которых берутся значения в точке переправы Якутск — Нижний Бестях.

Результат: папка data/ с CSV-файлами. Каждый файл подписан источником.
"""

import argparse
import datetime as dt
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from okno.regions import all_gauges  # noqa: E402 — посты всех подтверждённых регионов

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"

# Координаты приблизительные (центр населённого пункта / русло рядом).
# kind: crossing = ледовая переправа, district = центр арктического района.
POINTS = [
    {"id": "yakutsk_crossing", "name": "Переправа Якутск — Нижний Бестях", "lat": 62.00, "lon": 129.82, "kind": "crossing"},
    {"id": "yakutsk", "name": "Якутск", "lat": 62.03, "lon": 129.73, "kind": "city"},
    {"id": "zhigansk", "name": "Жиганск", "lat": 66.77, "lon": 123.37, "kind": "district"},
    {"id": "olenek", "name": "Оленёк", "lat": 68.50, "lon": 112.43, "kind": "district"},
    {"id": "saskylakh", "name": "Саскылах", "lat": 71.95, "lon": 114.08, "kind": "district"},
    {"id": "tiksi", "name": "Тикси", "lat": 71.64, "lon": 128.87, "kind": "district"},
    {"id": "batagay", "name": "Батагай", "lat": 67.65, "lon": 134.63, "kind": "district"},
    {"id": "chokurdakh", "name": "Чокурдах", "lat": 70.62, "lon": 147.90, "kind": "district"},
    {"id": "zyryanka", "name": "Зырянка", "lat": 65.73, "lon": 150.89, "kind": "district"},
    {"id": "chersky", "name": "Черский", "lat": 68.75, "lon": 161.33, "kind": "district"},
]

OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_mean,temperature_2m_min,temperature_2m_max,snowfall_sum,precipitation_sum"
ZENODO_RECORD = "17054619"


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- погода
def _get_weather_chunk(p, start, end, tries=8):
    params = {
        "latitude": p["lat"], "longitude": p["lon"],
        "start_date": start, "end_date": end,
        "daily": DAILY_VARS, "timezone": "auto",
    }
    for attempt in range(tries):
        try:
            r = requests.get(OPEN_METEO, params=params, timeout=120)
        except requests.exceptions.RequestException as e:  # обрыв соединения
            log(f"  {p['id']} {start[:4]}: {type(e).__name__}, повтор {attempt + 1}")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:  # лимит запросов
            log(f"  {p['id']} {start[:4]}: лимит запросов, жду")
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code >= 500:  # сервер временно недоступен
            log(f"  {p['id']} {start[:4]}: ошибка сервера {r.status_code}, повтор {attempt + 1}")
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        try:
            return pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
        except (ValueError, KeyError):  # пустой или битый ответ — бывает под нагрузкой
            log(f"  {p['id']} {start[:4]}: битый ответ, повтор {attempt + 1}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"не удалось скачать погоду {p['id']} {start}–{end}")


def default_end():
    # архив Open-Meteo/ERA5 отстаёт от сегодняшнего дня на несколько суток
    return (dt.date.today() - dt.timedelta(days=7)).isoformat()


def update_point(p, out: Path, start: str, end: str):
    """Скачивает ряд в файл; если файл уже есть — только недостающий хвост (для ежедневного обновления)."""
    old = pd.read_csv(out) if out.exists() else None
    p_start = start
    if old is not None:
        last = pd.Timestamp(old["date"].max())
        if last >= pd.Timestamp(end) - pd.Timedelta(days=1):
            log(f"погода {p['id']}: актуальна (до {last.date()})")
            return
        p_start = (last + pd.Timedelta(days=1)).date().isoformat()
    parts = []
    for chunk_start in range(int(p_start[:4]), int(end[:4]) + 1, 10):  # куски по 10 лет: длинные запросы сервер обрывает
        c_start = max(p_start, f"{chunk_start}-01-01")
        c_end = min(end, f"{chunk_start + 9}-12-31")
        parts.append(_get_weather_chunk(p, c_start, c_end))
        time.sleep(3)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "point_id", p["id"])
    df["source"] = "ERA5 via Open-Meteo archive API"
    df = df.dropna(subset=["temperature_2m_mean"])  # самые свежие дни архив иногда отдаёт пустыми
    if old is not None:
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("date", keep="last")
    df.to_csv(out, index=False)
    log(f"погода {p['id']}: {len(df)} дней, до {df['date'].max()}")
    time.sleep(2)


FAILED = []


def safe_update(p, out, start, end):
    """Сбой на одной точке не останавливает загрузку остальных; недокачанное подтянется в следующий раз."""
    try:
        update_point(p, out, start, end)
    except Exception as e:  # noqa: BLE001
        FAILED.append(p["id"])
        log(f"погода {p['id']}: НЕ скачана ({e}); пропускаю, докачается при следующем запуске")


def download_weather(start="1979-01-01", end=None):
    end = end or default_end()
    out_dir = DATA / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(POINTS).to_csv(DATA / "points.csv", index=False)
    for p in POINTS:
        if p["kind"] == "city":  # та же ячейка ERA5, что и переправа
            continue
        # Open-Meteo считает длинные запросы за много вызовов (лимит ~5000 в час с одного IP),
        # поэтому полный ряд с 1979 года берём только для переправы, для районов — с 2000 года
        p_start = start if p["kind"] == "crossing" else "2000-01-01"
        safe_update(p, out_dir / f"{p['id']}.csv", p_start, end)


HYDRO_DOI = "doi:10.18739/A22Z12Q02"  # новая версия набора doi:10.18739/A2PW9R
ADC = "https://arcticdata.io/metacat/d1/mn/v2"


def download_hydro():
    """Замеры толщины и даты ледовых явлений на 31 гидропосте (Arctic Data Center)."""
    out_dir = RAW / "hydro_ice"
    out_dir.mkdir(parents=True, exist_ok=True)
    q = {"q": f'isDocumentedBy:"{HYDRO_DOI}"', "fl": "id,fileName", "rows": 200, "wt": "json"}
    docs = requests.get(f"{ADC}/query/solr/", params=q, timeout=60).json()["response"]["docs"]
    files = [d for d in docs if d.get("fileName", "").endswith((".csv", ".txt"))]
    for d in files:
        dest = out_dir / d["fileName"]
        if dest.exists():
            continue
        r = requests.get(f"{ADC}/object/{d['id']}", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    log(f"гидропосты: {len(files)} файлов в {out_dir}")


# Якутские гидропосты набора Шикломанова (замеры толщины льда 1970–2012).
# Погода нужна, чтобы откалибровать формулу Стефана на реальных замерах.
# Первыми — Яна у Верхоянска (рядом зимник на Батагай) и Ботома (ближайший пост к Якутску).
GAUGE_PRIORITY = [3414, 3210, 3483, 3219, 3202, 3156, 3157, 3248, 3277, 3403, 3489, 3507, 3518]


# Посты вне Якутии — для проверки переносимости метода на другие регионы
# (ЯНАО, Красноярский край, Иркутская и Архангельская обл., Коми); ряд только до 2012 —
# дальше замеров нет.
GAUGE_OTHER = [11558, 11574, 9425, 9422, 9372, 9380, 9384, 3003, 3074, 3087, 3198,
               70334, 70378, 70275, 70302, 70366, 70043, 70047]


def download_gauge_weather(start="1979-01-01", end=None, codes=None):
    """Погода в точках постов. Если файл уже есть — докачивает только недостающий хвост
    (ряд нужен до сегодняшнего дня для живого прогноза текущего сезона)."""
    end = end or (dt.date.today() - dt.timedelta(days=7)).isoformat()
    st = pd.read_csv(RAW / "hydro_ice" / "River_ice_station_attributes.csv")
    st = st.set_index("Code")
    out_dir = DATA / "weather" / "gauges"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in (codes or all_gauges()):
        p = {"id": f"gauge_{code}", "lat": float(st.loc[code, "Lat"]), "lon": float(st.loc[code, "Long"])}
        safe_update(p, out_dir / f"gauge_{code}.csv", start, end)


def lena_basin_points():
    """Точки бассейна Лены (data/lena_points.csv, scripts/lena_points.py)."""
    f = DATA / "lena_points.csv"
    if not f.exists():
        return []
    return [{"id": r.point_id, "lat": r.lat, "lon": r.lon} for r in pd.read_csv(f).itertuples()]


def download_lena_weather(start="1995-01-01", end=None, max_new=8):
    """Погода в точках бассейна Лены: остальным — хвост до сегодня; новых (полная история с 1995) — не больше
    max_new за запуск, чтобы ежедневная задача не упёрлась в лимит Open-Meteo; остальные докачаются в следующие дни."""
    end = end or default_end()
    out_dir = DATA / "weather" / "lena"
    out_dir.mkdir(parents=True, exist_ok=True)
    new = 0
    for p in lena_basin_points():
        out = out_dir / f"{p['id']}.csv"
        if not out.exists():
            if new >= max_new:
                continue
            new += 1
        safe_update(p, out, start, end)


# ---------------------------------------------------------------- оперативная погода
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def download_recent(past_days=14, forecast_days=16):
    """Последние дни (архив ERA5 отстаёт на ~неделю) и прогноз на 16 дней из оперативного API Open-Meteo.
    Перезаписывается при каждом запуске; в обучении не участвует — только в живом прогнозе."""
    out_dir = DATA / "weather_recent"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(RAW / "hydro_ice" / "River_ice_station_attributes.csv").set_index("Code")
    pts = [p for p in POINTS if p["kind"] != "city"] + [
        {"id": f"gauge_{c}", "lat": float(st.loc[c, "Lat"]), "lon": float(st.loc[c, "Long"])} for c in all_gauges()] +         lena_basin_points()
    today = dt.date.today().isoformat()
    for p in pts:
        params = {"latitude": p["lat"], "longitude": p["lon"], "daily": DAILY_VARS, "timezone": "auto",
                  "past_days": past_days, "forecast_days": forecast_days}
        try:
            for attempt in range(5):
                r = requests.get(OPEN_METEO_FORECAST, params=params, timeout=60)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(20 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
        except Exception as e:  # noqa: BLE001
            FAILED.append(p["id"])
            log(f"оперативная погода {p['id']}: НЕ скачана ({e})")
            continue
        df.insert(0, "point_id", p["id"])
        df["is_forecast"] = df["date"] > today
        df["source"] = ["Open-Meteo прогноз" if f else "Open-Meteo оперативный анализ" for f in df["is_forecast"]]
        df["fetched"] = today
        df.to_csv(out_dir / f"{p['id']}.csv", index=False)
        time.sleep(0.5)
    log(f"оперативная погода: {len(pts) - len([f for f in FAILED])} точек, прогноз до {df['date'].max()}")


# ---------------------------------------------------------------- лёд (спутник)
def download_ice():
    RAW.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60).json()
    # только годовые слои фенологии; дневной архив Lena_RIC_3km.zip (~1 ГБ) не нужен
    files = [f for f in meta["files"] if f["key"].lower().startswith("lena") and "RIC" not in f["key"]]
    log("файлы Лены в записи Zenodo: " + ", ".join(f"{f['key']} ({f['size']/1e6:.0f} МБ)" for f in files))
    for f in files:
        dest = RAW / f["key"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            log(f"{f['key']}: уже скачан")
            continue
        url = f["links"]["self"]
        log(f"качаю {f['key']} ...")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    # распаковываем только годовые слои фенологии (дневные слои не нужны)
    tif_dir = RAW / "lena_phenology"
    tif_dir.mkdir(exist_ok=True)
    for z in RAW.glob("Lena*.zip"):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base.startswith(("freezeup_DOHY_", "breakup_DOHY_", "ice_duration_DAY_")) and base.endswith(".tif"):
                    (tif_dir / base).write_bytes(zf.read(name))
    log(f"распаковано слоёв фенологии: {len(list(tif_dir.glob('*.tif')))}")


def dohy_to_date(hy, dohy):
    """День гидрологического года (1 = 1 октября года hy-1) -> дата."""
    return dt.date(hy - 1, 10, 1) + dt.timedelta(days=int(round(dohy)) - 1)


def sample_tif(path, lat, lon, radius_px=3):
    """Медиана непустых пикселей в окне вокруг точки (русло узкое, окно ловит воду)."""
    import numpy as np
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        row, col = src.index(x, y)
        r0, c0 = max(row - radius_px, 0), max(col - radius_px, 0)
        win = rasterio.windows.Window(c0, r0, 2 * radius_px + 1, 2 * radius_px + 1)
        a = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        vals = a[np.isfinite(a)]
        if vals.size == 0:
            return None, 0, None
        return float(np.median(vals)), int(vals.size), float(vals.max() - vals.min())


def extract_ice_points():
    tif_dir = RAW / "lena_phenology"
    rows = []
    for p in [p for p in POINTS if p["kind"] in ("crossing", "city")]:
        for tif in sorted(tif_dir.glob("*.tif")):
            kind, unit, year = tif.stem.rsplit("_", 2)  # ice_duration_DAY_2020 -> ice_duration, DAY, 2020
            val, n, spread = sample_tif(tif, p["lat"], p["lon"])
            row = {"point_id": p["id"], "variable": kind, "hydro_year": int(year),
                   "value": val, "unit": unit, "n_pixels": n, "spread": spread,
                   "source": "Qiu et al. 2026, ESSD, doi:10.5281/zenodo.17054619"}
            if val is not None and unit == "DOHY":
                row["date"] = dohy_to_date(int(year), val).isoformat()
            rows.append(row)
    df = pd.DataFrame(rows)
    out = DATA / "lena_ice_phenology_points.csv"
    df.to_csv(out, index=False)
    log(f"сохранено {out.name}: {len(df)} строк, пустых значений {df['value'].isna().sum()}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", action="store_true")
    ap.add_argument("--ice", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--gauges", action="store_true", help="погода ERA5 для якутских гидропостов 1979–2012")
    ap.add_argument("--hydro", action="store_true", help="замеры льда на гидропостах (Arctic Data Center)")
    ap.add_argument("--gauges-other", action="store_true", help="погода для постов вне Якутии (тест переносимости)")
    ap.add_argument("--recent", action="store_true", help="последние дни и прогноз на 16 дней (оперативный API)")
    ap.add_argument("--lena", action="store_true", help="погода в точках бассейна Лены (data/lena_points.csv)")
    a = ap.parse_args()
    everything = not (a.weather or a.ice or a.extract_only or a.gauges or a.hydro or a.gauges_other or a.recent or a.lena)
    DATA.mkdir(exist_ok=True)
    if a.weather or everything:
        download_weather()
    if a.hydro or everything:
        download_hydro()
    if a.lena or everything:
        download_lena_weather()
    if a.recent or everything:
        download_recent()
    if a.gauges_other:
        download_gauge_weather(end="2012-12-31", codes=GAUGE_OTHER)
    if a.gauges or everything:
        download_gauge_weather()
    if a.ice or everything:
        download_ice()
    if a.ice or a.extract_only or everything:
        extract_ice_points()
    if FAILED:
        log(f"не скачано: {', '.join(FAILED)} — остальное обновлено")
    log("готово.")
