"""
Разбор замеров льда на гидропостах Росгидромета.

Источник: Shiklomanov A., Lammers R. et al. «Data of river ice timing and thickness
for selected river gauges in Russian pan-Arctic over 1970-2012», Arctic Data Center,
doi:10.18739/A22Z12Q02 (более ранняя версия doi:10.18739/A2PW9R).

Формат файла Gauge_XXXXX.csv:
- переводы строк CR (старый формат Mac);
- две строки заголовка: месяц (октябрь…май) и день (5, 10, 15, 20, 25, последний);
- далее по строке на гидрологический год Y (октябрь–декабрь относятся к Y-1, январь–май к Y):
  год, 48 значений толщины льда в см (пусто = не мерили, «Frozen» = промерзание до дна),
  затем 5 дат ДД/ММ/ГГГГ: первый лёд, ледостав, начало таяния, начало ледохода, конец льда.
"""

import calendar
import csv
import datetime as dt
import io
from pathlib import Path

import pandas as pd

from .paths import RAW

HYDRO_DIR = RAW / "hydro_ice"
SOURCE = "Shiklomanov et al., Arctic Data Center, doi:10.18739/A22Z12Q02"

MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]
DAYS = [5, 10, 15, 20, 25, None]  # None = последний день месяца
EVENTS = ["first_ice", "freezeup", "melt_start", "drift_start", "ice_end"]


def _measure_date(hy: int, month: int, day):
    year = hy - 1 if month >= 10 else hy
    if day is None:
        day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, day)


def _parse_date(s: str):
    s = s.replace(" ", "")  # встречается «23 /04/1987»
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def read_gauge(path: Path):
    """Возвращает (thickness, events) — два «длинных» DataFrame для одного поста."""
    code = int(path.stem.split("_")[1])
    text = path.read_text(encoding="latin-1").replace("\r\n", "\n").replace("\r", "\n")
    rows = list(csv.reader(io.StringIO(text)))
    thick, events = [], []
    for row in rows[2:]:
        if not row or not row[0].strip().isdigit():
            continue
        hy = int(row[0])
        vals = (row + [""] * 60)[1:49]
        for k, raw in enumerate(vals):
            raw = raw.strip()
            if not raw:
                continue
            month, day = MONTHS[k // 6], DAYS[k % 6]
            frozen = raw.lower() == "frozen"
            try:
                value = None if frozen else float(raw)
            except ValueError:
                continue
            thick.append({
                "gauge_code": code, "hydro_year": hy,
                "date": _measure_date(hy, month, day).isoformat(),
                "ice_cm": value, "frozen_to_bottom": frozen,
            })
        dates = (row + [""] * 60)[49:54]
        for name, raw in zip(EVENTS, dates):
            raw_s = raw.strip()
            d = _parse_date(raw_s)
            events.append({
                "gauge_code": code, "hydro_year": hy, "event": name,
                "date": d.isoformat() if d else None, "raw": raw_s,
            })
    return pd.DataFrame(thick), pd.DataFrame(events)


def read_stations():
    st = pd.read_csv(HYDRO_DIR / "River_ice_station_attributes.csv")
    st = st.rename(columns={"Code": "gauge_code", "Name": "name", "Lat": "lat", "Long": "lon",
                            "DArea": "drainage_km2", "Region": "region", "BasinName": "basin"})
    return st[["gauge_code", "name", "lat", "lon", "drainage_km2", "region", "basin"]]


def read_all():
    th, ev = [], []
    for p in sorted(HYDRO_DIR.glob("Gauge_*.csv")):
        t, e = read_gauge(p)
        th.append(t)
        ev.append(e)
    thick = pd.concat(th, ignore_index=True)
    events = pd.concat(ev, ignore_index=True)
    for df in (thick, events):
        df["source_type"] = "замер"
        df["source"] = SOURCE
    return read_stations(), thick, events
