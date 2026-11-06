"""
Геометрия зимников и ледовых переправ из OpenStreetMap для карты дашборда.

    python scripts/osm_roads.py

Запрос к Overpass API: way[winter_road=yes] и way[ice_road=yes] в прямоугольниках районов (по умолчанию —
все регионы сервиса). Линии упрощаются (каждая n-я точка) и сохраняются в data/osm/winter_roads.json:
[{"name": ..., "kind": "winter_road|ice_road", "lat": [...], "lon": [...]}]. Данные © участники OpenStreetMap, ODbL.
Overpass бывает перегружен — тогда скрипт повторяет запрос; крупный прямоугольник лучше делить на части.
"""

import json
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "osm" / "winter_roads.json"
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
def tiles(s, w, n, e, dlat, dlon):
    """Большой район режем на куски — Overpass не тянет всю Якутию одним запросом."""
    out, la = [], s
    while la < n:
        lo = w
        while lo < e:
            out.append((la, lo, min(la + dlat, n), min(lo + dlon, e)))
            lo += dlon
        la += dlat
    return out


# прямоугольники регионов сервиса (юг, запад, север, восток)
BOXES = {
    "yakutia": tiles(56.0, 105.0, 73.5, 163.0, 6.0, 15.0),
    "yanao": tiles(63.0, 64.0, 69.5, 84.0, 6.5, 10.0),
    "krasnoyarsk": tiles(56.0, 82.0, 70.0, 100.0, 7.0, 9.0),
    "irkutsk": tiles(52.0, 100.0, 60.5, 119.0, 8.5, 10.0),
    "arkhangelsk": tiles(60.0, 37.0, 66.5, 52.0, 6.5, 15.0),
}
STEP = 4  # оставляем каждую 4-ю точку линии


def query(box):
    s, w, n, e = box
    q = (f'[out:json][timeout:120];(way["winter_road"="yes"]({s},{w},{n},{e});'
         f'way["ice_road"="yes"]({s},{w},{n},{e}););out tags geom qt;')
    for attempt in range(6):
        for url in URLS:
            try:
                r = requests.post(url, data={"data": q}, timeout=180,
                                  headers={"User-Agent": "okno-zavoza-research/1.0"})
                if r.ok and r.headers.get("content-type", "").startswith("application/json"):
                    return r.json()["elements"]
            except requests.RequestException:
                pass
        time.sleep(20 * (attempt + 1))
    raise RuntimeError("Overpass недоступен")


def main():
    roads, seen = [], set()
    for region, boxes in BOXES.items():
      for box in boxes:
        try:
            els = query(box)
        except RuntimeError:
            print(f"  {region} {box}: Overpass не ответил, кусок пропущен")
            continue
        print(f"  {region} {box}: участков {len(els)}", flush=True)
        for el in els:
            if el.get("id") in seen:
                continue
            seen.add(el.get("id"))
            g = el.get("geometry") or []
            if len(g) < 2:
                continue
            pts = g[::STEP] + ([g[-1]] if (len(g) - 1) % STEP else [])
            t = el.get("tags", {})
            roads.append({"name": t.get("name", ""), "region": region,
                          "kind": "ice_road" if t.get("ice_road") == "yes" else "winter_road",
                          "lat": [round(p["lat"], 4) for p in pts], "lon": [round(p["lon"], 4) for p in pts]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(roads, ensure_ascii=False), encoding="utf-8")
    print(f"участков: {len(roads)}, точек: {sum(len(r['lat']) for r in roads)} → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
