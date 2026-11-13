"""Общие ориентиры М2; фактические точки русла записаны в data/s1/index.csv."""
LENA_POINTS = {
    "lensk": ("Ленск", 60.72, 114.92),
    "olekminsk": ("Олёкминск", 60.38, 120.42),
    "pokrovsk": ("Покровск", 61.48, 129.15),
    "yakutsk_crossing": ("Переправа Якутск — Нижний Бестях", 62.00, 129.82),
    "sangar": ("Сангар", 63.92, 127.47),
    "zhigansk": ("Жиганск", 66.77, 123.37),
    "kyusyur": ("Кюсюр", 70.68, 127.37),
}
LENA_NAMES = {key: point[0] for key, point in LENA_POINTS.items()}


def river_points(*, half_size_m=1500, index_path=None):
    """Common river locations; half_size controls each provider's search window."""
    import csv
    import math
    from pathlib import Path
    from .paths import DATA

    result = {key: {"name": name, "lat": lat, "lon": lon, "coordinate_source": "landmark"}
              for key, (name, lat, lon) in LENA_POINTS.items()}
    try:
        with Path(index_path or DATA / "s1" / "index.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("point") not in result:
                    continue
                try:
                    lat, lon = float(row["lat"]), float(row["lon"])
                    if not (math.isfinite(lat) and math.isfinite(lon) and -90 < lat < 90 and -180 <= lon <= 180):
                        continue
                except (ValueError, KeyError, TypeError):
                    continue
                result[row["point"]].update(lat=lat, lon=lon, coordinate_source="sentinel_1_index")
    except OSError:
        pass
    for point in result.values():
        dy = half_size_m / 111320
        dx = dy / math.cos(math.radians(point["lat"]))
        point["bbox"] = [point["lon"]-dx, point["lat"]-dy, point["lon"]+dx, point["lat"]+dy]
    return result
