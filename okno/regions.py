"""
Регионы сервиса. Новый регион добавляется сюда: список гидропостов (с историей дат ледовых
явлений), центр карты и подписи. Регион включается, только если проверка переносимости
(scripts/transfer_test.py) подтвердила, что модель там точнее календаря.

Коми не включена: на единственном посту модель хуже календаря (см. reports/transferability.md).
"""

from .northeast import STATIONS as _NE_STATIONS

REGIONS = {
    "yakutia": {
        "name": "Якутия", "center": (65.0, 132.0), "zoom": 2.6, "default": "gauge_3414",
        "gauges": [3156, 3157, 3202, 3210, 3219, 3248, 3277, 3403, 3414, 3483, 3489, 3507, 3518,
                   1018, 1367, 1578, 1801, 1802, 1805],
        "extra_points": ["yakutsk_crossing"],
        "note": "Основной регион (вкл. 6 постов Колымы PANGAEA и сводки ЯУГМС 2022–2026). "
                "Открытие: точнее календаря за 14–28 дней; закрытие у переправы Якутск и постов — с проверкой 2022–2026.",
    },
    "yanao": {
        "name": "ЯНАО", "center": (65.5, 73.0), "zoom": 4.0, "default": "gauge_11558",
        "gauges": [11558, 11574], "extra_points": [],
        "note": "Открытие: −15% ошибки к календарю за 14 дней; закрытие — на уровне календаря.",
    },
    "krasnoyarsk": {
        "name": "Красноярский край", "center": (62.5, 89.0), "zoom": 3.6, "default": "gauge_9425",
        "gauges": [9372, 9380, 9384, 9422, 9425], "extra_points": [],
        "note": "Открытие: −26% ошибки к календарю за 14 дней; закрытие — на уровне календаря.",
    },
    "irkutsk": {
        "name": "Иркутская обл.", "center": (56.5, 109.0), "zoom": 3.8, "default": "gauge_3198",
        "gauges": [3003, 3074, 3087, 3198], "extra_points": [],
        "note": "Открытие: −25% ошибки к календарю за 14 дней; закрытие — на уровне календаря.",
    },
    "arkhangelsk": {
        "name": "Архангельская обл.", "center": (63.3, 42.0), "zoom": 4.6, "default": "gauge_70334",
        "gauges": [70043, 70047, 70275, 70302, 70334, 70366], "extra_points": [],
        "note": "⚠ Подтверждено слабо: лёд под 20 т здесь бывает редко, сравнивать мало с чем; закрытие — на уровне календаря.",
    },
}

GAUGE_NAMES = {
    3156: "Нюя — Курум (Беченча)", 3157: "Большой Патом — Патома", 3202: "Намана — Мекимдя",
    3210: "Ботома — Бролог", 3219: "Алдан — Томмот", 3248: "Тимптон — Усть-Тимптон",
    3277: "Аллах-Юнь — Аллах-Юнь", 3403: "Малая Куонапка — Джалинда (Жилинда)",
    3414: "Яна — Верхоянск (зимник на Батагай)", 3483: "Бытантай — Асар", 3489: "Индигирка — Индигирский",
    3507: "Алгий — устье", 3518: "Нера — Ала-Чубук",
    11558: "Полуй — Полуй (район Салехарда)", 11574: "Пякупур — Тарко-Сале",
    9372: "Большой Пит — Брянка", 9380: "Кас — Александровский", 9384: "Сым — Сым",
    9422: "Ерачимо — Большой Порог", 9425: "Турухан — Янов Стан",
    3003: "Лена — Качуг", 3074: "Кута — Ново-Ильинка", 3087: "Киренга — Шорохово",
    3198: "Жуя — Светлый (Бодайбинский р-н)",
    70043: "Кодина — Кодино", 70047: "Солза — Сухие Пороги", 70275: "Вага — Филяевская",
    70302: "Лель — Зеленинская", 70334: "Пинега — Кулогоры", 70366: "Кулой — Кулой",
    70378: "Мезень — Макарьб",
}
GAUGE_NAMES.update({c: v[0] for c, v in _NE_STATIONS.items()})

POINT_NAMES = {"yakutsk_crossing": "Переправа Якутск — Нижний Бестях (Лена)"}


def basin_points() -> dict:
    """Точки бассейна Лены (data/lena_points.csv) → регион. Верховья Лены в Иркутской обл., остальное — Якутия
    (граница приближённая: западнее 119,5° в.д. и южнее 59,5° с.ш. — Иркутская обл.)."""
    from pathlib import Path
    import csv
    f = Path(__file__).resolve().parent.parent / "data" / "lena_points.csv"
    if not f.exists():
        return {}
    out = {}
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            la, lo = float(r["lat"]), float(r["lon"])
            out[r["point_id"]] = {"lat": la, "lon": lo,
                                  "region": "irkutsk" if (lo < 119.5 and la < 59.5) else "yakutia"}
    return out


def points_of(key: str) -> list[str]:
    """Все точки региона: переправы, гидропосты, точки бассейна Лены."""
    r = REGIONS[key]
    basin = [p for p, v in basin_points().items() if v["region"] == key]
    return list(r["extra_points"]) + [f"gauge_{c}" for c in r["gauges"]] + basin


def all_gauges() -> list[int]:
    return [g for r in REGIONS.values() for g in r["gauges"]]


def all_training_gauges() -> list[int]:
    """Все посты сервиса плюс посты PANGAEA (Магадан, Чукотка) для обучения М1."""
    base = all_gauges()
    extra = [c for c in _NE_STATIONS if c not in base]
    return base + extra


def region_of(point_id: str) -> str | None:
    if point_id.startswith("lena_"):
        b = basin_points().get(point_id)
        return b["region"] if b else None
    for key, r in REGIONS.items():
        if point_id in r["extra_points"]:
            return key
        if point_id.startswith("gauge_") and int(point_id.split("_")[1]) in r["gauges"]:
            return key
    return None


def label(point_id: str) -> str:
    if point_id in POINT_NAMES:
        return POINT_NAMES[point_id]
    if point_id.startswith("lena_"):
        b = basin_points().get(point_id, {})
        la, lo = f"{b.get('lat', 0):.1f}".replace(".", ","), f"{b.get('lon', 0):.1f}".replace(".", ",")
        return f"Бассейн Лены · {la}° с.ш., {lo}° в.д."
    code = int(point_id.split("_")[1])
    return GAUGE_NAMES.get(code, f"Гидропост {code}")
