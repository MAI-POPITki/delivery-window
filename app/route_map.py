"""Схемы двух пилотных путей из проекта и интерактивные слои карты.

В репозитории нет геометрии дорог. Линии соединяют только имеющиеся опорные
точки из data/points.csv и data/processed/gauges.csv и явно помечены как схемы.
"""

import csv
from dataclasses import dataclass
from html import escape

import pydeck as pdk

from okno.paths import DATA, PROCESSED
from okno.regions import POINT_NAMES, REGIONS, label


@dataclass(frozen=True)
class SchematicRoute:
    id: str
    name: str
    point_id: str
    path: tuple[tuple[float, float], ...]  # долгота, широта
    anchors: tuple[str, ...]
    color: tuple[int, int, int]
    note: str


def schematic_routes(region):
    """Только два пилота, названные в scripts/before_after.py; не все гидропосты."""
    if region != "yakutia":
        return []

    with (DATA / "points.csv").open(encoding="utf-8", newline="") as file:
        points = {row["id"]: row for row in csv.DictReader(file)}
    with (PROCESSED / "gauges.csv").open(encoding="utf-8", newline="") as file:
        gauges = {row["gauge_code"]: row for row in csv.DictReader(file)}

    def position(row):
        return float(row["lon"]), float(row["lat"])

    return [
        SchematicRoute(
            id="batagay",
            name="Зимник на Батагай",
            point_id="gauge_3414",
            path=(position(gauges["3414"]), position(points["batagay"])),
            anchors=("Гидропост Яна — Верхоянск", points["batagay"]["name"]),
            color=(37, 99, 235),
            note="Схема: гидропост Яна — Верхоянск → Батагай. Точная трасса зимника не задана.",
        ),
        SchematicRoute(
            id="yakutsk_crossing",
            name=POINT_NAMES["yakutsk_crossing"],
            point_id="yakutsk_crossing",
            path=(position(points["yakutsk"]), position(points["yakutsk_crossing"])),
            anchors=(points["yakutsk"]["name"], "Опорная точка переправы"),
            color=(124, 58, 237),
            note="Схема: Якутск → опорная точка переправы. Полная трасса и береговые въезды не заданы.",
        ),
    ]


def build_route_map(region, points, routes, *, selected_point=None, focus_route=None):
    """Цвет линий различает пути; цвет точек сохраняет вероятность из прогноза."""
    layers = []
    paths = []
    anchors = []
    for route in routes:
        highlighted = route.id == focus_route or (focus_route is None and route.point_id == selected_point)
        tooltip = (f"<b>{escape(route.name)} · схема</b><br>{escape(route.note)}"
                   f"<br>Прогноз рассчитывается в точке: {escape(label(route.point_id))}.")
        paths.append({
            "path": route.path,
            "color": route.color,
            "width": 8 if highlighted else 5,
            "tooltip": tooltip,
        })
        for position, name in zip(route.path, route.anchors):
            anchors.append({
                "position": position,
                "color": route.color,
                "tooltip": f"{tooltip}<br><b>Опорная точка схемы:</b> {escape(name)}",
            })

    if paths:
        # PathLayer проверяет наведение на всю линию, включая середину сегмента.
        layers.extend([
            pdk.Layer(
                "PathLayer", id="route-outlines", data=paths,
                get_path="path", get_color=[255, 255, 255, 230],
                get_width="width + 3", width_units=pdk.types.String("pixels"),
                cap_rounded=True, joint_rounded=True, pickable=False,
            ),
            pdk.Layer(
                "PathLayer", id="schematic-routes", data=paths,
                get_path="path", get_color="color", get_width="width", width_units=pdk.types.String("pixels"),
                cap_rounded=True, joint_rounded=True, pickable=True, auto_highlight=True,
                highlight_color=[255, 255, 255, 60],
            ),
            pdk.Layer(
                "ScatterplotLayer", id="route-anchors", data=anchors,
                get_position="position", get_fill_color="color", get_radius=4, radius_units=pdk.types.String("pixels"),
                stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True,
            ),
        ])

    markers = []
    for point in points:
        color = point["color"].lstrip("#")
        markers.append({
            "position": [point["lon"], point["lat"]],
            "color": [int(color[i:i + 2], 16) for i in (0, 2, 4)],
            "radius": 9 if point["point_id"] == selected_point else 6,
            "tooltip": point["txt"],
        })
    layers.append(pdk.Layer(
        "ScatterplotLayer", id="forecast-points", data=markers,
        get_position="position", get_fill_color="color", get_radius="radius", radius_units=pdk.types.String("pixels"),
        stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5,
        pickable=True, auto_highlight=True,
    ))

    focused = next((route for route in routes if route.id == focus_route), None)
    if focused:
        view = pdk.data_utils.compute_view(focused.path)
        view.zoom = min(view.zoom, 12)
    else:
        latitude, longitude = REGIONS[region]["center"]
        view = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=REGIONS[region]["zoom"])

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_provider="carto",
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        tooltip={
            "html": "{tooltip}",
            "style": {"backgroundColor": "#ffffff", "color": "#17212c", "maxWidth": "340px",
                      "fontSize": "13px", "padding": "12px", "borderRadius": "8px"},
        },
    )
