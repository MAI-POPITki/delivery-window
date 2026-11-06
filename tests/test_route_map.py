"""Проверки координат схем, наведения и карты в реальном дашборде."""

import csv
import json
from pathlib import Path
import sys
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.route_map import build_route_map, schematic_routes
from okno.regions import REGIONS


def source_rows(path, key):
    with path.open(encoding="utf-8", newline="") as file:
        return {row[key]: row for row in csv.DictReader(file)}


class RouteMapTests(unittest.TestCase):
    def setUp(self):
        self.routes = schematic_routes("yakutia")
        self.points = [{"point_id": "gauge_3414", "lon": 133.38, "lat": 67.34,
                        "color": "#c62828", "txt": "Гидропост: 25%"}]

    def spec(self, **kwargs):
        return json.loads(build_route_map("yakutia", self.points, self.routes, **kwargs).to_json())

    def test_only_two_pilots_and_coordinates_come_from_project_data(self):
        routes = {route.id: route for route in self.routes}
        self.assertEqual(set(routes), {"batagay", "yakutsk_crossing"})
        points = source_rows(ROOT / "data" / "points.csv", "id")
        gauges = source_rows(ROOT / "data" / "processed" / "gauges.csv", "gauge_code")
        position = lambda row: (float(row["lon"]), float(row["lat"]))
        self.assertEqual(routes["batagay"].path, (position(gauges["3414"]), position(points["batagay"])))
        self.assertEqual(routes["batagay"].point_id, "gauge_3414")
        self.assertEqual(routes["yakutsk_crossing"].path,
                         (position(points["yakutsk"]), position(points["yakutsk_crossing"])))
        for route in routes.values():
            self.assertIn("Схема", route.note)
            self.assertEqual(len(route.path), len(route.anchors))

    def test_other_regions_do_not_invent_routes_between_gauges(self):
        for region in REGIONS:
            if region != "yakutia":
                with self.subTest(region=region):
                    self.assertEqual(schematic_routes(region), [])

    def test_routes_have_distinct_colors_and_hover_along_the_line(self):
        layers = {layer["id"]: layer for layer in self.spec()["layers"]}
        paths = layers["schematic-routes"]
        self.assertEqual(paths["@@type"], "PathLayer")
        self.assertTrue(paths["pickable"])
        self.assertTrue(paths["autoHighlight"])
        self.assertEqual(paths["widthUnits"], "pixels")
        self.assertEqual(len({tuple(row["color"]) for row in paths["data"]}), 2)
        for route, row in zip(self.routes, paths["data"]):
            self.assertIn(route.name, row["tooltip"])
            self.assertIn("схема", row["tooltip"])
            self.assertIn(route.note, row["tooltip"])
        self.assertFalse(layers["route-outlines"]["pickable"])

    def test_forecast_markers_keep_their_probability_color_and_tooltip(self):
        layers = {layer["id"]: layer for layer in self.spec(selected_point="gauge_3414")["layers"]}
        points = layers["forecast-points"]
        self.assertEqual(len(points["data"]), len(self.points))
        self.assertEqual(points["data"][0]["color"], [198, 40, 40])
        self.assertEqual(points["data"][0]["tooltip"], "Гидропост: 25%")
        self.assertTrue(points["pickable"])
        self.assertEqual(points["radiusUnits"], "pixels")

    def test_focusing_a_route_zooms_in_and_highlights_only_that_route(self):
        normal = self.spec()
        for route in self.routes:
            with self.subTest(route=route.id):
                focused = self.spec(focus_route=route.id, selected_point="gauge_3414")
                view = focused["initialViewState"]
                self.assertGreater(view["zoom"], normal["initialViewState"]["zoom"])
                self.assertAlmostEqual(view["longitude"], sum(p[0] for p in route.path) / len(route.path))
                self.assertAlmostEqual(view["latitude"], sum(p[1] for p in route.path) / len(route.path))
                paths = next(layer for layer in focused["layers"] if layer["id"] == "schematic-routes")
                widths = [row["width"] for row in paths["data"]]
                self.assertGreater(widths[self.routes.index(route)], min(widths))

    def test_map_without_route_geometry_preserves_the_region_and_points(self):
        deck = build_route_map("yanao", self.points, [])
        spec = json.loads(deck.to_json())
        self.assertFalse(any(layer["@@type"] == "PathLayer" for layer in spec["layers"]))
        self.assertEqual(len(spec["layers"][0]["data"]), 1)
        self.assertEqual(spec["initialViewState"]["latitude"], REGIONS["yanao"]["center"][0])

    def test_schemes_are_visible_even_without_a_point_forecast(self):
        spec = json.loads(build_route_map("yakutia", [], self.routes).to_json())
        paths = next(layer for layer in spec["layers"] if layer["id"] == "schematic-routes")
        self.assertEqual(len(paths["data"]), 2)


class RouteMapAppTests(unittest.TestCase):
    def test_dashboard_focus_and_region_changes_render_the_correct_layers(self):
        app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90)
        app.session_state["dashboard_open"] = True
        app.run()
        self.assertEqual([e.message for e in app.exception], [])
        chart = app.get("deck_gl_json_chart")[0]
        self.assertEqual(json.loads(chart.proto.tooltip)["html"], "{tooltip}")
        spec = json.loads(chart.proto.json)
        paths = next(layer for layer in spec["layers"] if layer["id"] == "schematic-routes")
        self.assertEqual(len(paths["data"]), 2)
        self.assertTrue(all("схема" in row["tooltip"] for row in paths["data"]))
        self.assertTrue(any("Линии — схемы" in info.value for info in app.info))

        app.selectbox(key="map_focus_yakutia").select("yakutsk_crossing").run()
        self.assertEqual([e.message for e in app.exception], [])
        focused = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        self.assertGreater(focused["initialViewState"]["zoom"], spec["initialViewState"]["zoom"])

        app.sidebar.selectbox[0].select("yanao").run()
        self.assertEqual([e.message for e in app.exception], [])
        other_region = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        self.assertFalse(any(layer["@@type"] == "PathLayer" for layer in other_region["layers"]))
        self.assertEqual(len(other_region["layers"][0]["data"]), 2)

        app.sidebar.selectbox[0].select("yakutia").run()
        self.assertEqual([e.message for e in app.exception], [])
        restored = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        self.assertEqual(len(next(layer for layer in restored["layers"]
                                  if layer["id"] == "schematic-routes")["data"]), 2)


if __name__ == "__main__":
    unittest.main()
