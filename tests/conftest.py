"""Тесты не должны запускать сетевые фоновые загрузчики приложения и переписывать данные в репозитории.

Отдельные тесты фоновых загрузчиков включают их сами через patch.dict(os.environ, ...).
"""
import os

for _name in ("OKNO_HYDRO_AUTO_REFRESH", "OKNO_FLOOD_AUTO_REFRESH", "OKNO_LANDSAT_AUTO_REFRESH", "OKNO_GAOFEN_AUTO_REFRESH"):
    os.environ.setdefault(_name, "0")
