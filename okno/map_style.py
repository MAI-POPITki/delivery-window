"""Единая шкала благоприятности для карт проекта."""
AVAILABILITY_SCALE = "RdYlGn"
UNKNOWN_COLOR = "#9e9e9e"

# Подложка без иностранных подписей; названия участков добавляет приложение.
FLOOD_MAP_STYLE = dict(version=8,sources={'background':dict(type='raster',tileSize=256,
    tiles=['https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}'],
    attribution='Подложка: Esri, USGS, NOAA; дороги © OpenStreetMap contributors')},
    layers=[dict(id='background',type='raster',source='background')])
