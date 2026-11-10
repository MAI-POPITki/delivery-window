# Реальные спутниковые наблюдения М2 паводков

Источник: Copernicus Sentinel-2 L2A через
https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a.

`<point>/scenes/<item_id>/`:
- `dn.npz`: каналы B02/B03/B04/B08/B11 и SCL на единой сетке UTM 20 м.
  Это реальные DN после пространственного приведения; не исходная целая сцена.
- `product.xml`: исходные коэффициенты радиометрии данного продукта.
- `stac.json`: идентификатор, времена, CRS, исходные ссылки без временных токенов.
- `crop.json`: сетка, resampling и SHA-256 вырезки.

`<point>/runs/<item_id>/`: mask.npz (0 неизвестно, 1 суша, 2 вода, 3 новая вода),
rgb.png, baseline.png, overlay.png и manifest.json. В маске базы 0 неизвестно,
1 суша, 2 вода. Grid — UTM, transform в порядке a,b,c,d,e,f; overlay_coordinates
относится к PNG, отдельно перепроецированному в EPSG:4326.

Обновление/ограничения: [FLOOD_M2.md](../../../docs/FLOOD_M2.md).
Никакой независимой точности классификации затопления эти файлы не доказывают.
