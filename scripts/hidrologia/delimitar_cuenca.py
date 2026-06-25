"""Delimitación de cuenca/microcuenca sobre DEM con WhiteboxTools (D8 + snap pour point).

Portado y adaptado desde el repo WaterKu (branch `hidrologia`,
scripts/hidrologia/delimitar_cuenca.py). La lógica original era un script de
nivel de módulo acoplado a Snakemake (`snakemake.input`/`snakemake.output`) o a
un `config/config.yaml` legacy multi-cuenca; aquí se envuelve en
`delimit_watershed()` para poder llamarla directamente desde
`hidrologia_runner.py` con los parámetros de `config/config.hidrologia.yaml`,
preservando ambos modos de ejecución originales al final del archivo.
"""
import os

import geopandas as gpd
import rasterio
import whitebox
import yaml
from rasterio.crs import CRS
from rasterio.features import shapes
from shapely.geometry import shape


def get_cuenca_from_path(path):
    """Extraer nombre de cuenca del path de salida o entrada"""
    parts = path.replace('\\', '/').split('/')
    if 'results' in parts:
        idx = parts.index('results')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def delimit_watershed(dem_path: str, punto_shp: str, target_crs: str, salida_dir: str) -> str:
    """Delimita la cuenca de aporte a partir de un DEM y un punto de control/desagüe.

    Args:
        dem_path: DEM ya reproyectado a `target_crs`.
        punto_shp: shapefile de punto con el punto de control (desagüe de la laguna).
        target_crs: CRS métrico de trabajo (ej. "EPSG:32718").
        salida_dir: carpeta de salida (se crean los rásters/shapefiles intermedios aquí).

    Returns:
        Ruta al shapefile de la cuenca delimitada (`<salida_dir>/cuenca.shp`).
    """
    os.makedirs(salida_dir, exist_ok=True)
    dem_path = os.path.abspath(dem_path)
    punto_shp = os.path.abspath(punto_shp)
    salida_dir = os.path.abspath(salida_dir)
    watershed_shp = os.path.join(salida_dir, "cuenca.shp")

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = True
    wbt.work_dir = salida_dir

    filled_dem = os.path.join(salida_dir, "dem_filled.tif")
    flow_dir = os.path.join(salida_dir, "flow_dir.tif")
    flow_acc = os.path.join(salida_dir, "flow_acc.tif")
    snapped_pour = os.path.join(salida_dir, "punto_snapped.shp")
    watershed_raster = os.path.join(salida_dir, "cuenca.tif")

    target_epsg = int(target_crs.split(":")[-1])

    gdf_punto = gpd.read_file(punto_shp)
    if gdf_punto.crs is None or gdf_punto.crs.to_epsg() != target_epsg:
        gdf_punto = gdf_punto.to_crs(target_crs)
        gdf_punto.to_file(punto_shp)

    with rasterio.open(dem_path) as src:
        if src.crs is None or src.crs.to_epsg() != target_epsg:
            raise ValueError(f"Error: El DEM no está en {target_crs}. Reproyéctalo antes de continuar.")

    # NOTE: se usa el método de relleno de depresiones de Wang & Liu en vez de
    # breaching (ej. breach_depressions_least_cost) para evitar canales
    # artificiales en cuencas pequeñas como la de la laguna.
    wbt.fill_depressions_wang_and_liu(dem_path, filled_dem)
    wbt.d8_pointer(filled_dem, flow_dir)
    wbt.d8_flow_accumulation(filled_dem, flow_acc, "cells")
    # Snap del punto de desagüe a la celda de mayor acumulación de flujo dentro
    # de 10 celdas (reducido respecto al valor por defecto de 100 para no
    # desplazar de más el punto en una cuenca pequeña).
    wbt.snap_pour_points(punto_shp, flow_acc, snapped_pour, 10)
    wbt.watershed(flow_dir, snapped_pour, watershed_raster)

    with rasterio.open(watershed_raster) as src:
        mask = src.read(1) == 1
        affine = src.transform
        crs = src.crs or CRS.from_string(target_crs)

        resultados = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for s, v in shapes(src.read(1), mask=mask, transform=affine)
        )

        geometries, values = [], []
        for r in resultados:
            geometries.append(shape(r['geometry']))
            values.append(r['properties']['raster_val'])

    gdf_cuenca = gpd.GeoDataFrame({'raster_val': values, 'geometry': geometries}, crs=crs)
    gdf_cuenca.to_file(watershed_shp)
    return watershed_shp


if 'snakemake' in globals():
    from utils import load_cuenca_config

    _cuenca_id = get_cuenca_from_path(str(snakemake.output.cuenca_shp))
    _cuenca_config, _global_config = load_cuenca_config(_cuenca_id)
    delimit_watershed(
        dem_path=str(snakemake.input.dem_reproyectado),
        punto_shp=str(snakemake.input.punto_control),
        target_crs=_cuenca_config["dst_crs"],
        salida_dir=os.path.dirname(str(snakemake.output.cuenca_shp)),
    )
elif __name__ == "__main__":
    # Ejecución directa legacy (config/config.yaml de una sola cuenca)
    with open(os.path.join("config", "config.yaml"), "r", encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    _shp = delimit_watershed(
        dem_path=_config["dem_reproyectado"],
        punto_shp=f"data/{_config['punto_control']}",
        target_crs=_config["dst_crs"],
        salida_dir=os.path.dirname(_config["cuenca_shp"]),
    )
    print(f"Cuenca delimitada: {_shp}")
