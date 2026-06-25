"""Parámetros geomorfológicos de la cuenca (área, perímetro, Kc, Ff, densidad de
drenaje, Tc por Kirpich, etc.) a partir del DEM y el shapefile de cuenca ya
delimitada.

Portado y adaptado desde el repo WaterKu (branch `hidrologia`,
scripts/hidrologia/parametros_cuenca.py) — envuelto en
`compute_basin_parameters()` para uso directo desde `hidrologia_runner.py`,
preservando los modos de ejecución Snakemake/legacy al final del archivo.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import whitebox
import yaml
from shapely.geometry import LineString


def get_cuenca_from_path(path):
    """Extraer nombre de cuenca del path de salida o entrada"""
    parts = path.replace('\\', '/').split('/')
    if 'results' in parts:
        idx = parts.index('results')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def compute_basin_parameters(dem_path: str, cuenca_shp: str, target_crs: str, salida_excel: str) -> pd.DataFrame:
    """Calcula la morfometría de la cuenca y la guarda en `salida_excel`.

    Returns:
        DataFrame con los parámetros geomorfológicos (mismo contenido que el Excel).
    """
    os.makedirs(os.path.dirname(salida_excel) or ".", exist_ok=True)
    target_epsg = int(target_crs.split(":")[-1])

    work_dir = os.path.dirname(os.path.abspath(dem_path))
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.work_dir = work_dir

    cuenca = gpd.read_file(cuenca_shp).to_crs(epsg=target_epsg)

    area_km2 = cuenca.geometry.area.values[0] / 1e6
    perimetro_km = cuenca.geometry.length.values[0] / 1000

    with rasterio.open(dem_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, cuenca.geometry, crop=True)
        perfil = src.profile
        perfil.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    dem_recortado = os.path.join(work_dir, "dem_recortado.tif")
    with rasterio.open(dem_recortado, "w", **perfil) as dest:
        dest.write(out_image)

    filled_dem = os.path.join(work_dir, "filled_dem.tif")
    fdr = os.path.join(work_dir, "fdr.tif")
    accum = os.path.join(work_dir, "accum.tif")
    streams = os.path.join(work_dir, "streams.tif")
    stream_order_raster = os.path.join(work_dir, "stream_order.tif")

    if not os.path.exists(filled_dem):
        wbt.fill_depressions_wang_and_liu(dem_recortado, filled_dem)
    if not os.path.exists(fdr):
        wbt.d8_pointer(filled_dem, fdr)
    if not os.path.exists(accum):
        wbt.d8_flow_accumulation(filled_dem, accum, out_type="cells")

    with rasterio.open(accum) as src:
        acc_data = src.read(1)
        acc_data = acc_data[acc_data > 0]
        threshold = int(np.percentile(acc_data, 90))
        threshold = max(threshold, 10)

    rivers_shp = os.path.join(work_dir, "rivers.shp")
    if not os.path.exists(rivers_shp):
        wbt.extract_streams(accum, streams, threshold=threshold)
        wbt.strahler_stream_order(fdr, streams, stream_order_raster)
        wbt.raster_streams_to_vector(streams, fdr, rivers_shp)

    rivers = gpd.read_file(rivers_shp).set_crs(target_crs)

    long_total_cauces = rivers.length.sum() / 1000
    num_total_cauces = len(rivers)

    dem_array = out_image[0].astype(float)
    dem_array[dem_array == perfil['nodata']] = np.nan

    z_max = np.nanmax(dem_array)
    z_min = np.nanmin(dem_array)

    bounds = cuenca.total_bounds
    linea_diag = LineString([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
    long_cauce_principal = linea_diag.length / 1000

    kc = perimetro_km / (2 * (np.pi * area_km2) ** 0.5)
    ff = area_km2 / (long_cauce_principal ** 2)

    with rasterio.open(stream_order_raster) as src:
        stream_order_array = src.read(1).astype(float)
        stream_order_array[stream_order_array == src.nodata] = np.nan
        orden_cuenca = int(np.nanmax(stream_order_array)) if np.nanmax(stream_order_array) > 0 else 1
        num_orden_1 = np.count_nonzero(stream_order_array == 1)

    densidad_drenaje = long_total_cauces / area_km2
    frecuencia_rios = num_total_cauces / area_km2
    altitud_media = np.nanmean(dem_array)

    K = 0.28 * (perimetro_km / np.sqrt(area_km2))
    lado_mayor = (K * np.sqrt(area_km2) / 1.12) * (1 + np.sqrt(1 - (1.12 / K) ** 2))
    lado_menor = (K * np.sqrt(area_km2) / 1.12) * (1 - np.sqrt(1 - (1.12 / K) ** 2))
    pendiente_cuenca = (z_max - z_min) / (lado_mayor * 1000) * 100
    pendiente_cauce = (z_max - z_min) / (long_cauce_principal * 1000) * 100

    ext_escurr = 1 / (2 * densidad_drenaje)
    coef_torrencialidad = num_orden_1 / area_km2

    L_m = long_cauce_principal * 1000
    S = (z_max - z_min) / L_m
    Tc_kirpich = 0.0195 * (L_m ** 0.77) * (S ** -0.385) if S > 0 and L_m > 0 else np.nan

    datos = {
        "Área (Km²)": [round(area_km2, 3)],
        "Perímetro (Km)": [round(perimetro_km, 3)],
        "Longitud del cauce principal (Km)": [round(long_cauce_principal, 3)],
        "Longitud total de cauces (Km)": [round(long_total_cauces, 3)],
        "Número total de cauces": [num_total_cauces],
        "Coeficiente de compacidad (Kc)": [round(kc, 3)],
        "Factor de forma (Ff)": [round(ff, 3)],
        "Orden de la cuenca": [orden_cuenca],
        "Densidad de drenaje (Km/Km²)": [round(densidad_drenaje, 3)],
        "Extensión media del escurrimiento superficial (Km)": [round(ext_escurr, 3)],
        "Frecuencia de ríos (cauces/Km²)": [round(frecuencia_rios, 3)],
        "Altitud media de la cuenca (msnm)": [round(altitud_media, 3)],
        "Lado mayor del rectángulo equivalente (Km)": [round(lado_mayor, 3)],
        "Lado menor del rectángulo equivalente (Km)": [round(lado_menor, 3)],
        "Pendiente media del cauce principal (%)": [round(pendiente_cauce, 3)],
        "Pendiente media de la cuenca (%)": [round(pendiente_cuenca, 3)],
        "Coeficiente de torrencialidad (ríos/Km²)": [round(coef_torrencialidad, 3)],
        "Tiempo de concentración (Kirpich) (min)": [round(Tc_kirpich, 3)],
    }

    df = pd.DataFrame(datos).T
    df.columns = ["Valor"]
    df.to_excel(salida_excel, engine="openpyxl")
    return df


if 'snakemake' in globals():
    from utils import load_cuenca_config

    _cuenca_id = get_cuenca_from_path(str(snakemake.output.parametros_excel))
    _cuenca_config, _global_config = load_cuenca_config(_cuenca_id)
    compute_basin_parameters(
        dem_path=str(snakemake.input.dem_reproyectado),
        cuenca_shp=str(snakemake.input.cuenca_shp),
        target_crs=_cuenca_config.get("dst_crs", "EPSG:32718"),
        salida_excel=str(snakemake.output.parametros_excel),
    )
elif __name__ == "__main__":
    with open(os.path.join("config", "config.yaml"), "r", encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    compute_basin_parameters(
        dem_path=_config["dem_reproyectado"],
        cuenca_shp=_config["cuenca_shp"],
        target_crs=_config["dst_crs"],
        salida_excel=_config["parametros_excel"],
    )
