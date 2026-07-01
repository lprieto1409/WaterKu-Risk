"""Gráficos de apoyo de la cuenca: áreas parciales (DEM), curva hipsométrica,
perfil longitudinal del cauce principal y mapa base de red de drenaje.

Portado y adaptado desde el repo WaterKu (branch `hidrologia`,
scripts/hidrologia/graficos_cuenca.py) — envuelto en
`generar_graficos_cuenca()` para uso directo desde `hidrologia_runner.py`,
preservando los modos de ejecución Snakemake/legacy al final del archivo.
"""
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
import whitebox
import yaml


def get_cuenca_from_path(path):
    """Extraer nombre de cuenca del path de salida o entrada"""
    match = re.search(r'[/\\]([^/\\]+)[/\\]hidrologia[/\\]', path)
    if match:
        return match.group(1)
    return "cuenca_proyecto1"


def generar_graficos_cuenca(
    dem_path: str,
    cuenca_shp: str,
    target_crs: str,
    salida_dir: str,
    areas_parciales_path: str,
    curva_hipsometrica_path: str,
    perfil_longitudinal_path: str,
    mapa_base_drenaje_path: str,
) -> None:
    os.makedirs(salida_dir, exist_ok=True)
    temp_dir = os.path.dirname(os.path.abspath(dem_path))

    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = temp_dir
    wbt.verbose = False

    cuenca = gpd.read_file(cuenca_shp).to_crs(target_crs)
    with rasterio.open(dem_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, cuenca.geometry, crop=True)
        perfil = src.profile
        perfil.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    dem_clip = os.path.join(temp_dir, "dem_clip.tif")
    with rasterio.open(dem_clip, "w", **perfil) as dest:
        dest.write(out_image)

    filled = os.path.join(temp_dir, "filled.tif")
    fdr = os.path.join(temp_dir, "fdr.tif")
    accum = os.path.join(temp_dir, "accum.tif")
    streams_raster = os.path.join(temp_dir, "streams.tif")
    stream_order = os.path.join(temp_dir, "stream_order.tif")
    streams_vector = os.path.join(temp_dir, "streams.shp")
    wbt.fill_depressions_wang_and_liu(dem_clip, filled)
    wbt.d8_pointer(filled, fdr)
    wbt.d8_flow_accumulation(filled, accum, out_type="cells")

    with rasterio.open(accum) as src:
        acc_data = src.read(1)
        acc_data = acc_data[acc_data > 0]
        threshold = int(np.percentile(acc_data, 90))
        threshold = max(threshold, 10)

    wbt.extract_streams(accum, streams_raster, threshold=threshold)
    wbt.strahler_stream_order(fdr, streams_raster, stream_order)
    wbt.raster_streams_to_vector(streams_raster, fdr, streams_vector)

    elev = out_image[0].astype(float)
    elev[elev == perfil['nodata']] = np.nan
    pixel_area_sq_m = abs(perfil['transform'][0] * perfil['transform'][4]) if 'transform' in perfil else 900

    plt.figure(figsize=(8, 7))
    plt.imshow(elev, cmap='terrain', origin='lower',
               extent=[0, elev.shape[1] * np.sqrt(pixel_area_sq_m),
                       0, elev.shape[0] * np.sqrt(pixel_area_sq_m)])
    plt.colorbar(label='Altitud (msnm)')
    plt.title("Gráfico N° 01: Áreas parciales (Visualización del DEM)")
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.tight_layout()
    plt.savefig(areas_parciales_path, dpi=300)
    plt.close()

    elev_flat = elev.flatten()
    elev_flat = elev_flat[~np.isnan(elev_flat)]
    sorted_elevations = np.sort(elev_flat)
    cumulative_area_fraction = np.linspace(0, 1, len(sorted_elevations))
    altitud_media = np.mean(elev_flat)

    plt.figure(figsize=(7, 5))
    plt.plot(cumulative_area_fraction * 100, sorted_elevations, color='blue', linewidth=2, label="Curva hipsométrica")
    plt.axhline(altitud_media, color='red', linestyle='--', linewidth=1.8, label=f'Altitud media ≈ {altitud_media:.1f} m')
    plt.xlabel("Área acumulada (%)")
    plt.ylabel("Altitud (msnm)")
    plt.title("Gráfico N° 02: Curva Hipsométrica de la Cuenca")
    plt.grid(True)
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curva_hipsometrica_path, dpi=300)
    plt.close()

    streams = gpd.read_file(streams_vector)
    streams.set_crs(target_crs, inplace=True)
    longest = streams.geometry.apply(lambda g: g.length).idxmax()
    linea = streams.geometry.iloc[longest]
    coords = list(linea.coords)

    distancias = [0]
    perfil_longitudinal = []
    for i, (x, y) in enumerate(coords):
        col, row = ~out_transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < elev.shape[0] and 0 <= col < elev.shape[1]:
            z = elev[row, col]
            perfil_longitudinal.append(z)
            if i > 0:
                dx = np.hypot(x - coords[i - 1][0], y - coords[i - 1][1])
                distancias.append(distancias[-1] + dx)

    plt.figure(figsize=(8, 4))
    plt.plot(distancias, perfil_longitudinal, color='green', linewidth=2)
    plt.title("Gráfico N° 03: Perfil Longitudinal del Cauce Principal")
    plt.xlabel("Distancia a lo largo del cauce (m)")
    plt.ylabel("Altitud (msnm)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(perfil_longitudinal_path, dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 10))
    xmin = out_transform[2]
    xmax = xmin + elev.shape[1] * out_transform[0]
    ymax = out_transform[5]
    ymin = ymax + elev.shape[0] * out_transform[4]
    img = ax.imshow(elev, cmap='terrain', extent=(xmin, xmax, ymin, ymax), origin='upper')
    cuenca.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label='Límite de cuenca')
    streams.plot(ax=ax, color='blue', linewidth=1, label='Red de drenaje')
    plt.colorbar(img, ax=ax, label='Altitud (msnm)')
    ax.set_title("Gráfico N° 04: Mapa de Cuenca y Red de Drenaje")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(mapa_base_drenaje_path, dpi=300)
    plt.close()


if 'snakemake' in globals():
    from utils import load_cuenca_config

    _cuenca_id = get_cuenca_from_path(str(snakemake.output[0]))
    _cuenca_config, _global_config = load_cuenca_config(_cuenca_id)
    _salida_dir = os.path.abspath(os.path.dirname(str(snakemake.output[0])))
    generar_graficos_cuenca(
        dem_path=str(snakemake.input.dem_reproyectado),
        cuenca_shp=str(snakemake.input.cuenca_shp),
        target_crs=_cuenca_config["dst_crs"],
        salida_dir=_salida_dir,
        areas_parciales_path=str(snakemake.output.areas_parciales),
        curva_hipsometrica_path=str(snakemake.output.curva_hipsometrica),
        perfil_longitudinal_path=str(snakemake.output.perfil_longitudinal),
        mapa_base_drenaje_path=str(snakemake.output.mapa_base_drenaje),
    )
elif __name__ == "__main__":
    with open(os.path.join("config", "config.yaml"), "r", encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    generar_graficos_cuenca(
        dem_path=_config["dem_reproyectado"],
        cuenca_shp=_config["cuenca_shp"],
        target_crs=_config["dst_crs"],
        salida_dir=_config["graficos_dir"],
        areas_parciales_path=_config["areas_parciales"],
        curva_hipsometrica_path=_config["curva_hipsometrica"],
        perfil_longitudinal_path=_config["perfil_longitudinal"],
        mapa_base_drenaje_path=_config["mapa_base_drenaje"],
    )
