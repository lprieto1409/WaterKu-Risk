"""Orquestador de hidrología superficial — Fase 0 + Fase 1 de la Propuesta Arkel
(Laguna Colombina Sur, Valle Real).

Encadena los módulos portados desde el repo WaterKu original (branches
`hidrologia`, `enhanced-precipitation-analysis`, `recoleccion_datos`) más el
módulo `homogeneidad.py` construido nuevo para WaterKu-Risk:

    precipitacion_24h.py  ->  analisis_precipitacion.py  ->  precipitacion_maxima.py
            (SENAMHI txt)        (consistencia/máx. anual)      (frecuencia + KS)
                                                                        |
                                                                        v
    delimitar_cuenca.py  ->  parametros_cuenca.py  ->  hidrograma_scs.py / metodo_racional.py
       (DEM + desagüe)         (morfometría + Tc)        (escorrentía SCS-CN, "HEC-HMS nativo")
                                                                        ^
                                                                        |
                                                              curvas_IDF.py (IDF + hietograma)

Sigue el mismo patrón `process_*(cfg, output_dir)` que
`scripts/transport/transport_runner.py`. Cada paso se omite con un aviso claro
si los datos de entrada del proyecto (DEM, series SENAMHI crudas) todavía no
han llegado — ver `data/valle_real/README.md`.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # evita que plt.show() (en curvas_IDF.generar_grafico_hietograma) bloquee en headless

import pandas as pd
import yaml

_HIDRO_DIR = Path(__file__).resolve().parent
if str(_HIDRO_DIR) not in sys.path:
    sys.path.insert(0, str(_HIDRO_DIR))

import analisis_precipitacion  # noqa: E402
import precipitacion_24h  # noqa: E402
import precipitacion_maxima  # noqa: E402
import curvas_IDF  # noqa: E402
from homogeneidad import run_homogeneity_tests  # noqa: E402
from delimitar_cuenca import delimit_watershed  # noqa: E402
from parametros_cuenca import compute_basin_parameters  # noqa: E402
from graficos_cuenca import generar_graficos_cuenca  # noqa: E402
import hidrograma_scs  # noqa: E402
import metodo_racional  # noqa: E402

# avenida_de_diseño.py (enhanced-precipitation-analysis) implementa un cálculo
# de avenida de diseño autocontenido (IDF + SCS) alternativo al encadenado
# arriba (precipitacion_maxima -> curvas_IDF -> hidrograma_scs); no se usa en
# este orquestador para evitar dos rutas de cálculo distintas para el mismo
# resultado, pero queda portado en scripts/hidrologia/avenida_de_diseño.py
# como referencia/validación cruzada si se necesita.


def load_config(config_path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fase 0 + 1a: SENAMHI -> consistencia / máximos anuales / homogeneidad
# ---------------------------------------------------------------------------

def _senamhi_station_file(senamhi_dir: Path, estacion: str) -> Path | None:
    for ext in ("txt", "TXT", "dat"):
        candidates = list(senamhi_dir.glob(f"*{estacion}*.{ext}"))
        if candidates:
            return candidates[0]
    return None


def procesar_precipitacion_estaciones(cfg: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Para cada estación SENAMHI configurada: parsea el .txt crudo, completa
    datos faltantes y obtiene la serie de precipitación máxima 24h anual
    ('Precipitación' por año). Devuelve {estacion: DataFrame} solo para las
    estaciones con datos disponibles."""
    senamhi_dir = Path(cfg["fuentes"]["senamhi"]["directorio"])
    estaciones = cfg["fuentes"]["senamhi"]["estaciones"]
    out = output_dir / "00_precipitacion_anual"
    out.mkdir(parents=True, exist_ok=True)

    series_por_estacion: dict[str, pd.DataFrame] = {}
    for estacion in estaciones:
        archivo = _senamhi_station_file(senamhi_dir, estacion)
        if archivo is None:
            warnings.warn(
                f"[hidrologia] Sin archivo SENAMHI para la estación '{estacion}' en "
                f"{senamhi_dir} — se omite esta estación. Adquirir vía TUPA SENAMHI."
            )
            continue

        tabla_pivote, _, ok = precipitacion_24h.procesar_archivo_precipitacion(str(archivo))
        if not ok or tabla_pivote is None:
            warnings.warn(f"[hidrologia] No se pudo procesar el archivo SENAMHI de '{estacion}'.")
            continue

        df_mensual = tabla_pivote.reset_index()  # columnas: Año, Enero, Febrero, ...
        df_maximos, df_completo = analisis_precipitacion.completar_datos_faltantes_y_obtener_maximos(df_mensual)
        df_maximos, _umbral = analisis_precipitacion.eliminar_outliers_y_repetir(df_maximos, df_completo)

        excel_path = out / f"{estacion}_precipitacion_maxima_anual.xlsx"
        df_maximos.to_excel(excel_path, index=False)
        series_por_estacion[estacion] = df_maximos

    return series_por_estacion


def analizar_consistencia_homogeneidad(cfg: dict, series_por_estacion: dict[str, pd.DataFrame],
                                        output_dir: Path) -> dict:
    """Curva doble masa (estación objetivo vs. promedio de estaciones de
    referencia) + pruebas de homogeneidad (Mann-Kendall / Pettitt) por
    estación. No existe un módulo de curva doble masa multi-estación en
    ninguna branch del repo WaterKu original — se construye aquí, simple."""
    out = output_dir / "01_consistencia_homogeneidad"
    out.mkdir(parents=True, exist_ok=True)
    resultado: dict = {"homogeneidad": {}, "doble_masa": None}

    tests = cfg.get("homogeneidad", {}).get("tests", ["mann_kendall", "pettitt"])
    alpha = cfg.get("homogeneidad", {}).get("alpha", 0.05)
    for estacion, df in series_por_estacion.items():
        serie = df["Precipitación"].dropna().tolist()
        if len(serie) < 8:
            continue
        resultados = run_homogeneity_tests(serie, tests=tests, alpha=alpha)
        resultado["homogeneidad"][estacion] = [r.__dict__ for r in resultados]

    objetivo = cfg.get("consistencia", {}).get("estacion_objetivo")
    referencias = cfg.get("consistencia", {}).get("estaciones_referencia", [])
    if objetivo in series_por_estacion and any(r in series_por_estacion for r in referencias):
        df_obj = series_por_estacion[objetivo].set_index("Año")["Precipitación"]
        refs_disponibles = [series_por_estacion[r].set_index("Año")["Precipitación"]
                             for r in referencias if r in series_por_estacion]
        df_ref_mean = pd.concat(refs_disponibles, axis=1).mean(axis=1)
        comunes = df_obj.index.intersection(df_ref_mean.index)
        acumulado_obj = df_obj.loc[comunes].sort_index().cumsum()
        acumulado_ref = df_ref_mean.loc[comunes].sort_index().cumsum()
        df_doble_masa = pd.DataFrame({
            f"{objetivo}_acumulado": acumulado_obj,
            "referencia_acumulada": acumulado_ref,
        })
        df_doble_masa.to_excel(out / "curva_doble_masa.xlsx")
        resultado["doble_masa"] = str(out / "curva_doble_masa.xlsx")
    else:
        warnings.warn(
            "[hidrologia] No hay suficientes estaciones con datos para la curva doble masa "
            f"(objetivo={objetivo}, referencias={referencias})."
        )

    return resultado


# ---------------------------------------------------------------------------
# Fase 1b + 1c: frecuencia (Gumbel/Pearson III/Log-Pearson III/GEV/LogNormal) + IDF
# ---------------------------------------------------------------------------

def analizar_frecuencia_e_idf(cfg: dict, serie_objetivo: pd.Series, output_dir: Path) -> dict:
    """Ajusta distribuciones de precipitación máxima 24h, selecciona la mejor
    por KS, calcula P24h para los periodos de retorno configurados (10, 25,
    100 años) y construye las curvas IDF (Dick-Peschke) + hietograma de
    bloques alternos — el "HEC-HMS nativo" del PDF de la propuesta (ver
    decisión de diseño en el plan)."""
    out = output_dir / "02_frecuencia_idf"
    out.mkdir(parents=True, exist_ok=True)

    data = serie_objetivo.dropna().sort_values()
    (ks_test, dist_normal, dist_lognorm_2, dist_lognorm_3, dist_gumbel, dist_loggumbel,
     dist_pearson3, dist_logpearson3) = precipitacion_maxima.ajustar_distribuciones(data)
    mejor_distribucion, _ = precipitacion_maxima.elegir_mejor_distribucion(ks_test)

    distribuciones_params = {
        'normal': dist_normal, 'lognorm_2': dist_lognorm_2, 'lognorm_3': dist_lognorm_3,
        'gumbel': dist_gumbel, 'loggumbel': dist_loggumbel,
        'pearson3': dist_pearson3, 'logpearson3': dist_logpearson3,
    }
    periodos = cfg.get("frecuencia", {}).get("periodos_retorno", [10, 25, 100])
    p_max = {tr: precipitacion_maxima.calcular_precipitacion_max(
        tr, distribuciones_params[mejor_distribucion], mejor_distribucion, data) for tr in periodos}

    precipitacion_maxima.guardar_resultados(
        str(out / "resultados_precipitacion_max.xlsx"),
        str(out / "resultados_precipitacion_max.txt"),
        precipitacion_maxima.calcular_parametros_estadisticos(data),
        ks_test, mejor_distribucion, p_max,
    )

    df_intensidades, ecuacion, df_hietograma = curvas_IDF.main(
        archivo_entrada=str(out / "resultados_precipitacion_max.xlsx"),
        archivo_intensidades=str(out / "intensidades_idf.xlsx"),
        archivo_ecuacion=str(out / "ecuacion_idf.txt"),
        archivo_grafico=str(out / "curvas_idf.png"),
        archivo_hietograma=str(out / "hietograma.xlsx"),
        archivo_hietograma_png=str(out / "hietograma.png"),
        tr_especifico=max(periodos),
    )

    return {
        "mejor_distribucion": mejor_distribucion,
        "p_max_por_tr": p_max,
        "ecuacion_idf": ecuacion,
        "hietograma_excel": str(out / "hietograma.xlsx"),
    }


# ---------------------------------------------------------------------------
# Fase 1d + 1e: cuenca (DEM) + escorrentía SCS-CN
# ---------------------------------------------------------------------------

def delimitar_y_caracterizar_cuenca(cfg: dict, output_dir: Path) -> dict | None:
    cc = cfg.get("cuenca", {})
    dem_path = Path(cc.get("dem_path", ""))
    punto = cc.get("punto_desague")
    if not dem_path.exists() or not punto:
        warnings.warn(
            "[hidrologia] Falta el DEM de la cuenca y/o el punto de desagüe de la laguna "
            f"({dem_path}) — se omite delimitación de cuenca hasta recibir cartografía/topografía."
        )
        return None

    out = output_dir / "03_cuenca"
    out.mkdir(parents=True, exist_ok=True)
    target_crs = cfg["study_area"]["crs_metric"]

    punto_shp = out / "punto_desague.shp"
    import geopandas as gpd
    from shapely.geometry import Point
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(punto)], crs=target_crs).to_file(punto_shp)

    cuenca_shp = delimit_watershed(str(dem_path), str(punto_shp), target_crs, str(out))
    df_parametros = compute_basin_parameters(str(dem_path), cuenca_shp, target_crs,
                                              str(out / "parametros_cuenca.xlsx"))
    generar_graficos_cuenca(
        str(dem_path), cuenca_shp, target_crs, str(out),
        str(out / "areas_parciales.jpg"), str(out / "curva_hipsometrica.jpg"),
        str(out / "perfil_longitudinal.jpg"), str(out / "mapa_base_drenaje.jpg"),
    )
    return {"cuenca_shp": cuenca_shp, "parametros_excel": str(out / "parametros_cuenca.xlsx"),
            "parametros": df_parametros}


def calcular_escorrentia_scs(cfg: dict, cuenca_info: dict | None, frecuencia_info: dict,
                              output_dir: Path) -> dict | None:
    """Escorrentía SCS-CN + hidrograma unitario (equivalente nativo a HEC-HMS,
    ver decisión de diseño) y caudal pico por el método racional, para
    comparación."""
    if cuenca_info is None:
        warnings.warn("[hidrologia] Sin parámetros de cuenca — se omite el hidrograma SCS de ingreso a la laguna.")
        return None

    out = output_dir / "04_escorrentia_scs"
    out.mkdir(parents=True, exist_ok=True)
    cn = cfg.get("escorrentia_scs", {}).get("curve_number") or 85
    cuenca_config = {"usar_parametros_excel": True, "cn_default": cn}

    hidrograma_scs.procesar_cuenca_integrada(
        parametros_excel_path=cuenca_info["parametros_excel"],
        hietograma_excel_path=frecuencia_info["hietograma_excel"],
        cuenca_config=cuenca_config,
        output_dir=str(out),
        dt_horas=1.0,
    )

    area_ha, area_km2, tc_min = metodo_racional.leer_area_cuenca(cuenca_info["parametros_excel"], cuenca_config)
    return {"output_dir": str(out), "area_km2": area_km2, "tc_min": tc_min}


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def process_hidrologia(config_path, output_dir) -> None:
    cfg = load_config(config_path)
    out_base = cfg.get("output_base", "results")
    output_dir = Path(str(output_dir).format(output_base=out_base))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 66)
    print("  Hidrología superficial — Laguna Colombina Sur / Valle Real")
    print("=" * 66 + "\n")

    series_por_estacion = procesar_precipitacion_estaciones(cfg, output_dir)
    if not series_por_estacion:
        warnings.warn(
            "[hidrologia] No hay ninguna serie SENAMHI disponible todavía en "
            f"{cfg['fuentes']['senamhi']['directorio']}. Cargar los .txt de las 4 estaciones "
            "(Viques, Huayao, Santa_Ana, Ingenio) para correr el análisis completo."
        )
        print(f"\n  Sin datos de entrada todavía. Resultados parciales en: {output_dir}\n")
        return

    analizar_consistencia_homogeneidad(cfg, series_por_estacion, output_dir)

    objetivo = cfg.get("consistencia", {}).get("estacion_objetivo")
    serie_objetivo = (series_por_estacion.get(objetivo) or next(iter(series_por_estacion.values())))["Precipitación"]
    frecuencia_info = analizar_frecuencia_e_idf(cfg, serie_objetivo, output_dir)

    cuenca_info = delimitar_y_caracterizar_cuenca(cfg, output_dir)
    calcular_escorrentia_scs(cfg, cuenca_info, frecuencia_info, output_dir)

    print(f"\n  Listo. Resultados en: {output_dir}\n")


if "snakemake" in dir():
    cfg_path = snakemake.input.config  # noqa: F821
    out_dir = snakemake.output[0]  # noqa: F821
    process_hidrologia(cfg_path, out_dir)
