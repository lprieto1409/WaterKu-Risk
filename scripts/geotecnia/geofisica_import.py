"""Import de perfiles geofísicos MASW-2D y de tomografía de resistividad
eléctrica (ERT) YA PROCESADOS por ARKEL o su subcontratista de geofísica.

WaterKu-Risk no invierte geofísica cruda (curvas de dispersión MASW, datos de
resistividad aparente ERT) — esa inversión requiere software especializado de
terceros (ej. SeisImager/SW, RES2DINV/EarthImager) y no se reimplementa aquí
(ver decisión de diseño en el plan). Este módulo solo importa los resultados
ya interpretados (perfil Vs(z), sección de resistividad) para apoyar la
definición de límites de capas y nivel freático del modelo conceptual.

Construido nuevo para WaterKu-Risk.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_masw_profile(path: str | Path) -> pd.DataFrame:
    """Carga un perfil MASW-2D ya procesado (CSV con columnas
    `profundidad_m`, `vs_m_s`, y opcionalmente `x`, `y` si el perfil tiene
    varios sondeos de superficie a lo largo de la línea)."""
    df = pd.read_csv(path)
    columnas_requeridas = {"profundidad_m", "vs_m_s"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Perfil MASW {path} no tiene las columnas requeridas: {faltantes}")
    return df


def load_ert_section(path: str | Path) -> pd.DataFrame:
    """Carga una sección de resistividad ERT ya invertida (CSV/XYZ con
    columnas `x_m` o `distancia_m`, `profundidad_m`, `resistividad_ohm_m`),
    típicamente exportada desde RES2DINV/EarthImager."""
    df = pd.read_csv(path)
    columnas_requeridas = {"profundidad_m", "resistividad_ohm_m"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Sección ERT {path} no tiene las columnas requeridas: {faltantes}")
    return df


def correlate_geophysics_with_boreholes(masw: pd.DataFrame, ert: pd.DataFrame,
                                         boreholes: pd.DataFrame) -> dict:
    """Compara, por profundidad, los contrastes de Vs (MASW) y resistividad
    (ERT) contra los cambios litológicos observados en los sondeos
    (`boreholes`, salida de `sondeos.build_stratigraphic_profile`), como
    apoyo cualitativo a la definición de límites de capas y nivel freático
    del modelo conceptual.

    No determina automáticamente las unidades — es apoyo a la interpretación
    del hidrogeólogo (juicio de ingeniería, no derivado solo del código).
    """
    return {
        "rango_vs_m_s": (float(masw["vs_m_s"].min()), float(masw["vs_m_s"].max())) if len(masw) else None,
        "rango_resistividad_ohm_m": (
            float(ert["resistividad_ohm_m"].min()), float(ert["resistividad_ohm_m"].max())
        ) if len(ert) else None,
        "profundidad_maxima_masw_m": float(masw["profundidad_m"].max()) if len(masw) else None,
        "profundidad_maxima_ert_m": float(ert["profundidad_m"].max()) if len(ert) else None,
        "unidades_en_sondeos": sorted(boreholes["unidad_hidroestratigrafica"].unique().tolist())
            if len(boreholes) else [],
    }
