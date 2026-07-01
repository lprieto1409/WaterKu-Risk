"""Carga de logs de sondeo (perforaciones washboring + SPT) y construcción del
perfil hidroestratigráfico para el modelo conceptual.

Construido nuevo para WaterKu-Risk — no existe ningún módulo equivalente en el
repo WaterKu original. Formato de entrada esperado (CSV/XLSX por sondeo):
columnas `profundidad_desde_m`, `profundidad_hasta_m`, `n_spt` (opcional),
`descripcion_litologica`, `unidad_hidroestratigrafica`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BoreholeLog:
    borehole_id: str
    x: float
    y: float
    elevacion_boca_m: float
    estratos: pd.DataFrame  # profundidad_desde_m, profundidad_hasta_m, n_spt, descripcion_litologica, unidad_hidroestratigrafica
    nivel_freatico_m: float | None = None  # profundidad al nivel freático medido en el sondeo, si se registró


def load_borehole_log(path: str | Path, borehole_id: str, x: float, y: float,
                       elevacion_boca_m: float, nivel_freatico_m: float | None = None) -> BoreholeLog:
    """Carga el log de un sondeo (CSV o XLSX) con las columnas esperadas."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    columnas_requeridas = {"profundidad_desde_m", "profundidad_hasta_m", "descripcion_litologica"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Log de sondeo {path} no tiene las columnas requeridas: {faltantes}")
    if "unidad_hidroestratigrafica" not in df.columns:
        df["unidad_hidroestratigrafica"] = df["descripcion_litologica"]
    return BoreholeLog(borehole_id=borehole_id, x=x, y=y, elevacion_boca_m=elevacion_boca_m,
                        estratos=df, nivel_freatico_m=nivel_freatico_m)


def build_stratigraphic_profile(boreholes: list[BoreholeLog]) -> pd.DataFrame:
    """Combina los logs de varios sondeos en una tabla única de unidades
    hidroestratigráficas con cota top/base por sondeo, para construir las
    capas (layers) del modelo MODFLOW en `scripts/modflow/model_builder.py`.
    """
    filas = []
    for bh in boreholes:
        for _, estrato in bh.estratos.iterrows():
            filas.append({
                "borehole_id": bh.borehole_id,
                "x": bh.x,
                "y": bh.y,
                "unidad_hidroestratigrafica": estrato["unidad_hidroestratigrafica"],
                "top_m": bh.elevacion_boca_m - estrato["profundidad_desde_m"],
                "base_m": bh.elevacion_boca_m - estrato["profundidad_hasta_m"],
                "n_spt": estrato.get("n_spt"),
                "descripcion_litologica": estrato["descripcion_litologica"],
            })
    return pd.DataFrame(filas)
