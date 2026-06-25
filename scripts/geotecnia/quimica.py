"""Carga de resultados de laboratorio (sales solubles, sulfatos, cloruros)
para evaluar la agresividad del entorno — Fase 2 de la Propuesta Arkel.

Construido nuevo para WaterKu-Risk. Es un import/reporte tabular simple: la
evaluación de agresividad frente a concreto/estructuras es un criterio
normativo (ej. ACI 318 / RNE) que aplica el ingeniero, no un cálculo derivado
automáticamente aquí.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Umbrales referenciales de agresividad por sulfatos solubles en suelo,
# similares a los de ACI 318 / RNE E.060 (a confirmar por el ingeniero contra
# la norma vigente aplicable al proyecto).
_UMBRALES_SULFATOS_PPM = [
    (0, 1000, "despreciable"),
    (1000, 2000, "moderada"),
    (2000, 20000, "severa"),
    (20000, float("inf"), "muy_severa"),
]


def load_chemical_tests(path: str | Path) -> pd.DataFrame:
    """Carga resultados de laboratorio (CSV/XLSX) con columnas esperadas:
    `borehole_id`, `profundidad_m`, `sales_solubles_ppm`, `sulfatos_ppm`,
    `cloruros_ppm`."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if "sulfatos_ppm" in df.columns:
        df["clasificacion_agresividad_sulfatos"] = df["sulfatos_ppm"].apply(_clasificar_sulfatos)
    return df


def _clasificar_sulfatos(valor_ppm: float) -> str:
    for lo, hi, etiqueta in _UMBRALES_SULFATOS_PPM:
        if lo <= valor_ppm < hi:
            return etiqueta
    return "sin_clasificar"
