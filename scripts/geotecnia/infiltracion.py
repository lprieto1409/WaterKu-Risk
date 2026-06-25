"""Interpretación de ensayos de permeabilidad/infiltración in situ (Lefranc,
Porchet) y de laboratorio (permeámetro), bajo la ley de Darcy — Fase 2 de la
Propuesta Arkel.

Construido nuevo para WaterKu-Risk: no existe ningún módulo equivalente en el
repo WaterKu original (confirmado por exploración exhaustiva de las ~50
branches). WaterKu-Risk no ejecuta los ensayos (los ejecuta ARKEL en campo) ni
reinterpreta geofísica cruda — este módulo solo aplica las fórmulas estándar
de reducción de cada ensayo a conductividad hidráulica K.

Referencias:
    - Ensayo Lefranc (carga variable/constante), norma de referencia NF P94-132
      y práctica habitual peruana (estudios de mecánica de suelos/CISMID).
    - Ensayo Porchet (pozo cilíndrico, carga variable) — método estándar para
      ensayos de infiltración sobre el nivel freático.
    - Permeámetro de laboratorio (carga constante/variable) — ley de Darcy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class LefrancResult:
    borehole_id: str
    k_m_s: float
    metodo: str  # "carga_constante" | "carga_variable"


def k_from_lefranc_carga_constante(Q_m3_s: float, H_m: float, shape_factor_C: float) -> float:
    """K = Q / (C * H) — ensayo Lefranc a caudal constante.

    Args:
        Q_m3_s: caudal constante inyectado para mantener la sobreelevación H.
        H_m: sobreelevación de carga mantenida durante el ensayo (m).
        shape_factor_C: factor de forma de la cavidad de ensayo (m), según
            geometría (esférica o cilíndrica) — ver `shape_factor_spherical`
            / `shape_factor_cylindrical`.
    """
    return Q_m3_s / (shape_factor_C * H_m)


def k_from_lefranc_carga_variable(casing_area_m2: float, h1_m: float, h2_m: float,
                                   dt_s: float, shape_factor_C: float) -> float:
    """K = (A / (C * Δt)) * ln(h1/h2) — ensayo Lefranc a carga variable
    (decreciente), entre dos lecturas de nivel h1 (en t1) y h2 (en t2=t1+Δt).
    """
    return (casing_area_m2 / (shape_factor_C * dt_s)) * math.log(h1_m / h2_m)


def shape_factor_spherical(radius_m: float) -> float:
    """Factor de forma C para cavidad esférica de radio `radius_m`."""
    return 2.0 * math.pi * radius_m


def shape_factor_cylindrical(length_m: float, radius_m: float) -> float:
    """Factor de forma C para cavidad cilíndrica de longitud >> radio."""
    return 2.0 * math.pi * length_m / math.log(2.0 * length_m / radius_m)


def k_from_porchet(radius_m: float, h1_m: float, h2_m: float, dt_s: float) -> float:
    """K = (r / (2*Δt)) * ln[(2*h1 + r) / (2*h2 + r)] — ensayo Porchet (pozo
    cilíndrico de radio `radius_m`, carga decreciente entre h1 (t1) y h2
    (t1+Δt)), método estándar de ensayos de infiltración sobre el nivel
    freático.
    """
    return (radius_m / (2.0 * dt_s)) * math.log((2.0 * h1_m + radius_m) / (2.0 * h2_m + radius_m))


def k_from_lab_permeameter_carga_constante(volumen_m3: float, longitud_muestra_m: float,
                                            area_muestra_m2: float, carga_h_m: float,
                                            tiempo_s: float) -> float:
    """K = (V * L) / (A * H * t) — permeámetro de carga constante (ley de Darcy)."""
    return (volumen_m3 * longitud_muestra_m) / (area_muestra_m2 * carga_h_m * tiempo_s)


def k_from_lab_permeameter_carga_variable(area_bureta_m2: float, longitud_muestra_m: float,
                                           area_muestra_m2: float, h1_m: float, h2_m: float,
                                           dt_s: float) -> float:
    """K = (a * L) / (A * Δt) * ln(h1/h2) — permeámetro de carga variable."""
    return (area_bureta_m2 * longitud_muestra_m) / (area_muestra_m2 * dt_s) * math.log(h1_m / h2_m)


def summarize_k_by_unit(tests: pd.DataFrame, stratigraphy: pd.DataFrame) -> pd.DataFrame:
    """Asigna cada ensayo (`tests`, con columnas `borehole_id`, `profundidad_m`,
    `k_m_s`) a su unidad hidroestratigráfica (`stratigraphy`, salida de
    `scripts.geotecnia.sondeos.build_stratigraphic_profile`) por profundidad,
    y resume K representativa por unidad mediante la media geométrica
    (estándar para conductividad hidráulica, que se distribuye log-normal).

    Returns:
        DataFrame con columnas `unidad_hidroestratigrafica`, `k_m_s_geomean`,
        `n_ensayos` — insumo directo para el paquete NPF del modelo MODFLOW
        (`scripts/modflow/model_builder.py`).
    """
    merged_rows = []
    for _, t in tests.iterrows():
        candidatos = stratigraphy[
            (stratigraphy["borehole_id"] == t["borehole_id"]) &
            (stratigraphy["base_m"] <= t.get("elevacion_m", stratigraphy["top_m"])) &
            (stratigraphy["top_m"] >= t.get("elevacion_m", stratigraphy["base_m"]))
        ]
        unidad = candidatos.iloc[0]["unidad_hidroestratigrafica"] if len(candidatos) else "sin_clasificar"
        merged_rows.append({"unidad_hidroestratigrafica": unidad, "k_m_s": t["k_m_s"]})

    df = pd.DataFrame(merged_rows)
    if df.empty:
        return pd.DataFrame(columns=["unidad_hidroestratigrafica", "k_m_s_geomean", "n_ensayos"])

    resumen = df.groupby("unidad_hidroestratigrafica")["k_m_s"].agg(
        k_m_s_geomean=lambda s: math.exp(sum(math.log(v) for v in s if v > 0) / len(s)),
        n_ensayos="count",
    ).reset_index()
    return resumen
