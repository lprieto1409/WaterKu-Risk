"""Pruebas formales de homogeneidad de series hidroclimáticas (Mann-Kendall y
Pettitt), requeridas por la Fase 1 de la Propuesta Arkel ("pruebas de
homogeneidad") y no presentes en ninguna branch del repo WaterKu original
(`analisis_precipitacion.py` solo hace curva doble masa / regresión, no un
test estadístico formal de homogeneidad).

Construido nuevo para WaterKu-Risk, usando la librería `pymannkendall`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pymannkendall as mk


@dataclass
class HomogeneityResult:
    test: str
    is_homogeneous: bool
    trend_or_change: str
    p_value: float
    statistic: float
    change_point_index: int | None = None


def mann_kendall_test(series: Sequence[float], alpha: float = 0.05) -> HomogeneityResult:
    """Test de tendencia Mann-Kendall. Una serie homogénea no debe mostrar
    tendencia significativa al nivel `alpha`."""
    result = mk.original_test(series, alpha=alpha)
    return HomogeneityResult(
        test="mann_kendall",
        is_homogeneous=(result.trend == "no trend"),
        trend_or_change=result.trend,
        p_value=result.p,
        statistic=result.z,
    )


def pettitt_test(series: Sequence[float], alpha: float = 0.05) -> HomogeneityResult:
    """Test de Pettitt para detección de un punto de cambio (quiebre) en la
    media de la serie. Una serie homogénea no debe mostrar un quiebre
    significativo al nivel `alpha`."""
    result = mk.pettitt_test(series)
    is_significant = result.p < alpha
    return HomogeneityResult(
        test="pettitt",
        is_homogeneous=not is_significant,
        trend_or_change="quiebre" if is_significant else "sin quiebre",
        p_value=result.p,
        statistic=result.U,
        change_point_index=int(result.cp) if is_significant else None,
    )


def run_homogeneity_tests(series: Sequence[float], tests: Sequence[str] = ("mann_kendall", "pettitt"),
                           alpha: float = 0.05) -> list[HomogeneityResult]:
    """Corre la batería de pruebas de homogeneidad indicada en
    `config.hidrologia.yaml::homogeneidad.tests`."""
    dispatch = {"mann_kendall": mann_kendall_test, "pettitt": pettitt_test}
    return [dispatch[name](series, alpha=alpha) for name in tests if name in dispatch]
