"""Estadísticos de calibración ASTM D5981 (media de residuales, RMSE, RMSE
normalizado, cierre del balance de masa) — Fase 3 de la Propuesta Arkel.

Generaliza, sin modificarlos, los `_rmse`/`_mae` privados de
`scripts/modflow/modflow_runner.py:198-205` (usados hoy en
`plot_head_vs_piezometers`) y reutiliza `read_budget()` de ese mismo módulo
para el cierre de balance de masa. No existe una versión más completa
(RMSE normalizado, régimen transitorio) en ninguna branch del repo WaterKu
original — construido nuevo para WaterKu-Risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualStats:
    n: int
    mean_residual: float   # media (obs - sim); ASTM D5981 reporta este signo
    rmse: float
    rmse_normalized: float  # RMSE / (rango de los observados), adimensional
    regimen: str = "estacionario"  # "estacionario" | "transitorio"


def compute_residual_stats(obs: np.ndarray, sim: np.ndarray, regimen: str = "estacionario") -> ResidualStats:
    """Calcula los estadísticos de ajuste que pide ASTM D5981: media de
    residuales, RMSE y RMSE normalizado (RMSE / rango de los valores
    observados, criterio habitual de aceptación < 10%)."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_v, sim_v = obs[mask], sim[mask]
    if len(obs_v) == 0:
        raise ValueError("No hay pares observado/simulado válidos para calcular estadísticos de calibración.")

    residuales = obs_v - sim_v
    rmse = float(np.sqrt(np.mean(residuales ** 2)))
    rango_obs = float(obs_v.max() - obs_v.min())
    rmse_normalizado = rmse / rango_obs if rango_obs > 0 else float("nan")

    return ResidualStats(
        n=int(len(obs_v)),
        mean_residual=float(np.mean(residuales)),
        rmse=rmse,
        rmse_normalized=rmse_normalizado,
        regimen=regimen,
    )


def mass_balance_closure(budget: dict) -> dict:
    """Cierre del balance de masa del modelo a partir del dict que devuelve
    `modflow_runner.read_budget()` ({paquete: {"in": x, "out": y}}).

    Devuelve el mismo "PERCENT DISCREPANCY" que reporta MODFLOW-6 en el
    listing (.lst), recalculado por suma de los paquetes ya parseados (el
    parser de `read_budget` descarta intencionalmente las líneas TOTAL y
    PERCENT del listing, así que se recalculan aquí en vez de tocar ese
    parser).
    """
    total_in = sum(v.get("in", 0.0) for v in budget.values())
    total_out = sum(v.get("out", 0.0) for v in budget.values())
    promedio = (total_in + total_out) / 2.0
    discrepancia_pct = 100.0 * (total_in - total_out) / promedio if promedio > 0 else float("nan")
    return {
        "total_in": total_in,
        "total_out": total_out,
        "percent_discrepancy": discrepancia_pct,
    }


def calibration_report(stats_steady: ResidualStats | None = None,
                        stats_transient: list[ResidualStats] | None = None,
                        mass_balance: dict | None = None) -> dict:
    """Ensambla el reporte de calibración (insumo del entregable) combinando
    estadísticos en régimen estacionario, una serie de estadísticos
    transitorios (uno por periodo de estrés calibrado) y el cierre de balance
    de masa."""
    return {
        "estacionario": stats_steady.__dict__ if stats_steady else None,
        "transitorio": [s.__dict__ for s in stats_transient] if stats_transient else None,
        "balance_de_masa": mass_balance,
    }
