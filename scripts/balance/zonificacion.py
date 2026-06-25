"""Mapa base de apoyo para la zonificación hidrogeológica — Fase 5 de la
Propuesta Arkel (diagnóstico y recomendaciones).

Genera solo el mapa SIG base (nivel freático interpolado + unidades K) para
que el hidrogeólogo dibuje encima las zonas de manejo del terreno y control
de infiltración. El contenido de la zonificación (dónde está cada zona y por
qué) es juicio de ingeniería senior — no se deriva automáticamente aquí.
"""
from __future__ import annotations

import pandas as pd


def plot_zonificacion_hidrogeologica(napa_grid_csv: str, k_por_unidad: pd.DataFrame, output_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = pd.read_csv(napa_grid_csv).values

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    im = axes[0].imshow(grid, cmap="Blues_r", origin="upper")
    plt.colorbar(im, ax=axes[0], label="Elevación napa freática (m asl)", shrink=0.8)
    axes[0].set_title("Napa freática interpolada")

    if not k_por_unidad.empty:
        axes[1].barh(k_por_unidad["unidad_hidroestratigrafica"], k_por_unidad["k_m_s_geomean"])
        axes[1].set_xscale("log")
        axes[1].set_xlabel("K (m/s, media geométrica)")
        axes[1].set_title("Conductividad hidráulica por unidad")
    else:
        axes[1].text(0.5, 0.5, "Sin ensayos de permeabilidad todavía", ha="center", va="center")
        axes[1].axis("off")

    plt.suptitle("Mapa base de apoyo — Zonificación hidrogeológica (Fase 5)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
