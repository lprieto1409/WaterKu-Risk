"""
Visualización del módulo de transporte de contaminantes (ADE / MT3DMS).
Genera: mapas de pluma, breakthrough curves, balance de masa.
Backend no-interactivo (Agg) para compatibilidad con Snakemake.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mapa de concentración (pluma) por capa
# ---------------------------------------------------------------------------

def plot_plume_map(
    C2d: np.ndarray,
    time_years: float,
    layer: int,
    sources: list[dict],
    receptors: list[dict],
    cfg: dict,
    output_path: str | Path,
) -> None:
    """
    Mapa de concentración normalizada C/Cs para una capa y tiempo dados.
    Incluye isoconcentraciones (5 niveles), marcadores de emisores y asentamientos.
    """
    g = cfg["grid"]
    nrow, ncol = g["nrow"], g["ncol"]
    dx, dy = g["delr"], g["delc"]

    fig, ax = plt.subplots(figsize=(10, 8))

    Cmax = float(np.nanmax(C2d)) if np.nanmax(C2d) > 0 else 1.0
    Cnorm = C2d / Cmax

    extent = [0, ncol * dx / 1000, 0, nrow * dy / 1000]  # km
    im = ax.imshow(
        Cnorm,
        origin="upper",
        extent=extent,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        aspect="auto",
        alpha=0.85,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("C / Cmax  [-]", fontsize=9)

    # Isoconcentraciones
    levels = [0.01, 0.05, 0.10, 0.25, 0.50]
    X = np.linspace(0, ncol * dx / 1000, ncol)
    Y = np.linspace(nrow * dy / 1000, 0, nrow)
    try:
        cs = ax.contour(X, Y, Cnorm, levels=levels, colors="navy", linewidths=0.8, alpha=0.6)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
    except Exception:
        pass

    # Emisores como triángulos rojos
    for src in sources:
        x_km = (src["col"] + 0.5) * dx / 1000
        y_km = (nrow - src["row"] - 0.5) * dy / 1000
        ax.plot(x_km, y_km, "^r", ms=8 + 4 * src.get("hazard_weight", 1.0),
                label="Emisor" if src == sources[0] else "")
        ax.annotate(src["id"], (x_km, y_km), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="darkred")

    # Asentamientos como círculos azules
    for rec in receptors:
        x_km = (rec["col"] + 0.5) * dx / 1000
        y_km = (nrow - rec["row"] - 0.5) * dy / 1000
        marker = "o" if not rec.get("serves_under5", False) else "s"
        ax.plot(x_km, y_km, marker, color="steelblue", ms=6,
                label="Asentamiento" if rec == receptors[0] else "")
        ax.annotate(rec.get("name", rec["id"]), (x_km, y_km),
                    textcoords="offset points", xytext=(5, -8), fontsize=6, color="navy")

    ax.set_xlabel("Distancia X [km]", fontsize=9)
    ax.set_ylabel("Distancia Y [km]", fontsize=9)
    ax.set_title(
        f"Pluma de contaminantes — Capa {layer + 1} — t = {time_years:.0f} años\n"
        f"(ADE implícito, Manual MT3DMS Ec. 4)",
        fontsize=10,
    )

    handles = [
        plt.Line2D([0], [0], marker="^", color="r", linestyle="None", ms=8, label="Emisor"),
        plt.Line2D([0], [0], marker="o", color="steelblue", linestyle="None", ms=6, label="Asentamiento"),
        plt.Line2D([0], [0], marker="s", color="steelblue", linestyle="None", ms=6, label="Asentamiento (niños <5a)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Pluma guardada: {output_path}")


# ---------------------------------------------------------------------------
# Breakthrough curves (concentración vs tiempo en asentamientos)
# ---------------------------------------------------------------------------

def plot_breakthrough_curves(
    df: pd.DataFrame,
    settlements_gdf: gpd.GeoDataFrame | None,
    output_path: str | Path,
) -> None:
    """
    Curvas C/Cs vs tiempo [años] para cada asentamiento receptor.
    """
    time_col = "time_years"
    receptor_ids = [c for c in df.columns if c != time_col]
    if not receptor_ids:
        return

    # Mapa id → name desde GeoJSON
    name_map = {}
    if settlements_gdf is not None:
        for _, row in settlements_gdf.iterrows():
            name_map[str(row.get("id", ""))] = row.get("name", str(row.get("id", "")))

    n = len(receptor_ids)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols) if n > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.get_cmap("tab10")

    for idx, rid in enumerate(receptor_ids):
        ax = axes_flat[idx]
        label = name_map.get(str(rid), str(rid))
        color = cmap(idx % 10)
        ax.plot(df[time_col], df[rid], color=color, lw=1.8)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Tiempo [años]", fontsize=7)
        ax.set_ylabel("C / C_fuente [-]", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Breakthrough Curves — Concentración en asentamientos\n(Solver ADE, Manual MT3DMS Ec. 4)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Breakthrough curves guardadas: {output_path}")


# ---------------------------------------------------------------------------
# Balance de masa
# ---------------------------------------------------------------------------

def plot_mass_balance(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Gráfico de masa total disuelta, sorbida y decaída vs tiempo.
    Incluye porcentaje de discrepancia (Manual MT3DMS Ec. 133).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(df["time_years"], df["mass_dissolved_kg"], "b-", lw=1.8, label="Masa disuelta [kg]")
    ax1.plot(df["time_years"], df["mass_sorbed_kg"], "g--", lw=1.4, label="Masa sorbida [kg]")
    ax1.plot(df["time_years"], df["mass_total_kg"], "k-", lw=2.0, label="Masa total [kg]")
    if df["mass_decayed_kg"].max() > 0:
        ax1.plot(df["time_years"], df["mass_decayed_kg"], "r:", lw=1.4, label="Masa decaída [kg]")
    ax1.set_ylabel("Masa [kg]", fontsize=9)
    ax1.set_title(
        "Balance de masa — Módulo ADE (Manual MT3DMS Ec. 133)\n"
        "Discrepancy (%) = |IN − OUT| / [0.5(IN + OUT)] × 100",
        fontsize=9,
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["time_years"], df["discrepancy_pct"], "m-", lw=1.6)
    ax2.axhline(1.0, color="r", linestyle="--", lw=1, label="Umbral 1%")
    ax2.set_xlabel("Tiempo [años]", fontsize=9)
    ax2.set_ylabel("Discrepancia [%]", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Balance de masa guardado: {output_path}")
