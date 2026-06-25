"""MODPATH7 (particle tracking) — trayectorias y tiempos de tránsito del flujo
subterráneo, e interacción laguna-acuífero (Fase 3 de la Propuesta Arkel).

Confirmado por exploración exhaustiva de las ~50 branches del repo WaterKu
original que MODPATH nunca se integró (el único resultado de búsqueda fue una
analogía en un docstring, no código real) — construido enteramente nuevo para
WaterKu-Risk, usando `flopy.modpath.Modpath7` (ya disponible, sin
dependencias nuevas).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_modpath_model(gwf, model_dir: str, model_name: str,
                         particle_placement: str = "cell_centers", port: int = 1):
    """Crea el modelo MODPATH7 (forward tracking) ligado al modelo de flujo
    `gwf` ya construido/ejecutado por `model_builder.build_mf6_simulation`.
    """
    import flopy

    mp_name = f"{model_name}_mp"
    mp = flopy.modpath.Modpath7.create_mp7(
        modelname=mp_name,
        trackdir="forward",
        flowmodel=gwf,
        model_ws=str(model_dir),
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
        nodes=None,
    )
    return mp


def define_particle_groups(mp, zones_of_interest: list[str], lake_cells: list[tuple] | None = None):
    """Define los grupos de partículas. Para `"lake_cells"` (interacción
    laguna-acuífero, lo que pide la Fase 3 del PDF), se colocan partículas en
    las celdas conectadas al paquete LAK (`lake_cells`, salida de
    `model_builder.add_lake_package`)."""
    import flopy

    groups = []
    if "lake_cells" in zones_of_interest and lake_cells:
        sd = flopy.modpath.ParticleGroup(
            particlegroupname="lago_colombina_sur",
            particledata=flopy.modpath.ParticleData(lake_cells, drape=0, structured=True),
        )
        groups.append(sd)
    return groups


def run_modpath(mp) -> None:
    mp.write_input()
    success, _buff = mp.run_model(silent=False)
    if not success:
        raise RuntimeError("MODPATH7 no terminó correctamente — revisar el log de la corrida.")


def read_pathlines(model_dir: str, model_name: str) -> pd.DataFrame:
    import flopy.utils

    mp_name = f"{model_name}_mp"
    pathline_file = Path(model_dir) / f"{mp_name}.mppth"
    pf = flopy.utils.PathlineFile(str(pathline_file))
    return pd.DataFrame(pf.get_alldata())


def read_endpoints(model_dir: str, model_name: str) -> pd.DataFrame:
    """Lee los puntos finales de las trayectorias (tiempos de tránsito), pedidos
    explícitamente en la Fase 3 del PDF de la propuesta."""
    import flopy.utils

    mp_name = f"{model_name}_mp"
    endpoint_file = Path(model_dir) / f"{mp_name}.mpend"
    ef = flopy.utils.EndpointFile(str(endpoint_file))
    return pd.DataFrame(ef.get_alldata())


def plot_pathlines_map(pathlines: pd.DataFrame, gwf, output_path: Path) -> None:
    """Mapa de trayectorias en planta, mismo estilo que
    `modflow_runner.plot_head_map` (no se modifica ese archivo)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import flopy.plot

    fig, ax = plt.subplots(figsize=(10, 10))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pmv.plot_grid(linewidth=0.3, alpha=0.4)
    for pid in pathlines["particleid"].unique():
        traj = pathlines[pathlines["particleid"] == pid]
        ax.plot(traj["x"], traj["y"], lw=0.8, alpha=0.7, color="steelblue")
    ax.set_title("MODPATH7 — Trayectorias de partículas (laguna-acuífero)")
    ax.set_xlabel("Easting (m UTM)")
    ax.set_ylabel("Northing (m UTM)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def lake_interaction_summary(endpoints: pd.DataFrame, lake_cells: list[tuple]) -> dict:
    """Cuantifica la interacción laguna-acuífero a partir de los endpoints:
    cuántas partículas entran/salen por las celdas conectadas al paquete LAK
    y su tiempo de tránsito (`time` en el endpoint file), pedido en la Fase 3
    del PDF ("recarga, descarga y filtraciones")."""
    if endpoints.empty:
        return {"n_particulas": 0, "tiempo_transito_medio_s": None}

    lake_rc = {(c[1], c[2]) for c in lake_cells} if lake_cells else set()
    en_laguna = endpoints[endpoints.apply(
        lambda r: (int(r.get("row", -1)), int(r.get("column", -1))) in lake_rc, axis=1
    )] if lake_rc else endpoints

    return {
        "n_particulas": int(len(en_laguna)),
        "tiempo_transito_medio_s": float(en_laguna["time"].mean()) if "time" in en_laguna.columns and len(en_laguna) else None,
        "tiempo_transito_max_s": float(en_laguna["time"].max()) if "time" in en_laguna.columns and len(en_laguna) else None,
    }
