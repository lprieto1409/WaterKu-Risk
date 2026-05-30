#!/usr/bin/env python3
"""
MODFLOW-6 general runner and post-processor.

All paths and model constants are taken from a config dict so this module
works with any MODFLOW-6 model, not just a specific project.

CLI usage (reads config from config/config.modflow.yaml):
    pixi run python scripts/modflow/modflow_runner.py
    pixi run python scripts/modflow/modflow_runner.py --no-run
    pixi run python scripts/modflow/modflow_runner.py --layer 3

Config dict keys expected by process_modflow():
    mf6_exe      str  - path to mf6.exe
    model_dir    str  - directory containing mfsim.nam
    model_name   str  - base name for .bhd / .cbc / .lst files
    piezo_csv    str  - CSV with observed piezometric levels
    output_dir   str  - output directory for PNG plots
    layer        int  - model layer (1-based) for head map (default 1)
    grid         dict - xorigin, yorigin, angrot_deg, delr, delc, nlay, nrow, ncol
"""

import math
import re
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# MODFLOW execution
# ---------------------------------------------------------------------------

def run_modflow(exe: str, model_dir: str) -> None:
    exe_path = Path(exe)
    if not exe_path.exists():
        raise FileNotFoundError(
            f"mf6.exe not found at: {exe}\n"
            "Set 'mf6_exe' in config/config.modflow.yaml"
        )

    print(f"  Running: {exe}")
    print(f"  Working directory: {model_dir}\n")

    result = subprocess.run(
        [str(exe_path)],
        cwd=model_dir,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    print("\n".join(lines[-25:]))

    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"MODFLOW-6 exited with code {result.returncode}")

    for suffix in (".bhd", ".cbc"):
        fpath = Path(model_dir) / (Path(model_dir).name + suffix)
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected output file not found after run: {fpath}\n"
                + "\n".join(lines[-30:])
            )

    print("\n  MODFLOW run successful. Output files confirmed.\n")


# ---------------------------------------------------------------------------
# Model loading (flopy)
# ---------------------------------------------------------------------------

def patch_and_load_sim(model_dir: str, model_name: str):
    """Load MFSimulation patching the original machine's absolute paths."""
    import flopy

    nam_path = Path(model_dir) / "mfsim.nam"
    content = nam_path.read_text()
    patched = re.sub(r"'[A-Za-z]:[^']*[\\/]([^\\/]+)'", r"'\1'", content)

    tmp_nam = Path(model_dir) / "_mfsim_patched.nam"
    try:
        tmp_nam.write_text(patched)
        sim = flopy.mf6.MFSimulation.load(
            sim_name="_mfsim_patched",
            sim_ws=str(model_dir),
            verbosity_level=0,
        )
        gwf = sim.get_model(model_name)
        return sim, gwf
    except Exception as exc:
        print(f"  [warn] flopy load failed ({exc}); plotting will use manual grid.")
        return None, None
    finally:
        if tmp_nam.exists():
            tmp_nam.unlink()


# ---------------------------------------------------------------------------
# Binary output readers
# ---------------------------------------------------------------------------

def read_heads(model_dir: str, model_name: str) -> np.ndarray:
    import flopy.utils

    bhd_path = Path(model_dir) / f"{model_name}.bhd"
    hf = flopy.utils.HeadFile(str(bhd_path))
    head = hf.get_data(kstpkper=(0, 0)).astype(float)
    head[head >= 1e29] = np.nan
    return head


def read_budget(model_dir: str, model_name: str) -> dict:
    lst_path = Path(model_dir) / f"{model_name}.lst"
    text = lst_path.read_text(errors="replace")

    start = text.rfind("VOLUME BUDGET FOR ENTIRE MODEL")
    if start == -1:
        return {}
    block = text[start: start + 3000]

    budget = {}
    pkg_pat = re.compile(r"^\s+([A-Z_]+)\s*=\s*([\d.E+\-]+)", re.IGNORECASE)
    section = None
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("IN:"):
            section = "in"
            continue
        if stripped.startswith("OUT:"):
            section = "out"
            continue
        if section is None:
            continue
        m = pkg_pat.match(line)
        if m:
            key = m.group(1).upper()
            if key in ("TOTAL", "PERCENT", "STORAGE"):
                continue
            budget.setdefault(key, {"in": 0.0, "out": 0.0})
            budget[key][section] = float(m.group(2))
    return budget


# ---------------------------------------------------------------------------
# Piezometers
# ---------------------------------------------------------------------------

def load_piezometers(piezo_csv: str, grid_cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(piezo_csv)
    xorigin    = grid_cfg["xorigin"]
    yorigin    = grid_cfg["yorigin"]
    angle      = math.radians(grid_cfg["angrot_deg"])
    delr       = grid_cfg["delr"]
    delc       = grid_cfg["delc"]
    nrow       = grid_cfg["nrow"]

    rows, cols = [], []
    for _, r in df.iterrows():
        dx = r["Easting"] - xorigin
        dy = r["Northing"] - yorigin
        x_m =  dx * math.cos(angle) + dy * math.sin(angle)
        y_m = -dx * math.sin(angle) + dy * math.cos(angle)
        col = int(x_m / delr)
        row = int((nrow * delc - y_m) / delc)
        rows.append(max(0, min(row, nrow - 1)))
        cols.append(max(0, min(col, grid_cfg["ncol"] - 1)))

    df["row"]   = rows
    df["col"]   = cols
    df["layer"] = 0
    return df


def extract_simulated_heads(head: np.ndarray, piezo_df: pd.DataFrame) -> pd.DataFrame:
    df = piezo_df.copy()
    df["sim_head"] = [
        head[int(r["layer"]), int(r["row"]), int(r["col"])]
        for _, r in df.iterrows()
    ]
    return df


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _rmse(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def _mae(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return float(np.mean(np.abs(obs[mask] - sim[mask])))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_head_map(
    gwf,
    head: np.ndarray,
    piezo_df: pd.DataFrame,
    layer: int,
    output_path: Path,
    grid_cfg: dict,
) -> None:
    import flopy.plot

    layer0     = layer - 1
    head_layer = np.ma.masked_invalid(head[layer0])

    fig, ax = plt.subplots(figsize=(13, 10))

    if gwf is not None:
        pmv  = flopy.plot.PlotMapView(model=gwf, layer=layer0, ax=ax)
        quad = pmv.plot_array(head_layer, alpha=0.75, cmap="Blues_r")
        try:
            cs = pmv.contour_array(head_layer, levels=15,
                                   colors="navy", linewidths=0.5, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%d m")
        except Exception:
            pass
    else:
        quad = ax.imshow(head_layer, cmap="Blues_r",
                         origin="upper", aspect="auto")

    plt.colorbar(quad, ax=ax, label="Hydraulic Head (m asl)", shrink=0.8, pad=0.02)

    ax.scatter(
        piezo_df["Easting"], piezo_df["Northing"],
        c="red", s=35, zorder=5, marker="^",
        label=f"Piezometers (n={len(piezo_df)})",
        edgecolors="darkred", linewidths=0.5,
    )

    nlay = grid_cfg["nlay"]
    nrow = grid_cfg["nrow"]
    ncol = grid_cfg["ncol"]
    delr = grid_cfg["delr"]
    rot  = grid_cfg["angrot_deg"]

    ax.set_xlabel("Easting (m UTM)", fontsize=11)
    ax.set_ylabel("Northing (m UTM)", fontsize=11)
    ax.set_title(
        f"Hydraulic Head — Layer {layer}\n"
        f"{nlay}L × {nrow}R × {ncol}C  |  Δ={delr:.0f}m  |  rot={rot}°",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_head_vs_piezometers(piezo_df: pd.DataFrame, output_path: Path) -> None:
    valid = piezo_df.dropna(subset=["sim_head"])
    dry   = piezo_df[piezo_df["sim_head"].isna()]

    obs = valid["piezometricLevel"].values
    sim = valid["sim_head"].values

    rmse   = _rmse(obs, sim)
    mae    = _mae(obs, sim)
    lim_lo = min(obs.min(), sim.min()) - 80
    lim_hi = max(obs.max(), sim.max()) + 80

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.5, label="1:1", alpha=0.7)

    sc = ax.scatter(
        obs, sim,
        c=obs, cmap="viridis", s=60,
        edgecolors="black", linewidths=0.5,
        zorder=4, label=f"Wells (n={len(valid)})",
    )
    plt.colorbar(sc, ax=ax, label="Observed Head (m asl)", shrink=0.85)

    if len(dry) > 0:
        ax.scatter(
            dry["piezometricLevel"],
            [lim_lo + 20] * len(dry),
            marker="x", color="red", s=80, zorder=5,
            label=f"Dry cells (n={len(dry)})",
        )

    info = f"RMSE = {rmse:.1f} m\nMAE  = {mae:.1f} m\nn    = {len(valid)}"
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax.set_xlabel("Observed Head — piezometricLevel (m asl)", fontsize=12)
    ax.set_ylabel("Simulated Head — MODFLOW Layer 1 (m asl)", fontsize=12)
    ax.set_title(
        "Calibración: Carga Hidráulica Observada vs Simulada\nMODFLOW-6",
        fontsize=13,
    )
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_water_budget(budget: dict, output_path: Path) -> None:
    if not budget:
        print("  [warn] No budget data available — skipping water_budget.png")
        return

    packages  = list(budget.keys())
    inflows   = [budget[p]["in"] for p in packages]
    outflows  = [budget[p]["out"] for p in packages]
    total_in  = sum(inflows)
    total_out = sum(outflows)
    disc_pct  = abs(total_in - total_out) / total_in * 100 if total_in else 0

    x     = np.arange(len(packages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_in  = ax.bar(x - width / 2, inflows,  width, label="Inflow",
                      color="steelblue", edgecolor="black", linewidth=0.7)
    bars_out = ax.bar(x + width / 2, outflows, width, label="Outflow",
                      color="tomato",    edgecolor="black", linewidth=0.7)

    for bar in list(bars_in) + list(bars_out):
        h = bar.get_height()
        if h > 1e-6:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(total_in,  color="steelblue", ls="--", lw=1.2, alpha=0.6,
               label=f"Total IN  = {total_in:.4f} m³/s")
    ax.axhline(total_out, color="tomato",    ls="--", lw=1.2, alpha=0.6,
               label=f"Total OUT = {total_out:.4f} m³/s")

    ax.set_xlabel("Paquete de Condición de Borde", fontsize=12)
    ax.set_ylabel("Caudal (m³/s)", fontsize=12)
    ax.set_title(
        f"Balance Hídrico por Paquete — Discrepancia = {disc_pct:.2f}%\n"
        "MODFLOW-6 (estado estacionario)",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(packages, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


# ---------------------------------------------------------------------------
# Grid coordinate helper
# ---------------------------------------------------------------------------

def _grid_utm_coords(grid_cfg: dict):
    """Meshgrids X, Y en coordenadas UTM para el centro de cada celda."""
    nrow, ncol = grid_cfg["nrow"], grid_cfg["ncol"]
    delr, delc = grid_cfg["delr"], grid_cfg["delc"]
    xorigin, yorigin = grid_cfg["xorigin"], grid_cfg["yorigin"]
    angle = math.radians(grid_cfg["angrot_deg"])

    j_c = (np.arange(ncol) + 0.5) * delr
    i_c = (nrow - 0.5 - np.arange(nrow)) * delc
    J, I = np.meshgrid(j_c, i_c)

    X = xorigin + J * math.cos(angle) - I * math.sin(angle)
    Y = yorigin + J * math.sin(angle) + I * math.cos(angle)
    return X, Y


# ---------------------------------------------------------------------------
# 3D plots
# ---------------------------------------------------------------------------

def plot_3d_head_surface(
    head: np.ndarray,
    grid_cfg: dict,
    layer: int,
    output_path: Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    layer0 = layer - 1
    X, Y = _grid_utm_coords(grid_cfg)
    Z = head[layer0].copy()
    z_floor = np.nanmin(Z) - 20
    Z_plot = np.where(np.isfinite(Z), Z, z_floor)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z_plot, cmap="Blues_r", alpha=0.9,
        linewidth=0, antialiased=True, rstride=2, cstride=2,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Carga Hidráulica (m snm)")
    ax.set_xlabel("Easting (m UTM)", labelpad=8)
    ax.set_ylabel("Northing (m UTM)", labelpad=8)
    ax.set_zlabel("Carga (m)", labelpad=8)
    ax.set_title(
        f"Superficie Piezométrica 3D — Capa {layer}\n"
        f"MODFLOW-6  |  {grid_cfg['nrow']}×{grid_cfg['ncol']} celdas"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_3d_all_layers(
    head: np.ndarray,
    grid_cfg: dict,
    output_path: Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.patches import Patch

    nlay = grid_cfg["nlay"]
    X, Y = _grid_utm_coords(grid_cfg)
    cmaps = ["Blues_r", "Greens_r", "Oranges_r", "Purples_r", "Reds_r"]
    z_floor = np.nanmin(head[np.isfinite(head)]) - 20

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection="3d")

    for lay in range(nlay):
        Z = head[lay].copy()
        Z_plot = np.where(np.isfinite(Z), Z, z_floor)
        ax.plot_surface(
            X, Y, Z_plot, cmap=cmaps[lay], alpha=0.55,
            linewidth=0, antialiased=False, rstride=3, cstride=3,
        )

    ax.set_xlabel("Easting (m UTM)", labelpad=8)
    ax.set_ylabel("Northing (m UTM)", labelpad=8)
    ax.set_zlabel("Carga (m)", labelpad=8)
    ax.set_title("Superficies Piezométricas — 5 Capas\nMODFLOW-6")
    handles = [
        Patch(color=plt.get_cmap(c)(0.5), alpha=0.7, label=f"Capa {i + 1}")
        for i, c in enumerate(cmaps[:nlay])
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def _launch_interactive_viewer(
    head: np.ndarray,
    grid_cfg: dict,
    layer: int,
) -> None:
    """Abre una ventana nativa matplotlib 3D interactiva (rotar/zoom/pan)."""
    import tempfile

    layer0 = layer - 1
    X, Y = _grid_utm_coords(grid_cfg)
    Z = head[layer0].copy()
    Z[~np.isfinite(Z)] = np.nan

    tmp = Path(tempfile.mktemp(suffix=".npz"))
    np.savez(tmp, X=X, Y=Y, Z=Z)

    viewer_code = f"""
import numpy as np, os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

with np.load(r'{tmp}') as data:
    X, Y, Z = data['X'].copy(), data['Y'].copy(), data['Z'].copy()
os.remove(r'{tmp}')

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.95,
                       linewidth=0, antialiased=True, rstride=2, cstride=2)
fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label='Carga Hidraulica (m snm)')
ax.set_xlabel('Easting (m UTM)', labelpad=8)
ax.set_ylabel('Northing (m UTM)', labelpad=8)
ax.set_zlabel('Carga (m)', labelpad=8)
ax.set_title('Superficie Piezometrica 3D - Capa {layer}\\nMODFLOW-6')
plt.tight_layout()
plt.show()
"""
    subprocess.Popen([sys.executable, "-c", viewer_code])
    print("  Visualizador 3D abierto (puedes rotar con el mouse).")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(head: np.ndarray, piezo_df: pd.DataFrame,
                  budget: dict, nlay: int) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print("  Estadísticas de Carga Hidráulica por Capa")
    print(sep)
    print(f"  {'Capa':>5}  {'Min (m)':>10}  {'Media (m)':>10}  {'Max (m)':>10}  {'Std (m)':>9}")
    print(sep)
    for lay in range(nlay):
        data = head[lay][np.isfinite(head[lay])]
        if data.size == 0:
            print(f"  {lay+1:>5}  {'(sin datos)':>10}")
            continue
        print(f"  {lay+1:>5}  {data.min():>10.1f}  {data.mean():>10.1f}"
              f"  {data.max():>10.1f}  {data.std():>9.1f}")

    valid = piezo_df.dropna(subset=["sim_head"])
    if len(valid) > 0:
        obs  = valid["piezometricLevel"].values
        sim  = valid["sim_head"].values
        rmse = _rmse(obs, sim)
        mae  = _mae(obs, sim)
        print(f"\n{sep}")
        print("  Calibración — Observado vs Simulado")
        print(sep)
        print(f"  Wells totales  : {len(piezo_df)}")
        print(f"  Wells activos  : {len(valid)}")
        print(f"  Celdas secas   : {len(piezo_df) - len(valid)}")
        print(f"  RMSE           : {rmse:.2f} m")
        print(f"  MAE            : {mae:.2f} m")

    if budget:
        total_in  = sum(v["in"]  for v in budget.values())
        total_out = sum(v["out"] for v in budget.values())
        disc      = abs(total_in - total_out) / total_in * 100 if total_in else 0
        print(f"\n{sep}")
        print("  Balance Hídrico (m³/s)")
        print(sep)
        print(f"  {'Paquete':>8}  {'Entrada':>12}  {'Salida':>12}")
        print(sep)
        for pkg, vals in budget.items():
            print(f"  {pkg:>8}  {vals['in']:>12.4f}  {vals['out']:>12.4f}")
        print(sep)
        print(f"  {'TOTAL':>8}  {total_in:>12.4f}  {total_out:>12.4f}")
        print(f"  Discrepancia   : {disc:.3f}%")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_modflow(cfg: dict, no_run: bool = False) -> None:
    """Run MODFLOW and produce the standard PNG outputs."""
    model_dir  = cfg["model_dir"]
    model_name = cfg["model_name"]
    piezo_csv  = cfg["piezo_csv"]
    layer      = int(cfg.get("layer", 1))
    grid_cfg   = cfg["grid"]
    nlay       = int(grid_cfg["nlay"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not no_run:
        print("[1/6] Ejecutando MODFLOW-6...")
        run_modflow(cfg["mf6_exe"], model_dir)
    else:
        print("[1/6] --no-run: usando archivos binarios existentes.")

    print("[2/6] Cargando simulación con flopy...")
    _, gwf = patch_and_load_sim(model_dir, model_name)

    print("[3/6] Leyendo cabezas hidráulicas (.bhd)...")
    head = read_heads(model_dir, model_name)
    print(f"      Forma: {head.shape}  "
          f"(rango activo: {np.nanmin(head):.0f} – {np.nanmax(head):.0f} m)")

    print("[4/6] Leyendo balance hídrico (.lst)...")
    budget = read_budget(model_dir, model_name)
    if budget:
        for pkg, v in budget.items():
            print(f"      {pkg:>6}: IN={v['in']:.4f}  OUT={v['out']:.4f}  m³/s")

    print("[5/6] Cargando piezómetros y extrayendo cabezas simuladas...")
    piezo_df = load_piezometers(piezo_csv, grid_cfg)
    piezo_df = extract_simulated_heads(head, piezo_df)
    n_valid  = piezo_df["sim_head"].notna().sum()
    print(f"      {len(piezo_df)} piezómetros cargados  ({n_valid} en celdas activas)")

    print_summary(head, piezo_df, budget, nlay)

    print("[6/6] Generando gráficas...")
    plot_head_map(
        gwf, head, piezo_df, layer,
        output_path=out_dir / f"head_map_layer{layer}.png",
        grid_cfg=grid_cfg,
    )
    plot_head_vs_piezometers(piezo_df, output_path=out_dir / "head_vs_piezometers.png")
    plot_water_budget(budget,          output_path=out_dir / "water_budget.png")
    plot_3d_head_surface(
        head, grid_cfg, layer,
        output_path=out_dir / f"head_3d_layer{layer}.png",
    )
    plot_3d_all_layers(head, grid_cfg, output_path=out_dir / "heads_3d_all_layers.png")
    if cfg.get("interactive", True):
        _launch_interactive_viewer(head, grid_cfg, layer)

    print(f"\n  Listo. Resultados en: {out_dir}\n")

    if cfg.get("open_modelmuse"):
        _open_modelmuse(cfg["model_dir"], cfg["model_name"],
                        cfg.get("modelmuse_exe", ""))


# ---------------------------------------------------------------------------
# ModelMuse launcher
# ---------------------------------------------------------------------------

def _open_modelmuse(model_dir: str, model_name: str, modelmuse_exe: str) -> None:
    gpt     = Path(model_dir) / f"{model_name}.gpt"
    mm_exe  = Path(modelmuse_exe) if modelmuse_exe else Path(
        r"C:\Program Files\USGS\ModelMuse5\bin\ModelMuse.exe"
    )
    if not mm_exe.exists():
        print(f"  [warn] ModelMuse no encontrado en: {mm_exe}")
        return
    if not gpt.exists():
        print(f"  [warn] Archivo .gpt no encontrado: {gpt}")
        return
    print(f"  Abriendo ModelMuse con {gpt.name} ...")
    subprocess.Popen([str(mm_exe), str(gpt)])


# ---------------------------------------------------------------------------
# Snakemake entry-point
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    cfg = snakemake.params.cfg
    cfg["output_dir"] = cfg.get("output_dir", "results/hidrogeologia").format(
        output_base=snakemake.params.get("output_base", "results")
    )
    process_modflow(cfg, no_run=False)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import yaml

    _root = Path(__file__).resolve().parent.parent.parent
    _default_cfg = _root / "config" / "config.modflow.yaml"

    p = argparse.ArgumentParser(
        description="MODFLOW-6 runner and post-processor."
    )
    p.add_argument("--config", default=str(_default_cfg),
                   help="Path to config.modflow.yaml  (default: %(default)s)")
    p.add_argument("--no-run", action="store_true",
                   help="Skip running MODFLOW; use existing binary output files.")
    p.add_argument("--layer", type=int, default=None,
                   help="Override model layer (1-based) for head contour map.")
    p.add_argument("--open-modelmuse", action="store_true",
                   help="Launch ModelMuse with the .gpt project file after processing.")
    p.add_argument("--no-interactive", action="store_true",
                   help="No abrir el visualizador 3D interactivo.")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_base = cfg.get("output_base", "results")
    cfg["output_dir"] = cfg.get("output_dir", f"{out_base}/hidrogeologia").format(
        output_base=out_base
    )

    if args.layer is not None:
        cfg["layer"] = args.layer
    if args.open_modelmuse:
        cfg["open_modelmuse"] = True
    if args.no_interactive:
        cfg["interactive"] = False

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 66)
    print("  MODFLOW-6  —  WaterKu-Risk")
    print("=" * 66 + "\n")
    print(f"  Config   : {cfg_path}")
    print(f"  Modelo   : {cfg['model_dir']}")
    print(f"  Salida   : {cfg['output_dir']}\n")

    process_modflow(cfg, no_run=args.no_run)


if __name__ == "__main__":
    _cli()
