#!/usr/bin/env python3
"""
MODFLOW-6 XMI API — control en tiempo real desde Python.

Usa libmf6.dll (Extended Model Interface) para inicializar y avanzar el
modelo paso a paso, leer cabezas y hacer análisis de sensibilidad.

Todas las rutas y constantes de grilla vienen del config; no hay paths
hardcodeados en este módulo.

Requiere: modflowapi  (pip install modflowapi  o  pixi run pip install modflowapi)

CLI usage:
    pixi run python scripts/modflow/modflow_api.py
    pixi run python scripts/modflow/modflow_api.py --sensitivity
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# mfsim.nam patching (XMI requires bare filenames in the working directory)
# ---------------------------------------------------------------------------

def patch_mfsim(model_dir: str) -> Path:
    """Overwrite mfsim.nam with bare filenames; return backup path."""
    nam_path = Path(model_dir) / "mfsim.nam"
    backup   = Path(model_dir) / "mfsim.nam.orig"
    content  = nam_path.read_text()
    patched  = re.sub(r"'[A-Za-z]:[^']*[\\/]([^\\/]+)'", r"'\1'", content)
    backup.write_text(content)
    nam_path.write_text(patched)
    return backup


def restore_mfsim(backup: Path) -> None:
    """Restore original mfsim.nam from backup and remove backup file."""
    nam_path = backup.parent / "mfsim.nam"
    if backup.exists():
        nam_path.write_text(backup.read_text())
        backup.unlink()


# ---------------------------------------------------------------------------
# modflowapi import helper
# ---------------------------------------------------------------------------

def load_modflowapi():
    try:
        from modflowapi import ModflowApi
        return ModflowApi
    except ImportError:
        print(
            "\n[ERROR] modflowapi no está instalado.\n"
            "Instálalo con:\n"
            "    pip install modflowapi\n"
            "  o en el entorno pixi:\n"
            "    pixi run pip install modflowapi\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# XMI runners
# ---------------------------------------------------------------------------

def run_via_api(dll: str, model_dir: str, model_name: str,
                nlay: int, nrow: int, ncol: int) -> np.ndarray:
    """Run model via XMI; return head array of shape (nlay, nrow, ncol)."""
    ModflowApi = load_modflowapi()

    backup = patch_mfsim(model_dir)
    try:
        mf6 = ModflowApi(dll, working_directory=model_dir)
        mf6.initialize()

        t_end = mf6.get_end_time()
        t_cur = mf6.get_current_time()
        print(f"  Tiempo de simulación: {t_cur} -> {t_end} s  (estado estacionario)")

        while t_cur < t_end:
            mf6.update()
            t_cur = mf6.get_current_time()

        mf6.finalize()

        import flopy.utils
        bhd  = Path(model_dir) / f"{model_name}.bhd"
        hf   = flopy.utils.HeadFile(str(bhd))
        head = hf.get_data(kstpkper=(0, 0)).astype(float)
        head[head >= 1e29] = np.nan
        return head

    finally:
        restore_mfsim(backup)


def run_sensitivity(dll: str, model_dir: str, model_name: str,
                    nlay: int, nrow: int, ncol: int,
                    rch_multipliers: list,
                    mf6_component: str = "MODFLOW") -> dict:
    """Run model with each recharge multiplier; return {mult: head_array}."""
    ModflowApi = load_modflowapi()
    results    = {}

    for mult in rch_multipliers:
        print(f"  Corriendo con recarga x{mult:.1f} ...")
        backup = patch_mfsim(model_dir)
        try:
            mf6 = ModflowApi(dll, working_directory=model_dir)
            mf6.initialize()

            try:
                rch_tag = mf6.get_var_address("RECHARGE", mf6_component, "RCH_0")
                rch     = mf6.get_value_ptr(rch_tag)
                rch[:] *= mult
            except Exception as e:
                print(f"    [warn] No se pudo modificar RECHARGE: {e}")

            t_end = mf6.get_end_time()
            t_cur = mf6.get_current_time()
            while t_cur < t_end:
                mf6.update()
                t_cur = mf6.get_current_time()

            mf6.finalize()

            import flopy.utils
            bhd  = Path(model_dir) / f"{model_name}.bhd"
            hf   = flopy.utils.HeadFile(str(bhd))
            head = hf.get_data(kstpkper=(0, 0)).astype(float)
            head[head >= 1e29] = np.nan
            results[mult] = head

        finally:
            restore_mfsim(backup)

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sensitivity(results: dict, out_path: Path) -> None:
    multipliers = sorted(results.keys())
    mean_heads  = [np.nanmean(results[m][0]) for m in multipliers]

    fig  = plt.figure(figsize=(14, 5))
    ax0  = fig.add_subplot(1, 3, 1)
    ax0.plot(multipliers, mean_heads, "o-", color="steelblue", lw=2, ms=8)
    ax0.set_xlabel("Multiplicador de Recarga", fontsize=11)
    ax0.set_ylabel("Carga Media Capa 1 (m asl)", fontsize=11)
    ax0.set_title("Sensibilidad a la Recarga", fontsize=12)
    ax0.grid(True, alpha=0.3)
    for m, h in zip(multipliers, mean_heads):
        ax0.annotate(f"{h:.0f} m", (m, h),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)

    for idx, mult in enumerate([multipliers[0], multipliers[-1]]):
        ax   = fig.add_subplot(1, 3, idx + 2)
        h_l1 = np.ma.masked_invalid(results[mult][0])
        im   = ax.imshow(h_l1, cmap="Blues_r", origin="upper", aspect="auto")
        plt.colorbar(im, ax=ax, label="Head (m asl)", shrink=0.8)
        ax.set_title(f"Recarga x{mult:.1f}\nMedia={np.nanmean(results[mult][0]):.0f} m",
                     fontsize=11)
        ax.set_xlabel("Columna")
        ax.set_ylabel("Fila")

    plt.suptitle("Análisis de Sensibilidad — Recarga vs Carga Hidráulica (Capa 1)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out_path.name}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import yaml

    _root        = Path(__file__).resolve().parent.parent.parent
    _default_cfg = _root / "config" / "config.modflow.yaml"

    p = argparse.ArgumentParser(description="MODFLOW-6 XMI API.")
    p.add_argument("--config", default=str(_default_cfg),
                   help="Path to config.modflow.yaml  (default: %(default)s)")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run sensitivity analysis with rch_multipliers from config.")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_base   = cfg.get("output_base", "results")
    output_dir = Path(
        cfg.get("output_dir", f"{out_base}/hidrogeologia").format(output_base=out_base)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dll        = cfg["libmf6_dll"]
    model_dir  = cfg["model_dir"]
    model_name = cfg["model_name"]
    grid       = cfg["grid"]
    nlay, nrow, ncol = int(grid["nlay"]), int(grid["nrow"]), int(grid["ncol"])
    sensitivity_cfg  = cfg.get("sensitivity", {})

    dll_path = Path(dll)
    if not dll_path.exists():
        print(f"[ERROR] libmf6.dll no encontrada en: {dll_path}")
        sys.exit(1)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 66)
    print("  MODFLOW-6 XMI API  —  WaterKu-Risk")
    print("=" * 66 + "\n")
    print(f"  DLL : {dll}")
    print(f"  Dir : {model_dir}\n")

    if args.sensitivity:
        multipliers = sensitivity_cfg.get("rch_multipliers", [0.5, 1.0, 1.5, 2.0])
        mf6_comp    = sensitivity_cfg.get("mf6_component", "MODFLOW")
        print(f"[1/2] Análisis de sensibilidad (recarga {multipliers})...\n")
        results = run_sensitivity(
            dll, model_dir, model_name, nlay, nrow, ncol,
            multipliers, mf6_comp,
        )

        print("\n  Resumen:")
        print(f"  {'Mult':>6}  {'Media L1 (m)':>14}  {'Min L1 (m)':>12}  {'Max L1 (m)':>12}")
        print("  " + "-" * 50)
        for m in sorted(results):
            h = results[m][0]
            print(f"  {m:>6.1f}  {np.nanmean(h):>14.1f}  {np.nanmin(h):>12.1f}  {np.nanmax(h):>12.1f}")

        print("\n[2/2] Graficando sensibilidad...")
        plot_sensitivity(results, output_dir / "sensitivity_recharge.png")

    else:
        print("[1/2] Ejecutando modelo via XMI API...")
        head = run_via_api(dll, model_dir, model_name, nlay, nrow, ncol)
        print(f"\n  Cabezas leídas: {head.shape}  "
              f"(rango: {np.nanmin(head):.0f} – {np.nanmax(head):.0f} m)\n")

        print(f"  {'Capa':>5}  {'Media (m)':>10}  {'Std (m)':>9}")
        print("  " + "-" * 30)
        for lay in range(nlay):
            data = head[lay][np.isfinite(head[lay])]
            print(f"  {lay+1:>5}  {data.mean():>10.1f}  {data.std():>9.1f}")

        print("\n[2/2] Guardando mapa de cabezas (API)...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        layers_to_plot = [0, nlay // 2, nlay - 1]
        for ax, lay in zip(axes, layers_to_plot):
            h_l = np.ma.masked_invalid(head[lay])
            im  = ax.imshow(h_l, cmap="Blues_r", origin="upper", aspect="auto")
            plt.colorbar(im, ax=ax, label="Head (m asl)", shrink=0.85)
            ax.set_title(f"Capa {lay+1}", fontsize=12)
            ax.set_xlabel("Columna")
            ax.set_ylabel("Fila")
        plt.suptitle(f"Cabezas Hidráulicas via XMI API — Capas {', '.join(str(l+1) for l in layers_to_plot)}",
                     fontsize=13)
        plt.tight_layout()
        out = output_dir / "heads_xmi_api.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {out.name}")

    print(f"\n  Listo. Resultados en: {output_dir}\n")


if __name__ == "__main__":
    _cli()
