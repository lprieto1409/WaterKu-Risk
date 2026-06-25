#!/usr/bin/env python3
"""
WaterKu-Risk CLI — Módulos de hidrogeología compactos.

Subcomandos:
    modflow            Simulación MODFLOW-6 + gráficas 2D/3D (modelo Angascancha)
    api                MODFLOW-6 XMI API + análisis de sensibilidad
    transport          Transporte de contaminantes ADE (MT3DMS-style)
    benchmark          Benchmarks numéricos del Manual MT3DMS
    hidrologia         Hidrología superficial — Laguna Colombina Sur / Valle Real (Fase 0+1, Arkel)
    geotecnia          Modelo conceptual hidrogeológico — Valle Real (Fase 2, Arkel)
    modflow-vallereal  Construcción + ejecución + calibración MODFLOW-6 — Valle Real (Fase 3, Arkel)
    balance            Balance hídrico de la laguna — Valle Real (Fase 4, Arkel)

Uso rápido:
    python waterku_risk.py transport --demo
    python waterku_risk.py benchmark --1d
    python waterku_risk.py modflow --no-run --no-interactive
    python waterku_risk.py api --sensitivity
    python waterku_risk.py hidrologia
    python waterku_risk.py geotecnia
    python waterku_risk.py modflow-vallereal --modpath
    python waterku_risk.py balance
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _default_cfg(name: str) -> Path:
    return ROOT / "config" / name


# ---------------------------------------------------------------------------
# Subcomando: modflow
# ---------------------------------------------------------------------------

def cmd_modflow(args) -> None:
    import yaml
    from scripts.modflow.modflow_runner import process_modflow

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
    if args.layer:
        cfg["layer"] = args.layer
    if args.no_interactive:
        cfg["interactive"] = False

    print("\n" + "=" * 66)
    print("  MODFLOW-6  —  WaterKu-Risk")
    print("=" * 66 + "\n")
    print(f"  Config : {cfg_path}")
    print(f"  Salida : {cfg['output_dir']}\n")

    process_modflow(cfg, no_run=args.no_run)


# ---------------------------------------------------------------------------
# Subcomando: api
# ---------------------------------------------------------------------------

def cmd_api(args) -> None:
    import yaml
    from scripts.modflow.modflow_api import run_via_api, run_sensitivity, plot_sensitivity

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

    print("\n" + "=" * 66)
    print("  MODFLOW-6 XMI API  —  WaterKu-Risk")
    print("=" * 66 + "\n")

    if args.sensitivity:
        sens_cfg   = cfg.get("sensitivity", {})
        # `parametros` (ASTM D5611, multi-parámetro) tiene prioridad; si no
        # está, se usa el legacy `rch_multipliers` (solo recarga).
        param_spec = sens_cfg.get("parametros") or {"RECHARGE": sens_cfg.get("rch_multipliers", [0.5, 1.0, 1.5, 2.0])}
        mf6_comp   = sens_cfg.get("mf6_component", "MODFLOW")
        results = run_sensitivity(dll, model_dir, model_name, nlay, nrow, ncol,
                                  param_spec, mf6_comp)
        plot_sensitivity(results, output_dir / "sensitivity_analysis.png")
    else:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        head = run_via_api(dll, model_dir, model_name, nlay, nrow, ncol)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        layers_to_plot = [0, nlay // 2, nlay - 1]
        for ax, lay in zip(axes, layers_to_plot):
            h_l = np.ma.masked_invalid(head[lay])
            im  = ax.imshow(h_l, cmap="Blues_r", origin="upper", aspect="auto")
            plt.colorbar(im, ax=ax, label="Head (m asl)", shrink=0.85)
            ax.set_title(f"Capa {lay+1}", fontsize=12)
        plt.suptitle("Cabezas Hidráulicas via XMI API", fontsize=13)
        plt.tight_layout()
        out = output_dir / "heads_xmi_api.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {out.name}")

    print(f"\n  Listo. Resultados en: {output_dir}\n")


# ---------------------------------------------------------------------------
# Subcomando: transport
# ---------------------------------------------------------------------------

def cmd_transport(args) -> None:
    import yaml
    import geopandas as gpd
    from scripts.transport.transport_runner import (
        load_config, build_synthetic_velocity, load_modflow_velocity,
        run_transport, map_emitters_to_grid, map_settlements_to_grid,
        process_transport,
    )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    if args.demo:
        cfg = load_config(cfg_path)
        cfg["paths"]["modflow_bhd"] = ""
        cfg["paths"]["modflow_cbc"] = ""

    output_dir = ROOT / "results" / "hidrogeologia" / "mt3dms"

    print("\n" + "=" * 66)
    print("  Transporte de Contaminantes (ADE)  —  WaterKu-Risk")
    print("=" * 66 + "\n")
    if args.demo:
        print("  Modo: DEMO (flujo sintético uniforme)\n")
    else:
        print(f"  Config: {cfg_path}\n")

    process_transport(cfg_path, output_dir)


# ---------------------------------------------------------------------------
# Subcomando: hidrologia  (Valle Real — Fase 0 + Fase 1 de la Propuesta Arkel)
# ---------------------------------------------------------------------------

def cmd_hidrologia(args) -> None:
    from scripts.hidrologia.hidrologia_runner import process_hidrologia

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    process_hidrologia(cfg_path, "{output_base}/hidrologia_vallereal")


# ---------------------------------------------------------------------------
# Subcomando: geotecnia  (Valle Real — Fase 2 interpretada de la Propuesta Arkel)
# ---------------------------------------------------------------------------

def cmd_geotecnia(args) -> None:
    from scripts.geotecnia.geotecnia_runner import process_geotecnia

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    process_geotecnia(cfg_path, "{output_base}/geotecnia_vallereal")


# ---------------------------------------------------------------------------
# Subcomando: modflow-vallereal  (Valle Real — Fase 3 de la Propuesta Arkel)
# ---------------------------------------------------------------------------

def cmd_modflow_vallereal(args) -> None:
    from scripts.modflow.model_builder import process_modflow_vallereal

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    resultado = process_modflow_vallereal(cfg_path, "{output_base}/modflow_vallereal")

    if args.modpath and resultado:
        import yaml
        from scripts.modflow.modpath_runner import (
            build_modpath_model, run_modpath, read_endpoints, lake_interaction_summary,
        )

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("\n[MODPATH] Construyendo y ejecutando particle tracking...")
        mp = build_modpath_model(resultado["gwf"], cfg["model_dir"], cfg["model_name"])
        run_modpath(mp)
        endpoints = read_endpoints(cfg["model_dir"], cfg["model_name"])
        resumen = lake_interaction_summary(endpoints, cfg.get("lake", {}).get("connectiondata", []))
        print(f"  Interacción laguna-acuífero: {resumen}")


# ---------------------------------------------------------------------------
# Subcomando: balance  (Valle Real — Fase 4 de la Propuesta Arkel)
# ---------------------------------------------------------------------------

def cmd_balance(args) -> None:
    from scripts.balance.balance_runner import process_balance

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    out_dir = Path(args.config).resolve().parent.parent / "results" / "balance_vallereal"
    process_balance(cfg_path, out_dir)


# ---------------------------------------------------------------------------
# Subcomando: benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args) -> None:
    from scripts.transport.benchmarks import run_benchmark_1d, run_benchmark_3d

    output_dir = ROOT / "results" / "hidrogeologia" / "mt3dms"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 66)
    print("  Benchmarks MT3DMS  —  WaterKu-Risk")
    print("=" * 66 + "\n")

    if args.b1d or (not args.b1d and not args.b3d):
        print("[Benchmark 1D] MT3DMS Cap. 7 Figura 31 ...")
        run_benchmark_1d(output_dir / "benchmark_1d.png")

    if args.b3d or (not args.b1d and not args.b3d):
        print("[Benchmark 3D] MT3DMS Cap. 7 Figura 39 ...")
        run_benchmark_3d(output_dir / "benchmark_3d.png")

    print(f"\n  Resultados en: {output_dir}\n")


# ---------------------------------------------------------------------------
# Argparse principal
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(
        prog="waterku_risk",
        description="WaterKu-Risk — Módulos de hidrogeología compactos.",
    )
    sub = p.add_subparsers(dest="command", metavar="SUBCOMANDO")
    sub.required = True

    # --- modflow ---
    pm = sub.add_parser("modflow", help="Simulación MODFLOW-6 + gráficas 2D/3D")
    pm.add_argument("--config", default=str(_default_cfg("config.modflow.yaml")))
    pm.add_argument("--no-run", action="store_true",
                    help="Usar archivos binarios existentes (sin re-ejecutar)")
    pm.add_argument("--layer", type=int, default=None,
                    help="Capa 1-based para el mapa de cabezas")
    pm.add_argument("--no-interactive", action="store_true",
                    help="No abrir el visualizador 3D interactivo")
    pm.set_defaults(func=cmd_modflow)

    # --- api ---
    pa = sub.add_parser("api", help="MODFLOW-6 XMI API + análisis de sensibilidad")
    pa.add_argument("--config", default=str(_default_cfg("config.modflow.yaml")))
    pa.add_argument("--sensitivity", action="store_true",
                    help="Ejecutar análisis de sensibilidad a la recarga")
    pa.set_defaults(func=cmd_api)

    # --- transport ---
    pt = sub.add_parser("transport", help="Transporte de contaminantes ADE")
    pt.add_argument("--config", default=str(_default_cfg("config.transport.yaml")))
    pt.add_argument("--demo", action="store_true",
                    help="Modo demo: flujo sintético uniforme (sin MODFLOW)")
    pt.set_defaults(func=cmd_transport)

    # --- hidrologia (Valle Real) ---
    ph = sub.add_parser("hidrologia", help="Hidrología superficial — Laguna Colombina Sur / Valle Real")
    ph.add_argument("--config", default=str(_default_cfg("config.hidrologia.yaml")))
    ph.set_defaults(func=cmd_hidrologia)

    # --- geotecnia (Valle Real) ---
    pg = sub.add_parser("geotecnia", help="Modelo conceptual hidrogeológico — Laguna Colombina Sur / Valle Real")
    pg.add_argument("--config", default=str(_default_cfg("config.geotecnia.yaml")))
    pg.set_defaults(func=cmd_geotecnia)

    # --- modflow-vallereal (Valle Real) ---
    pv = sub.add_parser("modflow-vallereal", help="Construcción + ejecución + calibración MODFLOW-6 — Valle Real")
    pv.add_argument("--config", default=str(_default_cfg("config.modflow_vallereal.yaml")))
    pv.add_argument("--modpath", action="store_true",
                    help="Encadenar MODPATH7 (particle tracking) tras correr el modelo de flujo")
    pv.set_defaults(func=cmd_modflow_vallereal)

    # --- balance (Valle Real) ---
    pbal = sub.add_parser("balance", help="Balance hídrico de la laguna — Valle Real")
    pbal.add_argument("--config", default=str(_default_cfg("config.balance.yaml")))
    pbal.set_defaults(func=cmd_balance)

    # --- benchmark ---
    pb = sub.add_parser("benchmark", help="Benchmarks numéricos MT3DMS")
    pb.add_argument("--1d", dest="b1d", action="store_true", help="Benchmark 1D (Figura 31)")
    pb.add_argument("--3d", dest="b3d", action="store_true", help="Benchmark 3D (Figura 39)")
    pb.set_defaults(func=cmd_benchmark)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
