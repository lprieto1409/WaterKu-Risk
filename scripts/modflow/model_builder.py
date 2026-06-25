"""Construcción de un modelo MODFLOW-6 nuevo (Laguna Colombina Sur, Valle
Real) vía `flopy.mf6` a partir del modelo conceptual de `geotecnia_runner`
(Fase 3 de la Propuesta Arkel).

A diferencia del modelo "Angascancha" (`config/config.modflow.yaml`), que ya
viene armado externamente (ej. en ModelMuse) y que `modflow_runner.py` solo
ejecuta/post-procesa, este módulo SÍ construye el modelo desde cero —
confirmado que no existe un "model builder" equivalente en ninguna de las ~50
branches del repo WaterKu original.

Tras `write_and_validate()`, el modelo queda en `model_dir` listo para
ejecutarse con las funciones YA EXISTENTES `modflow_runner.run_modflow()` /
`read_heads()` / `read_budget()`, sin modificar ese archivo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_grid_from_conceptual_model(conceptual_model: dict, cfg_grid: dict,
                                      lake_polygon=None) -> dict:
    """Deriva la geometría de la grilla (DIS) a partir del modelo conceptual
    (capas hidroestratigráficas de `geotecnia_runner`) y de los parámetros de
    celda de `config.modflow_vallereal.yaml::grid`.

    Si el modelo conceptual no trae estratigrafía todavía (campo de ARKEL
    pendiente), usa una malla mínima de 1 capa como placeholder para poder
    ensayar el flujo de construcción del modelo sin bloquear el desarrollo.

    Returns:
        dict con `nlay`, `nrow`, `ncol`, `delr`, `delc`, `top` (2D) y `botm`
        (3D, una capa por unidad hidroestratigráfica).
    """
    delr = cfg_grid.get("delr", 10.0)
    delc = cfg_grid.get("delc", 10.0)
    nrow = cfg_grid.get("nrow")
    ncol = cfg_grid.get("ncol")
    if nrow is None or ncol is None:
        raise ValueError(
            "config.modflow_vallereal.yaml::grid.nrow/ncol no están definidos — "
            "se derivan del polígono de la laguna + área de aporte una vez se tenga "
            "la topografía de detalle (pendiente de ARKEL)."
        )

    unidades = conceptual_model.get("unidades_hidroestratigraficas") or ["unidad_unica"]
    nlay = max(len(unidades), cfg_grid.get("nlay", 1))

    estratigrafia = conceptual_model.get("estratigrafia") or []
    if estratigrafia:
        top_value = max(e["top_m"] for e in estratigrafia)
        base_min = min(e["base_m"] for e in estratigrafia)
    else:
        top_value, base_min = 0.0, -50.0 * nlay

    top = np.full((nrow, ncol), top_value, dtype=float)
    espesor_capa = (top_value - base_min) / nlay
    botm = np.stack([top_value - espesor_capa * (k + 1) for k in range(nlay)])
    botm = np.broadcast_to(botm[:, None, None], (nlay, nrow, ncol)).copy()

    return {"nlay": nlay, "nrow": nrow, "ncol": ncol, "delr": delr, "delc": delc,
            "top": top, "botm": botm, "unidades": unidades}


def _k_array_from_conceptual(conceptual_model: dict, grid: dict, k_default: float = 1e-5) -> np.ndarray:
    """Arreglo de conductividad hidráulica K11 por capa, a partir de
    `conceptual_model["conductividad_hidraulica_por_unidad"]`
    (`scripts/geotecnia/infiltracion.summarize_k_by_unit`); usa `k_default`
    [m/s] para las unidades sin ensayos todavía."""
    k_por_unidad = {row["unidad_hidroestratigrafica"]: row["k_m_s_geomean"]
                     for row in conceptual_model.get("conductividad_hidraulica_por_unidad", [])}
    nlay, nrow, ncol = grid["nlay"], grid["nrow"], grid["ncol"]
    k = np.full((nlay, nrow, ncol), k_default, dtype=float)
    for i, unidad in enumerate(grid["unidades"][:nlay]):
        k[i, :, :] = k_por_unidad.get(unidad, k_default)
    return k


def build_mf6_simulation(cfg: dict, conceptual_model: dict):
    """Ensambla la simulación MODFLOW-6 (TDIS, IMS, GWF: DIS/NPF/IC/RCH/CHD)
    para Valle Real. Devuelve el objeto `flopy.mf6.MFSimulation` (todavía sin
    escribir a disco — usar `write_and_validate`)."""
    import flopy

    model_dir = Path(cfg["model_dir"])
    model_name = cfg["model_name"]
    grid = build_grid_from_conceptual_model(conceptual_model, cfg["grid"])

    sim = flopy.mf6.MFSimulation(sim_name=model_name, sim_ws=str(model_dir),
                                  exe_name=cfg.get("mf6_exe"))
    flopy.mf6.ModflowTdis(sim, time_units="seconds", nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="MODERATE", outer_dvclose=1e-4, inner_dvclose=1e-5)

    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=grid["nlay"], nrow=grid["nrow"], ncol=grid["ncol"],
        delr=grid["delr"], delc=grid["delc"], top=grid["top"], botm=grid["botm"],
        xorigin=cfg["grid"].get("xorigin") or 0.0, yorigin=cfg["grid"].get("yorigin") or 0.0,
        angrot=cfg["grid"].get("angrot_deg", 0.0),
    )

    napa = conceptual_model.get("napa_freatica") or {}
    strt_value = napa.get("rango_elevacion_napa_m", [grid["top"].mean()])[0]
    flopy.mf6.ModflowGwfic(gwf, strt=strt_value)

    k = _k_array_from_conceptual(conceptual_model, grid)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k, save_flows=True)

    recarga_m_s = cfg.get("recharge_m_s", 1e-9)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recarga_m_s)

    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{model_name}.cbc",
        head_filerecord=f"{model_name}.bhd",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    if cfg.get("lake", {}).get("enabled"):
        add_lake_package(sim, gwf, cfg["lake"], grid)

    return sim, gwf


def add_lake_package(sim, gwf, lake_cfg: dict, grid: dict):
    """Añade el paquete LAK6 (laguna Colombina Sur) — confirmado ausente en
    todas las branches del repo WaterKu original. La laguna se resuelve como
    objeto hidráulico propio (tabla stage-area-volumen + conexiones LAKE-GWF
    por celda), y MODFLOW calcula nativamente el intercambio laguna-acuífero
    (GWF, lluvia, evaporación, escorrentía) en el balance de masa del modelo —
    decisión de diseño del plan (ver sección de decisiones).

    `lake_cfg["stage_area_volume_table"]` y las celdas de conexión dependen
    de la topografía de detalle de la laguna (pendiente de ARKEL); con
    `stage_area_volume_table=None` se usa una tabla mínima placeholder a partir
    de `lake_cfg["area_m2"]` solo para poder ensayar la construcción del
    modelo.
    """
    import flopy

    area_m2 = lake_cfg.get("area_m2", 13858)
    tabla = lake_cfg.get("stage_area_volume_table")
    if tabla is None:
        # Placeholder: relación lineal área-profundidad simplificada para una
        # laguna somera, SOLO para poder escribir/validar el modelo antes de
        # tener la topografía real.
        stage_top = float(grid["top"].mean())
        tabla = [
            (stage_top - 3.0, 0.2 * area_m2, 0.2 * area_m2 * 1.0),
            (stage_top - 1.5, 0.6 * area_m2, 0.6 * area_m2 * 2.5),
            (stage_top, area_m2, area_m2 * 4.0),
        ]

    lake_no = 0
    packagedata = [(lake_no, tabla[-1][0], 1, "lake_colombina_sur")]  # (lakeno, strt, nlakeconn, boundname)
    connectiondata = lake_cfg.get("connectiondata") or [
        (lake_no, 0, (0, grid["nrow"] // 2, grid["ncol"] // 2), "HORIZONTAL", lake_cfg.get("bed_leakance", 1e-6),
         0.0, 0.0, grid["delr"], grid["delc"])
    ]

    flopy.mf6.ModflowGwflak(
        gwf, nlakes=1, noutlets=0,
        packagedata=packagedata,
        connectiondata=connectiondata,
        tables=[(lake_no, tabla)] if tabla else None,
        budget_filerecord=f"{gwf.name}.lak.cbc",
    )


def write_and_validate(sim) -> bool:
    """Escribe la simulación a disco y corre `sim.check()` — paso obligatorio
    antes de pasar a calibración/transitorio (ver sección de verificación del
    plan)."""
    sim.write_simulation()
    checks = sim.check()
    ok = all(c.summary_array.size == 0 for c in checks) if checks else True
    return ok


def process_modflow_vallereal(config_path, output_dir) -> dict:
    """Orquestador Fase 3: construye el modelo MODFLOW-6 de Valle Real desde
    el modelo conceptual de `geotecnia_runner`, lo ejecuta (reutilizando
    `modflow_runner.run_modflow`/`read_heads`/`read_budget`) y calcula los
    estadísticos de calibración (`calibration.py`). Sigue el mismo patrón
    `process_*(cfg, output_dir)` que el resto de runners de WaterKu-Risk.
    """
    import json
    import sys

    import pandas as pd
    import yaml

    _MF_DIR = Path(__file__).resolve().parent
    if str(_MF_DIR) not in sys.path:
        sys.path.insert(0, str(_MF_DIR))
    import modflow_runner  # noqa: E402
    import calibration  # noqa: E402

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_base = cfg.get("output_base", "results")
    output_dir = Path(str(output_dir).format(output_base=out_base))
    output_dir.mkdir(parents=True, exist_ok=True)

    conceptual_path = Path(cfg["conceptual_model_json"])
    if not conceptual_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo conceptual ({conceptual_path}) — corre primero "
            "`python waterku_risk.py geotecnia` (Fase 2) con los entregables de ARKEL."
        )
    with open(conceptual_path, encoding="utf-8") as f:
        conceptual_model = json.load(f)

    sim, gwf = build_mf6_simulation(cfg, conceptual_model)
    ok = write_and_validate(sim)
    if not ok:
        print("[model_builder] AVISO: sim.check() reportó observaciones — revisar antes de calibrar.")

    model_dir, model_name = cfg["model_dir"], cfg["model_name"]
    modflow_runner.run_modflow(cfg["mf6_exe"], model_dir)
    head = modflow_runner.read_heads(model_dir, model_name)
    budget = modflow_runner.read_budget(model_dir, model_name)

    resultado = {"head_shape": head.shape, "balance_de_masa": calibration.mass_balance_closure(budget)}

    piezo_csv = cfg.get("piezo_csv")
    if piezo_csv and Path(piezo_csv).exists():
        piezo_df = modflow_runner.load_piezometers(piezo_csv, cfg["grid"])
        piezo_df = modflow_runner.extract_simulated_heads(head, piezo_df)
        stats = calibration.compute_residual_stats(
            piezo_df["piezometricLevel"].values, piezo_df["sim_head"].values,
            regimen=cfg.get("calibracion", {}).get("regimen", ["estacionario"])[0],
        )
        resultado["calibracion"] = calibration.calibration_report(
            stats_steady=stats, mass_balance=resultado["balance_de_masa"]
        )
        pd.DataFrame([stats.__dict__]).to_excel(output_dir / "calibracion_resumen.xlsx", index=False)
    else:
        print("[model_builder] Sin piezómetros observados todavía — se omite calibración cuantitativa.")

    with open(output_dir / "resumen_modflow_vallereal.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2, default=str)

    # `gwf` se devuelve para encadenar MODPATH (scripts/modflow/modpath_runner.py)
    # en la misma corrida sin tener que recargar la simulación; no se incluye
    # en el JSON anterior (no es serializable).
    resultado["gwf"] = gwf
    return resultado


if "snakemake" in dir():
    cfg_path = snakemake.input.config  # noqa: F821
    out_dir = Path(snakemake.output[0]).parent  # noqa: F821
    process_modflow_vallereal(cfg_path, out_dir)
