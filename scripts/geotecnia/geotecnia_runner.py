"""Orquestador de geotecnia — Fase 2 (interpretada) de la Propuesta Arkel
(Laguna Colombina Sur, Valle Real).

Importa e interpreta los trabajos de campo ejecutados por ARKEL (logs de
sondeo washboring+SPT, ensayos Lefranc/Porchet, perfiles MASW-2D/ERT ya
procesados, química de suelos) e interpola los niveles freáticos observados,
produciendo el **modelo conceptual** (JSON) que alimenta
`scripts/modflow/model_builder.py` (Fase 3).

Sigue el mismo patrón `process_*(cfg, output_dir)` que
`scripts/transport/transport_runner.py` y `scripts/hidrologia/hidrologia_runner.py`.
Cada paso se omite con un aviso claro si los entregables de campo de ARKEL
todavía no han llegado (ver `data/valle_real/README.md`).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

_GEOTEC_DIR = Path(__file__).resolve().parent
if str(_GEOTEC_DIR) not in sys.path:
    sys.path.insert(0, str(_GEOTEC_DIR))

from sondeos import load_borehole_log, build_stratigraphic_profile  # noqa: E402
from infiltracion import summarize_k_by_unit  # noqa: E402
from geofisica_import import load_masw_profile, load_ert_section, correlate_geophysics_with_boreholes  # noqa: E402
from quimica import load_chemical_tests  # noqa: E402
from interpolacion_niveles import PiezometricInterpolator  # noqa: E402


def load_config(config_path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_sondeos(cfg: dict) -> list:
    sondeos_dir = Path(cfg["sondeos"]["directorio"])
    archivos = cfg["sondeos"].get("archivos", [])
    if not archivos:
        warnings.warn(
            f"[geotecnia] Sin logs de sondeo configurados en {sondeos_dir} — "
            "pendiente de las 2 perforaciones washboring+SPT de ARKEL."
        )
        return []
    boreholes = []
    for entry in archivos:
        boreholes.append(load_borehole_log(
            path=sondeos_dir / entry["archivo"], borehole_id=entry["borehole_id"],
            x=entry["x"], y=entry["y"], elevacion_boca_m=entry["elevacion_boca_m"],
            nivel_freatico_m=entry.get("nivel_freatico_m"),
        ))
    return boreholes


def interpolar_napa_freatica(cfg: dict, boreholes: list, output_dir: Path) -> dict | None:
    wells_file = Path(cfg["interpolacion_napa"]["wells_file"])
    if not wells_file.exists():
        warnings.warn(f"[geotecnia] Sin archivo de piezómetros ({wells_file}) — se omite interpolación de napa.")
        return None

    df_wells = pd.read_csv(wells_file)
    interpolator = PiezometricInterpolator.from_dataframe(
        df_wells, cell_size=cfg["interpolacion_napa"].get("cell_size", 5.0)
    )
    method = cfg["interpolacion_napa"].get("method", "idw")
    resultado = (interpolator.idw(power=cfg["interpolacion_napa"].get("idw_power", 2.0))
                 if method == "idw" else interpolator.ordinary_kriging())

    out = output_dir / "napa_freatica"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultado["water_elev"]).to_csv(out / "water_table_elevation.csv", index=False)
    return {
        "metodo": method,
        "n_pozos": len(df_wells),
        "rango_elevacion_napa_m": (float(resultado["water_elev"].min()), float(resultado["water_elev"].max())),
        "grid_csv": str(out / "water_table_elevation.csv"),
    }


def interpretar_geofisica(cfg: dict, stratigraphy: pd.DataFrame) -> dict | None:
    gcfg = cfg.get("geofisica", {})
    masw_files = gcfg.get("masw", {}).get("perfiles", [])
    ert_files = gcfg.get("ert", {}).get("secciones", [])
    if not masw_files or not ert_files:
        warnings.warn(
            "[geotecnia] Sin perfiles MASW-2D/ERT procesados todavía — pendiente de ARKEL/subcontratista de geofísica."
        )
        return None

    base = Path(gcfg["directorio"])
    masw = pd.concat([load_masw_profile(base / f) for f in masw_files], ignore_index=True)
    ert = pd.concat([load_ert_section(base / f) for f in ert_files], ignore_index=True)
    return correlate_geophysics_with_boreholes(masw, ert, stratigraphy)


def construir_modelo_conceptual(cfg: dict, output_dir: Path) -> dict:
    """Construye el modelo conceptual hidroestratigráfico (capas, K por
    unidad, nivel freático, bordes) a partir de los pasos anteriores y lo
    serializa a `<output_dir>/modelo_conceptual.json` — input directo de
    `scripts/modflow/model_builder.py` (Fase 3)."""
    boreholes = cargar_sondeos(cfg)
    stratigraphy = build_stratigraphic_profile(boreholes) if boreholes else pd.DataFrame()

    k_por_unidad = pd.DataFrame()
    infiltracion_dir = Path(cfg["infiltracion"]["directorio"])
    ensayos_paths = (cfg["infiltracion"].get("ensayos_lefranc", [])
                      + cfg["infiltracion"].get("ensayos_porchet", [])
                      + cfg["infiltracion"].get("ensayos_permeametro", []))
    if ensayos_paths and not stratigraphy.empty:
        tests = pd.concat([pd.read_csv(infiltracion_dir / f) for f in ensayos_paths], ignore_index=True)
        k_por_unidad = summarize_k_by_unit(tests, stratigraphy)
    elif not ensayos_paths:
        warnings.warn(
            "[geotecnia] Sin resultados de ensayos Lefranc/Porchet/permeámetro todavía "
            "— pendiente de ARKEL. El modelo conceptual quedará sin K observada."
        )

    geofisica_info = interpretar_geofisica(cfg, stratigraphy)
    napa_info = interpolar_napa_freatica(cfg, boreholes, output_dir)

    quimica_dir = Path(cfg["quimica"]["directorio"])
    quimica_ensayos = cfg["quimica"].get("ensayos", [])
    quimica_df = (pd.concat([load_chemical_tests(quimica_dir / f) for f in quimica_ensayos], ignore_index=True)
                  if quimica_ensayos else pd.DataFrame())

    modelo_conceptual = {
        "unidades_hidroestratigraficas": sorted(stratigraphy["unidad_hidroestratigrafica"].unique().tolist())
            if not stratigraphy.empty else [],
        "estratigrafia": stratigraphy.to_dict(orient="records") if not stratigraphy.empty else [],
        "conductividad_hidraulica_por_unidad": k_por_unidad.to_dict(orient="records") if not k_por_unidad.empty else [],
        "napa_freatica": napa_info,
        "geofisica": geofisica_info,
        "agresividad_quimica": quimica_df.to_dict(orient="records") if not quimica_df.empty else [],
        "condiciones_de_borde": None,  # a definir por el hidrogeólogo a partir de la geometría de la cuenca/laguna
    }
    return modelo_conceptual


def process_geotecnia(config_path, output_dir) -> None:
    cfg = load_config(config_path)
    out_base = cfg.get("output_base", "results")
    output_dir = Path(str(output_dir).format(output_base=out_base))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 66)
    print("  Geotecnia / modelo conceptual — Laguna Colombina Sur / Valle Real")
    print("=" * 66 + "\n")

    modelo_conceptual = construir_modelo_conceptual(cfg, output_dir)
    out_json = output_dir / "modelo_conceptual.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(modelo_conceptual, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  Modelo conceptual: {out_json}\n")
    if not modelo_conceptual["unidades_hidroestratigraficas"]:
        print("  AVISO: modelo conceptual incompleto — faltan entregables de campo de ARKEL.\n")


if "snakemake" in dir():
    cfg_path = snakemake.input.config  # noqa: F821
    out_dir = Path(snakemake.output[0]).parent  # noqa: F821
    process_geotecnia(cfg_path, out_dir)
