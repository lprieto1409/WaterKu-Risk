"""Orquestador del balance hídrico de la Laguna Colombina Sur — Fase 4 de la
Propuesta Arkel. Sigue el mismo patrón `process_*(cfg, output_dir)` que el
resto de runners de WaterKu-Risk.

Consume: escorrentía mensual (aproximada por coeficiente de escorrentía sobre
la precipitación mensual de Fase 1 — ver nota más abajo), evapotranspiración
Penman-Monteith (`evapotranspiracion.py`, Fase 1/ERA5) y el budget mensual del
paquete LAK del modelo MODFLOW (Fase 3, `model_builder.process_modflow_vallereal`).

NOTA DE INGENIERÍA: el hidrograma SCS de `hidrograma_scs.py` (Fase 1) da el
caudal de DISEÑO para una tormenta puntual (T=10/25/100 años) — una magnitud
distinta a la escorrentía MENSUAL PROMEDIO que necesita este balance. Por eso
aquí se estima la escorrentía mensual con un coeficiente de escorrentía simple
(`config.balance.yaml::balance.coeficiente_escorrentia_mensual`, por defecto
0.30) sobre la precipitación mensual — es una aproximación de gabinete
razonable para un balance de cuenca pequeña, pero la elección final del
coeficiente (o de un método más riguroso, ej. SCS-CN mensual) es juicio del
hidrogeólogo, no derivable solo del código.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

_BALANCE_DIR = Path(__file__).resolve().parent
if str(_BALANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_BALANCE_DIR))

from evapotranspiracion import calcular_eto_diario_era5, agregar_eto_mensual  # noqa: E402
from balance_laguna import (  # noqa: E402
    compute_inflows, compute_outflows, monthly_annual_balance,
    sensitivity_balance, plot_balance_monthly, plot_balance_annual,
)


def load_config(config_path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _escorrentia_mensual_desde_precipitacion(precip_mensual_mm: list[float], area_m2: float,
                                              coeficiente: float = 0.30) -> list[float]:
    return [coeficiente * (p_mm / 1000.0) * area_m2 for p_mm in precip_mensual_mm]


def _cargar_eto_mensual(cfg: dict) -> list[float] | None:
    salidas = cfg.get("salidas", {}).get("evapotranspiracion", {})
    if salidas.get("fuente_clima") != "era5":
        return None
    era5_dir = Path(salidas.get("era5_dir", ""))
    archivos = list(era5_dir.glob("*.csv")) if era5_dir.exists() else []
    if not archivos:
        warnings.warn(f"[balance] Sin datos ERA5 en {era5_dir} — se omite evapotranspiración Penman-Monteith.")
        return None

    df_era5 = pd.read_csv(archivos[0])
    df_diario = calcular_eto_diario_era5(df_era5, latitud=cfg["ubicacion"]["latitud"],
                                          elevacion_m=cfg["ubicacion"]["elevacion_m"])
    df_mensual = agregar_eto_mensual(df_diario)
    return df_mensual.groupby("mes")["ETo_mm_mes"].mean().reindex(range(1, 13)).tolist()


def _cargar_budget_lak(cfg: dict) -> tuple[list[float], list[float]] | None:
    resultados_dir = Path(cfg["entradas"]["aporte_subterraneo"]["resultados_dir"])
    resumen_path = resultados_dir / "resumen_modflow_vallereal.json"
    if not resumen_path.exists():
        warnings.warn(
            f"[balance] Sin resultados de MODFLOW/LAK todavía ({resumen_path}) — "
            "corre primero `python waterku_risk.py modflow-vallereal` (Fase 3)."
        )
        return None
    warnings.warn(
        "[balance] El modelo MODFLOW de Valle Real corre hoy en régimen estacionario "
        "(un único periodo) — el budget LAK mensual requiere extender la simulación a "
        "transitorio con 12 periodos de estrés (pendiente, ver Fase D del plan)."
    )
    return None


def process_balance(config_path, output_dir) -> dict:
    cfg = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 66)
    print("  Balance hídrico — Laguna Colombina Sur / Valle Real")
    print("=" * 66 + "\n")

    area_m2 = 13858.0  # área de la laguna, ver config.modflow_vallereal.yaml::lake.area_m2

    precip_mensual_mm = cfg.get("precipitacion_mensual_mm")
    if not precip_mensual_mm:
        warnings.warn(
            "[balance] Sin serie de precipitación mensual configurada — "
            "corre primero `python waterku_risk.py hidrologia` (Fase 1) y completa "
            "config.balance.yaml::precipitacion_mensual_mm."
        )
        print(f"\n  Sin datos de entrada todavía. Resultados parciales en: {output_dir}\n")
        return {}

    coef = cfg.get("balance", {}).get("coeficiente_escorrentia_mensual", 0.30)
    escorrentia_m3 = _escorrentia_mensual_desde_precipitacion(precip_mensual_mm, area_m2, coef)

    eto_mensual_mm = _cargar_eto_mensual(cfg) or [0.0] * 12
    budget_lak = _cargar_budget_lak(cfg)
    aporte_subterraneo_m3, filtracion_m3 = budget_lak if budget_lak else ([0.0] * 12, [0.0] * 12)

    inflows = compute_inflows(escorrentia_m3, aporte_subterraneo_m3)
    outflows = compute_outflows(eto_mensual_mm, area_m2, filtracion_m3)
    balance = monthly_annual_balance(inflows, outflows)

    excel_path = output_dir / cfg.get("salida", {}).get("excel_filename", "balance_hidrico_laguna.xlsx")
    balance.to_excel(excel_path, index=False)
    plot_balance_monthly(balance, output_dir / "balance_mensual.png")
    plot_balance_annual(balance, output_dir / "balance_anual.png")

    variables = cfg.get("balance", {}).get("variables_sensibilidad", [])
    perturbaciones = cfg.get("balance", {}).get("perturbaciones", [-0.2, -0.1, 0.0, 0.1, 0.2])
    if variables:
        sens = sensitivity_balance(inflows, outflows, variables, perturbaciones)
        sens.to_excel(output_dir / "sensibilidad_balance.xlsx", index=False)

    print(f"\n  Listo. Resultados en: {output_dir}\n")
    return {"excel": str(excel_path)}


if "snakemake" in dir():
    cfg_path = snakemake.input.config  # noqa: F821
    out_dir = snakemake.output[0]  # noqa: F821
    process_balance(cfg_path, out_dir)
