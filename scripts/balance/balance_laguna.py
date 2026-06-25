"""Balance hídrico mensual/anual de la Laguna Colombina Sur — Fase 4 de la
Propuesta Arkel: entradas (escorrentía superficial de
`scripts/hidrologia/hidrograma_scs.py` + aporte subterráneo del paquete LAK
de `scripts/modflow/model_builder.py`) frente a salidas (evaporación
Penman-Monteith de `evapotranspiracion.py` + infiltración/filtraciones del
paquete LAK), con análisis de sensibilidad de las variables críticas.

A diferencia de `balance_hidrico.py` (Thornthwaite-Mather, portado desde la
branch `claude/water-balance-module` para balance de humedad de SUELO en una
cuenca genérica), este módulo resuelve el balance de un CUERPO DE AGUA (la
laguna) cuyo almacenamiento ya lo resuelve MODFLOW vía el paquete LAK — no
existe un módulo equivalente en ninguna branch del repo WaterKu original,
construido nuevo para WaterKu-Risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BalanceLagunaMensual:
    mes: int
    escorrentia_m3: float
    aporte_subterraneo_m3: float
    evaporacion_m3: float
    filtracion_m3: float
    entradas_m3: float
    salidas_m3: float
    delta_almacenamiento_m3: float


def compute_inflows(escorrentia_mensual_m3: list[float], gwf_to_lake_mensual_m3: list[float]) -> pd.DataFrame:
    """Entradas mensuales: escorrentía superficial (Fase 1, hidrograma SCS
    integrado a volumen mensual) + aporte subterráneo (budget GWF->LAK del
    paquete LAK, Fase 3)."""
    if len(escorrentia_mensual_m3) != len(gwf_to_lake_mensual_m3):
        raise ValueError("Escorrentía y aporte subterráneo deben tener el mismo número de meses.")
    return pd.DataFrame({
        "mes": range(1, len(escorrentia_mensual_m3) + 1),
        "escorrentia_m3": escorrentia_mensual_m3,
        "aporte_subterraneo_m3": gwf_to_lake_mensual_m3,
    }).assign(entradas_m3=lambda d: d["escorrentia_m3"] + d["aporte_subterraneo_m3"])


def compute_outflows(eto_mensual_mm: list[float], area_laguna_m2: float,
                      lake_to_gwf_mensual_m3: list[float]) -> pd.DataFrame:
    """Salidas mensuales: evaporación (ETo Penman-Monteith [mm/mes] × área de
    espejo de agua) + filtración/infiltración (budget LAK->GWF del paquete
    LAK, Fase 3)."""
    if len(eto_mensual_mm) != len(lake_to_gwf_mensual_m3):
        raise ValueError("Evapotranspiración y filtración deben tener el mismo número de meses.")
    evaporacion_m3 = [eto_mm / 1000.0 * area_laguna_m2 for eto_mm in eto_mensual_mm]
    return pd.DataFrame({
        "mes": range(1, len(eto_mensual_mm) + 1),
        "evaporacion_m3": evaporacion_m3,
        "filtracion_m3": lake_to_gwf_mensual_m3,
    }).assign(salidas_m3=lambda d: d["evaporacion_m3"] + d["filtracion_m3"])


def monthly_annual_balance(inflows: pd.DataFrame, outflows: pd.DataFrame) -> pd.DataFrame:
    """Combina entradas y salidas en la tabla de balance mensual (con fila de
    totales anuales) que pide la Fase 4 del PDF."""
    df = inflows.merge(outflows, on="mes")
    df["delta_almacenamiento_m3"] = df["entradas_m3"] - df["salidas_m3"]

    totales = df.drop(columns="mes").sum(numeric_only=True)
    fila_anual = {"mes": "ANUAL", **totales.to_dict()}
    return pd.concat([df, pd.DataFrame([fila_anual])], ignore_index=True)


def sensitivity_balance(inflows: pd.DataFrame, outflows: pd.DataFrame,
                         variables: list[str], perturbaciones: list[float]) -> pd.DataFrame:
    """Análisis de sensibilidad del balance anual ante perturbaciones
    porcentuales de las variables críticas (`config.balance.yaml::balance.
    variables_sensibilidad`), pedido en la Fase 4 del PDF. No existe un
    análisis de sensibilidad de balance hídrico en ninguna branch del repo
    WaterKu original — construido nuevo.
    """
    columnas = {
        "escorrentia": ("entradas", "escorrentia_m3"),
        "aporte_subterraneo": ("entradas", "aporte_subterraneo_m3"),
        "evapotranspiracion": ("salidas", "evaporacion_m3"),
        "bed_leakance": ("salidas", "filtracion_m3"),
    }
    base = monthly_annual_balance(inflows, outflows)
    base_anual = base[base["mes"] == "ANUAL"].iloc[0]
    delta_base = base_anual["delta_almacenamiento_m3"]

    filas = []
    for variable in variables:
        if variable not in columnas:
            continue
        lado, col = columnas[variable]
        for pert in perturbaciones:
            in2 = inflows.copy()
            out2 = outflows.copy()
            if lado == "entradas":
                in2[col] = in2[col] * (1 + pert)
                in2["entradas_m3"] = in2["escorrentia_m3"] + in2["aporte_subterraneo_m3"]
            else:
                out2[col] = out2[col] * (1 + pert)
                out2["salidas_m3"] = out2["evaporacion_m3"] + out2["filtracion_m3"]

            anual = monthly_annual_balance(in2, out2)
            delta_pert = anual[anual["mes"] == "ANUAL"].iloc[0]["delta_almacenamiento_m3"]
            filas.append({
                "variable": variable, "perturbacion_pct": pert * 100,
                "delta_almacenamiento_anual_m3": delta_pert,
                "variacion_respecto_base_m3": delta_pert - delta_base,
            })
    return pd.DataFrame(filas)


def plot_balance_monthly(df_balance: pd.DataFrame, output_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mensual = df_balance[df_balance["mes"] != "ANUAL"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(mensual["mes"], mensual["entradas_m3"], width=0.4, label="Entradas", align="edge")
    ax.bar(mensual["mes"], -mensual["salidas_m3"], width=-0.4, label="Salidas", align="edge")
    ax.plot(mensual["mes"], mensual["delta_almacenamiento_m3"], "o-", color="black", label="Δ almacenamiento")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Volumen (m³)")
    ax.set_title("Balance hídrico mensual — Laguna Colombina Sur")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_balance_annual(df_balance: pd.DataFrame, output_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anual = df_balance[df_balance["mes"] == "ANUAL"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(["Entradas", "Salidas"], [anual["entradas_m3"], anual["salidas_m3"]],
           color=["steelblue", "indianred"])
    ax.set_ylabel("Volumen anual (m³)")
    ax.set_title(f"Balance anual — Δ = {anual['delta_almacenamiento_m3']:.0f} m³")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
