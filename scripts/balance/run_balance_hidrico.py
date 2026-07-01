#!/usr/bin/env python3
"""
Demo ejecutable del pipeline ETo + Balance Hídrico mensual.

Caso ejemplo: cuenca andina en Huancavelica (Perú) a ~3500 msnm.
Usa valores climáticos típicos de la región para demostrar el flujo completo:

    1. Precipitación mensual (fuente: registros SENAMHI/ANA)
    2. ETo mensual vía Penman-Monteith FAO-56 (datos ERA5 sintéticos)
    3. Balance hídrico mensual Thornthwaite-Mather
    4. Clasificación ejecutiva de la cuenca
    5. Exportación a Excel con tabla ANA-lista

Uso:
    python scripts/hidrologia/run_balance_hidrico.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from balance_hidrico import balance_mensual, clasificar_cuenca
from evapotranspiracion import (
    EntradaETo,
    eto_penman_monteith,
    presion_vapor_saturacion,
    radiacion_cielo_despejado,
    radiacion_extraterrestre,
    radiacion_neta_onda_corta,
    radiacion_neta_onda_larga,
    radiacion_solar_desde_nubosidad,
    viento_a_2m,
)


def eto_mensual_cuenca_andina(latitud: float, elevacion_m: float) -> list[float]:
    """Calcula ETo mensual típica usando valores climáticos de sierra sur de Perú."""
    # Valores mensuales representativos (datos ERA5 promedio multi-anual)
    t_media = [12.5, 12.3, 12.1, 11.8, 10.5, 9.2, 9.0, 10.1, 11.5, 12.8, 13.0, 12.8]
    t_max = [18.0, 17.8, 17.5, 17.2, 16.0, 15.0, 14.8, 16.0, 17.5, 18.5, 18.8, 18.5]
    t_min = [7.0, 6.8, 6.5, 5.5, 3.0, 1.5, 1.0, 2.0, 4.5, 6.5, 7.0, 7.2]
    t_dew = [9.0, 9.2, 8.5, 6.5, 3.0, 1.0, 0.5, 2.0, 4.5, 6.5, 8.0, 9.0]
    u10 = [2.5, 2.3, 2.4, 2.8, 3.2, 3.5, 3.5, 3.3, 3.0, 2.8, 2.6, 2.5]
    presion_kpa = [68.0] * 12  # ~3500 msnm
    nubosidad = [0.75, 0.70, 0.65, 0.45, 0.25, 0.20, 0.20, 0.30, 0.45, 0.55, 0.65, 0.75]

    dias_medios = [15, 46, 75, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    dias_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    eto_mensual = []
    for m in range(12):
        ra = radiacion_extraterrestre(latitud, dias_medios[m])
        rso = radiacion_cielo_despejado(ra, elevacion_m)
        rs = radiacion_solar_desde_nubosidad(ra, nubosidad[m])
        ea = presion_vapor_saturacion(t_dew[m])
        rns = radiacion_neta_onda_corta(rs)
        rnl = radiacion_neta_onda_larga(t_max[m], t_min[m], ea, rs, rso)
        rn = max(0.0, rns - rnl)

        u2 = viento_a_2m(u10[m])
        entrada = EntradaETo(
            t_mean_c=t_media[m], t_max_c=t_max[m], t_min_c=t_min[m],
            t_dew_c=t_dew[m], u2_mps=u2, presion_kpa=presion_kpa[m],
            rn_mj=rn,
        )
        eto_dia = eto_penman_monteith(entrada)
        eto_mensual.append(eto_dia * dias_mes[m])
    return eto_mensual


def main():
    # Configuración del caso: cuenca andina Huancavelica
    print("=" * 70)
    print("WaterKu - Balance Hídrico Mensual")
    print("Caso: Cuenca andina, Huancavelica - Perú (3500 msnm)")
    print("=" * 70)

    latitud = -13.5
    elevacion = 3500.0

    # Precipitación mensual típica de sierra sur (mm/mes)
    precipitacion = [140, 130, 110, 50, 20, 10, 10, 20, 50, 80, 100, 130]

    # ETo mensual desde clima sintético representativo
    eto = eto_mensual_cuenca_andina(latitud, elevacion)

    print(f"\nPrecipitación anual: {sum(precipitacion):.0f} mm")
    print(f"ETo anual: {sum(eto):.0f} mm")
    print(f"Relación P/ETo: {sum(precipitacion) / sum(eto):.2f}")

    # Balance hídrico mensual
    df_balance = balance_mensual(
        precipitacion=precipitacion,
        eto=eto,
        capacidad_campo_mm=100.0,
        cn=80,
        fraccion_recarga=0.30,
    )

    print("\n" + "─" * 70)
    print("BALANCE HÍDRICO MENSUAL")
    print("─" * 70)
    cols_display = [
        "mes_nombre", "precipitacion_mm", "eto_mm", "etr_mm",
        "almacenamiento_mm", "deficit_mm", "excedente_mm",
        "escorrentia_mm", "recarga_mm",
    ]
    print(df_balance[cols_display].to_string(index=False))

    # Clasificación ejecutiva
    resultado = clasificar_cuenca(df_balance)
    print("\n" + "─" * 70)
    print("CLASIFICACIÓN EJECUTIVA")
    print("─" * 70)
    print(f"Estado de la cuenca     : {resultado.estado}")
    print(f"Índice de aridez (ETo/P): {resultado.indice_aridez:.2f}")
    print(f"Meses con déficit       : {resultado.meses_deficit}")
    print(f"Meses con excedente     : {resultado.meses_excedente}")
    print(f"Recarga anual estimada  : {resultado.recarga_anual_mm:.0f} mm/año")
    print(f"\n{resultado.descripcion}")

    # Exportar a Excel
    output_dir = Path("results/hidrologia")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "balance_hidrico_mensual.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_balance.to_excel(writer, sheet_name="Balance_Mensual", index=False)
        resumen = pd.DataFrame({
            "Indicador": [
                "Estado", "Índice de aridez", "Meses con déficit",
                "Meses con excedente", "Recarga anual (mm)",
                "Precipitación anual (mm)", "ETo anual (mm)",
                "ETr anual (mm)", "Déficit anual (mm)",
                "Excedente anual (mm)",
            ],
            "Valor": [
                resultado.estado,
                f"{resultado.indice_aridez:.2f}",
                resultado.meses_deficit,
                resultado.meses_excedente,
                f"{resultado.recarga_anual_mm:.0f}",
                f"{df_balance['precipitacion_mm'].sum():.0f}",
                f"{df_balance['eto_mm'].sum():.0f}",
                f"{df_balance['etr_mm'].sum():.0f}",
                f"{df_balance['deficit_mm'].sum():.0f}",
                f"{df_balance['excedente_mm'].sum():.0f}",
            ],
        })
        resumen.to_excel(writer, sheet_name="Resumen_Ejecutivo", index=False)

    print(f"\n✓ Balance exportado a: {output_file}")

    # Generar reporte PDF ejecutivo
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from reportes.generador_pdf import MetadataReporte, generar_reporte_balance

    metadata = MetadataReporte(
        proyecto="Cuenca Andina - Caso Demo",
        cliente="WaterKu — Demo Huancavelica",
        latitud=latitud,
        longitud=-75.2,
        elevacion_m=elevacion,
    )
    pdf_file = output_dir / "reporte_balance_hidrico.pdf"
    generar_reporte_balance(df_balance, resultado, metadata, pdf_file)
    print(f"✓ Reporte PDF generado: {pdf_file}")
    print("=" * 70)


if __name__ == "__main__":
    # Permite correr desde la raíz del proyecto
    script_dir = Path(__file__).parent
    os.sys.path.insert(0, str(script_dir))
    main()
