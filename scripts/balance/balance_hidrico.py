"""
Balance hídrico mensual - Método Thornthwaite-Mather.

Calcula el balance hídrico mensual de una cuenca combinando:
    - Precipitación mensual (input: estación, CHIRPS o ERA5)
    - Evapotranspiración potencial (ETo de Penman-Monteith FAO-56)
    - Escorrentía (estimada vía Curve Number SCS o proporción fija)
    - Almacenamiento en el suelo (capacidad de campo)

Produce la tabla estándar que requiere ANA en estudios hidrológicos:
    P, ETo, ETr, Δalmacenamiento, excedente, déficit, escorrentía, recarga.

Referencia:
    Thornthwaite, C.W., Mather, J.R. (1957). Instructions and tables for
    computing potential evapotranspiration and the water balance.

Uso típico:
    >>> from balance_hidrico import balance_mensual
    >>> resultado = balance_mensual(
    ...     precipitacion=[120, 90, 80, 40, 10, 5, 5, 15, 50, 90, 110, 130],
    ...     eto=[90, 85, 80, 70, 65, 55, 60, 75, 85, 95, 95, 90],
    ...     capacidad_campo_mm=100,
    ...     cn=85,
    ... )
    >>> resultado.head()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


MESES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


# =============================================================================
# Escorrentía - método SCS Curve Number
# =============================================================================

def escorrentia_scs(precipitacion_mm: float, cn: float) -> float:
    """Escorrentía directa mensual [mm] por método SCS Curve Number.

    ⚠️ NOTA DE DISEÑO: esta función se expone como UTILIDAD pero NO se
    usa directamente en el balance mensual de Thornthwaite-Mather.

    ¿Por qué? SCS-CN es un método EVENT-BASED, diseñado para caudales
    de diseño a partir de una tormenta específica. Aplicarlo a totales
    mensuales consume agua artificialmente antes del balance de suelo,
    dando resultados incorrectos para cuencas húmedas.

    Se deja como función standalone para:
        1. Estudios de caudales de diseño (SCS Unit Hydrograph)
        2. Futuro: ajustar dinámicamente `fraccion_recarga` según el CN
           del terreno (terrenos impermeables → menos recarga)

    Fórmula SCS:
        S = 25400/CN - 254       (retención máxima potencial, mm)
        Ia = 0.2·S               (abstracción inicial)
        Q = (P - Ia)² / (P - Ia + S)   si P > Ia, sino Q = 0

    Valores típicos de CN:
        60-70: bosque/pastizal, suelo bien drenado
        75-85: cultivos, suelo franco
        85-95: urbano, roca expuesta, suelo saturado

    Args:
        precipitacion_mm: Precipitación del evento/mes [mm]
        cn: Curve Number (30-100)
    """
    if precipitacion_mm <= 0 or cn <= 0:
        return 0.0
    s = 25400.0 / cn - 254.0   # S: retención potencial máxima [mm]
    ia = 0.2 * s               # Ia: abstracción inicial (intercepción, etc.)
    # Si la lluvia no supera la abstracción inicial, no hay escorrentía
    if precipitacion_mm <= ia:
        return 0.0
    return np.power(precipitacion_mm - ia, 2) / (precipitacion_mm - ia + s)


# =============================================================================
# Balance hídrico Thornthwaite-Mather
# =============================================================================

@dataclass
class ResultadoBalance:
    """Clasificación del balance hídrico anual de la cuenca."""
    estado: str               # "EXCEDENTARIA" | "EQUILIBRADA" | "DEFICITARIA"
    indice_aridez: float      # ETo_anual / P_anual (>1 = árido, <1 = húmedo)
    meses_deficit: int        # Cantidad de meses con déficit > 0
    meses_excedente: int      # Cantidad de meses con excedente > 0
    recarga_anual_mm: float   # Recarga potencial al acuífero [mm/año]
    descripcion: str          # Texto ejecutivo para reporte


def _clasificar_anual(df: pd.DataFrame) -> ResultadoBalance:
    """Genera la clasificación ejecutiva a partir del DataFrame mensual."""
    p_anual = float(df["precipitacion_mm"].sum())
    eto_anual = float(df["eto_mm"].sum())
    deficit_anual = float(df["deficit_mm"].sum())
    excedente_anual = float(df["excedente_mm"].sum())
    recarga_anual = float(df["recarga_mm"].sum())

    indice_aridez = eto_anual / max(p_anual, 1e-6)
    meses_deficit = int((df["deficit_mm"] > 1.0).sum())
    meses_excedente = int((df["excedente_mm"] > 1.0).sum())

    if excedente_anual > deficit_anual * 1.5:
        estado = "EXCEDENTARIA"
        descripcion = (
            f"La cuenca presenta {meses_excedente} meses con excedente hídrico "
            f"y una recarga anual estimada de {recarga_anual:.0f} mm. "
            f"Disponibilidad hídrica favorable."
        )
    elif deficit_anual > excedente_anual * 1.5:
        estado = "DEFICITARIA"
        descripcion = (
            f"La cuenca presenta déficit en {meses_deficit} meses del año "
            f"(índice de aridez {indice_aridez:.2f}). Requiere fuentes "
            f"complementarias o almacenamiento para uso productivo."
        )
    else:
        estado = "EQUILIBRADA"
        descripcion = (
            f"La cuenca mantiene un balance equilibrado con "
            f"{meses_excedente} meses de excedente y {meses_deficit} de déficit. "
            f"Gestión estacional recomendada."
        )

    return ResultadoBalance(
        estado=estado,
        indice_aridez=indice_aridez,
        meses_deficit=meses_deficit,
        meses_excedente=meses_excedente,
        recarga_anual_mm=recarga_anual,
        descripcion=descripcion,
    )


def balance_mensual(
    precipitacion: Sequence[float],
    eto: Sequence[float],
    capacidad_campo_mm: float = 100.0,
    cn: float = 75.0,
    almacenamiento_inicial_mm: Optional[float] = None,
    fraccion_recarga: float = 0.30,
) -> pd.DataFrame:
    """Balance hídrico mensual por el método de Thornthwaite-Mather.

    Args:
        precipitacion: 12 valores mensuales de precipitación [mm/mes]
        eto: 12 valores mensuales de evapotranspiración potencial [mm/mes]
        capacidad_campo_mm: Capacidad de campo del suelo [mm]
            (típico: 50-150 mm según textura)
        cn: Curve Number del SCS para separar escorrentía
        almacenamiento_inicial_mm: Almacenamiento inicial; si None, se
            estabiliza iterando el año hasta convergencia
        fraccion_recarga: Fracción del excedente que va a recarga subterránea
            (el resto es escorrentía superficial); típico 0.20-0.40

    Returns:
        DataFrame con 12 filas (meses) y columnas:
            mes, precipitacion_mm, eto_mm, etr_mm, variacion_almac_mm,
            almacenamiento_mm, deficit_mm, excedente_mm,
            escorrentia_mm, recarga_mm
    """
    p = np.asarray(precipitacion, dtype=float)
    e = np.asarray(eto, dtype=float)
    if len(p) != 12 or len(e) != 12:
        raise ValueError("Se requieren 12 valores mensuales de P y ETo")
    if not (30 <= cn <= 100):
        raise ValueError("CN debe estar entre 30 y 100")
    if not (0.0 <= fraccion_recarga <= 1.0):
        raise ValueError("fraccion_recarga debe estar en [0, 1]")

    almac_prev = (
        almacenamiento_inicial_mm
        if almacenamiento_inicial_mm is not None
        else capacidad_campo_mm / 2.0
    )

    def _correr_ano(almac_inicial: float):
        # ALGORITMO THORNTHWAITE-MATHER (1957)
        # ===========================================
        # Simula el estado del agua en el suelo mes a mes. La "capacidad de
        # campo" (CC) es cuánta agua puede retener el suelo antes de que
        # el excedente drene por gravedad hacia escorrentía/recarga.
        #
        # Estados posibles cada mes:
        #   Mes húmedo (P >= ETo): el suelo recibe más agua de la que
        #       pierde por evaporación. ETo se cumple completo (ETr = ETo),
        #       el sobrante llena el suelo y si hay más, sale como excedente.
        #   Mes seco (P < ETo): la atmósfera demanda más agua de la que
        #       llueve. La demanda excedente la saca del suelo en forma
        #       exponencial: al principio es fácil (suelo mojado), después
        #       se hace cada vez más difícil (suelo seco).
        almac = almac_inicial
        filas = []
        for mes in range(12):
            p_mes = p[mes]
            e_mes = e[mes]

            if p_mes >= e_mes:
                # MES HÚMEDO
                etr = e_mes                                                  # Todo lo que pide ETo se cumple
                sobrante = p_mes - e_mes                                     # Agua que sobra después de ETr
                almac_nuevo = min(almac + sobrante, capacidad_campo_mm)      # El suelo se llena hasta CC
                excedente = (almac + sobrante) - almac_nuevo                 # Lo que no cabe, sale
                deficit = 0.0
            else:
                # MES SECO
                deficit_atm = e_mes - p_mes                                  # La atmósfera "pide" esta cantidad
                # Extracción exponencial: a mayor déficit y menor CC, más
                # difícil es que el suelo ceda agua. Modelo empírico
                # Thornthwaite que funciona bien para suelos típicos.
                retiro = almac * (1.0 - np.exp(-deficit_atm / max(capacidad_campo_mm, 1.0)))
                almac_nuevo = max(0.0, almac - retiro)                       # El suelo baja (no puede <0)
                etr = p_mes + retiro                                         # ETr = lluvia + lo que pudo ceder
                excedente = 0.0                                              # No hay sobrante en mes seco
                deficit = e_mes - etr                                        # Lo que ETo pidió pero no se cumplió

            # PARTICIÓN DEL EXCEDENTE
            # El excedente (agua que ya no cabe en el suelo) se divide:
            #   recarga → infiltra profundo, alimenta el acuífero
            #   escorrentía → corre por superficie hacia el río
            # La proporción depende del terreno (permeabilidad, geología).
            recarga = excedente * fraccion_recarga
            escorrentia_total = excedente - recarga
            filas.append({
                "mes": mes + 1,
                "mes_nombre": MESES_ES[mes],
                "precipitacion_mm": round(p_mes, 2),
                "eto_mm": round(e_mes, 2),
                "etr_mm": round(etr, 2),
                "variacion_almac_mm": round(almac_nuevo - almac, 2),
                "almacenamiento_mm": round(almac_nuevo, 2),
                "deficit_mm": round(deficit, 2),
                "excedente_mm": round(excedente, 2),
                "escorrentia_mm": round(escorrentia_total, 2),
                "recarga_mm": round(recarga, 2),
            })
            almac = almac_nuevo
        return pd.DataFrame(filas), almac

    # ITERACIÓN DE CONVERGENCIA
    # ========================
    # Problema: al iniciar enero, ¿qué almacenamiento asumimos? Si
    # arrancamos con el valor "equivocado", los primeros meses dan
    # resultados sesgados.
    #
    # Solución: correr el año varias veces reutilizando el almacenamiento
    # de diciembre como nuevo enero inicial, hasta que converja
    # (tolerancia 0.1 mm). Para climas estacionales típicos basta 2-3
    # iteraciones. Max 10 como salvaguarda.
    if almacenamiento_inicial_mm is None:
        for _ in range(10):
            df, almac_final = _correr_ano(almac_prev)
            if abs(almac_final - almac_prev) < 0.1:
                break
            almac_prev = almac_final
    else:
        # Si el usuario especifica el inicial, no iteramos (una sola pasada)
        df, _ = _correr_ano(almac_prev)

    return df


def clasificar_cuenca(df_balance: pd.DataFrame) -> ResultadoBalance:
    """Genera la clasificación anual ejecutiva desde el DataFrame mensual."""
    return _clasificar_anual(df_balance)


# =============================================================================
# Integración con outputs existentes de WaterKu
# =============================================================================

def balance_desde_series(
    df_precipitacion: pd.DataFrame,
    df_eto: pd.DataFrame,
    capacidad_campo_mm: float = 100.0,
    cn: float = 75.0,
    fraccion_recarga: float = 0.30,
    columna_precip: str = "precipitacion_mm",
    columna_eto: str = "ETo_mm_mes",
    columna_mes: str = "mes",
) -> tuple[pd.DataFrame, ResultadoBalance]:
    """Balance hídrico desde DataFrames mensuales (promedios multi-anuales).

    Args:
        df_precipitacion: 12 filas con columnas [mes, precipitacion_mm]
        df_eto: 12 filas con columnas [mes, ETo_mm_mes]
        (resto igual que balance_mensual)

    Returns:
        (df_mensual, clasificacion_anual)
    """
    p_df = df_precipitacion.sort_values(columna_mes)
    e_df = df_eto.sort_values(columna_mes)

    if len(p_df) != 12 or len(e_df) != 12:
        raise ValueError(
            "Se requieren 12 filas mensuales en cada DataFrame. "
            "Usa groupby('mes').mean() para promediar años múltiples."
        )

    df = balance_mensual(
        precipitacion=p_df[columna_precip].tolist(),
        eto=e_df[columna_eto].tolist(),
        capacidad_campo_mm=capacidad_campo_mm,
        cn=cn,
        fraccion_recarga=fraccion_recarga,
    )
    return df, clasificar_cuenca(df)
