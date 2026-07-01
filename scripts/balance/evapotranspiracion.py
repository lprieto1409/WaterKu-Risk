"""
Evapotranspiración de referencia (ETo) - Método Penman-Monteith FAO-56.

Implementa el método estándar de FAO-56 (Allen et al., 1998) para calcular
evapotranspiración potencial a partir de datos ERA5. Cuando no hay datos
directos de radiación solar, usa el método Angström-Prescott modificado con
nubosidad como proxy.

Referencia:
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
    FAO Irrigation and Drainage Paper No. 56: Crop Evapotranspiration.

Variables ERA5 requeridas (todas ya disponibles en WaterKu):
    - t2m: Temperatura a 2m (K)
    - d2m: Punto de rocío a 2m (K)
    - u10, v10: Viento a 10m (m/s)
    - sp: Presión superficial (Pa)
    - tcc: Cobertura nubosa total (0-1) [opcional, para estimar Rn]
    - ssr: Radiación solar neta (J/m²) [opcional, preferido si disponible]

Uso típico:
    >>> import pandas as pd
    >>> from evapotranspiracion import calcular_eto_diario_era5
    >>> df_era5 = pd.read_csv("station_era5.csv")
    >>> df_eto = calcular_eto_diario_era5(df_era5, latitud=-13.5, elevacion=3200)
    >>> df_eto["ETo_mm_dia"].head()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# Constantes físicas (FAO-56, Allen et al. 1998)
# =============================================================================
# GSC: Constante solar. Energía solar incidente en el tope de la atmósfera,
#      usada para calcular radiación extraterrestre Ra. FAO-56 eq. (28).
GSC = 0.0820          # [MJ/(m²·min)]

# SIGMA: Constante de Stefan-Boltzmann. Relaciona temperatura^4 con emisión
#        de radiación térmica. Aparece en el cálculo de radiación neta de
#        onda larga (FAO-56 eq. 39).
SIGMA = 4.903e-9      # [MJ/(K⁴·m²·día)]

# ALBEDO_REF: Albedo del cultivo de referencia (pasto verde de 0.12m, bien
#             regado). Asumido constante en FAO-56 para simplificar el
#             cálculo de radiación neta de onda corta (Rns).
ALBEDO_REF = 0.23     # [adimensional]

# Conversión Kelvin ↔ Celsius. ERA5 viene en K, usuarios suelen pensar en °C.
KELVIN_0 = 273.15


# =============================================================================
# Funciones de presión de vapor y constantes termodinámicas
# =============================================================================

def presion_vapor_saturacion(t_celsius: float | np.ndarray) -> float | np.ndarray:
    """Presión de vapor de saturación e°(T) [kPa]. Ecuación FAO-56 (11).

    La presión de saturación es la cantidad máxima de vapor de agua que
    el aire puede contener a una temperatura dada. Cuando el aire real
    tiene menos, existe un déficit (es - ea) que impulsa la evaporación.

    La fórmula es una aproximación polinómica de la ecuación de Tetens:
        e°(T) = 0.6108 · exp(17.27·T / (T + 237.3))

    Validada contra FAO-56 Tabla 2.3: e°(15°C)=1.705, e°(25°C)=3.168 kPa.
    """
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def pendiente_curva_vapor(t_celsius: float | np.ndarray) -> float | np.ndarray:
    """Pendiente de la curva de presión de vapor Δ [kPa/°C]. FAO-56 (13).

    Δ es la derivada de e°(T) respecto a T, o sea cuánto cambia la presión
    de saturación por cada grado de cambio de temperatura. Aparece en el
    término "radiativo" de Penman-Monteith (peso de Rn vs viento).

    A más temperatura, Δ es mayor (exponencial), lo que explica por qué
    la ETo aumenta mucho en climas cálidos aún con humedad alta.
    """
    es = presion_vapor_saturacion(t_celsius)
    return 4098.0 * es / np.power(t_celsius + 237.3, 2)


def constante_psicrometrica(presion_kpa: float | np.ndarray) -> float | np.ndarray:
    """Constante psicrométrica γ [kPa/°C]. FAO-56 (8).

    γ relaciona la presión parcial de vapor con la temperatura del aire.
    Depende (débilmente) de la presión atmosférica, que cambia con la
    altitud: a nivel del mar (101.3 kPa) γ ≈ 0.0674; en sierra alta
    (~68 kPa a 3500m) γ ≈ 0.0452.

    Esto importa: a mayor altitud γ es menor, lo que reduce ligeramente
    el término aerodinámico de Penman-Monteith (menos relevancia del
    viento vs la radiación).
    """
    return 0.665e-3 * presion_kpa


def viento_a_2m(u10: float | np.ndarray) -> float | np.ndarray:
    """Conversión de viento de 10m a 2m (perfil logarítmico). FAO-56 (47).

    ERA5 reporta viento a 10m (u10, v10), pero Penman-Monteith pide
    viento a 2m sobre el cultivo. Sobre superficie "abierta" típica,
    el perfil logarítmico estándar da: u₂ ≈ u₁₀ · 0.748.

    Fórmula: u₂ = u₁₀ · 4.87 / ln(67.8·10 - 5.42) ≈ u₁₀ · 0.748.
    """
    return u10 * 4.87 / math.log(67.8 * 10.0 - 5.42)


# =============================================================================
# Radiación — estimación cuando no hay datos directos
# =============================================================================

def radiacion_extraterrestre(latitud: float, dia_del_ano: int) -> float:
    """Radiación extraterrestre Ra [MJ/(m²·día)]. FAO-56 (21-25).

    Ra es la energía solar que llega al tope de la atmósfera en ese punto
    y día. Es 100% determinista (astronomía): depende solo de latitud y
    día del año, no del clima. Es la referencia contra la cual se compara
    la radiación real en superficie para inferir cobertura nubosa.

    Método:
        1. dr = distancia relativa Tierra-Sol (oscila 0.97-1.03 en el año)
        2. δ = declinación solar (-23.45° en solsticio de invierno a +23.45°
                en verano)
        3. ωs = ángulo horario de puesta de sol (depende de latitud y δ)
        4. Ra = integral diaria del flujo solar con esos parámetros

    Args:
        latitud: Latitud en grados decimales (negativa para hemisferio sur)
        dia_del_ano: Día del año (1-366)

    Valores típicos en ecuador: 36-38 MJ/m²/día según época.
    En Huancavelica (-13.5°S) en enero: ~40 MJ/m²/día (verano austral).
    """
    phi = math.radians(latitud)

    # dr: factor de corrección por distancia variable Tierra-Sol.
    # Mayor en enero (perihelio), menor en julio (afelio).
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * dia_del_ano / 365.0)

    # δ: declinación solar. En equinoccios (día 81/265) es 0; en solsticios
    # alcanza ±0.409 rad (±23.45°). Esto determina la duración del día.
    delta = 0.409 * math.sin(2.0 * math.pi * dia_del_ano / 365.0 - 1.39)

    # ωs: ángulo horario de puesta del sol. acos(arg) donde arg > 1 o < -1
    # indica día/noche polar (sol nunca se pone o nunca sale). Clip evita
    # NaN por errores de redondeo cerca de los polos.
    arg = -math.tan(phi) * math.tan(delta)
    arg = max(-1.0, min(1.0, arg))
    ws = math.acos(arg)

    # Integral diaria del flujo solar sobre el arco diurno.
    ra = (24.0 * 60.0 / math.pi) * GSC * dr * (
        ws * math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )
    return ra


def radiacion_cielo_despejado(ra: float, elevacion_m: float) -> float:
    """Radiación solar de cielo despejado Rso [MJ/(m²·día)]. FAO-56 (37)."""
    return (0.75 + 2e-5 * elevacion_m) * ra


def radiacion_solar_desde_nubosidad(ra: float, cobertura_nubosa: float) -> float:
    """Radiación solar estimada Rs vía Angström-Prescott con nubosidad como proxy.

    ¿POR QUÉ ESTA FUNCIÓN EXISTE?
    ERA5 en su configuración base de WaterKu no descarga radiación solar
    directa (sería otro TB de datos). Sí descarga `tcc` (total cloud cover),
    que es un proxy: a más nubes, menos radiación.

    La fórmula original de Angström-Prescott usa horas de sol (n/N):
        Rs = (as + bs·n/N) · Ra

    Donde as=0.25, bs=0.50 son valores recomendados FAO para uso universal.
    Reinterpretamos n/N ≈ 1 - cobertura_nubosa:
        - Cielo despejado (tcc=0): n/N=1 → Rs = 0.75·Ra (máximo realista)
        - Cielo cubierto (tcc=1): n/N=0 → Rs = 0.25·Ra (mínimo, solo difusa)

    Precisión: ~10% menor que con mediciones directas de Rs. Es aceptable
    para estudios de balance mensual. Si el usuario tiene datos ERA5 de
    radiación (`ssr`), usar `usar_radiacion_directa=True`.

    Args:
        ra: Radiación extraterrestre [MJ/(m²·día)]
        cobertura_nubosa: Fracción de cobertura nubosa (0 = despejado, 1 = cubierto)
    """
    # Clip por seguridad: ERA5 puede tener valores fuera [0,1] por interpolación.
    cobertura = np.clip(cobertura_nubosa, 0.0, 1.0)
    n_sobre_N = 1.0 - cobertura
    return (0.25 + 0.50 * n_sobre_N) * ra


def radiacion_neta_onda_corta(rs: float | np.ndarray) -> float | np.ndarray:
    """Radiación neta de onda corta Rns [MJ/(m²·día)]. FAO-56 (38)."""
    return (1.0 - ALBEDO_REF) * rs


def radiacion_neta_onda_larga(
    t_max_c: float | np.ndarray,
    t_min_c: float | np.ndarray,
    ea_kpa: float | np.ndarray,
    rs: float | np.ndarray,
    rso: float | np.ndarray,
) -> float | np.ndarray:
    """Radiación neta de onda larga Rnl [MJ/(m²·día)]. FAO-56 (39)."""
    tmax_k4 = np.power(t_max_c + KELVIN_0, 4)
    tmin_k4 = np.power(t_min_c + KELVIN_0, 4)
    ratio = np.clip(rs / np.maximum(rso, 1e-6), 0.0, 1.0)
    return SIGMA * ((tmax_k4 + tmin_k4) / 2.0) * (
        0.34 - 0.14 * np.sqrt(np.maximum(ea_kpa, 0.0))
    ) * (1.35 * ratio - 0.35)


# =============================================================================
# Cálculo principal de ETo
# =============================================================================

@dataclass
class EntradaETo:
    """Entradas diarias para Penman-Monteith FAO-56 (unidades SI estándar)."""
    t_mean_c: float       # Temperatura media [°C]
    t_max_c: float        # Temperatura máxima [°C]
    t_min_c: float        # Temperatura mínima [°C]
    t_dew_c: float        # Punto de rocío [°C]
    u2_mps: float         # Viento a 2m [m/s]
    presion_kpa: float    # Presión atmosférica [kPa]
    rn_mj: float          # Radiación neta [MJ/(m²·día)]
    g_mj: float = 0.0     # Flujo de calor en suelo (≈0 para escala diaria)


def eto_penman_monteith(e: EntradaETo) -> float:
    """ETo [mm/día] según Penman-Monteith FAO-56 (ecuación 6).

    ESTA ES LA ECUACIÓN CENTRAL DEL MÓDULO. Todo lo demás existe para
    alimentarla con datos correctos.

    Estructura:
        ETo = [término_radiativo + término_aerodinámico] / denominador

    1. Término radiativo: 0.408·Δ·(Rn-G)
       Convierte energía radiativa en equivalente de evaporación (mm).
       0.408 es el factor que convierte MJ/m² a mm de agua evaporada
       (calor latente de vaporización del agua).
       A más radiación y más Δ, más evaporación.

    2. Término aerodinámico: γ·(900/(T+273))·u₂·(es-ea)
       Cuantifica el "poder evaporante" del aire: depende del viento
       (u₂) y el déficit de presión de vapor (es - ea).
       Si el aire está saturado (es ≈ ea), este término se anula.

    3. Denominador: Δ + γ·(1 + 0.34·u₂)
       Factor de ponderación que ajusta el peso relativo del viento
       sobre la resistencia estomática del cultivo de referencia.

    Returns:
        ETo en mm/día. Valores típicos:
        - Frío húmedo (sierra alta en invierno): 1-3 mm/día
        - Templado (valle interandino): 3-5 mm/día
        - Árido cálido (costa en verano): 6-10 mm/día
    """
    # Parámetros termodinámicos a la temperatura media del día
    delta = pendiente_curva_vapor(e.t_mean_c)
    gamma = constante_psicrometrica(e.presion_kpa)

    # FAO-56 recomienda usar el promedio de e°(Tmax) y e°(Tmin) en vez de
    # e°(Tmean) porque la saturación es no-lineal: el promedio preserva
    # mejor el valor real en días con alta amplitud térmica (desiertos).
    es = (presion_vapor_saturacion(e.t_max_c)
          + presion_vapor_saturacion(e.t_min_c)) / 2.0

    # La presión real se estima desde el punto de rocío: a Tdew el aire
    # está saturado, entonces ea = e°(Tdew).
    ea = presion_vapor_saturacion(e.t_dew_c)

    # Ecuación de Penman-Monteith FAO-56 (6):
    numerador = (
        0.408 * delta * (e.rn_mj - e.g_mj)                                  # Término radiativo
        + gamma * (900.0 / (e.t_mean_c + 273.0)) * e.u2_mps * (es - ea)     # Término aerodinámico
    )
    denominador = delta + gamma * (1.0 + 0.34 * e.u2_mps)
    return numerador / denominador


# =============================================================================
# Integración con datos ERA5 (pipeline WaterKu)
# =============================================================================

def _celsius(series: pd.Series) -> pd.Series:
    """Convierte K a °C si la magnitud sugiere Kelvin."""
    return series - KELVIN_0 if series.mean() > 150 else series


def _pascales_a_kpa(series: pd.Series) -> pd.Series:
    """Convierte Pa a kPa si la magnitud sugiere Pascales."""
    return series / 1000.0 if series.mean() > 1000 else series


def calcular_eto_diario_era5(
    df: pd.DataFrame,
    latitud: float,
    elevacion_m: float = 0.0,
    columna_fecha: str = "date",
    usar_radiacion_directa: bool = False,
) -> pd.DataFrame:
    """Calcula ETo diario a partir de un DataFrame con columnas ERA5.

    Columnas esperadas (nombres estándar del pipeline ML de WaterKu):
        - t2m (temperatura 2m, K o °C)
        - d2m (punto de rocío 2m, K o °C)
        - u10, v10 (viento 10m, m/s)
        - sp (presión superficial, Pa o kPa)
        - tcc (cobertura nubosa, 0-1) [requerida si no hay ssr]
        - ssr (radiación solar neta en J/m²) [opcional]

    Args:
        df: DataFrame con datos diarios de ERA5
        latitud: Latitud del punto (grados decimales, sur = negativo)
        elevacion_m: Elevación del punto [m] (default 0)
        columna_fecha: Nombre de la columna de fecha
        usar_radiacion_directa: Si True y existe columna ssr, la usa
            directamente en vez de estimar por Angström-Prescott

    Returns:
        DataFrame con columnas originales más:
            - ETo_mm_dia: Evapotranspiración de referencia [mm/día]
            - Rn_MJ: Radiación neta calculada [MJ/(m²·día)]
            - metodo_radiacion: "directa" | "angstrom_prescott"
    """
    if columna_fecha not in df.columns:
        raise ValueError(f"Columna de fecha '{columna_fecha}' no encontrada")

    out = df.copy()
    fecha = pd.to_datetime(out[columna_fecha])

    # Normalización de unidades
    t2m_c = _celsius(out["t2m"])
    d2m_c = _celsius(out["d2m"])
    sp_kpa = _pascales_a_kpa(out["sp"])

    # Aproximar Tmax/Tmin si solo hay media (ERA5 diario agregado)
    if "t2m_max" in out.columns and "t2m_min" in out.columns:
        t_max_c = _celsius(out["t2m_max"])
        t_min_c = _celsius(out["t2m_min"])
    else:
        amplitud = 3.0  # °C, valor por defecto conservador
        t_max_c = t2m_c + amplitud
        t_min_c = t2m_c - amplitud

    # Velocidad del viento a 10m, luego a 2m
    if "wind10" in out.columns:
        u10 = out["wind10"]
    else:
        u10 = np.sqrt(np.power(out["u10"], 2) + np.power(out["v10"], 2))
    u2 = viento_a_2m(u10)

    # Radiación neta — directa desde ERA5 o estimada
    metodo_rad = []
    rn_vals = []
    for idx, row in out.iterrows():
        dia_ano = int(fecha.iloc[idx].dayofyear)
        ra = radiacion_extraterrestre(latitud, dia_ano)
        rso = radiacion_cielo_despejado(ra, elevacion_m)

        if usar_radiacion_directa and "ssr" in out.columns and pd.notna(row["ssr"]):
            # ERA5 ssr viene en J/m² acumulado; convertir a MJ/(m²·día)
            rs = float(row["ssr"]) / 1e6
            metodo_rad.append("directa")
        else:
            tcc = float(row.get("tcc", 0.5))
            rs = radiacion_solar_desde_nubosidad(ra, tcc)
            metodo_rad.append("angstrom_prescott")

        ea = presion_vapor_saturacion(float(d2m_c.iloc[idx]))
        rns = radiacion_neta_onda_corta(rs)
        rnl = radiacion_neta_onda_larga(
            float(t_max_c.iloc[idx]), float(t_min_c.iloc[idx]),
            float(ea), rs, rso
        )
        rn_vals.append(max(0.0, rns - rnl))

    out["Rn_MJ"] = rn_vals
    out["metodo_radiacion"] = metodo_rad

    # Cálculo de ETo punto a punto
    eto_vals = []
    for idx in range(len(out)):
        entrada = EntradaETo(
            t_mean_c=float(t2m_c.iloc[idx]),
            t_max_c=float(t_max_c.iloc[idx]),
            t_min_c=float(t_min_c.iloc[idx]),
            t_dew_c=float(d2m_c.iloc[idx]),
            u2_mps=float(u2.iloc[idx]),
            presion_kpa=float(sp_kpa.iloc[idx]),
            rn_mj=float(rn_vals[idx]),
        )
        eto_vals.append(max(0.0, eto_penman_monteith(entrada)))
    out["ETo_mm_dia"] = eto_vals
    return out


def agregar_eto_mensual(
    df_diario: pd.DataFrame,
    columna_fecha: str = "date",
) -> pd.DataFrame:
    """Agrega ETo diaria a totales mensuales [mm/mes]."""
    if "ETo_mm_dia" not in df_diario.columns:
        raise ValueError("El DataFrame debe contener ETo_mm_dia")
    tmp = df_diario.copy()
    tmp[columna_fecha] = pd.to_datetime(tmp[columna_fecha])
    tmp = tmp.set_index(columna_fecha)
    mensual = tmp["ETo_mm_dia"].resample("MS").sum().to_frame("ETo_mm_mes")
    mensual["mes"] = mensual.index.month
    mensual["anio"] = mensual.index.year
    return mensual.reset_index()
