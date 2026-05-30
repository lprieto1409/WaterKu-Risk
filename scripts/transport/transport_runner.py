"""
Solver ADE (Advección-Dispersión-Reacción) para transporte de contaminantes.
Implementa la Ecuación de Transporte MT3DMS (Manual Ec. 4) en Python puro.

Ecuación resuelta (con LEA y sorción lineal, Manual Ec. 5-6):
    R·θ·∂C/∂t = ∂/∂xᵢ(θ·Dᵢⱼ·∂C/∂xⱼ) - ∂(θ·vᵢ·C)/∂xᵢ + qs·Cs - λ₁·θ·C

donde:
    R  = 1 + (ρb/θ)·Kd    retardation factor (Manual Ec. 6 + 13)
    Dᵢⱼ= tensor de dispersión (Manual Ec. 10a-f)
    λ₁ = decaimiento de primer orden (RCT)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import yaml

from scripts.transport.transport_plot import (
    plot_breakthrough_curves,
    plot_mass_balance,
    plot_plume_map,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Campo de velocidades
# ---------------------------------------------------------------------------

def build_synthetic_velocity(cfg: dict) -> dict[str, np.ndarray]:
    """Genera un campo de velocidad uniforme para el modo demo (sin MODFLOW)."""
    g = cfg["grid"]
    nlay, nrow, ncol = g["nlay"], g["nrow"], g["ncol"]
    sf = cfg["synthetic_flow"]
    shape = (nlay, nrow, ncol)
    return {
        "vx": np.full(shape, sf["vx"]),
        "vy": np.full(shape, sf["vy"]),
        "vz": np.full(shape, sf["vz"]),
    }


def load_modflow_velocity(bhd_path: str, cbc_path: str, cfg: dict) -> dict[str, np.ndarray]:
    """Lee el campo de velocidades desde outputs binarios de MODFLOW-6."""
    try:
        import flopy.utils as fpu
    except ImportError:
        warnings.warn("flopy no instalado. Usando flujo sintético.", stacklevel=2)
        return build_synthetic_velocity(cfg)

    g = cfg["grid"]
    theta = cfg["transport"]["porosity"]

    hf = fpu.HeadFile(bhd_path)
    heads = hf.get_data(kstpkper=(0, 0))
    heads[heads >= 1e29] = np.nan

    # Gradiente hidráulico → velocidad de Darcy / θ (approx. isotrópico)
    K = 1e-5  # conductividad hidráulica por defecto [m/s] (ajustar según modelo)
    dx = g["delr"]
    dy = g["delc"]

    dh_dx = np.gradient(heads, dx, axis=2)
    dh_dy = np.gradient(heads, dy, axis=1)

    vx = -K * dh_dx / theta
    vy = -K * dh_dy / theta
    vz = np.zeros_like(vx)

    return {"vx": vx, "vy": vy, "vz": vz}


# ---------------------------------------------------------------------------
# Emisores → celdas de la grilla
# ---------------------------------------------------------------------------

def _utm_to_ij(easting: float, northing: float, cfg: dict) -> tuple[int, int] | None:
    """Convierte coordenadas UTM a índices (fila, columna) de la grilla MODFLOW."""
    g = cfg["grid"]
    ang = math.radians(g["angrot_deg"])
    dx = easting - g["xorigin"]
    dy = northing - g["yorigin"]

    # Rotación inversa
    local_x = dx * math.cos(-ang) - dy * math.sin(-ang)
    local_y = dx * math.sin(-ang) + dy * math.cos(-ang)

    col = int(local_x / g["delr"])
    row = int((g["nrow"] * g["delc"] - local_y) / g["delc"])

    if 0 <= row < g["nrow"] and 0 <= col < g["ncol"]:
        return row, col
    return None


def map_emitters_to_grid(emitters_gdf: gpd.GeoDataFrame, cfg: dict) -> list[dict]:
    """
    Mapea emisores GeoJSON a celdas de la grilla.
    Retorna lista de {row, col, lay, hazard_weight, category}.
    """
    crs_m = cfg["study_area"]["crs_metric"]
    gdf = emitters_gdf.to_crs(crs_m)

    sources = []
    for _, row in gdf.iterrows():
        pt = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
        ij = _utm_to_ij(pt.x, pt.y, cfg)
        if ij is None:
            warnings.warn(f"Emisor '{row.get('id', '?')}' fuera de la grilla.", stacklevel=2)
            continue
        sources.append({
            "row": ij[0],
            "col": ij[1],
            "lay": 0,  # primera capa (zona vadosa/superior)
            "hazard_weight": float(row.get("hazard_weight", 1.0)),
            "category": row.get("category", "unknown"),
            "id": row.get("id", "?"),
        })
    return sources


def map_settlements_to_grid(settlements_gdf: gpd.GeoDataFrame, cfg: dict) -> list[dict]:
    """Mapea asentamientos a celdas de la grilla para extraer concentraciones."""
    crs_m = cfg["study_area"]["crs_metric"]
    gdf = settlements_gdf.to_crs(crs_m)

    receptors = []
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ij = _utm_to_ij(centroid.x, centroid.y, cfg)
        if ij is None:
            continue
        receptors.append({
            "row": ij[0],
            "col": ij[1],
            "lay": 0,
            "name": row.get("name", "?"),
            "id": row.get("id", "?"),
            "serves_under5": bool(row.get("serves_under5", False)),
        })
    return receptors


# ---------------------------------------------------------------------------
# Tensor de dispersión (Manual MT3DMS Ec. 10a-f, 2D simplificado)
# ---------------------------------------------------------------------------

def build_dispersion_coeffs(velocity: dict, cfg: dict) -> dict[str, np.ndarray]:
    """
    Calcula Dxx, Dyy, Dxy del tensor de dispersión isotrópico (Manual Ec. 10a-f).
    Usa componentes 2D (capa activa) + contribución vertical despreciada.
    """
    t = cfg["transport"]
    aL = t["alpha_L"]
    aT = t["alpha_T"]
    Dm = t["D_molecular"]

    vx = velocity["vx"]
    vy = velocity["vy"]
    vmag = np.sqrt(vx**2 + vy**2) + 1e-20  # evitar división por cero

    # Manual Ec. 10a, 10b, 10d
    Dxx = aL * vx**2 / vmag + aT * vy**2 / vmag + Dm
    Dyy = aL * vy**2 / vmag + aT * vx**2 / vmag + Dm
    Dxy = (aL - aT) * vx * vy / vmag

    return {"Dxx": Dxx, "Dyy": Dyy, "Dxy": Dxy}


# ---------------------------------------------------------------------------
# Ensamblaje de la matriz dispersión (Crank-Nicolson implícito)
# ---------------------------------------------------------------------------

def _assemble_dispersion_matrix(D: dict, theta: float, dx: float, dy: float,
                                 nrow: int, ncol: int) -> sp.csr_matrix:
    """
    Ensambla la matriz de dispersión 2D (Crank-Nicolson, diferencias centradas).
    Opera sobre una sola capa (nrow × ncol).
    """
    N = nrow * ncol

    def idx(r, c):
        return r * ncol + c

    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        vals.append(v)

    Dxx = D["Dxx"]
    Dyy = D["Dyy"]

    for r in range(nrow):
        for c in range(ncol):
            ii = idx(r, c)
            # Coeficientes dispersión X — Neumann no-flujo en bordes
            Dx_e = theta * 0.5 * (Dxx[r, c] + Dxx[r, c + 1]) / dx**2 if c + 1 < ncol else 0.0
            Dx_w = theta * 0.5 * (Dxx[r, c] + Dxx[r, c - 1]) / dx**2 if c - 1 >= 0 else 0.0
            # Coeficientes dispersión Y — Neumann no-flujo en bordes
            Dy_n = theta * 0.5 * (Dyy[r, c] + Dyy[r - 1, c]) / dy**2 if r - 1 >= 0 else 0.0
            Dy_s = theta * 0.5 * (Dyy[r, c] + Dyy[r + 1, c]) / dy**2 if r + 1 < nrow else 0.0

            diag = Dx_e + Dx_w + Dy_n + Dy_s
            add(ii, ii, diag)

            if c + 1 < ncol:
                add(ii, idx(r, c + 1), -Dx_e)
            if c - 1 >= 0:
                add(ii, idx(r, c - 1), -Dx_w)
            if r - 1 >= 0:
                add(ii, idx(r - 1, c), -Dy_n)
            if r + 1 < nrow:
                add(ii, idx(r + 1, c), -Dy_s)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


# ---------------------------------------------------------------------------
# Advección upwind implícito 2D
# ---------------------------------------------------------------------------

def _assemble_advection_matrix(vx2d: np.ndarray, vy2d: np.ndarray,
                                theta: float, dx: float, dy: float,
                                nrow: int, ncol: int) -> sp.csr_matrix:
    """
    Matriz advección con esquema upwind de primer orden (Manual Grupo C — FD implícito).
    Estable para Cr ≤ 1.
    """
    N = nrow * ncol

    def idx(r, c):
        return r * ncol + c

    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        vals.append(v)

    for r in range(nrow):
        for c in range(ncol):
            ii = idx(r, c)
            vx = vx2d[r, c]
            vy = vy2d[r, c]

            # Advección X — upwind según signo de vx
            if vx >= 0:
                ax = theta * vx / dx
                add(ii, ii, ax)
                if c - 1 >= 0:
                    add(ii, idx(r, c - 1), -ax)
            else:
                ax = theta * abs(vx) / dx
                add(ii, ii, ax)
                if c + 1 < ncol:
                    add(ii, idx(r, c + 1), -ax)

            # Advección Y — upwind según signo de vy
            if vy >= 0:
                ay = theta * vy / dy
                add(ii, ii, ay)
                if r - 1 >= 0:
                    add(ii, idx(r - 1, c), -ay)
            else:
                ay = theta * abs(vy) / dy
                add(ii, ii, ay)
                if r + 1 < nrow:
                    add(ii, idx(r + 1, c), -ay)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


# ---------------------------------------------------------------------------
# Solver principal (operator-splitting: Adv → Disp → Reacción)
# ---------------------------------------------------------------------------

def run_transport(cfg: dict, velocity: dict,
                  emitters_gdf: gpd.GeoDataFrame,
                  settlements_gdf: gpd.GeoDataFrame) -> dict:
    """
    Resuelve la ADE con operator-splitting implícito:
      1. Advección upwind de primer orden implícito  (Manual Grupo C — FD)
      2. Dispersión implícita (diferencias centradas, ensamblaje Crank-Nicolson)
      3. Decaimiento analítico exacto: C *= exp(-λ₁·Δt/R)

    Retorna:
        {
            "concentration_snapshots": list[np.ndarray],
            "breakthrough":            pd.DataFrame (tiempo × asentamiento),
            "mass_balance":            pd.DataFrame (tiempo, masa_total, sorbed, ...),
            "output_times_s":          list[float],
            "sources":                 list[dict],
            "receptors":               list[dict],
        }
    """
    g = cfg["grid"]
    t = cfg["transport"]
    s = cfg["simulation"]

    nlay = g["nlay"]
    nrow = g["nrow"]
    ncol = g["ncol"]
    dx = g["delr"]
    dy = g["delc"]
    theta = t["porosity"]
    rho_b = t["bulk_density"]
    Kd = t["Kd"]
    lam1 = t["lambda_decay"]
    Cs = t["source_concentration"]

    # Retardation factor R = 1 + (ρb/θ)·Kd  (Manual Ec. 6 + 13)
    R = 1.0 + (rho_b / theta) * Kd

    # Paso de tiempo adaptativo por Courant
    vmax = max(
        float(np.max(np.abs(velocity["vx"]))),
        float(np.max(np.abs(velocity["vy"]))),
        1e-20,
    )
    dt_courant = s["courant_max"] * min(dx, dy) / vmax
    dt_max = s["dt_max_days"] * 86400.0  # días → segundos
    dt = min(dt_courant, dt_max)

    T_total = s["total_time_years"] * 365.25 * 86400.0
    output_times_s = [y * 365.25 * 86400.0 for y in s["output_times_years"]]

    # Estado inicial: concentración cero en todo el dominio
    C = np.zeros((nlay, nrow, ncol), dtype=float)

    sources = map_emitters_to_grid(emitters_gdf, cfg)
    receptors = map_settlements_to_grid(settlements_gdf, cfg)

    # Índices de celdas fuente (Dirichlet: C = hazard_weight × Cs)
    source_mask = {}
    for src in sources:
        k, i, j = src["lay"], src["row"], src["col"]
        source_mask[(k, i, j)] = src["hazard_weight"] * Cs

    # Matrices dispersión y advección (estacionarias, solver 2D usando capa 0)
    if nlay > 1:
        warnings.warn(
            f"Este solver 2D usa solo la capa 0 para la física (velocidad/dispersión). "
            f"nlay={nlay} capas son avanzadas como campos independientes con la misma matriz.",
            stacklevel=2,
        )
    D_coeff = build_dispersion_coeffs(
        {"vx": velocity["vx"][0], "vy": velocity["vy"][0]}, cfg
    )
    A_disp = _assemble_dispersion_matrix(
        {"Dxx": D_coeff["Dxx"], "Dyy": D_coeff["Dyy"]},
        theta, dx, dy, nrow, ncol,
    )
    A_adv = _assemble_advection_matrix(
        velocity["vx"][0], velocity["vy"][0],
        theta, dx, dy, nrow, ncol,
    )
    N2d = nrow * ncol
    I = sp.eye(N2d, format="csr")

    # Penalización diagonal para Dirichlet consistente en celdas fuente
    A_src = sp.lil_matrix((N2d, N2d))
    for (_, i, j) in source_mask:
        A_src[i * ncol + j, i * ncol + j] = 1e8
    A_src = A_src.tocsr()

    # Sistema implícito combinado
    coef = R * theta / dt
    A_sys = coef * I + A_adv + A_disp + A_src
    factored = spla.factorized(A_sys)  # LU una sola vez

    # Almacenamiento de resultados
    concentrations_out = []
    breakthrough_rows = []
    mass_rows = []

    t_curr = 0.0
    next_out_idx = 0

    mass_injected_accum = 0.0
    mass_decayed_accum = 0.0

    while t_curr < T_total - 1e-6:
        dt_step = min(dt, T_total - t_curr)

        if abs(dt_step - dt) > 1e-6:
            coef_step = R * theta / dt_step
            A_sys_step = coef_step * I + A_adv + A_disp + A_src
            factored_step = spla.factorized(A_sys_step)
            active_coef = coef_step
            active_factored = factored_step
        else:
            active_coef = coef
            active_factored = factored

        for k in range(nlay):
            c2d = C[k].ravel().copy()
            rhs = active_coef * c2d

            # Condición Dirichlet en fuentes
            for (kk, i, j), val in source_mask.items():
                if kk == k:
                    flat = i * ncol + j
                    rhs[flat] += val * 1e8

            c_new = active_factored(rhs)

            # Decaimiento analítico exacto (Manual RCT)
            if lam1 > 0:
                c_new *= math.exp(-lam1 * dt_step / R)

            # Fijar Dirichlet estrictamente
            for (kk, i, j), val in source_mask.items():
                if kk == k:
                    c_new[i * ncol + j] = val

            c_new = np.maximum(c_new, 0.0)
            C[k] = c_new.reshape(nrow, ncol)

        t_curr += dt_step

        # Balance de masa
        V_cell = dx * dy * g["layer_thickness"]
        mass_dissolved = float(np.sum(C)) * V_cell * theta
        mass_sorbed = float(np.sum(C)) * V_cell * rho_b * Kd
        mass_total = mass_dissolved + mass_sorbed
        mass_decayed_accum += lam1 * mass_dissolved * dt_step
        for val in source_mask.values():
            mass_injected_accum += val * V_cell * theta * dt_step
        discrepancy = (
            abs(mass_total - mass_injected_accum) /
            (0.5 * (mass_total + mass_injected_accum) + 1e-20) * 100
        )

        mass_rows.append({
            "time_years": t_curr / (365.25 * 86400.0),
            "mass_dissolved_kg": mass_dissolved,
            "mass_sorbed_kg": mass_sorbed,
            "mass_total_kg": mass_total,
            "mass_decayed_kg": mass_decayed_accum,
            "discrepancy_pct": discrepancy,
        })

        # Breakthrough en asentamientos
        bt_row = {"time_years": t_curr / (365.25 * 86400.0)}
        for rec in receptors:
            k, i, j = rec["lay"], rec["row"], rec["col"]
            c_val = float(C[k, i, j])
            if rec.get("serves_under5", False):
                c_val *= 1.5  # multiplicador salud infantil
            bt_row[rec["id"]] = c_val
        breakthrough_rows.append(bt_row)

        # Guardar snapshot en tiempos de salida
        if next_out_idx < len(output_times_s) and t_curr >= output_times_s[next_out_idx]:
            concentrations_out.append(C.copy())
            next_out_idx += 1

    # Asegurar snapshot final
    if len(concentrations_out) == 0:
        concentrations_out.append(C.copy())

    return {
        "concentration_snapshots": concentrations_out,
        "output_times_s": output_times_s[:len(concentrations_out)],
        "breakthrough": pd.DataFrame(breakthrough_rows),
        "mass_balance": pd.DataFrame(mass_rows),
        "sources": sources,
        "receptors": receptors,
        "grid": g,
        "dt_used_days": dt / 86400.0,
        "R": R,
    }


# ---------------------------------------------------------------------------
# Pipeline principal (entrada Snakemake o CLI)
# ---------------------------------------------------------------------------

def process_transport(config_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    paths = cfg["paths"]

    # Cargar emisores y asentamientos
    emitters_gdf = gpd.read_file(paths["emitters_geojson"])
    settlements_gdf = gpd.read_file(paths["settlements_geojson"])

    if len(emitters_gdf) == 0:
        warnings.warn(
            f"[MT3DMS] {paths['emitters_geojson']} no tiene features. "
            "Agrega emisores al GeoJSON para simular fuentes de contaminación. "
            "Ejecutando sin fuentes (concentración = 0 en todo el dominio).",
            stacklevel=2,
        )
    if len(settlements_gdf) == 0:
        warnings.warn(
            f"[MT3DMS] {paths['settlements_geojson']} no tiene features. "
            "Agrega asentamientos al GeoJSON para obtener breakthrough curves.",
            stacklevel=2,
        )

    # Cargar campo de velocidades
    if paths.get("modflow_bhd") and Path(paths["modflow_bhd"]).exists():
        velocity = load_modflow_velocity(paths["modflow_bhd"], paths["modflow_cbc"], cfg)
    else:
        velocity = build_synthetic_velocity(cfg)

    # Resolver ADE
    results = run_transport(cfg, velocity, emitters_gdf, settlements_gdf)

    # Generar visualizaciones
    s = cfg["simulation"]
    active_lay = s.get("active_layer", 0)
    for snap, t_s in zip(results["concentration_snapshots"], results["output_times_s"]):
        t_yr = t_s / (365.25 * 86400.0)
        label = f"{t_yr:.0f}a"
        plot_plume_map(
            snap[active_lay], t_yr, active_lay,
            results["sources"], results["receptors"],
            cfg,
            output_dir / f"plume_map_{label}.png",
        )

    plot_breakthrough_curves(
        results["breakthrough"],
        settlements_gdf,
        output_dir / "breakthrough_curves.png",
    )
    plot_mass_balance(
        results["mass_balance"],
        output_dir / "mass_balance.png",
    )

    # Exportar Excel — 3 hojas
    xlsx_path = output_dir / "concentration_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        bt = results["breakthrough"].copy()
        bt.insert(0, "time_years", bt.pop("time_years"))
        bt.to_excel(writer, sheet_name="Breakthrough_Curves", index=False)

        results["mass_balance"].to_excel(writer, sheet_name="Mass_Balance", index=False)

        t = cfg["transport"]
        params = pd.DataFrame([
            ["Porosidad θ [-]", t["porosity"]],
            ["Densidad aparente ρb [kg/m³]", t["bulk_density"]],
            ["Dispersividad αL [m]", t["alpha_L"]],
            ["Dispersividad αT [m]", t["alpha_T"]],
            ["Coef. difusión D* [m²/s]", t["D_molecular"]],
            ["Coef. distribución Kd [m³/kg]", t["Kd"]],
            ["Decaimiento λ₁ [1/s]", t["lambda_decay"]],
            ["Factor de retardo R [-]", results["R"]],
            ["Δt usado [días]", results["dt_used_days"]],
            ["Tiempo total [años]", cfg["simulation"]["total_time_years"]],
        ], columns=["Parámetro", "Valor"])
        params.to_excel(writer, sheet_name="Parametros_ADE", index=False)

    print(f"[MT3DMS] Resultados guardados en: {output_dir}")


# ---------------------------------------------------------------------------
# Entrada Snakemake
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    cfg_path = snakemake.input.config  # noqa: F821
    out_dir = Path(snakemake.output.summary).parent  # noqa: F821
    process_transport(cfg_path, out_dir)
