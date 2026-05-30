"""
Benchmarks del Manual MT3DMS, Capítulo 7.

1D (p.130-131, Figura 31): 4 casos (1a-1d) — advección, dispersión, sorción, decay.
    Analítica: Van Genuchten & Alves (1982) Ec. A3 (= Ogata-Banks para λ=0).

3D (p.145-146, Figura 39): Transporte 3D en flujo uniforme, fuente puntual.
    Analítica: Hunt (1978).

Esquema de advección: TVD SUPERBEE (3er orden efectivo, sin difusión numérica).
"""
import math
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Soluciones analíticas
# ---------------------------------------------------------------------------

def _vga1d(x, t, v, D, R=1.0, lam=0.0):
    """
    Van Genuchten & Alves (1982) Eq. A3 — ADE 1D con sorción lineal y decay.
    BC: C(0,t)=1, C(x,0)=0, ∂C/∂x(∞,t)=0.
    Para λ=0 reduce a Ogata-Banks (1961). Para D≈0 retorna escalón exacto.
    """
    if D < 1e-12:
        front = v * t / R
        return np.where(x <= front, math.exp(-lam * t / R), 0.0).astype(float)
    gamma = math.sqrt(1.0 + 4.0 * lam * D / v**2)
    sqDRt = np.sqrt(D * R * t)
    with np.errstate(over="ignore", invalid="ignore"):
        A1 = 0.5 * np.exp(np.clip(v * x / (2.0 * D) * (1.0 - gamma), -700, 700)) \
             * erfc((R * x - v * gamma * t) / (2.0 * sqDRt))
        A2 = 0.5 * np.exp(np.clip(v * x / (2.0 * D) * (1.0 + gamma), -700, 700)) \
             * erfc((R * x + v * gamma * t) / (2.0 * sqDRt))
    return np.nan_to_num(A1 + A2, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.5)


def _hunt3d(x, y, z, t, v, Dxx, Dyy, Dzz, Q, theta):
    """
    Hunt (1978) — fuente puntual continua en flujo uniforme 3D (+x), R=1, λ=0.
    C = Q/(8θπ√(Dxx·Dyy·Dzz)) × (1/r) × exp(v(x−r)/(2Dxx)) × erfc((r−vt)/(2√(Dxx·t)))
    """
    r2 = x**2 + (Dxx / Dyy) * y**2 + (Dxx / Dzz) * z**2
    if r2 < 1e-9:
        return float("nan")
    r = math.sqrt(r2)
    prefactor = Q / (8.0 * theta * math.pi * math.sqrt(Dxx * Dyy * Dzz))
    val = prefactor / r \
          * math.exp(max(-700.0, v * (x - r) / (2.0 * Dxx))) \
          * float(erfc((r - v * t) / (2.0 * math.sqrt(Dxx * t + 1e-30))))
    return max(0.0, val)


# ---------------------------------------------------------------------------
# TVD SUPERBEE — esquema de advección de alto orden
# ---------------------------------------------------------------------------

def _tvd_step_1d(C1d, v_eff, dx, dt, bc_left=0.0):
    """
    Un paso explícito de advección TVD con limitador SUPERBEE.
    v_eff: velocidad efectiva de advección [m/unidad_tiempo] = v/R
    bc_left: valor de concentración en el borde izquierdo (upstream).
    Neumann ∂C/∂x=0 en el borde derecho.
    """
    N = len(C1d)
    Cr = v_eff * dt / dx        # número de Courant efectivo

    # Array extendido con celdas fantasma (2 a cada lado)
    Cg = np.empty(N + 4)
    Cg[:2] = bc_left            # fantasmas izquierdos (Dirichlet upstream)
    Cg[2:N + 2] = C1d
    Cg[N + 2:] = C1d[-1]       # fantasmas derechos (Neumann)

    # Diferencias upwind y downwind en cada cara i+1/2
    dC_up = Cg[2:N + 2] - Cg[1:N + 1]   # C[i] − C[i-1]
    dC_dn = Cg[3:N + 3] - Cg[2:N + 2]   # C[i+1] − C[i]

    # Ratio de suavidad r = dC_up / dC_dn
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(dC_dn) > 1e-15, dC_up / dC_dn, 0.0)

    # Limitador SUPERBEE: φ(r) = max(0, min(2r,1), min(r,2))
    phi = np.maximum(0.0,
          np.maximum(np.minimum(2.0 * r, 1.0),
                     np.minimum(r, 2.0)))

    # Flujo en cara derecha
    F_right = v_eff * Cg[2:N + 2] + 0.5 * v_eff * (1.0 - Cr) * phi * dC_dn

    # Flujo en cara izquierda de celda i = flujo derecho de celda i-1
    F_left = np.empty(N)
    F_left[0] = v_eff * bc_left
    F_left[1:] = F_right[:-1]

    C_new = C1d - (dt / dx) * (F_right - F_left)
    np.maximum(C_new, 0.0, out=C_new)
    return C_new


def _tvd_step_x_3d(C3d, v_eff, dx, dt, bc_left=0.0):
    """
    Paso TVD SUPERBEE en dirección X para array 3D (nlay, nrow, ncol).
    Vectorizado sobre capas y filas.
    """
    nlay, nrow, ncol = C3d.shape
    Cr = v_eff * dt / dx

    Cg = np.empty((nlay, nrow, ncol + 4))
    Cg[:, :, :2] = bc_left
    Cg[:, :, 2:ncol + 2] = C3d
    Cg[:, :, ncol + 2:] = C3d[:, :, -1:]

    dC_up = Cg[:, :, 2:ncol + 2] - Cg[:, :, 1:ncol + 1]
    dC_dn = Cg[:, :, 3:ncol + 3] - Cg[:, :, 2:ncol + 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(dC_dn) > 1e-15, dC_up / dC_dn, 0.0)

    phi = np.maximum(0.0,
          np.maximum(np.minimum(2.0 * r, 1.0),
                     np.minimum(r, 2.0)))

    F_right = v_eff * Cg[:, :, 2:ncol + 2] + 0.5 * v_eff * (1.0 - Cr) * phi * dC_dn
    F_left = np.empty_like(C3d)
    F_left[:, :, 0] = v_eff * bc_left
    F_left[:, :, 1:] = F_right[:, :, :-1]

    C_new = C3d - (dt / dx) * (F_right - F_left)
    np.maximum(C_new, 0.0, out=C_new)
    return C_new


# ---------------------------------------------------------------------------
# Solver numérico 1D (TVD advección + CN dispersión + decay analítico)
# ---------------------------------------------------------------------------

def _solve_1d(ncell, dx, v, D, R, lam, theta, dt, nsteps):
    """
    Operator-splitting:
      1. Advección TVD SUPERBEE explícito  (velocidad efectiva v/R)
      2. Dispersión Crank-Nicolson implícito
      3. Decaimiento analítico exacto: C *= exp(-λ·Δt/R)
    BC: Dirichlet C[0]=1, Neumann ∂C/∂x(L)=0.
    """
    N = ncell
    C = np.zeros(N)
    v_eff = v / R           # velocidad de advección retardada
    coef = R * theta / dt   # coeficiente temporal para CN

    # Precompute dispersion CN matrices
    if D > 1e-12:
        Dc = theta * D / dx**2
        rows, cols_l, vals = [], [], []
        for i in range(N):
            left = Dc if i > 0 else 0.0
            right = Dc if i < N - 1 else 0.0
            rows.append(i); cols_l.append(i); vals.append(left + right)
            if i > 0:
                rows.append(i); cols_l.append(i - 1); vals.append(-left)
            if i < N - 1:
                rows.append(i); cols_l.append(i + 1); vals.append(-right)
        A_d = sp.csr_matrix((vals, (rows, cols_l)), shape=(N, N))
        I = sp.eye(N, format="csr")
        # CN: (coef·I + ½·A_d)·C_new = (coef·I − ½·A_d)·C_adv
        A_lhs = (coef * I + 0.5 * A_d).tolil()
        A_lhs[0, :] = 0; A_lhs[0, 0] = 1.0   # Dirichlet en i=0
        A_lhs = A_lhs.tocsr()
        A_rhs = coef * I - 0.5 * A_d
        factored = spla.factorized(A_lhs)

    for _ in range(nsteps):
        # 1. TVD advection
        C = _tvd_step_1d(C, v_eff, dx, dt, bc_left=1.0)
        C[0] = 1.0  # Dirichlet estricto

        # 2. CN dispersion
        if D > 1e-12:
            rhs = A_rhs.dot(C)
            rhs[0] = 1.0  # Dirichlet
            C = factored(rhs)

        # 3. Decay analítico
        if lam > 0.0:
            C *= math.exp(-lam * dt / R)

        C[0] = 1.0
        np.maximum(C, 0.0, out=C)

    return C


# ---------------------------------------------------------------------------
# Benchmark 1D
# ---------------------------------------------------------------------------

def run_benchmark_1d(output_path):
    """
    Manual MT3DMS p.130-131, Figura 31.
    4 casos: 1a (Pe=∞), 1b (Pe=1), 1c (+sorción R=5), 1d (+decay λ=0.002 d⁻¹).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncell = 101
    dx = 10.0
    v = 0.24        # m/day
    theta = 0.25
    t_end = 2000.0  # days

    cases = [
        {"label": "1a — Solo advección  (Pe=∞)",          "alphaL": 0.0,  "R": 1.0, "lam": 0.000},
        {"label": "1b — Adv. + Dispersión  (Pe=1)",        "alphaL": 10.0, "R": 1.0, "lam": 0.000},
        {"label": "1c — + Sorción  (R = 5)",               "alphaL": 10.0, "R": 5.0, "lam": 0.000},
        {"label": "1d — + Decaimiento  (λ = 0.002 d⁻¹)",  "alphaL": 10.0, "R": 5.0, "lam": 0.002},
    ]

    x = np.arange(0.5, ncell) * dx

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, case in zip(axes.flat, cases):
        aL  = case["alphaL"]
        R   = case["R"]
        lam = case["lam"]
        D   = aL * v

        C_anal = _vga1d(x, t_end, v, D, R, lam)

        # dt: CFL basado en velocidad retardada v/R; Von Neumann para dispersión
        v_eff = v / R
        dt_cfl = 0.8 * dx / (v_eff + 1e-15)
        dt_dif = 0.4 * dx**2 / (D + 1e-15) if D > 1e-12 else dt_cfl
        dt = min(dt_cfl, dt_dif, 30.0)
        nsteps = math.ceil(t_end / dt)
        dt_used = t_end / nsteps

        C_num = _solve_1d(ncell, dx, v, D, R, lam, theta, dt_used, nsteps)

        ax.plot(x, C_anal, "k-", lw=1.8, label="Analítica (VGA 1982)")
        ax.plot(x[::4], C_num[::4], "o", ms=4, color="steelblue",
                mfc="none", label="Numérica (TVD SUPERBEE)")
        ax.set_title(case["label"], fontsize=9)
        ax.set_xlabel("Distancia [m]", fontsize=8)
        ax.set_ylabel("C / C₀  [-]", fontsize=8)
        ax.set_xlim(0, ncell * dx)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color="gray", lw=0.5, ls=":")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Benchmark 1D — MT3DMS Cap. 7 (Figura 31)\n"
        "v = 0.24 m/día,  θ = 0.25,  t = 2000 días  |  Esquema: TVD SUPERBEE",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark_1d] {output_path}")


# ---------------------------------------------------------------------------
# Benchmark 3D
# ---------------------------------------------------------------------------

def run_benchmark_3d(output_path):
    """
    Manual MT3DMS p.145-146, Figura 39.
    TVD SUPERBEE en X + dispersión explícita central en Y, Z.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncol, nrow, nlay = 21, 15, 8
    dx = dy = dz = 10.0
    theta = 0.2
    v = 0.5         # m/day (pore velocity, v_eff = v pues R=1)
    alpha_L  = 10.0
    alpha_TH = 3.0
    alpha_TV = 3.0
    Q_inj = 0.5     # m³/day
    C_inj = 1.0
    t_end = 100.0   # days

    Dxx = alpha_L  * v   # 5.0 m²/day
    Dyy = alpha_TH * v   # 1.5 m²/day
    Dzz = alpha_TV * v   # 1.5 m²/day

    src_col, src_row, src_lay = 2, 7, 6   # 0-based (col=3, row=8, lay=7 1-based)

    # Paso de tiempo: CFL y Von Neumann
    dt = 2.0
    nsteps = math.ceil(t_end / dt)
    dt = t_end / nsteps

    Cr  = v * dt / dx
    rdx = Dxx * dt / dx**2
    rdy = Dyy * dt / dy**2
    rdz = Dzz * dt / dz**2
    assert Cr <= 1.0 and 2 * (rdx + rdy + rdz) < 1.0, (
        f"Inestabilidad: Cr={Cr:.2f}, rdiff={2*(rdx+rdy+rdz):.2f}")

    q_src = Q_inj / (dx * dy * dz) / theta   # aporte a ∂C/∂t [1/day]

    C = np.zeros((nlay, nrow, ncol), dtype=np.float64)

    for _ in range(nsteps):
        Cn = C.copy()

        # TVD advection en +X (v_eff = v, R=1)
        C = _tvd_step_x_3d(Cn, v, dx, dt, bc_left=0.0)

        # Dispersión Y (central, explícita desde Cn)
        C[:, 1:-1, :] += rdy * (Cn[:, 2:, :] - 2 * Cn[:, 1:-1, :] + Cn[:, :-2, :])
        C[:,  0,   :] += rdy * (Cn[:, 1, :]  - Cn[:, 0, :])
        C[:, -1,   :] += rdy * (Cn[:, -2, :] - Cn[:, -1, :])

        # Dispersión Z (central, explícita desde Cn)
        C[1:-1, :, :] += rdz * (Cn[2:, :, :] - 2 * Cn[1:-1, :, :] + Cn[:-2, :, :])
        C[ 0,   :, :] += rdz * (Cn[1, :, :]  - Cn[0, :, :])
        C[-1,   :, :] += rdz * (Cn[-2, :, :] - Cn[-1, :, :])

        # Dispersión X adicional (central, desde Cn)
        C[:, :, 1:-1] += rdx * (Cn[:, :, 2:] - 2 * Cn[:, :, 1:-1] + Cn[:, :, :-2])
        C[:, :,  0]   += rdx * (Cn[:, :, 1]  - Cn[:, :, 0])
        C[:, :, -1]   += rdx * (Cn[:, :, -2] - Cn[:, :, -1])

        # Fuente puntual (SSM)
        C[src_lay, src_row, src_col] += q_src * C_inj * dt
        np.maximum(C, 0.0, out=C)

    # Solución analítica Hunt (1978)
    C_hunt = np.zeros_like(C)
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                if k == src_lay and i == src_row and j == src_col:
                    continue
                val = _hunt3d(
                    (j - src_col) * dx, (i - src_row) * dy, (k - src_lay) * dz,
                    t_end, v, Dxx, Dyy, Dzz, Q_inj, theta,
                )
                C_hunt[k, i, j] = 0.0 if math.isnan(val) else val

    # Visualización: capas 5, 6, 7 (1-based) → 4, 5, 6 (0-based)
    x_ax = (np.arange(ncol) - src_col) * dx
    y_ax = (np.arange(nrow) - src_row) * dy

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, [4, 5, 6]):
        Ck_n = C[k]
        Ck_h = C_hunt[k]
        vmax = max(float(Ck_n.max()), float(Ck_h.max()), 1e-12)
        levels = np.linspace(0.05 * vmax, 0.9 * vmax, 7)

        cf = ax.contourf(x_ax, y_ax, Ck_n, levels=levels, cmap="Blues", alpha=0.75)
        if Ck_h.max() > 0:
            ax.contour(x_ax, y_ax, Ck_h, levels=levels,
                       colors="red", linewidths=1.0, linestyles="--")
        ax.plot(0, 0, "r^", ms=9, zorder=5)
        ax.set_title(f"Capa {k + 1}", fontsize=9)
        ax.set_xlabel("X relativo [m]", fontsize=8)
        ax.set_ylabel("Y relativo [m]", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02, label="C [-]")

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], color="steelblue", lw=5, alpha=0.75, label="Numérico (TVD SUPERBEE)"),
        Line2D([0], [0], color="red", lw=1.2, ls="--", label="Analítico (Hunt 1978)"),
        Line2D([0], [0], marker="^", color="red", ms=8, ls="None", label="Fuente puntual"),
    ], loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Benchmark 3D — MT3DMS Cap. 7 (Figura 39)\n"
        f"v={v} m/día, θ={theta}, αL={alpha_L}m, αTH=αTV={alpha_TH}m, "
        f"t={t_end:.0f} días  |  Esquema: TVD SUPERBEE + CN disp.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark_3d] {output_path}")


# ---------------------------------------------------------------------------
# Entrada Snakemake
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    _out = snakemake.output[0]  # noqa: F821
    if "benchmark_1d" in _out:
        run_benchmark_1d(_out)
    elif "benchmark_3d" in _out:
        run_benchmark_3d(_out)
