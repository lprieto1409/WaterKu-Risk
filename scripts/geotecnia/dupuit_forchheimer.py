"""
Dupuit-Forchheimer steady-state flow in unconfined aquifers.

Assumptions:
    - Horizontal flow (Dupuit assumption).
    - Homogeneous, isotropic aquifer.
    - Steady-state conditions.

Equations:
    Unconfined, no recharge:
        q = (K * (h1^2 - h2^2)) / (2 * L)            (per unit width)
    Unconfined with uniform recharge N (m/s):
        h(x)^2 = h1^2 + ((h2^2 - h1^2) / L) * x + (N / K) * x * (L - x)
        Water divide: xd = L/2 - (K * (h1^2 - h2^2)) / (2 * N * L)

Reference:
    Bear, J. (1979). Hydraulics of Groundwater.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DupuitForchheimerResult:
    x: np.ndarray            # positions along the flow line (m)
    h: np.ndarray            # water table elevation above the impervious base (m)
    q: float                 # discharge per unit width (m^2/s)
    water_divide: Optional[float]  # location of water divide (m), None if N==0


class DupuitForchheimer:
    """Steady-state unconfined flow between two fixed-head boundaries."""

    def __init__(
        self,
        hydraulic_conductivity: float,
        length: float,
        h_upstream: float,
        h_downstream: float,
        recharge: float = 0.0,
    ) -> None:
        """
        Args:
            hydraulic_conductivity: K in m/s.
            length: distance L between boundaries (m).
            h_upstream: head h1 at x=0 above the impervious base (m).
            h_downstream: head h2 at x=L above the impervious base (m).
            recharge: uniform recharge N in m/s (default 0).
        """
        if hydraulic_conductivity <= 0:
            raise ValueError("hydraulic_conductivity must be positive.")
        if length <= 0:
            raise ValueError("length must be positive.")
        if h_upstream < 0 or h_downstream < 0:
            raise ValueError("Heads must be non-negative.")

        self.K = float(hydraulic_conductivity)
        self.L = float(length)
        self.h1 = float(h_upstream)
        self.h2 = float(h_downstream)
        self.N = float(recharge)

    def profile(self, n_points: int = 101) -> DupuitForchheimerResult:
        """Compute the water-table profile h(x) and discharge per unit width."""
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")
        x = np.linspace(0.0, self.L, n_points)

        h1_sq, h2_sq = self.h1 ** 2, self.h2 ** 2

        if self.N == 0.0:
            h_sq = h1_sq + ((h2_sq - h1_sq) / self.L) * x
            q = self.K * (h1_sq - h2_sq) / (2.0 * self.L)
            divide = None
        else:
            h_sq = (
                h1_sq
                + ((h2_sq - h1_sq) / self.L) * x
                + (self.N / self.K) * x * (self.L - x)
            )
            q_at_x = self.K * (h1_sq - h2_sq) / (2.0 * self.L) - self.N * (
                x - self.L / 2.0
            )
            q = float(q_at_x[0])
            divide = float(
                self.L / 2.0
                - (self.K * (h1_sq - h2_sq)) / (2.0 * self.N * self.L)
            )
            if divide < 0 or divide > self.L:
                divide = None

        h_sq = np.clip(h_sq, 0.0, None)
        h = np.sqrt(h_sq)
        return DupuitForchheimerResult(x=x, h=h, q=float(q), water_divide=divide)

    def discharge_per_unit_width(self) -> float:
        """Return q (m^2/s) without computing the full profile (N=0 only)."""
        if self.N != 0.0:
            return self.profile().q
        return self.K * (self.h1 ** 2 - self.h2 ** 2) / (2.0 * self.L)

    def max_water_table(self) -> float:
        """Maximum water-table elevation along the profile."""
        return float(self.profile().h.max())
