"""
Piezometric Level Interpolation (Phase 1)

Interpolates water table elevations and depths from piezometric observations
(wells, test pits) to a regular grid using IDW and Ordinary Kriging.

Typical use: real-estate lot subdivisions (e.g. Los Portales) where the
question is "at what depth is the water table across this plot?"

Methodology references:
    - Shepard, D. (1968). A two-dimensional interpolation function for
      irregularly-spaced data.
    - Kitanidis, P.K. (1997). Introduction to Geostatistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PiezometricObservation:
    """A single piezometric observation.

    Attributes:
        well_id: identifier of the well/test pit.
        x, y: projected coordinates (UTM, meters).
        ground_elev: ground surface elevation (m.a.s.l.).
        water_depth: depth to water table below ground (m, positive downward).
    """

    well_id: str
    x: float
    y: float
    ground_elev: float
    water_depth: float

    @property
    def water_elev(self) -> float:
        """Water table elevation (m.a.s.l.)."""
        return self.ground_elev - self.water_depth


class PiezometricInterpolator:
    """Interpolate piezometric levels to a regular grid."""

    def __init__(
        self,
        observations: Sequence[PiezometricObservation],
        cell_size: float = 5.0,
    ) -> None:
        if len(observations) < 1:
            raise ValueError("At least one piezometric observation is required.")
        self.observations = list(observations)
        self.cell_size = float(cell_size)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        cell_size: float = 5.0,
        columns: Optional[dict] = None,
    ) -> "PiezometricInterpolator":
        """Build an interpolator from a DataFrame.

        Expected columns (override via ``columns`` mapping):
            well_id, x, y, ground_elev, water_depth
        """
        cols = {
            "well_id": "well_id",
            "x": "x",
            "y": "y",
            "ground_elev": "ground_elev",
            "water_depth": "water_depth",
        }
        if columns:
            cols.update(columns)

        observations = [
            PiezometricObservation(
                well_id=str(row[cols["well_id"]]),
                x=float(row[cols["x"]]),
                y=float(row[cols["y"]]),
                ground_elev=float(row[cols["ground_elev"]]),
                water_depth=float(row[cols["water_depth"]]),
            )
            for _, row in df.iterrows()
        ]
        return cls(observations, cell_size=cell_size)

    def _xy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([o.x for o in self.observations], dtype=float)
        y = np.array([o.y for o in self.observations], dtype=float)
        h = np.array([o.water_elev for o in self.observations], dtype=float)
        return x, y, h

    def build_grid(self, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Construct a regular grid covering all observations."""
        x, y, _ = self._xy_arrays()
        x_min, x_max = x.min() - padding, x.max() + padding
        y_min, y_max = y.min() - padding, y.max() + padding

        if x_min == x_max:
            x_max = x_min + self.cell_size
        if y_min == y_max:
            y_max = y_min + self.cell_size

        xs = np.arange(x_min, x_max + self.cell_size, self.cell_size)
        ys = np.arange(y_min, y_max + self.cell_size, self.cell_size)
        return xs, ys

    def idw(
        self,
        power: float = 2.0,
        padding: float = 0.0,
        smoothing: float = 1e-9,
    ) -> dict:
        """Inverse Distance Weighting interpolation of water table elevation.

        Args:
            power: IDW exponent (typically 2).
            padding: extra meters around the data bounding box.
            smoothing: added to distances to avoid singularity at data points.

        Returns:
            dict with keys: ``xs``, ``ys``, ``water_elev`` (2D grid).
        """
        xs, ys = self.build_grid(padding=padding)
        x_obs, y_obs, h_obs = self._xy_arrays()

        xx, yy = np.meshgrid(xs, ys)
        grid = np.empty_like(xx, dtype=float)

        for j in range(xx.shape[0]):
            for i in range(xx.shape[1]):
                dx = xx[j, i] - x_obs
                dy = yy[j, i] - y_obs
                dist = np.sqrt(dx * dx + dy * dy) + smoothing

                if np.any(dist < 1e-6):
                    grid[j, i] = h_obs[np.argmin(dist)]
                    continue

                weights = 1.0 / np.power(dist, power)
                grid[j, i] = np.sum(weights * h_obs) / np.sum(weights)

        return {"xs": xs, "ys": ys, "water_elev": grid}

    def ordinary_kriging(
        self,
        variogram: str = "spherical",
        range_: Optional[float] = None,
        sill: Optional[float] = None,
        nugget: float = 0.0,
        padding: float = 0.0,
    ) -> dict:
        """Ordinary Kriging with a simple analytic variogram model.

        Kept dependency-free (no ``pykrige``) by solving the Kriging
        system directly. Sufficient for small piezometric datasets
        (< ~200 wells).

        Args:
            variogram: ``"spherical"``, ``"exponential"``, or ``"gaussian"``.
            range_: variogram range ``a`` (meters). Defaults to half the
                longest distance between observations.
            sill: variogram sill. Defaults to sample variance.
            nugget: variogram nugget.
        """
        xs, ys = self.build_grid(padding=padding)
        x_obs, y_obs, h_obs = self._xy_arrays()
        n = len(h_obs)

        if range_ is None:
            dx = x_obs[:, None] - x_obs[None, :]
            dy = y_obs[:, None] - y_obs[None, :]
            range_ = float(np.sqrt(dx * dx + dy * dy).max() / 2.0 or 1.0)
        if sill is None:
            sill = float(np.var(h_obs)) if n > 1 else 1.0
            if sill == 0.0:
                sill = 1.0

        gamma = _variogram_fn(variogram, range_, sill, nugget)

        # Build (n+1) x (n+1) Kriging system with Lagrange multiplier.
        A = np.zeros((n + 1, n + 1), dtype=float)
        for i in range(n):
            for j in range(n):
                d = math.hypot(x_obs[i] - x_obs[j], y_obs[i] - y_obs[j])
                A[i, j] = gamma(d)
        A[n, :n] = 1.0
        A[:n, n] = 1.0
        A[n, n] = 0.0

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # Fallback to IDW if the Kriging matrix is singular
            return self.idw(padding=padding)

        xx, yy = np.meshgrid(xs, ys)
        grid = np.empty_like(xx, dtype=float)
        variance = np.empty_like(xx, dtype=float)

        b = np.zeros(n + 1, dtype=float)
        b[n] = 1.0
        for jj in range(xx.shape[0]):
            for ii in range(xx.shape[1]):
                for k in range(n):
                    d = math.hypot(xx[jj, ii] - x_obs[k], yy[jj, ii] - y_obs[k])
                    b[k] = gamma(d)
                w = A_inv @ b
                grid[jj, ii] = float(np.dot(w[:n], h_obs))
                variance[jj, ii] = float(np.dot(w[:n], b[:n]) + w[n])

        return {
            "xs": xs,
            "ys": ys,
            "water_elev": grid,
            "variance": variance,
        }

    def depth_to_water(
        self,
        water_elev_grid: np.ndarray,
        ground_elev_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute depth-to-water (m) from ground and water elevation grids."""
        if water_elev_grid.shape != ground_elev_grid.shape:
            raise ValueError(
                "Ground elevation grid must match the water elevation grid shape."
            )
        return ground_elev_grid - water_elev_grid

    def flat_ground_depth(self, water_elev_grid: np.ndarray) -> np.ndarray:
        """Depth-to-water assuming flat ground equal to the mean observed elevation.

        Useful for quick-look analyses where no DEM is available.
        """
        mean_ground = float(np.mean([o.ground_elev for o in self.observations]))
        return mean_ground - water_elev_grid


def _variogram_fn(model: str, range_: float, sill: float, nugget: float):
    model = model.lower()
    if model == "spherical":
        def gamma(h: float) -> float:
            if h <= 0.0:
                return 0.0
            if h >= range_:
                return nugget + sill
            ratio = h / range_
            return nugget + sill * (1.5 * ratio - 0.5 * ratio ** 3)
    elif model == "exponential":
        def gamma(h: float) -> float:
            if h <= 0.0:
                return 0.0
            return nugget + sill * (1.0 - math.exp(-3.0 * h / range_))
    elif model == "gaussian":
        def gamma(h: float) -> float:
            if h <= 0.0:
                return 0.0
            return nugget + sill * (1.0 - math.exp(-3.0 * (h / range_) ** 2))
    else:
        raise ValueError(f"Unknown variogram model: {model!r}")
    return gamma


def contour_levels(grid: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Generate isohypse contour levels at a given step (meters)."""
    g_min = float(np.nanmin(grid))
    g_max = float(np.nanmax(grid))
    start = math.floor(g_min / step) * step
    stop = math.ceil(g_max / step) * step
    if stop <= start:
        stop = start + step
    return np.arange(start, stop + step, step)
