"""
WaterKu Napa Freatica (Groundwater/Water Table) Module

Phase 1: Piezometric interpolation and water table mapping (IDW, kriging)
Phase 2: Analytical groundwater calculations
    - Dupuit-Forchheimer aquifer flow
"""

from .interpolacion_niveles import PiezometricInterpolator
from .dupuit_forchheimer import DupuitForchheimer

__all__ = [
    "PiezometricInterpolator",
    "DupuitForchheimer",
]
