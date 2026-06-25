"""
WaterKu Napa Freatica (Groundwater/Water Table) Module

Phase 1: Piezometric interpolation and water table mapping (IDW, kriging)
Phase 2: Analytical groundwater calculations
    - Dupuit-Forchheimer aquifer flow
    - Hooghoudt / Ernst drain spacing for water table drawdown
    - Pumping tests: Theis, Cooper-Jacob, Thiem
"""

from .interpolacion_niveles import PiezometricInterpolator
from .dupuit_forchheimer import DupuitForchheimer
from .drenaje_subterraneo import DrainSpacingDesign
from .pruebas_bombeo import PumpingTestAnalysis

__all__ = [
    "PiezometricInterpolator",
    "DupuitForchheimer",
    "DrainSpacingDesign",
    "PumpingTestAnalysis",
]
