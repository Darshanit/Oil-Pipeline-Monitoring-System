"""
Feature engineering subpackage.

Provides specialized modules for pressure and flow feature extraction.
"""

from src.features.pressure_features import (
    compute_dp_dt,
    compute_rolling_variance,
    compute_station_differentials,
    compute_hydraulic_gradients,
)
from src.features.flow_features import (
    compute_raw_flow_imbalance,
    compute_linepack_rate,
    compute_corrected_flow_imbalance,
)

__all__ = [
    "compute_dp_dt",
    "compute_rolling_variance",
    "compute_station_differentials",
    "compute_hydraulic_gradients",
    "compute_raw_flow_imbalance",
    "compute_linepack_rate",
    "compute_corrected_flow_imbalance",
]

