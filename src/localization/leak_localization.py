"""
Leak Localization Module (Backward Compatibility Facade).

Re-exports localization functions from src.detection.localization.
"""

from src.config import DEFAULT_CONFIG
from src.detection.localization import (
    localize_leak_from_npw,
    localize_leak_from_hydraulic_drop,
    estimate_leak_location,
)

STATIONS_KM = list(DEFAULT_CONFIG.station_positions_km)
WAVE_SPEED_KM_S = DEFAULT_CONFIG.wave_speed_km_s

__all__ = [
    "STATIONS_KM",
    "WAVE_SPEED_KM_S",
    "localize_leak_from_npw",
    "localize_leak_from_hydraulic_drop",
    "estimate_leak_location",
]
