"""
Detection subpackage.

Aggregates ML anomaly detection, flow evidence, NPW detection,
pressure evidence, evidence fusion, and localization.
"""

from src.detection.isolation_forest import ModeIsolationForestDetector, FEATURE_COLS
from src.detection.flow_evidence import calculate_flow_evidence_score
from src.detection.npw import detect_negative_pressure_waves
from src.detection.pressure_evidence import calculate_pressure_evidence_score
from src.detection.localization import (
    localize_leak_from_npw,
    localize_leak_from_hydraulic_drop,
    estimate_leak_location
)
from src.fusion.fusion import fuse_evidence_signals, get_recommended_action
from src.detection.mode_gating import OperatingModeGating, apply_operating_mode_gating

__all__ = [
    "ModeIsolationForestDetector",
    "FEATURE_COLS",
    "calculate_flow_evidence_score",
    "detect_negative_pressure_waves",
    "calculate_pressure_evidence_score",
    "localize_leak_from_npw",
    "localize_leak_from_hydraulic_drop",
    "estimate_leak_location",
    "fuse_evidence_signals",
    "get_recommended_action",
    "OperatingModeGating",
    "apply_operating_mode_gating",
]
