"""
Centralized Configuration Module for Oil Pipeline Edge AI.

Defines thresholds, fusion weights, persistence criteria, hysteresis margins,
line-pack dynamics, acoustic NPW wave speed, and physical resolution bounds.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration for pipeline monitoring, anomaly detection,
    evidence fusion, and localization.
    """
    # Pipeline Geometry
    pipeline_length_km: float = 100.0
    station_positions_km: Tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    sampling_rate_hz: float = 10.0
    dt_sec: float = 0.1  # 1 / sampling_rate_hz

    # Acoustic Wave Propagation (NPW)
    wave_speed_km_s: float = 1.0  # 1000 m/s in crude oil / petroleum
    npw_dp_dt_threshold: float = -1.20  # bar/s drop rate to trigger NPW front
    npw_smoothing_window: int = 5  # 5-sample centered rolling average (0.5s)
    npw_min_event_separation_sec: float = 30.0  # cooldown between distinct NPW events
    npw_evidence_duration_sec: float = 25.0  # duration to maintain elevated NPW evidence

    # Physical Resolution & Accuracy Limits
    # Quantization step = (wave_speed * dt) / 2 = 1.0 km/s * 0.1s / 2 = 0.05 km (50m)
    spatial_quantization_km: float = 0.05
    # Arrival smearing from 0.5s smoothing filter: +/- (1.0 km/s * 0.25s) / 2 ~= +/- 0.125 to 0.25 km
    filter_temporal_uncertainty_sec: float = 0.5
    empirical_localization_margin_km: float = 2.5

    # Line-Pack Storage Mass Balance
    # Default linepack coefficient K_lp: m3/h per (bar/s)
    default_linepack_coeff: float = 15.0

    # Flow Imbalance Scoring & Persistence
    flow_imbalance_threshold_m3h: float = 8.0
    flow_noise_suppression_m3h: float = 3.0
    flow_sigmoid_steepness: float = 0.15
    flow_rolling_window_samples: int = 20  # 2.0s rolling persistence filter to reject noise spikes

    # Operational Transient Settling & Gating
    transient_settling_sec: float = 25.0  # settling period for pump start/stop transients
    valve_transient_settling_sec: float = 20.0  # settling period for valve maneuvers

    # Pressure Evidence Diagnostics
    pressure_grad_tolerance_bar_per_km: float = 0.008
    pressure_dp_dt_threshold_bar_s: float = -0.40
    pressure_var_threshold_bar2: float = 0.04

    # Isolation Forest ML Parameters
    iforest_contamination: float = 0.03
    iforest_n_estimators: int = 100
    iforest_random_state: int = 42

    # Evidence Fusion Weights (ML, FLOW, NPW, PRESSURE must sum to 1.0)
    fusion_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ml": 0.30,
            "flow": 0.35,
            "npw": 0.15,
            "pressure": 0.20
        }
    )

    # Per-Signal Thresholds for Agreement Rule
    # At least two independent signals must exceed their thresholds before WARNING or CRITICAL
    signal_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "ml": 0.55,
            "flow": 0.50,
            "npw": 0.50,
            "pressure": 0.45
        }
    )
    min_agreement_signals: int = 2

    # Phase 4 Alert State Machine Parameters
    warning_persistence_n: int = 15    # N samples required to confirm WARNING
    critical_persistence_m: int = 25   # M samples required to confirm CRITICAL
    monitoring_threshold: float = 25.0  # Fused % threshold for MONITORING
    warning_threshold: float = 50.0     # Fused % threshold for WARNING
    critical_threshold: float = 75.0    # Fused % threshold for CRITICAL
    hysteresis_warning_recovery: float = 38.0   # Fused % drop required to clear WARNING
    hysteresis_critical_recovery: float = 62.0  # Fused % drop required to exit CRITICAL
    cooldown_steps: int = 10            # Steps required in de-escalation cooldown

    # Legacy Persistence Criteria (for backward compatibility)
    persistence_n_samples: int = 40
    persistence_m_triggers: int = 25

    # Legacy Alert Thresholds & Hysteresis (for backward compatibility)
    leak_confidence_threshold: float = 0.65
    suspect_confidence_threshold: float = 0.35
    hysteresis_leak_recovery: float = 0.55
    hysteresis_suspect_recovery: float = 0.25

    def __post_init__(self):
        """Validates configuration parameters."""
        total_weight = sum(self.fusion_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Fusion weights must sum to 1.0, got {total_weight}")

        if self.critical_persistence_m < self.warning_persistence_n:
            # M should generally be >= N, but enforce persistence_m_triggers compatibility
            pass

        if self.persistence_m_triggers > self.persistence_n_samples:
            raise ValueError(
                f"persistence_m_triggers ({self.persistence_m_triggers}) cannot exceed "
                f"persistence_n_samples ({self.persistence_n_samples})"
            )

        if self.hysteresis_leak_recovery >= self.leak_confidence_threshold:
            raise ValueError("hysteresis_leak_recovery must be less than leak_confidence_threshold")

        if self.hysteresis_suspect_recovery >= self.suspect_confidence_threshold:
            raise ValueError("hysteresis_suspect_recovery must be less than suspect_confidence_threshold")

        if self.hysteresis_warning_recovery >= self.warning_threshold:
            raise ValueError("hysteresis_warning_recovery must be less than warning_threshold")

        if self.hysteresis_critical_recovery >= self.critical_threshold:
            raise ValueError("hysteresis_critical_recovery must be less than critical_threshold")


# Default global singleton configuration instance
DEFAULT_CONFIG = PipelineConfig()

