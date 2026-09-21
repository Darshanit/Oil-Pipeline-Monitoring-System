"""
Unit tests for PipelineConfig and Centralized Configuration.
"""

import pytest
from dataclasses import FrozenInstanceError
from src.config import PipelineConfig, DEFAULT_CONFIG


def test_default_config_values():
    cfg = DEFAULT_CONFIG
    assert cfg.pipeline_length_km == 100.0
    assert len(cfg.station_positions_km) == 6
    assert cfg.wave_speed_km_s == 1.0
    assert cfg.sampling_rate_hz == 10.0
    assert cfg.dt_sec == 0.1

    # Weights (ML, FLOW, NPW, PRESSURE)
    assert sum(cfg.fusion_weights.values()) == pytest.approx(1.0)
    assert "ml" in cfg.fusion_weights
    assert "flow" in cfg.fusion_weights
    assert "npw" in cfg.fusion_weights
    assert "pressure" in cfg.fusion_weights

    # Phase 4 Persistence & Hysteresis
    assert cfg.warning_persistence_n == 15
    assert cfg.critical_persistence_m == 25
    assert cfg.warning_threshold == 50.0
    assert cfg.critical_threshold == 75.0
    assert cfg.hysteresis_warning_recovery < cfg.warning_threshold
    assert cfg.hysteresis_critical_recovery < cfg.critical_threshold
    assert cfg.cooldown_steps > 0
    assert len(cfg.signal_thresholds) >= 4
    assert cfg.min_agreement_signals == 2

    # Persistence & Hysteresis (backward compatibility)
    assert cfg.persistence_n_samples == 40
    assert cfg.persistence_m_triggers == 25
    assert cfg.leak_confidence_threshold == 0.65
    assert cfg.suspect_confidence_threshold == 0.35
    assert cfg.hysteresis_leak_recovery < cfg.leak_confidence_threshold
    assert cfg.hysteresis_suspect_recovery < cfg.suspect_confidence_threshold

    # Linepack & NPW
    assert cfg.default_linepack_coeff == 15.0
    assert cfg.npw_dp_dt_threshold == -1.20
    assert cfg.spatial_quantization_km == 0.05  # 50 meters


def test_config_immutability():
    cfg = DEFAULT_CONFIG
    with pytest.raises(FrozenInstanceError):
        cfg.wave_speed_km_s = 2.0  # type: ignore


def test_config_validation_weights_sum():
    with pytest.raises(ValueError, match="Fusion weights must sum to 1.0"):
        PipelineConfig(fusion_weights={"ml": 0.5, "flow": 0.2, "npw": 0.1})


def test_config_validation_persistence():
    with pytest.raises(ValueError, match="persistence_m_triggers .* cannot exceed persistence_n_samples"):
        PipelineConfig(persistence_n_samples=20, persistence_m_triggers=25)


def test_config_validation_hysteresis():
    with pytest.raises(ValueError, match="hysteresis_leak_recovery must be less than leak_confidence_threshold"):
        PipelineConfig(leak_confidence_threshold=0.60, hysteresis_leak_recovery=0.65)

    with pytest.raises(ValueError, match="hysteresis_suspect_recovery must be less than suspect_confidence_threshold"):
        PipelineConfig(suspect_confidence_threshold=0.30, hysteresis_suspect_recovery=0.35)

