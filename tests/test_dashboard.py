"""
Tests for Phase 7: Rebuilt Dashboard UI components and Data Contract Integration.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.contract import DetectionResult, AlertState
from dashboard.app import (
    render_pipeline_svg,
    evaluate_system_health,
    build_detection_result
)


@pytest.fixture
def sample_detection_result():
    """Provides a sample DetectionResult with a leak event."""
    pressures = {f"P_st{i+1}": 55.0 - i * 6.0 for i in range(6)}
    contributions = {"ml": 25.0, "flow": 30.0, "npw": 15.0, "pressure": 18.0}
    return DetectionResult(
        timestamp=45.0,
        mode="Steady-Flowing",
        pressures=pressures,
        flow_in=500.0,
        flow_out=470.0,
        flow_imbalance_raw=30.0,
        flow_imbalance_corrected=28.5,
        anomaly_score=0.85,
        flow_evidence=0.90,
        npw_evidence=1.0,
        pressure_evidence=0.80,
        fusion_confidence=88.0,
        contributions=contributions,
        alert_state=AlertState.CRITICAL.value,
        persistence=6,
        leak_km=78.2,
        segment="SEGMENT 4 (S4-S5: 60-80 km)",
        is_estimate=True,
        alert_duration=12.5,
        attention_required=True,
        suppression_reason=None
    )


def test_render_pipeline_svg_structure(sample_detection_result):
    """Verifies that render_pipeline_svg produces valid SVG with all required stations and telemetry."""
    svg_html = render_pipeline_svg(sample_detection_result, gt_km=78.0)
    
    assert "<svg" in svg_html
    assert "</svg>" in svg_html
    assert "pipeline-svg-container" in svg_html
    
    # Check all stations S1-S6 are present
    for i in range(1, 7):
        assert f"S{i}" in svg_html
        p_val = sample_detection_result.pressures[f"P_st{i}"]
        assert f"{p_val:.1f} bar" in svg_html

    # Check all 5 segments are marked
    for seg_idx in range(1, 6):
        assert f"SEG {seg_idx}" in svg_html

    # Check estimated leak and actual leak beacons
    assert "EST: 78.2 KM" in svg_html
    assert "ACTUAL: 78.0 KM" in svg_html


def test_build_detection_result_mapping():
    """Verifies build_detection_result properly maps row values into DetectionResult contract."""
    timestamps = [10.0, 11.0, 12.0]
    data = {
        "timestamp": timestamps,
        "operating_mode": ["Steady-Flowing", "Steady-Flowing", "Steady-Flowing"],
        "P_st1": [60.0, 60.0, 58.0],
        "P_st2": [52.0, 52.0, 50.0],
        "P_st3": [45.0, 45.0, 43.0],
        "P_st4": [38.0, 38.0, 36.0],
        "P_st5": [30.0, 30.0, 26.0],
        "P_st6": [22.0, 22.0, 20.0],
        "flow_in": [500.0, 500.0, 500.0],
        "flow_out": [498.0, 498.0, 460.0],
        "corrected_flow_imbalance": [0.5, 0.5, 38.0],
        "ml_evidence_score": [0.05, 0.05, 0.82],
        "flow_evidence_score": [0.02, 0.02, 0.95],
        "npw_evidence_score": [0.0, 0.0, 1.0],
        "pressure_evidence_score": [0.04, 0.04, 0.78],
        "confidence_pct": [8.0, 8.0, 86.5],
        "status": ["NORMAL", "NORMAL", "LEAK DETECTED"],
        "persistence_count": [0, 0, 3],
        "estimated_leak_km": [np.nan, np.nan, 78.4],
        "ground_truth_km": [np.nan, np.nan, 78.0]
    }
    df = pd.DataFrame(data)
    weights = {"ml": 0.30, "flow": 0.35, "npw": 0.15, "pressure": 0.20}

    curr_row = df.iloc[-1]
    res = build_detection_result(curr_row, df, weights=weights)

    assert isinstance(res, DetectionResult)
    assert res.alert_state == AlertState.CRITICAL.value
    assert res.timestamp == 12.0
    assert res.fusion_confidence == 86.5
    assert res.leak_km == 78.4
    assert res.segment == "SEGMENT 4 (S4-S5: 60-80 km)"
    assert res.persistence == 3
    assert "ml" in res.contributions
    assert "flow" in res.contributions
    assert "npw" in res.contributions
    assert "pressure" in res.contributions


def test_evaluate_system_health(sample_detection_result):
    """Verifies that evaluate_system_health runs real checks on all 6 components."""
    weights = {"ml": 0.30, "flow": 0.35, "npw": 0.15, "pressure": 0.20}
    health = evaluate_system_health(sample_detection_result, weights)

    required_checks = [
        "ML MODEL",
        "PRESSURE DATA",
        "FLOW DATA",
        "NPW DETECTOR",
        "FUSION ENGINE",
        "EDGE RUNTIME"
    ]

    for comp in required_checks:
        assert comp in health
        assert health[comp]["ok"] is True
        assert len(health[comp]["details"]) > 0


def test_mode_gating_monitoring_state():
    """Verifies that transient operating modes are mapped to MONITORING state."""
    df = pd.DataFrame({
        "timestamp": [20.0],
        "operating_mode": ["Transient_Operation"],
        "status": ["NORMAL"],
        "confidence_pct": [15.0],
        "P_st1": [60.0], "P_st2": [52.0], "P_st3": [45.0],
        "P_st4": [38.0], "P_st5": [30.0], "P_st6": [22.0],
        "flow_in": [500.0], "flow_out": [500.0],
        "corrected_flow_imbalance": [0.0],
        "ml_evidence_score": [0.1],
        "flow_evidence_score": [0.0],
        "npw_evidence_score": [0.0],
        "pressure_evidence_score": [0.0]
    })
    weights = {"ml": 0.30, "flow": 0.35, "npw": 0.15, "pressure": 0.20}
    res = build_detection_result(df.iloc[0], df, weights=weights)

    assert res.alert_state == AlertState.MONITORING.value


def test_compute_npw_replay_data():
    """Verifies acoustic wave arrival times, delta arrival times, and localization formula."""
    from dashboard.app import compute_npw_replay_data

    # Test leak at 78.0 km
    npw_info = compute_npw_replay_data(leak_km=78.0, wave_speed_km_s=1.0)

    assert npw_info["leak_km"] == 78.0
    assert npw_info["wave_speed"] == 1.0

    # Check arrival times at S4 (60 km, dist=18 km) and S5 (80 km, dist=2 km)
    assert npw_info["arrivals"]["S4"]["t_arr"] == 18.0
    assert npw_info["arrivals"]["S5"]["t_arr"] == 2.0
    assert npw_info["pair"] == ("S4", "S5")
    assert npw_info["delta_t"] == -16.0
    assert npw_info["calculated_km"] == 78.0
    assert "x_L =" in npw_info["formula"]


def test_get_presenter_commentary():
    """Verifies that rich engineering commentary is generated for all scenarios."""
    from dashboard.app import get_presenter_commentary

    scenarios = [
        "1. NORMAL",
        "2. SMALL CHRONIC LEAK",
        "3. LARGE LEAK",
        "4. PUMP TRANSIENT",
        "5. VALVE EVENT"
    ]

    for sc in scenarios:
        comment_early = get_presenter_commentary(sc, t=20.0)
        comment_late = get_presenter_commentary(sc, t=55.0)
        assert len(comment_early) > 20
        assert len(comment_late) > 20
        assert "T=" in comment_early


def test_pipeline_flow_motion_classes(sample_detection_result):
    """Verifies that render_pipeline_svg selects proper flow animation classes according to flow rate and mode."""
    # 1. Normal flowing (500 m3/h)
    svg_flowing = render_pipeline_svg(sample_detection_result, motion_mode="FULL")
    assert "flow-flowing" in svg_flowing

    # 2. Ramping flow (>550 m3/h)
    ramping_res = DetectionResult(
        timestamp=sample_detection_result.timestamp,
        mode="Ramping_Operation",
        pressures=sample_detection_result.pressures,
        flow_in=620.0,
        flow_out=600.0,
        flow_imbalance_raw=20.0,
        flow_imbalance_corrected=10.0,
        anomaly_score=0.2,
        flow_evidence=0.1,
        npw_evidence=0.0,
        pressure_evidence=0.1,
        fusion_confidence=15.0,
        contributions={},
        alert_state=AlertState.MONITORING.value,
        persistence=0
    )
    svg_ramping = render_pipeline_svg(ramping_res, motion_mode="FULL")
    assert "flow-ramping" in svg_ramping

    # 3. Shut-in mode (0 flow)
    shutin_res = DetectionResult(
        timestamp=sample_detection_result.timestamp,
        mode="SHUT-IN",
        pressures=sample_detection_result.pressures,
        flow_in=0.0,
        flow_out=0.0,
        flow_imbalance_raw=0.0,
        flow_imbalance_corrected=0.0,
        anomaly_score=0.05,
        flow_evidence=0.0,
        npw_evidence=0.0,
        pressure_evidence=0.0,
        fusion_confidence=5.0,
        contributions={},
        alert_state=AlertState.MONITORING.value,
        persistence=0
    )
    svg_shutin = render_pipeline_svg(shutin_res, motion_mode="FULL")
    assert "flow-shutin" in svg_shutin

    # 4. Motion OFF mode
    svg_off = render_pipeline_svg(sample_detection_result, motion_mode="OFF")
    assert "flow-shutin" in svg_off

    # 5. NPW Replay ripples
    svg_npw = render_pipeline_svg(sample_detection_result, motion_mode="FULL", npw_replay_active=True)
    assert "acoustic-ripple" in svg_npw
