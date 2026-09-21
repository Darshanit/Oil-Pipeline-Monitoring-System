"""
Unit tests for Detection Modules (Isolation Forest, Flow Imbalance, NPW, Fusion, Mode Gating).

Tests:
  1. normal operation baseline
  2. flowing operational mode
  3. ramping operational mode with line-pack dynamic correction
  4. shut-in operational mode
  5. small leak detection via flow & pressure evidence
  6. large leak detection via acoustic NPW & multi-sensor fusion
  7. pump transient gating & suppression
  8. valve event gating & suppression
  9. NPW wave front detection, propagation order, wave speed, & prototype limitation notice
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_generation import (
    generate_pipeline_dataset,
    generate_shut_in_dataset,
    generate_pump_transient_dataset,
    generate_valve_event_dataset
)
from src.preprocessing import extract_pipeline_features
from src.contract import AlertState
from src.detection import (
    ModeIsolationForestDetector,
    calculate_flow_evidence_score,
    calculate_pressure_evidence_score,
    detect_negative_pressure_waves,
    fuse_evidence_signals,
    OperatingModeGating,
    apply_operating_mode_gating
)


@pytest.fixture(scope="module")
def sample_features():
    df_raw = generate_pipeline_dataset(duration_sec=600.0, dt_pressure=0.1, random_seed=42)
    return extract_pipeline_features(df_raw)


# ==============================================================================
# EXISTING BASELINE UNIT TESTS (MAINTAINED)
# ==============================================================================

def test_isolation_forest_training_and_bounds(sample_features, tmp_path):
    model_dir = str(tmp_path / "models")
    detector = ModeIsolationForestDetector(model_dir=None, auto_load=False)
    detector.fit(sample_features, model_dir=model_dir)

    scores = detector.predict_anomaly_scores(sample_features)
    assert len(scores) == len(sample_features)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_flow_imbalance_scoring(sample_features):
    scores = calculate_flow_evidence_score(sample_features)
    assert len(scores) == len(sample_features)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

    normal_score = scores[sample_features["event_ground_truth"] == "NORMAL"].mean()
    large_leak_score = scores[sample_features["event_ground_truth"] == "LARGE_LEAK"].mean()
    assert large_leak_score > normal_score


def test_npw_detection_trigger(sample_features):
    scores, events = detect_negative_pressure_waves(sample_features)
    assert len(scores) == len(sample_features)
    assert len(events) >= 1
    assert any(ev["start_time"] >= 520.0 for ev in events)


def test_evidence_fusion_output(sample_features):
    ml = np.full(100, 0.2)
    flow = np.full(100, 0.1)
    npw = np.full(100, 0.0)

    # Simulate leak at sample index 40
    ml[40:] = 0.90
    flow[40:] = 0.85
    npw[40:70] = 0.95

    df_res, _ = fuse_evidence_signals(ml, flow, npw)
    assert "confidence_pct" in df_res.columns
    assert "status" in df_res.columns
    assert df_res["confidence_pct"].iloc[10] < 40.0
    assert df_res["confidence_pct"].iloc[85] > 65.0
    assert df_res["status"].iloc[85] == "LEAK DETECTED"


# ==============================================================================
# PHASE 3 — 9 SPECIFIC OPERATIONAL SCENARIO TESTS
# ==============================================================================

def test_scenario_normal(sample_features):
    """
    1. NORMAL: Under steady-state normal conditions, all detection signals
    (ML anomaly, flow evidence, pressure evidence, NPW) must remain low,
    yielding NORMAL operational alert status without false alarms.
    """
    normal_mask = (sample_features["event_ground_truth"] == "NORMAL") & (sample_features["timestamp"] < 140.0)
    df_normal = sample_features[normal_mask].copy()

    detector = ModeIsolationForestDetector()
    ml_scores = detector.predict_anomaly_scores(df_normal)
    flow_scores = calculate_flow_evidence_score(df_normal)
    pressure_scores = calculate_pressure_evidence_score(df_normal)
    npw_scores, npw_events = detect_negative_pressure_waves(df_normal)

    # Signal verifications: all normalized between 0 and 1
    assert np.all(ml_scores >= 0.0) and np.all(ml_scores <= 1.0)
    assert np.all(flow_scores >= 0.0) and np.all(flow_scores <= 1.0)
    assert np.all(pressure_scores >= 0.0) and np.all(pressure_scores <= 1.0)
    assert np.all(npw_scores == 0.0)
    assert len(npw_events) == 0

    # Mean evidence scores should be very low during baseline normal
    assert ml_scores.mean() < 0.35
    assert flow_scores.mean() < 0.15
    assert pressure_scores.mean() < 0.25

    # Evidence fusion verification
    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores,
        npw_scores=npw_scores,
        pressure_scores=pressure_scores,
        timestamps=df_normal["timestamp"].values,
        operating_modes=df_normal["operating_mode"].values
    )
    assert (df_fused["status"] == AlertState.NORMAL.value).all()
    assert df_fused["confidence_pct"].mean() < 25.0


def test_scenario_flowing(sample_features):
    """
    2. FLOWING: Steady-state flowing pipeline selects the 'Flowing' Isolation Forest
    model (with case-insensitive matching) and produces low normalized anomaly score.
    """
    df_flowing = sample_features[sample_features["operating_mode"] == "Flowing"].iloc[:100].copy()

    detector = ModeIsolationForestDetector()
    assert "Flowing" in detector.models

    # Test with uppercase and title case modes
    df_flowing["operating_mode"] = "FLOWING"
    scores_upper = detector.predict_anomaly_scores(df_flowing)

    df_flowing["operating_mode"] = "Flowing"
    scores_title = detector.predict_anomaly_scores(df_flowing)

    # Scores must be identical regardless of mode string casing
    np.testing.assert_allclose(scores_upper, scores_title)
    assert np.all(scores_upper >= 0.0) and np.all(scores_upper <= 1.0)
    assert scores_upper.mean() < 0.35

    # Single-sample scoring check
    row_dict = df_flowing.iloc[0].to_dict()
    single_score = detector.score_single(row_dict, mode="FLOWING")
    assert 0.0 <= single_score <= 0.35


def test_scenario_ramping(sample_features):
    """
    3. RAMPING: Flow ramp activates the 'Ramping' Isolation Forest model.
    Line-pack dynamic correction accounts for transient fluid accumulation,
    and gating suppresses ramp-induced transient false alarms.
    """
    ramp_mask = sample_features["operating_mode"] == "Ramping"
    df_ramp = sample_features[ramp_mask].copy()

    # Model selection check
    detector = ModeIsolationForestDetector()
    assert "Ramping" in detector.models

    ml_scores = detector.predict_anomaly_scores(df_ramp)
    assert np.all(ml_scores >= 0.0) and np.all(ml_scores <= 1.0)

    # Line-pack correction check: configurable coefficient
    flow_scores_corr = calculate_flow_evidence_score(df_ramp, linepack_coeff=15.0)
    assert np.all(flow_scores_corr >= 0.0) and np.all(flow_scores_corr <= 1.0)
    # Line-pack corrected flow evidence stays controlled despite rapid flow increase
    assert flow_scores_corr.mean() < 0.30

    # Operational gating during ramp transition
    gating = OperatingModeGating()
    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores_corr,
        npw_scores=np.zeros(len(df_ramp)),
        operating_modes=df_ramp["operating_mode"],
        event_types=df_ramp["event_ground_truth"],
        timestamps=df_ramp["timestamp"].values,
        gating=gating
    )

    # Verify that ramping transients are suppressed and never escalate to LEAK DETECTED (CRITICAL)
    assert not any(df_fused["status"] == AlertState.LEAK_DETECTED.value)
    assert any("ramping transient" in str(r) for r in df_fused["suppression_reason"] if r)


def test_scenario_shut_in():
    """
    4. SHUT-IN: Pipeline shut-in condition (zero flow, static line-pack head)
    activates the 'Shut-in' Isolation Forest model and yields stable baseline normal status.
    """
    df_raw_shutin = generate_shut_in_dataset(duration_sec=60.0)
    df_shutin = extract_pipeline_features(df_raw_shutin)

    detector = ModeIsolationForestDetector()
    assert "Shut-in" in detector.models

    # Mode-specific scoring
    ml_scores = detector.predict_anomaly_scores(df_shutin)
    assert np.all(ml_scores >= 0.0) and np.all(ml_scores <= 1.0)
    assert ml_scores.mean() < 0.35

    flow_scores = calculate_flow_evidence_score(df_shutin)
    assert np.all(flow_scores < 0.10)

    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores,
        npw_scores=np.zeros(len(df_shutin))
    )
    assert (df_fused["status"] == AlertState.NORMAL.value).all()


def test_scenario_small_leak(sample_features):
    """
    5. SMALL LEAK: Gradual small chronic leak (~16 m3/h at 57 km) produces:
    - Sustained line-pack corrected flow imbalance with elevated flow evidence.
    - Localized hydraulic gradient deflection (pressure evidence).
    - NO acoustic NPW trigger (gradual onset has no sharp rarefaction front).
    - Status reaches SUSPECTED ANOMALY or LEAK DETECTED without suppression.
    """
    small_leak_window = sample_features[
        (sample_features["timestamp"] >= 345.0) & (sample_features["timestamp"] <= 500.0)
    ].copy()

    detector = ModeIsolationForestDetector()
    ml_scores = detector.predict_anomaly_scores(small_leak_window)
    flow_scores = calculate_flow_evidence_score(small_leak_window)
    pressure_scores = calculate_pressure_evidence_score(small_leak_window)
    npw_scores, npw_events = detect_negative_pressure_waves(small_leak_window)

    # Flow evidence reflects continuous mass deficit
    assert flow_scores.mean() > 0.60

    # Pressure evidence captures gradient distortion
    assert pressure_scores.mean() > 0.35

    # Gradual small leak must NOT trigger acoustic NPW
    assert np.all(npw_scores == 0.0)
    assert len(npw_events) == 0

    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores,
        npw_scores=npw_scores,
        pressure_scores=pressure_scores
    )

    # Escalates to SUSPECTED ANOMALY or LEAK DETECTED
    abnormal_count = (df_fused["status"].isin([AlertState.SUSPECTED_ANOMALY.value, AlertState.LEAK_DETECTED.value])).sum()
    assert abnormal_count > len(df_fused) * 0.70
    assert not any(df_fused["suppression_reason"].notna() & (df_fused["suppression_reason"] != ""))


def test_scenario_large_leak(sample_features):
    """
    6. LARGE LEAK: Sudden major rupture (~85 m3/h at 78 km) causes:
    - Rapid acoustic Negative Pressure Wave (NPW) arrival front.
    - Significant line-pack corrected flow imbalance (> 80 m3/h).
    - Strong hydraulic gradient kink across stations.
    - Elevated ML anomaly score.
    - Fused confidence reaching LEAK DETECTED (> 65%).
    """
    large_leak_window = sample_features[sample_features["timestamp"] >= 525.0].copy()

    detector = ModeIsolationForestDetector()
    ml_scores = detector.predict_anomaly_scores(large_leak_window)
    flow_scores = calculate_flow_evidence_score(large_leak_window)
    pressure_scores = calculate_pressure_evidence_score(large_leak_window)
    npw_scores, npw_events = detect_negative_pressure_waves(sample_features)

    # Large leak window has elevated evidence across all modalities
    assert flow_scores.mean() > 0.85
    assert pressure_scores.mean() > 0.50
    assert ml_scores.mean() > 0.65

    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores,
        npw_scores=npw_scores[sample_features["timestamp"] >= 525.0],
        pressure_scores=pressure_scores
    )

    # Fused status transitions to LEAK DETECTED (CRITICAL)
    leak_detected_count = (df_fused["status"] == AlertState.LEAK_DETECTED.value).sum()
    assert leak_detected_count > len(df_fused) * 0.80
    assert df_fused["confidence_pct"].max() > 70.0


def test_scenario_pump_transient():
    """
    7. PUMP TRANSIENT: Secondary booster pump start induces hydraulic pressure surge
    and temporary flow surge. Operating mode gating activates, applies settling period,
    returns 'Suppressed: pump-start transient', and prevents alert state from becoming CRITICAL.
    """
    df_raw = generate_pump_transient_dataset(duration_sec=80.0, transient_start_sec=20.0)
    df_feats = extract_pipeline_features(df_raw)

    gating = OperatingModeGating(pump_settling_sec=25.0)

    # Synthesize momentarily elevated evidence caused by surge
    n = len(df_feats)
    ml_scores = np.full(n, 0.70)
    flow_scores = np.full(n, 0.65)
    npw_scores = np.zeros(n)

    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores,
        npw_scores=npw_scores,
        timestamps=df_feats["timestamp"].values,
        operating_modes=df_feats["operating_mode"],
        event_types=df_feats["event_ground_truth"],
        gating=gating
    )

    # Verify suppression reason formatting
    transient_rows = df_fused[df_fused["suppression_reason"].notna()]
    assert len(transient_rows) > 0
    assert (transient_rows["suppression_reason"] == "Suppressed: pump-start transient").all()

    # Critical requirement: A pump transient must NOT automatically become CRITICAL (LEAK DETECTED)
    assert not any(transient_rows["status"] == AlertState.LEAK_DETECTED.value)


def test_scenario_valve_event():
    """
    8. VALVE EVENT: Valve throttling maneuver causes localized pressure step and flow dip.
    Gating suppresses leak evidence during settling window, returns
    'Suppressed: valve maneuver transient', and prevents alert state from becoming CRITICAL.
    """
    df_raw = generate_valve_event_dataset(duration_sec=80.0, valve_event_start_sec=20.0)
    df_feats = extract_pipeline_features(df_raw)

    gating = OperatingModeGating(valve_settling_sec=20.0)

    n = len(df_feats)
    ml_scores = np.full(n, 0.72)
    flow_scores = np.full(n, 0.68)
    npw_scores = np.zeros(n)

    df_fused, _ = fuse_evidence_signals(
        ml_scores=ml_scores,
        flow_scores=flow_scores,
        npw_scores=npw_scores,
        timestamps=df_feats["timestamp"].values,
        operating_modes=df_feats["operating_mode"],
        event_types=df_feats["event_ground_truth"],
        gating=gating
    )

    # Verify valve suppression reason
    transient_rows = df_fused[df_fused["suppression_reason"].notna()]
    assert len(transient_rows) > 0
    assert (transient_rows["suppression_reason"] == "Suppressed: valve maneuver transient").all()

    # Critical requirement: Valve transient must NOT become CRITICAL
    assert not any(transient_rows["status"] == AlertState.LEAK_DETECTED.value)


def test_scenario_npw(sample_features):
    """
    9. NPW: Sharp negative pressure slope (dP/dt) detection verifies:
    - Detection of sharp negative slopes
    - Event time
    - Station arrival times
    - Affected stations list
    - Wave propagation order (ordered by arrival time)
    - Estimated acoustic wave speed (km/s)
    - NPW evidence normalized in [0.0, 1.0]
    - Prototype limitation notice clearly marked
    """
    scores, events = detect_negative_pressure_waves(sample_features)

    # Score bounds
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert scores.max() >= 0.75

    # At least one major NPW event triggered by rupture at t=520s
    assert len(events) >= 1
    ev = next(e for e in events if e["start_time"] >= 520.0)

    # 1. Event time
    assert "event_time" in ev
    assert 520.0 <= ev["event_time"] <= 535.0

    # 2. Station arrival times
    assert "arrival_times" in ev
    assert isinstance(ev["arrival_times"], dict)
    assert len(ev["arrival_times"]) >= 2

    # 3. Affected stations
    assert "affected_stations" in ev
    assert len(ev["affected_stations"]) >= 2
    # Large leak at 78 km should affect Station 5 (80 km) and Station 4 (60 km) or Station 6 (100 km)
    assert any("st5" in s.lower() or "st4" in s.lower() or "st6" in s.lower() for s in ev["affected_stations"])

    # 4. Wave propagation order: sorted chronologically by arrival time
    assert "wave_propagation_order" in ev
    prop_order = ev["wave_propagation_order"]
    assert len(prop_order) == len(ev["arrival_times"])
    for i in range(len(prop_order) - 1):
        st_curr = prop_order[i]
        st_next = prop_order[i + 1]
        assert ev["arrival_times"][st_curr] <= ev["arrival_times"][st_next]

    # 5. Estimated wave speed
    assert "estimated_wave_speed" in ev
    assert 0.5 <= ev["estimated_wave_speed"] <= 2.0  # physical range for liquid pipeline

    # 6. NPW evidence
    assert "npw_evidence" in ev
    assert 0.0 <= ev["npw_evidence"] <= 1.0
    assert ev["npw_evidence"] >= 0.75

    # 7. Prototype limitation disclosure
    assert "prototype_limitation" in ev
    assert "PROTOTYPE LIMITATION" in ev["prototype_limitation"]
    assert "10 Hz" in ev["prototype_limitation"]

