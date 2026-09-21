"""
Deterministic Demonstration Scenarios Unit Tests (Phase 5).

Tests:
  1. Repeatability: Scenarios are 100% deterministic across multiple runs with the same seed.
  2. Large Leak Detection: Evaluates rupture detection, acoustic NPW wave, ground truth (78 km),
     estimated location, and localization error.
  3. Small Leak Behavior: Evaluates pinhole chronic leak honestly (sustained flow imbalance,
     subtle gradient deflection, no acoustic shockwave).
  4. Pump Suppression: Verifies mode gating suppresses pump start surge, returns
     'Suppressed: pump-start transient', and prevents false CRITICAL alerts.
  5. Valve Suppression: Verifies valve maneuver transient is recognized and suppressed
     rather than blindly alarming.
"""

import pytest
import numpy as np
import pandas as pd
from src.simulation import (
    ScenarioType,
    generate_scenario_data,
    list_available_scenarios
)
from src.pipeline.run_detection import execute_detection_pipeline
from src.contract import AlertState


def test_scenario_list_and_types():
    """Verifies all 5 canonical demonstration scenarios are registered."""
    scenarios = list_available_scenarios()
    assert len(scenarios) == 5
    assert "NORMAL" in scenarios
    assert "SMALL CHRONIC LEAK" in scenarios
    assert "LARGE LEAK" in scenarios
    assert "PUMP TRANSIENT" in scenarios
    assert "VALVE EVENT" in scenarios


def test_scenario_repeatability():
    """
    Every scenario must be deterministic:
    Re-generating data and re-executing pipeline with the same fixed seed
    produces identical telemetry and identical pipeline outputs.
    """
    for sc in ScenarioType:
        df1 = generate_scenario_data(sc, duration_sec=40.0, random_seed=42)
        df2 = generate_scenario_data(sc, duration_sec=40.0, random_seed=42)

        # Telemetry equality
        pd.testing.assert_frame_equal(df1, df2)

        # Pipeline execution repeatability
        res1 = execute_detection_pipeline(df1)
        res2 = execute_detection_pipeline(df2)

        np.testing.assert_allclose(res1["confidence_pct"].values, res2["confidence_pct"].values)
        assert list(res1["status"]) == list(res2["status"])
        assert list(res1["suppression_reason"]) == list(res2["suppression_reason"])


def test_large_leak_detection():
    """
    LARGE LEAK:
    Demonstrates major rupture detection:
      - Triggers acoustic Negative Pressure Wave (NPW) arrival
      - Transitions status to LEAK DETECTED (CRITICAL)
      - Reports ground truth location (78.0 km)
      - Reports estimated location and localization error within acceptable margin
    """
    df_raw = generate_scenario_data(ScenarioType.LARGE_LEAK, duration_sec=70.0, random_seed=42)
    res = execute_detection_pipeline(df_raw)

    # 1. Pipeline status escalation
    leak_detected_rows = res[res["status"] == AlertState.LEAK_DETECTED.value]
    assert len(leak_detected_rows) > 0, "Large leak scenario must trigger LEAK DETECTED status."

    # 2. NPW acoustic evidence triggered
    assert res["npw_evidence_score"].max() >= 0.70, "Acoustic NPW evidence must be elevated."

    # 3. Ground truth verification (78.0 km)
    gt_leak = res.loc[res["event_ground_truth"] == "LARGE_LEAK", "ground_truth_km"].dropna()
    assert len(gt_leak) > 0
    assert (gt_leak == 78.0).all()

    # 4. Estimated location and localization error
    est_locations = leak_detected_rows["estimated_leak_km"].dropna()
    assert len(est_locations) > 0, "Estimated location must be computed during LEAK DETECTED."

    errors = leak_detected_rows["localization_error_km"].dropna()
    assert len(errors) > 0, "Localization error must be computed."
    median_err = errors.median()
    assert median_err <= 3.5, f"Median localization error ({median_err} km) must be within 3.5 km."


def test_small_leak_behavior():
    """
    SMALL CHRONIC LEAK:
    Demonstrates honest reporting for a pinhole leak (~16 m3/h at 57 km):
      - Sustained mass flow imbalance and subtle gradient shift
      - Elevated anomaly / suspect status
      - NO false acoustic NPW trigger (honest physical evaluation)
      - Reports ground truth location (57.0 km)
    """
    df_raw = generate_scenario_data(ScenarioType.SMALL_CHRONIC_LEAK, duration_sec=70.0, random_seed=42)
    res = execute_detection_pipeline(df_raw)

    leak_window = res[res["timestamp"] >= 45.0]

    # Elevated flow imbalance evidence
    assert leak_window["flow_evidence_score"].mean() > 0.40

    # Honest reporting: gradual small leak does NOT trigger acoustic shockwave
    assert res["npw_evidence_score"].max() < 0.30

    # Ground truth is 57.0 km
    gt_leak = res.loc[res["event_ground_truth"] == "SMALL_LEAK", "ground_truth_km"].dropna()
    assert len(gt_leak) > 0
    assert (gt_leak == 57.0).all()

    # Elevated confidence reaches SUSPECTED ANOMALY or LEAK DETECTED
    elevated_count = (res["status"].isin([AlertState.SUSPECTED_ANOMALY.value, AlertState.LEAK_DETECTED.value])).sum()
    assert elevated_count > 0


def test_pump_suppression():
    """
    PUMP TRANSIENT:
    Demonstrates that mode gating prevents a false CRITICAL leak during pump surge:
      - Recognizes pump start operational transient
      - Attaches 'Suppressed: pump-start transient'
      - Caps fused confidence strictly below leak threshold
      - Status NEVER becomes CRITICAL (LEAK DETECTED)
    """
    df_raw = generate_scenario_data(ScenarioType.PUMP_TRANSIENT, duration_sec=60.0, random_seed=42)
    res = execute_detection_pipeline(df_raw)

    # Verify suppression reason attached
    suppressed_rows = res[res["suppression_reason"].notna() & (res["suppression_reason"] != "")]
    assert len(suppressed_rows) > 0
    assert (suppressed_rows["suppression_reason"] == "Suppressed: pump-start transient").all()

    # Critical requirement: Mode gating prevents a false CRITICAL leak
    critical_leaks = res[res["status"] == AlertState.LEAK_DETECTED.value]
    assert len(critical_leaks) == 0, "Pump transient must NOT trigger CRITICAL (LEAK DETECTED)."
    assert res["confidence_pct"].max() < 65.0, "Confidence must be capped below leak threshold (65%)."


def test_valve_suppression():
    """
    VALVE EVENT:
    Demonstrates that the system recognizes an operational transient rather than blindly alarming:
      - Recognizes valve maneuver operational transition
      - Attaches 'Suppressed: valve maneuver transient'
      - Prevents alert state from escalating to CRITICAL
      - Caps confidence below leak threshold
    """
    df_raw = generate_scenario_data(ScenarioType.VALVE_EVENT, duration_sec=60.0, random_seed=42)
    res = execute_detection_pipeline(df_raw)

    # Verify suppression reason attached
    suppressed_rows = res[res["suppression_reason"].notna() & (res["suppression_reason"] != "")]
    assert len(suppressed_rows) > 0
    assert (suppressed_rows["suppression_reason"] == "Suppressed: valve maneuver transient").all()

    # Critical requirement: System recognizes transient rather than blindly alarming
    critical_leaks = res[res["status"] == AlertState.LEAK_DETECTED.value]
    assert len(critical_leaks) == 0, "Valve maneuver must NOT trigger CRITICAL (LEAK DETECTED)."
    assert res["confidence_pct"].max() < 65.0, "Confidence must be capped below leak threshold (65%)."


def test_normal_scenario():
    """
    NORMAL:
    Steady nominal flow produces clean baseline with all detection signals low,
    zero false alarms, and 100% NORMAL status.
    """
    df_raw = generate_scenario_data(ScenarioType.NORMAL, duration_sec=60.0, random_seed=42)
    res = execute_detection_pipeline(df_raw)

    assert (res["status"] == AlertState.NORMAL.value).all()
    assert res["confidence_pct"].max() < 25.0
    assert (res["npw_evidence_score"] == 0.0).all()
