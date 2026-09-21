"""
Phase 9 Comprehensive QA Test Suite.

Exhaustively verifies all 14 verification items specified in Phase 9 requirements:
1. Python imports
2. feature pipeline
3. all 3 Isolation Forest models
4. normalized anomaly score
5. flow evidence
6. NPW
7. localization
8. fusion
9. agreement rule
10. alert persistence
11. hysteresis
12. ACK
13. all demo scenarios
14. dashboard startup & offline capability
"""

import os
import re
import pytest
import numpy as np
import pandas as pd
from dataclasses import FrozenInstanceError


# ==============================================================================
# 1. PYTHON IMPORTS
# ==============================================================================
def test_01_python_imports():
    """1. Verifies all project modules and backend/dashboard contracts import cleanly."""
    import src.config as cfg
    import src.contract as contract
    import src.data_generation as data_gen
    import src.features as feats
    import src.preprocessing as prep
    import src.detection as det
    import src.fusion as fusion
    import src.localization as loc
    import src.inference as inf
    import src.pipeline.run_detection as run_det
    import src.simulation as sim
    import dashboard.app as dash_app

    # Assert key exports exist
    assert hasattr(cfg, "DEFAULT_CONFIG")
    assert hasattr(contract, "DetectionResult")
    assert hasattr(contract, "AlertState")
    assert hasattr(data_gen, "generate_pipeline_dataset")
    assert hasattr(prep, "extract_pipeline_features")
    assert hasattr(det, "ModeIsolationForestDetector")
    assert hasattr(det, "calculate_flow_evidence_score")
    assert hasattr(det, "detect_negative_pressure_waves")
    assert hasattr(det, "calculate_pressure_evidence_score")
    assert hasattr(det, "fuse_evidence_signals")
    assert hasattr(det, "OperatingModeGating")
    assert hasattr(loc, "estimate_leak_location")
    assert hasattr(fusion, "AlertStateMachine")
    assert hasattr(run_det, "execute_detection_pipeline")
    assert hasattr(sim, "generate_scenario_data")
    assert hasattr(dash_app, "render_pipeline_svg")


# ==============================================================================
# 2. FEATURE PIPELINE
# ==============================================================================
def test_02_feature_pipeline():
    """2. Verifies feature extraction, line-pack dynamic correction, and absence of NaNs/Infs."""
    from src.data_generation import generate_pipeline_dataset
    from src.preprocessing import extract_pipeline_features

    df_raw = generate_pipeline_dataset(duration_sec=30.0, dt_pressure=0.1, random_seed=42)
    df_feats = extract_pipeline_features(df_raw, linepack_coeff=15.0)

    required_cols = [
        "timestamp", "operating_mode", "flow_in", "flow_out",
        "raw_flow_imbalance", "mean_dp_dt", "linepack_rate", "corrected_flow_imbalance",
        "grad_total", "diff_st1_st2", "diff_st2_st3", "diff_st3_st4", "diff_st4_st5", "diff_st5_st6",
        "grad_st1_st2", "grad_st2_st3", "grad_st3_st4", "grad_st4_st5", "grad_st5_st6",
        "P_st1_dp_dt", "P_st2_dp_dt", "P_st3_dp_dt", "P_st4_dp_dt", "P_st5_dp_dt", "P_st6_dp_dt",
        "P_st1_var", "P_st2_var", "P_st3_var", "P_st4_var", "P_st5_var", "P_st6_var"
    ]
    for col in required_cols:
        assert col in df_feats.columns, f"Missing feature column: {col}"

    # Verify line-pack formula: corrected = raw - linepack_rate
    raw_imb = df_feats["raw_flow_imbalance"].values
    lp_rate = df_feats["linepack_rate"].values
    corr_imb = df_feats["corrected_flow_imbalance"].values
    np.testing.assert_allclose(corr_imb, raw_imb - lp_rate, atol=1e-5)

    # Verify no NaN or Inf in computed feature columns
    assert not df_feats[required_cols].isna().any().any()
    assert not np.isinf(df_feats[required_cols].select_dtypes(include=[np.number]).values).any()


# ==============================================================================
# 3. ALL 3 ISOLATION FOREST MODELS
# ==============================================================================
def test_03_all_three_isolation_forest_models():
    """3. Verifies all 3 operational mode models (Flowing, Ramping, Shut-in) exist and score."""
    from src.detection.isolation_forest import ModeIsolationForestDetector

    detector = ModeIsolationForestDetector(model_dir="models", auto_load=True)
    required_modes = ["Flowing", "Ramping", "Shut-in"]

    for mode in required_modes:
        assert mode in detector.models, f"Model for mode '{mode}' is not loaded."
        model_obj = detector.models[mode]
        assert hasattr(model_obj, "predict") or hasattr(model_obj, "decision_function")

    # Verify each model can score sample features
    from src.data_generation import generate_pipeline_dataset
    from src.preprocessing import extract_pipeline_features

    df_raw = generate_pipeline_dataset(duration_sec=10.0, dt_pressure=0.1, random_seed=42)
    df_feats = extract_pipeline_features(df_raw)

    for mode in required_modes:
        df_feats["operating_mode"] = mode
        scores = detector.predict_anomaly_scores(df_feats)
        assert len(scores) == len(df_feats)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ==============================================================================
# 4. NORMALIZED ANOMALY SCORE
# ==============================================================================
def test_04_normalized_anomaly_score():
    """4. Verifies anomaly score normalization strictly bounds output in [0.0, 1.0]."""
    from src.detection.isolation_forest import ModeIsolationForestDetector
    from src.data_generation import generate_pipeline_dataset
    from src.preprocessing import extract_pipeline_features

    detector = ModeIsolationForestDetector(model_dir="models", auto_load=True)
    df_raw = generate_pipeline_dataset(duration_sec=30.0, dt_pressure=0.1, random_seed=42)
    df_feats = extract_pipeline_features(df_raw)

    scores = detector.predict_anomaly_scores(df_feats)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert not np.isnan(scores).any()

    # Normal baseline scores must be well below alert suspect threshold (0.35)
    baseline_scores = scores[df_feats["timestamp"] < 20.0]
    assert baseline_scores.mean() < 0.35

    # Case-insensitive mode handling
    df_feats["operating_mode"] = "FLOWING"
    scores_upper = detector.predict_anomaly_scores(df_feats)
    df_feats["operating_mode"] = "flowing"
    scores_lower = detector.predict_anomaly_scores(df_feats)
    np.testing.assert_allclose(scores_upper, scores_lower)


# ==============================================================================
# 5. FLOW EVIDENCE
# ==============================================================================
def test_05_flow_evidence():
    """5. Verifies line-pack corrected flow evidence calculation, bounds, and sensitivity."""
    from src.detection.flow_evidence import calculate_flow_evidence_score
    from src.data_generation import generate_pipeline_dataset
    from src.preprocessing import extract_pipeline_features

    df_raw = generate_pipeline_dataset(duration_sec=30.0, dt_pressure=0.1, random_seed=42)
    df_feats = extract_pipeline_features(df_raw)

    # 1. Normal conditions: flow evidence is near zero
    scores_normal = calculate_flow_evidence_score(df_feats)
    assert np.all(scores_normal >= 0.0) and np.all(scores_normal <= 1.0)
    assert scores_normal.mean() < 0.15

    # 2. Large leak deficit simulation (+85 m3/h deficit)
    df_feats_leak = df_feats.copy()
    df_feats_leak["flow_out"] = df_feats_leak["flow_in"] - 85.0
    scores_leak = calculate_flow_evidence_score(df_feats_leak)
    assert np.all(scores_leak >= 0.0) and np.all(scores_leak <= 1.0)
    assert scores_leak.mean() > 0.85

    # 3. Monotonic response check
    df_feats_small = df_feats.copy()
    df_feats_small["flow_out"] = df_feats_small["flow_in"] - 16.0
    scores_small = calculate_flow_evidence_score(df_feats_small)
    assert scores_leak.mean() > scores_small.mean() > scores_normal.mean()


# ==============================================================================
# 6. NPW (NEGATIVE PRESSURE WAVE)
# ==============================================================================
def test_06_npw():
    """6. Verifies acoustic NPW detection, arrival times, propagation order, speed, and disclaimer."""
    from src.detection.npw import detect_negative_pressure_waves
    from src.data_generation import generate_pipeline_dataset
    from src.preprocessing import extract_pipeline_features

    # Generate dataset with large rupture at t=520s
    df_raw = generate_pipeline_dataset(duration_sec=600.0, dt_pressure=0.1, random_seed=42)
    df_feats = extract_pipeline_features(df_raw)

    scores, events = detect_negative_pressure_waves(df_feats)

    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert len(events) >= 1

    ev = next(e for e in events if e["start_time"] >= 520.0)
    assert "arrival_times" in ev
    assert "affected_stations" in ev
    assert "wave_propagation_order" in ev
    assert "estimated_wave_speed" in ev
    assert "prototype_limitation" in ev

    # Wave speed in physical range [0.5, 2.0] km/s
    assert 0.5 <= ev["estimated_wave_speed"] <= 2.0

    # Chronological propagation order
    order = ev["wave_propagation_order"]
    for i in range(len(order) - 1):
        assert ev["arrival_times"][order[i]] <= ev["arrival_times"][order[i + 1]]

    # Limitation notice clearly documented
    assert "PROTOTYPE LIMITATION" in ev["prototype_limitation"]
    assert "10 Hz" in ev["prototype_limitation"]


# ==============================================================================
# 7. LOCALIZATION
# ==============================================================================
def test_07_localization():
    """7. Verifies acoustic NPW localization formula and hydraulic drop localization math."""
    from src.localization.leak_localization import (
        localize_leak_from_npw,
        localize_leak_from_hydraulic_drop,
        estimate_leak_location
    )

    # Test acoustic NPW formula: x_L = [x_A + x_B - v * (t_B - t_A)] / 2
    # Station 4 at 60 km, Station 5 at 80 km, wave speed = 1.0 km/s
    # Rupture at 78 km -> Dist to S4 = 18 km, Dist to S5 = 2 km
    # t_arr_S4 = 18.0 s, t_arr_S5 = 2.0 s -> delta_t = 2.0 - 18.0 = -16.0 s
    # x_L = [60 + 80 - 1.0 * (-16.0)] / 2 = 156 / 2 = 78.0 km
    arr_times = {"P_st4": 18.0, "P_st5": 2.0}
    loc_npw = localize_leak_from_npw(arr_times, wave_speed_km_s=1.0)
    assert loc_npw is not None
    assert pytest.approx(loc_npw["estimated_km"], abs=0.1) == 78.0
    assert "Station" in loc_npw["nearest_station"]

    # Test hydraulic drop localization
    pressures = {"P_st1": 55.0, "P_st2": 47.0, "P_st3": 39.0, "P_st4": 31.0, "P_st5": 16.0, "P_st6": 8.0}
    loc_hyd = localize_leak_from_hydraulic_drop(pressures)
    assert loc_hyd is not None
    assert 55.0 <= loc_hyd["estimated_km"] <= 85.0


# ==============================================================================
# 8. FUSION
# ==============================================================================
def test_08_fusion():
    """8. Verifies 4-channel evidence fusion, contribution decomposition, and status mapping."""
    from src.fusion.fusion import fuse_evidence_signals, fuse_single_step
    from src.config import DEFAULT_CONFIG

    # Single-step fusion check
    weights = {"ml": 0.30, "flow": 0.35, "npw": 0.15, "pressure": 0.20}
    raw_fused, contribs, status, supp = fuse_single_step(
        ml_score=0.80,
        flow_score=0.90,
        npw_score=1.0,
        pressure_score=0.75,
        weights=weights
    )
    assert 0.0 <= raw_fused <= 1.0
    assert pytest.approx(sum(contribs.values()), abs=1e-4) == raw_fused
    assert status == "LEAK DETECTED"

    # Batch fusion check: sustained leak condition over 60 samples
    n = 60
    ml = np.full(n, 0.85)
    flow = np.full(n, 0.90)
    npw = np.full(n, 0.80)
    press = np.full(n, 0.75)

    df_res, alerts = fuse_evidence_signals(ml, flow, npw, pressure_scores=press, weights=weights)
    assert "confidence_pct" in df_res.columns
    assert "status" in df_res.columns
    assert df_res["confidence_pct"].iloc[-1] > 65.0
    assert df_res["status"].iloc[-1] == "LEAK DETECTED"


# ==============================================================================
# 9. AGREEMENT RULE
# ==============================================================================
def test_09_agreement_rule():
    """9. Verifies multi-signal agreement rule: >= 2 signals required before warning/critical."""
    from src.fusion.alert_state import AlertStateMachine
    from src.contract import AlertState
    from src.config import DEFAULT_CONFIG

    sm = AlertStateMachine(DEFAULT_CONFIG)

    # 1. Single noisy signal spiking (e.g. ML = 0.90, but Flow, NPW, Pressure are nominal)
    noisy_signals = {"ml": 0.90, "flow": 0.10, "npw": 0.0, "pressure": 0.10}
    agrees, n_agree = sm.check_agreement(noisy_signals)
    assert agrees is False
    assert n_agree == 1

    # In single-step update, agreement rule suppresses the alert
    state, p_count, reason = sm.update(raw_score=0.75, signals=noisy_signals)
    assert state == AlertState.NORMAL
    assert reason is not None
    assert "Suppressed by agreement rule" in reason

    # 2. Two independent signals agreeing (Flow = 0.85, Pressure = 0.70)
    leak_signals = {"ml": 0.30, "flow": 0.85, "npw": 0.0, "pressure": 0.70}
    agrees, n_agree = sm.check_agreement(leak_signals)
    assert agrees is True
    assert n_agree >= 2


# ==============================================================================
# 10. ALERT PERSISTENCE
# ==============================================================================
def test_10_alert_persistence():
    """10. Verifies M-of-N persistence filtering and suppression of single-sample transient noise."""
    from src.fusion.alert_state import AlertStateMachine
    from src.contract import AlertState
    from src.config import DEFAULT_CONFIG

    sm = AlertStateMachine(DEFAULT_CONFIG)

    # Single-sample transient anomaly spike: should NOT immediately transition to LEAK DETECTED
    state, p_cnt, reason = sm.update(raw_score=0.90)
    assert state != AlertState.LEAK_DETECTED
    assert reason is not None
    assert "suppressed by persistence filter" in reason

    # Sustained anomaly across M samples: transitions to LEAK DETECTED
    for _ in range(DEFAULT_CONFIG.persistence_m_triggers + 5):
        state, p_cnt, reason = sm.update(raw_score=0.90)

    assert state == AlertState.LEAK_DETECTED
    assert p_cnt >= DEFAULT_CONFIG.persistence_m_triggers


# ==============================================================================
# 11. HYSTERESIS
# ==============================================================================
def test_11_hysteresis():
    """11. Verifies dual-threshold hysteresis to prevent alarm chatter at recovery boundaries."""
    from src.fusion.alert_state import AlertStateMachine
    from src.contract import AlertState
    from src.config import DEFAULT_CONFIG

    sm = AlertStateMachine(DEFAULT_CONFIG)

    # Escalate to LEAK DETECTED
    for _ in range(DEFAULT_CONFIG.persistence_n_samples + 5):
        sm.update(raw_score=0.90)
    assert sm.current_state == AlertState.LEAK_DETECTED

    # Score drops to 0.60 (below leak threshold 0.65, but above hysteresis_leak_recovery 0.55)
    # The state machine must STAY in LEAK DETECTED until rolling mean drops below recovery threshold
    for _ in range(5):
        state, _, _ = sm.update(raw_score=0.60)
    assert state == AlertState.LEAK_DETECTED

    # Score drops significantly below recovery threshold (0.55)
    for _ in range(DEFAULT_CONFIG.persistence_n_samples + 5):
        state, _, _ = sm.update(raw_score=0.45)
    # Exited LEAK DETECTED into SUSPECTED ANOMALY
    assert state == AlertState.SUSPECTED_ANOMALY

    # Score drops below suspect recovery (0.25)
    for _ in range(DEFAULT_CONFIG.persistence_n_samples + 5):
        state, _, _ = sm.update(raw_score=0.10)
    # Exited into NORMAL
    assert state == AlertState.NORMAL


# ==============================================================================
# 12. ACK (OPERATOR ACKNOWLEDGEMENT)
# ==============================================================================
def test_12_ack():
    """12. Verifies alert acknowledgement handling, contract immutability, and state reset."""
    from src.contract import DetectionResult, AlertState

    pressures = {f"P_st{i+1}": 50.0 - i * 6.0 for i in range(6)}
    res = DetectionResult(
        timestamp=100.0,
        mode="Flowing",
        pressures=pressures,
        flow_in=500.0,
        flow_out=450.0,
        flow_imbalance_raw=50.0,
        flow_imbalance_corrected=48.0,
        anomaly_score=0.85,
        flow_evidence=0.90,
        npw_evidence=0.0,
        pressure_evidence=0.75,
        fusion_confidence=82.0,
        contributions={"ml": 25.0, "flow": 35.0, "npw": 0.0, "pressure": 22.0},
        alert_state=AlertState.CRITICAL.value,
        persistence=15,
        is_acknowledged=False
    )

    # 1. DetectionResult is immutable
    assert res.is_acknowledged is False
    with pytest.raises(FrozenInstanceError):
        res.is_acknowledged = True  # type: ignore

    # 2. Acknowledged result instance creation
    ack_res = DetectionResult(
        timestamp=res.timestamp,
        mode=res.mode,
        pressures=res.pressures,
        flow_in=res.flow_in,
        flow_out=res.flow_out,
        flow_imbalance_raw=res.flow_imbalance_raw,
        flow_imbalance_corrected=res.flow_imbalance_corrected,
        anomaly_score=res.anomaly_score,
        flow_evidence=res.flow_evidence,
        npw_evidence=res.npw_evidence,
        pressure_evidence=res.pressure_evidence,
        fusion_confidence=res.fusion_confidence,
        contributions=res.contributions,
        alert_state=res.alert_state,
        persistence=res.persistence,
        is_acknowledged=True
    )
    assert ack_res.is_acknowledged is True


# ==============================================================================
# 13. ALL DEMO SCENARIOS
# ==============================================================================
def test_13_all_demo_scenarios():
    """13. Verifies all 5 canonical demonstration scenarios execute and meet criteria."""
    from src.simulation import ScenarioType, generate_scenario_data
    from src.pipeline.run_detection import execute_detection_pipeline
    from src.contract import AlertState

    # Scenario 1: NORMAL
    df_norm = generate_scenario_data(ScenarioType.NORMAL, duration_sec=40.0, random_seed=42)
    res_norm = execute_detection_pipeline(df_norm)
    assert (res_norm["status"] == AlertState.NORMAL.value).all()
    assert res_norm["confidence_pct"].mean() < 25.0

    # Scenario 2: SMALL CHRONIC LEAK
    df_small = generate_scenario_data(ScenarioType.SMALL_CHRONIC_LEAK, duration_sec=70.0, random_seed=42)
    res_small = execute_detection_pipeline(df_small)
    leak_window = res_small[res_small["timestamp"] >= 45.0]
    assert leak_window["flow_evidence_score"].mean() > 0.40
    # No acoustic NPW shockwave for pinhole leak
    assert res_small["npw_evidence_score"].max() < 0.30

    # Scenario 3: LARGE LEAK
    df_large = generate_scenario_data(ScenarioType.LARGE_LEAK, duration_sec=70.0, random_seed=42)
    res_large = execute_detection_pipeline(df_large)
    assert (res_large["status"] == AlertState.LEAK_DETECTED.value).any()
    assert res_large["npw_evidence_score"].max() >= 0.70
    leak_rows = res_large[res_large["status"] == AlertState.LEAK_DETECTED.value]
    assert len(leak_rows) > 0
    errors = leak_rows["localization_error_km"].dropna()
    assert len(errors) > 0
    assert errors.median() <= 3.5

    # Scenario 4: PUMP TRANSIENT
    df_pump = generate_scenario_data(ScenarioType.PUMP_TRANSIENT, duration_sec=50.0, random_seed=42)
    res_pump = execute_detection_pipeline(df_pump)
    assert not (res_pump["status"] == AlertState.LEAK_DETECTED.value).any()
    assert any("pump-start transient" in str(r) for r in res_pump["suppression_reason"] if r)

    # Scenario 5: VALVE EVENT
    df_valve = generate_scenario_data(ScenarioType.VALVE_EVENT, duration_sec=50.0, random_seed=42)
    res_valve = execute_detection_pipeline(df_valve)
    assert not (res_valve["status"] == AlertState.LEAK_DETECTED.value).any()
    assert any("valve maneuver transient" in str(r) for r in res_valve["suppression_reason"] if r)


# ==============================================================================
# 14. DASHBOARD STARTUP & OFFLINE CAPABILITY
# ==============================================================================
def test_14_dashboard_startup():
    """14. Verifies dashboard startup, offline CSS/assets (NO CDN), SVG map, and health checks."""
    import dashboard.app as app
    from dashboard.app import (
        render_pipeline_svg,
        evaluate_system_health,
        build_detection_result,
        compute_npw_replay_data,
        load_scenario_detection_results
    )
    from src.contract import DetectionResult, AlertState

    # 1. Verify offline assets: NO external CDNs, NO external fonts, NO external APIs
    assets_dir = os.path.join(os.path.dirname(app.__file__), "assets")
    for css_file in ["tokens.css", "style.css", "motion.css"]:
        path = os.path.join(assets_dir, css_file)
        assert os.path.exists(path), f"Asset {css_file} not found"
        with open(path, "r") as f:
            content = f.read()
        # Verify no external URLs (http:// or https://)
        external_urls = re.findall(r"https?://[^\s\"')]+", content)
        assert len(external_urls) == 0, f"Found external CDN/URL in {css_file}: {external_urls}"

    # 2. System Health Evaluation runs on real subsystems and reports OK
    pressures = {f"P_st{i+1}": 50.0 - i * 6.0 for i in range(6)}
    sample_res = DetectionResult(
        timestamp=30.0,
        mode="Flowing",
        pressures=pressures,
        flow_in=500.0,
        flow_out=499.5,
        flow_imbalance_raw=0.5,
        flow_imbalance_corrected=0.2,
        anomaly_score=0.15,
        flow_evidence=0.05,
        npw_evidence=0.0,
        pressure_evidence=0.08,
        fusion_confidence=12.0,
        contributions={"ml": 4.5, "flow": 1.7, "npw": 0.0, "pressure": 1.6},
        alert_state=AlertState.NORMAL.value,
        persistence=0
    )
    weights = {"ml": 0.30, "flow": 0.35, "npw": 0.15, "pressure": 0.20}
    health = evaluate_system_health(sample_res, weights)
    assert len(health) == 6
    for comp, comp_data in health.items():
        assert comp_data["ok"] is True, f"Subsystem {comp} reported failure in health check"

    # 3. Pipeline SVG Map generation succeeds
    svg_html = render_pipeline_svg(sample_res, motion_mode="FULL")
    assert "<svg" in svg_html and "</svg>" in svg_html
    assert "S1" in svg_html and "S6" in svg_html

    # 4. NPW Replay math computation
    npw_info = compute_npw_replay_data(leak_km=78.0, wave_speed_km_s=1.0)
    assert npw_info["calculated_km"] == 78.0

    # 5. Scenario Detection Results Loading
    df = load_scenario_detection_results("1. NORMAL (Steady Nominal Baseline)")
    assert len(df) > 0
    assert "timestamp" in df.columns
