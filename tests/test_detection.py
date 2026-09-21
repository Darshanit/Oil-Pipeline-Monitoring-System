"""
Unit tests for Detection Modules (Isolation Forest, Flow Imbalance, NPW, Fusion).
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_generation import generate_pipeline_dataset
from src.preprocessing import extract_pipeline_features
from src.detection import (
    ModeIsolationForestDetector,
    calculate_flow_evidence_score,
    detect_negative_pressure_waves,
    fuse_evidence_signals
)


@pytest.fixture
def sample_features(tmp_path):
    df_raw = generate_pipeline_dataset(duration_sec=600.0, dt_pressure=0.1, random_seed=42)
    return extract_pipeline_features(df_raw)


def test_isolation_forest_training_and_bounds(sample_features, tmp_path):
    model_dir = str(tmp_path / "models")
    detector = ModeIsolationForestDetector()
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

    # Large leak window should have higher flow evidence score than normal window
    normal_score = scores[sample_features["event_ground_truth"] == "NORMAL"].mean()
    large_leak_score = scores[sample_features["event_ground_truth"] == "LARGE_LEAK"].mean()
    assert large_leak_score > normal_score


def test_npw_detection_trigger(sample_features):
    scores, events = detect_negative_pressure_waves(sample_features)
    assert len(scores) == len(sample_features)
    # Large leak at t=520s should trigger NPW event
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
