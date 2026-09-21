"""
Unit tests for DetectionResult contract and AlertState enum.
"""

import pytest
from dataclasses import FrozenInstanceError
from src.contract import DetectionResult, AlertState


@pytest.fixture
def sample_detection_result():
    return DetectionResult(
        timestamp=520.4,
        mode="Flowing",
        pressures={"P_st1": 49.5, "P_st2": 41.2, "P_st3": 33.1, "P_st4": 25.0, "P_st5": 15.2, "P_st6": 7.1},
        flow_in=500.2,
        flow_out=415.1,
        flow_imbalance_raw=85.1,
        flow_imbalance_corrected=84.3,
        anomaly_score=0.92,
        flow_evidence=0.95,
        npw_evidence=0.88,
        pressure_evidence=0.82,
        fusion_confidence=91.4,
        contributions={"ml": 0.368, "flow": 0.285, "npw": 0.264},
        alert_state=AlertState.LEAK_DETECTED.value,
        persistence=32,
        leak_km=78.2,
        segment="Station 5 (80 km)",
        npw_information={"start_time": 520.0, "peak_score": 0.88},
        is_estimate=True,
        suppression_reason=None
    )


def test_detection_result_fields(sample_detection_result):
    r = sample_detection_result
    assert r.timestamp == 520.4
    assert r.mode == "Flowing"
    assert len(r.pressures) == 6
    assert r.flow_in == 500.2
    assert r.flow_out == 415.1
    assert r.flow_imbalance_raw == 85.1
    assert r.flow_imbalance_corrected == 84.3
    assert r.anomaly_score == 0.92
    assert r.flow_evidence == 0.95
    assert r.npw_evidence == 0.88
    assert r.pressure_evidence == 0.82
    assert r.fusion_confidence == 91.4
    assert r.contributions["ml"] == pytest.approx(0.368)
    assert r.alert_state == "LEAK DETECTED"
    assert r.persistence == 32
    assert r.leak_km == 78.2
    assert r.segment == "Station 5 (80 km)"
    assert r.is_estimate is True
    assert r.suppression_reason is None
    assert r.npw_information["start_time"] == 520.0


def test_detection_result_immutability(sample_detection_result):
    with pytest.raises(FrozenInstanceError):
        sample_detection_result.anomaly_score = 0.50  # type: ignore

    with pytest.raises(FrozenInstanceError):
        sample_detection_result.alert_state = "NORMAL"  # type: ignore


def test_detection_result_serialization(sample_detection_result):
    d = sample_detection_result.to_dict()
    assert isinstance(d, dict)
    assert d["timestamp"] == 520.4
    assert d["alert_state"] == "LEAK DETECTED"
    assert d["pressures"]["P_st1"] == 49.5
    assert d["is_estimate"] is True


def test_detection_result_from_row():
    row = {
        "timestamp": 120.0,
        "operating_mode": "Flowing",
        "P_st1": 50.0, "P_st2": 42.0, "P_st3": 34.0,
        "P_st4": 26.0, "P_st5": 18.0, "P_st6": 10.0,
        "flow_in": 500.0,
        "flow_out": 499.8,
        "raw_flow_imbalance": 0.2,
        "corrected_flow_imbalance": 0.1,
        "ml_evidence_score": 0.15,
        "flow_evidence_score": 0.05,
        "npw_evidence_score": 0.0,
        "pressure_evidence_score": 0.02,
        "confidence_pct": 8.5,
        "status": "NORMAL",
        "estimated_leak_km": None,
        "nearest_station": None
    }
    result = DetectionResult.from_row(row)
    assert result.timestamp == 120.0
    assert result.alert_state == AlertState.NORMAL.value
    assert result.leak_km is None
    assert result.is_estimate is False


def test_alert_state_enum():
    assert AlertState.NORMAL.value == "NORMAL"
    assert AlertState.SUSPECTED_ANOMALY.value == "SUSPECTED ANOMALY"
    assert AlertState.LEAK_DETECTED.value == "LEAK DETECTED"

