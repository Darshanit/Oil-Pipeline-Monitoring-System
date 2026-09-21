"""
Evidence Fusion & Persistence Engine Module.

Combines independent evidence streams:
  1. ML Anomaly Score (Isolation Forest per mode)
  2. Line-pack Corrected Flow Imbalance Score
  3. Negative Pressure Wave (NPW) Acoustic Wave Score

Applies weighted fusion and rolling persistence filtering to produce actionable alerts.
Completely decoupled from any UI framework.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.contract import AlertState


def fuse_single_step(
    ml_score: float,
    flow_score: float,
    npw_score: float,
    pressure_score: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    timestamp: float = 0.0,
    mode: str = "FLOWING",
    known_event: Optional[str] = None,
    gating: Optional[Any] = None
) -> Tuple[float, Dict[str, float], str, Optional[str]]:
    """
    Computes weighted fusion, contribution breakdown, alert status, and transient suppression
    for a single time sample.
    """
    if weights is None:
        weights = DEFAULT_CONFIG.fusion_weights

    w_ml = weights.get("ml", 0.4)
    w_flow = weights.get("flow", 0.3)
    w_npw = weights.get("npw", 0.3)
    total_w = w_ml + w_flow + w_npw
    if total_w > 0:
        w_ml, w_flow, w_npw = w_ml / total_w, w_flow / total_w, w_npw / total_w

    raw_fused = (w_ml * ml_score) + (w_flow * flow_score) + (w_npw * npw_score)
    contributions = {
        "ml": float(w_ml * ml_score),
        "flow": float(w_flow * flow_score),
        "npw": float(w_npw * npw_score)
    }
    raw_fused = float(np.clip(raw_fused, 0.0, 1.0))

    suppression_reason = None
    if gating is not None:
        raw_fused, status, suppression_reason = gating.apply_gating_single(
            raw_confidence=raw_fused,
            timestamp=timestamp,
            mode=mode,
            known_event=known_event
        )
    else:
        if raw_fused >= DEFAULT_CONFIG.leak_confidence_threshold:
            status = AlertState.LEAK_DETECTED.value
        elif raw_fused >= DEFAULT_CONFIG.suspect_confidence_threshold:
            status = AlertState.SUSPECTED_ANOMALY.value
        else:
            status = AlertState.NORMAL.value

    return raw_fused, contributions, status, suppression_reason


def fuse_evidence_signals(
    ml_scores: np.ndarray,
    flow_scores: np.ndarray,
    npw_scores: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    persistence_window_samples: int = DEFAULT_CONFIG.persistence_n_samples,
    leak_threshold: float = DEFAULT_CONFIG.leak_confidence_threshold,
    suspect_threshold: float = DEFAULT_CONFIG.suspect_confidence_threshold,
    pressure_scores: Optional[np.ndarray] = None,
    operating_modes: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
    event_types: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
    timestamps: Optional[np.ndarray] = None,
    gating: Optional[Any] = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Combines normalized evidence streams, applies rolling persistence filtering,
    and enforces operating mode gating with transient suppression.

    Parameters:
      ml_scores: Numpy array of ML anomaly evidence in [0, 1]
      flow_scores: Numpy array of flow imbalance evidence in [0, 1]
      npw_scores: Numpy array of NPW wave evidence in [0, 1]
      weights: Dictionary of weights {"ml": ..., "flow": ..., "npw": ...}
      persistence_window_samples: Window length for smoothing (40 samples = 4s at 10 Hz)
      leak_threshold: Threshold to declare LEAK DETECTED (0.65)
      suspect_threshold: Threshold to declare SUSPECTED ANOMALY (0.35)
      pressure_scores: Optional array of pressure evidence scores
      operating_modes: Optional operating modes ('Flowing', 'Ramping', 'Shut-in')
      event_types: Optional event indicators ('TRANSIENT', 'VALVE_EVENT', etc.)
      timestamps: Optional timestamps array
      gating: Optional OperatingModeGating instance

    Returns:
      Tuple of (df_results, alerts_history)
    """
    if weights is None:
        weights = DEFAULT_CONFIG.fusion_weights

    w_ml = weights.get("ml", 0.4)
    w_flow = weights.get("flow", 0.3)
    w_npw = weights.get("npw", 0.3)

    total_w = w_ml + w_flow + w_npw
    if total_w > 0:
        w_ml, w_flow, w_npw = w_ml / total_w, w_flow / total_w, w_npw / total_w

    raw_fused = (w_ml * ml_scores) + (w_flow * flow_scores) + (w_npw * npw_scores)

    s_series = pd.Series(raw_fused)
    persistent_fused = s_series.rolling(
        window=persistence_window_samples, min_periods=1
    ).mean().values

    confidence_pct = np.round(persistent_fused * 100.0, 1)

    n_samples = len(ml_scores)
    status_list = []
    alerts_history = []

    for i in range(n_samples):
        conf = persistent_fused[i]

        if conf >= leak_threshold:
            status = AlertState.LEAK_DETECTED.value
        elif conf >= suspect_threshold:
            status = AlertState.SUSPECTED_ANOMALY.value
        else:
            status = AlertState.NORMAL.value

        status_list.append(status)

    df_dict = {
        "ml_evidence_score": np.round(ml_scores, 3),
        "flow_evidence_score": np.round(flow_scores, 3),
        "npw_evidence_score": np.round(npw_scores, 3),
        "raw_fused_score": np.round(raw_fused, 3),
        "persistent_fused_score": np.round(persistent_fused, 3),
        "confidence_pct": confidence_pct,
        "status": status_list
    }

    if pressure_scores is not None:
        df_dict["pressure_evidence_score"] = np.round(pressure_scores, 3)

    df_results = pd.DataFrame(df_dict)

    # Apply Operating Mode Gating if events or modes provided
    if gating is None and (event_types is not None or operating_modes is not None):
        from src.detection.mode_gating import OperatingModeGating
        gating = OperatingModeGating(leak_threshold=leak_threshold, suspect_threshold=suspect_threshold)

    if gating is not None:
        df_results = gating.apply_gating_batch(
            df_results=df_results,
            timestamps=timestamps,
            modes=operating_modes,
            event_ground_truth=event_types
        )
    else:
        df_results["suppression_reason"] = None

    return df_results, alerts_history


def get_recommended_action(status: str, location_info: Dict) -> str:
    """Generates operator action recommendation string for control room response."""
    if status == AlertState.LEAK_DETECTED.value:
        loc_str = location_info.get("search_region", "unknown region")
        near_st = location_info.get("nearest_station", "")
        return f"CRITICAL ALERT: Isolate pipeline segment around {loc_str} (near {near_st}). Dispatch emergency field inspection team immediately."
    elif status == AlertState.SUSPECTED_ANOMALY.value:
        loc_str = location_info.get("search_region", "monitored section")
        near_st = location_info.get("nearest_station", "")
        return f"WARNING: Elevated multi-sensor anomaly around {loc_str} (near {near_st}). Verify telemetry and monitor flow balance."
    else:
        return "NORMAL OPERATION: All parameters within standard operational baseline. No action required."

