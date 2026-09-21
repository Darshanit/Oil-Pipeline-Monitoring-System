"""
Evidence Fusion & Persistence Engine Module.

Fuses three independent evidence streams:
  1. ML Anomaly Score (Isolation Forest)
  2. Line-pack Corrected Flow Imbalance Score
  3. Negative Pressure Wave (NPW) Score

Applies configurable weighted combination and persistence filtering to produce robust control room alerts.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "ml": 0.40,
    "flow": 0.30,
    "npw": 0.30
}


def fuse_evidence_signals(
    ml_scores: np.ndarray,
    flow_scores: np.ndarray,
    npw_scores: np.ndarray,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
    persistence_window_samples: int = 40,  # 4 seconds at 10 Hz
    leak_threshold: float = 0.65,
    suspect_threshold: float = 0.35
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Combines normalized evidence streams and applies persistence logic.

    Returns:
      Tuple of:
        - df_results: DataFrame with fused scores, confidence %, status, and persistence flags
        - alerts_history: List of trigger event alerts with recommended actions
    """
    n_samples = len(ml_scores)
    w_ml = weights.get("ml", 0.4)
    w_flow = weights.get("flow", 0.3)
    w_npw = weights.get("npw", 0.3)

    total_w = w_ml + w_flow + w_npw
    w_ml, w_flow, w_npw = w_ml / total_w, w_flow / total_w, w_npw / total_w

    raw_fused = (w_ml * ml_scores) + (w_flow * flow_scores) + (w_npw * npw_scores)

    s_series = pd.Series(raw_fused)
    persistent_fused = s_series.rolling(
        window=persistence_window_samples, min_periods=1
    ).mean().values

    confidence_pct = np.round(persistent_fused * 100.0, 1)

    status_list = []
    alerts_history = []

    for i in range(n_samples):
        conf = persistent_fused[i]

        if conf >= leak_threshold:
            status = "LEAK DETECTED"
        elif conf >= suspect_threshold:
            status = "SUSPECTED ANOMALY"
        else:
            status = "NORMAL"

        status_list.append(status)

    df_results = pd.DataFrame({
        "ml_evidence_score": np.round(ml_scores, 3),
        "flow_evidence_score": np.round(flow_scores, 3),
        "npw_evidence_score": np.round(npw_scores, 3),
        "raw_fused_score": np.round(raw_fused, 3),
        "persistent_fused_score": np.round(persistent_fused, 3),
        "confidence_pct": confidence_pct,
        "status": status_list
    })

    return df_results, alerts_history


def get_recommended_action(status: str, location_info: Dict) -> str:
    """Generates human-readable recommendation for control room operators."""
    if status == "LEAK DETECTED":
        loc_str = location_info.get("search_region", "unknown region")
        near_st = location_info.get("nearest_station", "")
        return f"CRITICAL ALERT: Isolate pipeline segment around {loc_str} (near {near_st}). Dispatch emergency field inspection team immediately."
    elif status == "SUSPECTED ANOMALY":
        loc_str = location_info.get("search_region", "monitored section")
        near_st = location_info.get("nearest_station", "")
        return f"WARNING: Elevated multi-sensor anomaly around {loc_str} (near {near_st}). Verify telemetry and monitor flow balance."
    else:
        return "NORMAL OPERATION: All parameters within standard operational baseline. No action required."


def main():
    n = 100
    ml = np.full(n, 0.2)
    flow = np.full(n, 0.1)
    npw = np.full(n, 0.0)
    ml[50:] = 0.85
    flow[50:] = 0.90
    npw[50:70] = 0.95

    df_res, _ = fuse_evidence_signals(ml, flow, npw)
    print(f"[FUSION TEST] Baseline confidence: {df_res['confidence_pct'].iloc[10]}%")
    print(f"[FUSION TEST] Leak confidence: {df_res['confidence_pct'].iloc[60]}%")


if __name__ == "__main__":
    main()
