"""
Flow Imbalance Detection & Evidence Scoring Module.

Calculates normalized flow evidence score S_Flow in [0, 1] from line-pack corrected flow imbalance.
Distinguishes between operational line-pack transients and true hydraulic mass imbalances.
"""

import numpy as np
import pandas as pd


def calculate_flow_evidence_score(
    df_features: pd.DataFrame,
    threshold_m3h: float = 8.0,
    steepness: float = 0.15
) -> np.ndarray:
    """
    Computes continuous flow imbalance evidence score S_Flow in [0, 1] range.

    Parameters:
      df_features: DataFrame containing 'corrected_flow_imbalance' column
      threshold_m3h: Flow imbalance threshold (m3/h) above which evidence rises steeply
      steepness: Sigmoid growth rate parameter

    Returns:
      Numpy array of flow evidence scores in [0, 1].
    """
    corrected_imbalance = df_features["corrected_flow_imbalance"].values

    # Sigmoidal mapping centered around threshold_m3h
    # Imbalance <= 0 m3/h -> ~0.0
    # Imbalance == threshold_m3h (8 m3/h) -> ~0.5
    # Imbalance >= 30 m3/h -> ~0.99
    scores = 1.0 / (1.0 + np.exp(-steepness * (corrected_imbalance - threshold_m3h)))

    # Ensure baseline noise (< 3 m3/h) remains suppressed near 0.0
    scores = np.where(corrected_imbalance < 3.0, scores * 0.1, scores)

    return np.clip(scores, 0.0, 1.0)


def main():
    """Test flow evidence calculation on features dataset."""
    import os
    proc_path = os.path.join(os.path.dirname(__file__), "../../data/processed/pipeline_features.csv")
    if os.path.exists(proc_path):
        df_proc = pd.read_csv(proc_path)
        flow_scores = calculate_flow_evidence_score(df_proc)
        df_proc["flow_evidence_score"] = flow_scores
        print(f"[FLOW DETECT] Normal mean score: {flow_scores[df_proc['event_ground_truth']=='NORMAL'].mean():.3f}")
        print(f"[FLOW DETECT] Small leak mean score: {flow_scores[df_proc['event_ground_truth']=='SMALL_LEAK'].mean():.3f}")
        print(f"[FLOW DETECT] Large leak mean score: {flow_scores[df_proc['event_ground_truth']=='LARGE_LEAK'].mean():.3f}")


if __name__ == "__main__":
    main()
