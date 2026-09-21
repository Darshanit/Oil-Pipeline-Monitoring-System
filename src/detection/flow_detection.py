"""
Flow Imbalance Detection & Evidence Scoring Module (Backward Compatibility Facade).

Re-exports flow evidence calculation from src.detection.flow_evidence.
"""

from src.detection.flow_evidence import calculate_flow_evidence_score

__all__ = ["calculate_flow_evidence_score"]


def main():
    """Test flow evidence calculation on features dataset."""
    import os
    import pandas as pd
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
