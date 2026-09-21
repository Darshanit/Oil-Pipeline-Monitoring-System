"""
Anomaly Scoring Module.

Scores incoming pipeline telemetry against mode-specific Isolation Forest models.
Normalizes raw decision function scores into continuous [0, 1] anomaly evidence.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.inference.loader import load_mode_models, ModelArtifact

DEFAULT_FEATURE_COLS = [
    "diff_st1_st2", "diff_st2_st3", "diff_st3_st4", "diff_st4_st5", "diff_st5_st6",
    "grad_st1_st2", "grad_st2_st3", "grad_st3_st4", "grad_st4_st5", "grad_st5_st6",
    "grad_total",
    "P_st1_dp_dt", "P_st2_dp_dt", "P_st3_dp_dt", "P_st4_dp_dt", "P_st5_dp_dt", "P_st6_dp_dt",
    "P_st1_var", "P_st2_var", "P_st3_var", "P_st4_var", "P_st5_var", "P_st6_var",
    "corrected_flow_imbalance"
]


class AnomalyScorer:
    """
    Evaluates ML anomaly scores using mode-specific models.
    """

    def __init__(self, models: Optional[Dict[str, ModelArtifact]] = None, model_dir: str = "models"):
        if models is None:
            self.models = load_mode_models(model_dir=model_dir)
        else:
            self.models = models

    def score_single(self, features: Dict[str, float], mode: str = "Flowing") -> float:
        """
        Calculates normalized anomaly score [0, 1] for a single sample dict.
        """
        artifact = self.models.get(mode) or self.models.get("Flowing")
        if artifact is None:
            return 0.0

        feat_vals = np.array([[features.get(f, 0.0) for f in artifact.features]])
        raw_score = artifact.model.decision_function(feat_vals)[0]

        s_min = artifact.norm_params.get("min", -0.2)
        s_max = artifact.norm_params.get("max", 0.2)

        if s_max == s_min:
            return 0.5

        norm = (s_max - raw_score) / (s_max - s_min)
        return float(np.clip(norm, 0.0, 1.0))

    def score_batch(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Vectorized computation of normalized anomaly score S_ML in [0, 1] across a DataFrame.
        """
        n_samples = len(df_features)
        ml_scores = np.zeros(n_samples)

        for mode, artifact in self.models.items():
            mode_mask = df_features["operating_mode"] == mode
            if not np.any(mode_mask):
                continue

            X_mode = df_features.loc[mode_mask, artifact.features].values
            raw_scores = artifact.model.decision_function(X_mode)

            s_min = artifact.norm_params["min"]
            s_max = artifact.norm_params["max"]
            if s_max == s_min:
                normalized = np.full(len(raw_scores), 0.5)
            else:
                normalized = (s_max - raw_scores) / (s_max - s_min)

            ml_scores[mode_mask] = np.clip(normalized, 0.0, 1.0)

        # Fallback for unmodeled rows
        unmodeled = ml_scores == 0.0
        if np.any(unmodeled) and "Flowing" in self.models:
            artifact = self.models["Flowing"]
            X_unmodeled = df_features.loc[unmodeled, artifact.features].values
            raw_scores = artifact.model.decision_function(X_unmodeled)
            s_min, s_max = artifact.norm_params["min"], artifact.norm_params["max"]
            normalized = (s_max - raw_scores) / (s_max - s_min) if s_max != s_min else 0.5
            ml_scores[unmodeled] = np.clip(normalized, 0.0, 1.0)

        return ml_scores

