"""
Operating Mode Specific Isolation Forest Anomaly Detection Module.

Trains and evaluates separate Isolation Forest models for each operational mode:
  - 'Flowing': Steady-state commercial operation
  - 'Ramping': Flow acceleration / deceleration line-pack dynamics
  - 'Shut-in': Closed pipeline / zero-flow conditions

Score Normalization Documentation:
----------------------------------
Raw Isolation Forest decision function values s(x) follow the scikit-learn convention:
  - Positive values (up to ~ +0.20): inliers / typical baseline behavior.
  - Negative values (down to ~ -0.30): outliers / anomalous data points.

To transform this into a standardized evidence score S_ML in [0.0, 1.0]:
  S_ML = clip((s_max - s(x)) / (s_max - s_min), 0.0, 1.0)
where s_min and s_max correspond to the 1st and 99th percentiles of raw scores observed
on clean, non-leak baseline data for that specific operating mode:
  - 0.0 -> Fully normal (inlier, s(x) >= s_max)
  - 1.0 -> Highly anomalous (outlier, s(x) <= s_min)

CRITICAL OPERATIONAL LIMITATION NOTICE:
---------------------------------------
Isolation Forest alone does NOT detect every leak. Specifically:
  1. Small chronic leaks (e.g. < 2-4% of total throughput) produce subtle pressure and flow
     deviations that frequently fall within standard hydraulic variance envelopes and
     measurement noise distributions.
  2. Slow creeping leaks develop gradually without abrupt multidimensional feature separation.
Reliable pipeline protection strictly requires multi-sensor evidence fusion combining
the mode-specific Isolation Forest with line-pack corrected mass flow balance and
acoustic Negative Pressure Wave (NPW) arrival detection.
"""

import os
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Features selected for ML anomaly detection
FEATURE_COLS = [
    "diff_st1_st2", "diff_st2_st3", "diff_st3_st4", "diff_st4_st5", "diff_st5_st6",
    "grad_st1_st2", "grad_st2_st3", "grad_st3_st4", "grad_st4_st5", "grad_st5_st6",
    "grad_total",
    "P_st1_dp_dt", "P_st2_dp_dt", "P_st3_dp_dt", "P_st4_dp_dt", "P_st5_dp_dt", "P_st6_dp_dt",
    "P_st1_var", "P_st2_var", "P_st3_var", "P_st4_var", "P_st5_var", "P_st6_var",
    "corrected_flow_imbalance"
]


def normalize_mode_name(mode: Union[str, Any]) -> str:
    """
    Normalizes arbitrary mode strings into canonical titles:
      - 'Flowing', 'Ramping', 'Shut-in'
    Handles case variations ('FLOWING', 'ramping', 'SHUT-IN') and separators ('shut_in').
    """
    m = str(mode).strip().upper().replace("-", "_")
    if "FLOW" in m:
        return "Flowing"
    elif "RAMP" in m:
        return "Ramping"
    elif "SHUT" in m:
        return "Shut-in"
    return str(mode).strip().capitalize()


class ModeIsolationForestDetector:
    """
    Manages mode-specific Isolation Forest models for pipeline anomaly detection.
    Selects the active model corresponding to the pipeline's operational state
    ('Flowing', 'Ramping', 'Shut-in').
    """

    def __init__(
        self,
        contamination: float = 0.03,
        random_state: int = 42,
        model_dir: Optional[str] = "models",
        auto_load: bool = True
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.score_min_max: Dict[str, Dict[str, float]] = {}

        if auto_load and model_dir and os.path.exists(model_dir):
            self.load_models(model_dir)

    def fit(self, df_features: pd.DataFrame, model_dir: str = "models") -> None:
        """
        Trains Isolation Forest models for each distinct operating mode present in df_features.
        Filters training data to include only clean baseline windows (event_ground_truth in ['NORMAL', 'RAMP']).
        """
        os.makedirs(model_dir, exist_ok=True)
        raw_modes = df_features["operating_mode"].unique()

        for raw_mode in raw_modes:
            canon_mode = normalize_mode_name(raw_mode)

            # Clean baseline data filter (exclude leaks and transient disturbances for pure baseline)
            mode_mask = df_features["operating_mode"].apply(normalize_mode_name) == canon_mode
            clean_mask = mode_mask & (
                df_features["event_ground_truth"].isin(["NORMAL", "RAMP"])
            )
            df_mode_clean = df_features[clean_mask]

            if len(df_mode_clean) < 10:
                # Fallback if insufficient clean samples for mode
                df_mode_clean = df_features[mode_mask]

            if len(df_mode_clean) == 0:
                continue

            X_train = df_mode_clean[FEATURE_COLS].values

            clf = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            clf.fit(X_train)

            # Record baseline min/max raw decision function scores for normalization
            raw_scores = clf.decision_function(X_train)
            norm_params = {
                "min": float(np.percentile(raw_scores, 1)),
                "max": float(np.percentile(raw_scores, 99))
            }
            self.score_min_max[canon_mode] = norm_params

            mode_slug = canon_mode.lower().replace(" ", "_").replace("-", "_")
            model_path = os.path.join(model_dir, f"isolation_forest_{mode_slug}.joblib")
            joblib.dump(
                {
                    "model": clf,
                    "features": FEATURE_COLS,
                    "norm_params": norm_params
                },
                model_path
            )
            self.models[canon_mode] = clf
            print(f"[ML TRAIN] Trained Isolation Forest for mode '{canon_mode}' on {len(X_train)} samples. Saved to '{model_path}'.")

    def load_models(self, model_dir: str = "models") -> None:
        """Loads trained Isolation Forest models from joblib files."""
        for canon_mode in ["Flowing", "Ramping", "Shut-in"]:
            slug_underscore = canon_mode.lower().replace(" ", "_").replace("-", "_")
            slug_hyphen = canon_mode.lower().replace(" ", "-").replace("_", "-")

            paths_to_try = [
                os.path.join(model_dir, f"isolation_forest_{slug_underscore}.joblib"),
                os.path.join(model_dir, f"isolation_forest_{slug_hyphen}.joblib")
            ]

            loaded = False
            for model_path in paths_to_try:
                if os.path.exists(model_path):
                    try:
                        data = joblib.load(model_path)
                        self.models[canon_mode] = data["model"]
                        self.score_min_max[canon_mode] = data["norm_params"]
                        loaded = True
                        break
                    except Exception as e:
                        print(f"[ML LOAD WARNING] Failed loading '{model_path}': {e}")

            if loaded:
                # Also store uppercase/lowercase aliases for O(1) lookup
                self.models[canon_mode.upper()] = self.models[canon_mode]
                self.score_min_max[canon_mode.upper()] = self.score_min_max[canon_mode]

    def score_single(self, features_dict: Dict[str, float], mode: str = "Flowing") -> float:
        """
        Calculates normalized anomaly score in [0.0, 1.0] for a single sample dictionary.
        """
        canon_mode = normalize_mode_name(mode)
        clf = self.models.get(canon_mode) or self.models.get("Flowing")
        norm_p = self.score_min_max.get(canon_mode) or self.score_min_max.get("Flowing")

        if clf is None or norm_p is None:
            return 0.0

        feat_vals = np.array([[features_dict.get(f, 0.0) for f in FEATURE_COLS]])
        raw_score = clf.decision_function(feat_vals)[0]

        s_min, s_max = norm_p["min"], norm_p["max"]
        if s_max == s_min:
            return 0.5

        # Normalization: 0.0 -> normal baseline, 1.0 -> highly anomalous outlier
        normalized = (s_max - raw_score) / (s_max - s_min)
        return float(np.clip(normalized, 0.0, 1.0))

    def predict_anomaly_scores(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Vectorized computation of normalized anomaly score S_ML in [0.0, 1.0] range.
        Selects model according to each row's operational mode ('Flowing', 'Ramping', 'Shut-in').
        """
        n_samples = len(df_features)
        ml_scores = np.zeros(n_samples)

        if "operating_mode" not in df_features.columns:
            normalized_modes = pd.Series(["Flowing"] * n_samples, index=df_features.index)
        else:
            normalized_modes = df_features["operating_mode"].apply(normalize_mode_name)

        unique_modes = normalized_modes.unique()

        for canon_mode in unique_modes:
            mode_mask = (normalized_modes == canon_mode).values
            if not np.any(mode_mask):
                continue

            clf = self.models.get(canon_mode) or self.models.get("Flowing")
            norm_p = self.score_min_max.get(canon_mode) or self.score_min_max.get("Flowing")

            if clf is None or norm_p is None:
                continue

            X_mode = df_features.loc[mode_mask, FEATURE_COLS].values
            raw_scores = clf.decision_function(X_mode)

            s_min, s_max = norm_p["min"], norm_p["max"]
            if s_max == s_min:
                normalized = np.full(len(raw_scores), 0.5)
            else:
                # Normalization: 0.0 -> normal, 1.0 -> highly anomalous
                normalized = (s_max - raw_scores) / (s_max - s_min)

            ml_scores[mode_mask] = np.clip(normalized, 0.0, 1.0)

        # Fallback for any unmodeled rows
        unmodeled = ml_scores == 0.0
        if np.any(unmodeled) and "Flowing" in self.models:
            clf = self.models["Flowing"]
            norm_p = self.score_min_max.get("Flowing", {"min": -0.1, "max": 0.2})
            X_unmodeled = df_features.loc[unmodeled, FEATURE_COLS].values
            raw_scores = clf.decision_function(X_unmodeled)
            s_min, s_max = norm_p["min"], norm_p["max"]
            normalized = (s_max - raw_scores) / (s_max - s_min) if s_max != s_min else 0.5
            ml_scores[unmodeled] = np.clip(normalized, 0.0, 1.0)

        return ml_scores


def main():
    """Validates mode-specific models and outputs sample anomaly predictions."""
    proc_path = os.path.join(os.path.dirname(__file__), "../../data/processed/pipeline_features.csv")
    model_dir = os.path.join(os.path.dirname(__file__), "../../models")

    detector = ModeIsolationForestDetector(model_dir=model_dir)

    if os.path.exists(proc_path):
        df_proc = pd.read_csv(proc_path)
        scores = detector.predict_anomaly_scores(df_proc)
        df_proc["ml_anomaly_score"] = scores
        if "event_ground_truth" in df_proc.columns:
            print(f"[ML PREDICT] Mean ML anomaly score during NORMAL: {scores[df_proc['event_ground_truth']=='NORMAL'].mean():.3f}")
            print(f"[ML PREDICT] Mean ML anomaly score during LARGE_LEAK: {scores[df_proc['event_ground_truth']=='LARGE_LEAK'].mean():.3f}")


if __name__ == "__main__":
    main()

