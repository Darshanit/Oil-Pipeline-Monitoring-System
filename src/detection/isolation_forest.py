"""
Operating Mode Specific Isolation Forest Anomaly Detection Module.

Trains separate Isolation Forest models for each operating mode ('Flowing', 'Ramping', 'Shut-in').
Excludes known leak and transient event windows during training to learn pure baseline behavior.
Converts decision scores into a normalized [0, 1] anomaly evidence score.
"""

import os
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


class ModeIsolationForestDetector:
    """
    Manages mode-specific Isolation Forest models.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.score_min_max = {}  # for score normalization per mode

    def fit(self, df_features: pd.DataFrame, model_dir: str = "models") -> None:
        """
        Trains Isolation Forest models for each distinct operating mode present in df_features.
        Filters training data to include only clean baseline windows (event_ground_truth in ['NORMAL', 'RAMP']).
        """
        os.makedirs(model_dir, exist_ok=True)
        modes = df_features["operating_mode"].unique()

        for mode in modes:
            # Clean baseline data filter (exclude leaks and transient disturbances for pure baseline)
            clean_mask = (df_features["operating_mode"] == mode) & (
                df_features["event_ground_truth"].isin(["NORMAL", "RAMP"])
            )
            df_mode_clean = df_features[clean_mask]

            if len(df_mode_clean) < 10:
                # Fallback if insufficient clean samples for mode
                df_mode_clean = df_features[df_features["operating_mode"] == mode]

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
            self.score_min_max[mode] = {
                "min": float(np.percentile(raw_scores, 1)),
                "max": float(np.percentile(raw_scores, 99))
            }

            mode_slug = mode.lower().replace(" ", "_")
            model_path = os.path.join(model_dir, f"isolation_forest_{mode_slug}.joblib")
            joblib.dump(
                {
                    "model": clf,
                    "features": FEATURE_COLS,
                    "norm_params": self.score_min_max[mode]
                },
                model_path
            )
            self.models[mode] = clf
            print(f"[ML TRAIN] Trained Isolation Forest for mode '{mode}' on {len(X_train)} samples. Saved to '{model_path}'.")

    def load_models(self, model_dir: str = "models") -> None:
        """Loads trained Isolation Forest models from joblib files."""
        for mode in ["Flowing", "Ramping", "Shut-in"]:
            mode_slug = mode.lower().replace(" ", "_")
            model_path = os.path.join(model_dir, f"isolation_forest_{mode_slug}.joblib")
            if os.path.exists(model_path):
                data = joblib.load(model_path)
                self.models[mode] = data["model"]
                self.score_min_max[mode] = data["norm_params"]
                print(f"[ML LOAD] Loaded model for mode '{mode}' from '{model_path}'.")

    def predict_anomaly_scores(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Vectorized computation of normalized anomaly score S_ML in [0, 1] range.
        """
        n_samples = len(df_features)
        ml_scores = np.zeros(n_samples)

        for mode, clf in self.models.items():
            mode_mask = df_features["operating_mode"] == mode
            if not np.any(mode_mask):
                continue

            X_mode = df_features.loc[mode_mask, FEATURE_COLS].values
            raw_scores = clf.decision_function(X_mode)

            norm_p = self.score_min_max[mode]
            s_min, s_max = norm_p["min"], norm_p["max"]
            if s_max == s_min:
                normalized = np.full(len(raw_scores), 0.5)
            else:
                normalized = (s_max - raw_scores) / (s_max - s_min)

            ml_scores[mode_mask] = np.clip(normalized, 0.0, 1.0)

        # Fallback for unmodeled rows
        unmodeled = ml_scores == 0.0
        if np.any(unmodeled) and "Flowing" in self.models:
            X_unmodeled = df_features.loc[unmodeled, FEATURE_COLS].values
            clf = self.models["Flowing"]
            raw_scores = clf.decision_function(X_unmodeled)
            norm_p = self.score_min_max["Flowing"]
            s_min, s_max = norm_p["min"], norm_p["max"]
            normalized = (s_max - raw_scores) / (s_max - s_min) if s_max != s_min else 0.5
            ml_scores[unmodeled] = np.clip(normalized, 0.0, 1.0)

        return ml_scores


def main():
    """Trains mode-specific models and outputs sample anomaly predictions."""
    proc_path = os.path.join(os.path.dirname(__file__), "../../data/processed/pipeline_features.csv")
    model_dir = os.path.join(os.path.dirname(__file__), "../../models")

    if not os.path.exists(proc_path):
        from src.preprocessing.create_features import main as run_proc
        run_proc()

    df_proc = pd.read_csv(proc_path)
    detector = ModeIsolationForestDetector()
    detector.fit(df_proc, model_dir=model_dir)

    scores = detector.predict_anomaly_scores(df_proc)
    df_proc["ml_anomaly_score"] = scores
    print(f"[ML PREDICT] Mean ML anomaly score during NORMAL: {scores[df_proc['event_ground_truth']=='NORMAL'].mean():.3f}")
    print(f"[ML PREDICT] Mean ML anomaly score during LARGE_LEAK: {scores[df_proc['event_ground_truth']=='LARGE_LEAK'].mean():.3f}")


if __name__ == "__main__":
    main()
