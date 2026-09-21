"""
Flow Evidence Scoring Module.

Calculates normalized flow evidence score S_Flow in [0, 1] from line-pack corrected flow mass balance:
    Corrected Imbalance = (Flow_in - Linepack_Rate) - Flow_out
    Linepack_Rate = K_lp * (d P_avg / dt)

Uses configurable line-pack coefficient and rolling persistence filtering so isolated
noisy samples do not cause false alarms.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG
from src.features.flow_features import compute_linepack_rate, compute_raw_flow_imbalance


def calculate_flow_evidence_score(
    df_or_imbalance: Union[pd.DataFrame, np.ndarray, pd.Series],
    linepack_coeff: float = DEFAULT_CONFIG.default_linepack_coeff,
    rolling_window_samples: int = DEFAULT_CONFIG.flow_rolling_window_samples,
    threshold_m3h: float = DEFAULT_CONFIG.flow_imbalance_threshold_m3h,
    steepness: float = DEFAULT_CONFIG.flow_sigmoid_steepness,
    noise_suppression_m3h: float = DEFAULT_CONFIG.flow_noise_suppression_m3h,
    dt_sec: float = DEFAULT_CONFIG.dt_sec
) -> np.ndarray:
    """
    Computes continuous flow imbalance evidence score S_Flow in [0.0, 1.0] range.

    Parameters:
      df_or_imbalance: DataFrame containing telemetry ('flow_in', 'flow_out', and pressure data)
                       or precomputed 'corrected_flow_imbalance', or numpy array / Series.
      linepack_coeff: Dynamic line-pack storage coefficient K_lp (m3/h per bar/s). Configurable.
      rolling_window_samples: Number of samples in rolling window to ensure persistence and
                              reject isolated noise spikes (e.g. 20 samples = 2.0s at 10 Hz).
      threshold_m3h: Center threshold (m3/h) where evidence reaches 0.5.
      steepness: Logistic growth parameter for smooth sigmoidal evidence scaling.
      noise_suppression_m3h: Cutoff below which baseline noise is strongly suppressed.
      dt_sec: Sampling interval in seconds (default 0.1s for 10 Hz).

    Returns:
      Numpy array of flow evidence scores in [0.0, 1.0].
    """
    if isinstance(df_or_imbalance, pd.DataFrame):
        df = df_or_imbalance
        # Calculate corrected flow imbalance from corrected inlet flow - outlet flow if flow columns exist
        if "flow_in" in df.columns and "flow_out" in df.columns:
            flow_in = df["flow_in"].values
            flow_out = df["flow_out"].values

            # Line-pack rate from average pressure rate of change
            if "mean_dp_dt" in df.columns:
                mean_dp = df["mean_dp_dt"].values
            else:
                station_cols = [c for c in df.columns if c.startswith("P_st") and not c.endswith(("_dp_dt", "_var"))]
                if station_cols:
                    dp_dts = [np.gradient(df[col].values, dt_sec) for col in station_cols]
                    mean_dp = np.mean(dp_dts, axis=0)
                else:
                    mean_dp = np.zeros(len(df))

            lp_rate = compute_linepack_rate(mean_dp, linepack_coeff=linepack_coeff)
            corrected_inlet_flow = flow_in - lp_rate
            corrected_imbalance = corrected_inlet_flow - flow_out
        elif "corrected_flow_imbalance" in df.columns:
            corrected_imbalance = df["corrected_flow_imbalance"].values
        elif "raw_flow_imbalance" in df.columns:
            corrected_imbalance = df["raw_flow_imbalance"].values
        else:
            raise ValueError("DataFrame must contain flow telemetry or imbalance columns.")
    else:
        corrected_imbalance = np.asarray(df_or_imbalance, dtype=float)

    # Apply rolling persistence filter so one isolated noisy sample does not spike evidence
    if rolling_window_samples > 1 and len(corrected_imbalance) >= rolling_window_samples:
        s = pd.Series(corrected_imbalance)
        smoothed_imbalance = s.rolling(
            window=rolling_window_samples,
            min_periods=1
        ).mean().values
    else:
        smoothed_imbalance = corrected_imbalance

    # Sigmoidal mapping centered at threshold_m3h
    scores = 1.0 / (1.0 + np.exp(-steepness * (smoothed_imbalance - threshold_m3h)))

    # Suppress noise below noise_suppression_m3h
    scores = np.where(smoothed_imbalance < noise_suppression_m3h, scores * 0.1, scores)

    return np.clip(scores, 0.0, 1.0)


