"""
Pressure Evidence Scoring Module.

Evaluates spatial hydraulic gradient abnormalities, station pressure differences,
pressure slopes (dP/dt decompression rates), and rolling pressure variances.
Provides normalized pressure evidence score S_Pressure in [0.0, 1.0].
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG
from src.features.pressure_features import extract_pressure_features


def calculate_pressure_evidence_score(
    df_features: Union[pd.DataFrame, Dict[str, float], pd.Series],
    expected_gradient_bar_per_km: float = 0.40,
    gradient_tolerance_bar_per_km: float = DEFAULT_CONFIG.pressure_grad_tolerance_bar_per_km,
    dp_dt_threshold_bar_s: float = DEFAULT_CONFIG.pressure_dp_dt_threshold_bar_s,
    var_threshold_bar2: float = DEFAULT_CONFIG.pressure_var_threshold_bar2,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Computes normalized evidence of abnormal pressure behavior along the pipeline
    combining:
      1. Station pressures and adjacent station pressure differentials
      2. Section hydraulic gradients and total pipeline gradient
      3. Negative pressure slopes (dP/dt rate of pressure loss)
      4. Rolling pressure variances (turbulence and dynamic instability)

    Parameters:
      df_features: DataFrame containing pressure features or raw station pressures,
                   or a single sample dictionary / Series.
      expected_gradient_bar_per_km: Nominal hydraulic gradient (bar/km).
      gradient_tolerance_bar_per_km: Dispersion threshold above which gradient is abnormal.
      dp_dt_threshold_bar_s: Rate of pressure drop triggering depressurization evidence.
      var_threshold_bar2: Rolling variance threshold indicative of dynamic disturbance.
      weights: Optional weighting for sub-components (grad, slope, var).

    Returns:
      Numpy array of pressure evidence scores in [0.0, 1.0].
    """
    if weights is None:
        weights = {"grad": 0.55, "slope": 0.25, "var": 0.20}

    w_grad = weights.get("grad", 0.55)
    w_slope = weights.get("slope", 0.25)
    w_var = weights.get("var", 0.20)
    total_w = w_grad + w_slope + w_var
    if total_w > 0:
        w_grad, w_slope, w_var = w_grad / total_w, w_slope / total_w, w_var / total_w

    # Convert single dict or Series to DataFrame if needed
    is_single = False
    if isinstance(df_features, dict):
        df = pd.DataFrame([df_features])
        is_single = True
    elif isinstance(df_features, pd.Series):
        df = pd.DataFrame([df_features.to_dict()])
        is_single = True
    else:
        df = df_features

    # If engineered pressure features not present, extract them from raw station pressures
    has_grads = any(c.startswith("grad_st") for c in df.columns)
    has_dp_dt = any(c.endswith("_dp_dt") for c in df.columns)
    has_var = any(c.endswith("_var") for c in df.columns)

    if not (has_grads and has_dp_dt and has_var):
        station_cols = [c for c in df.columns if c.startswith("P_st") and not c.endswith(("_dp_dt", "_var"))]
        if station_cols:
            df = extract_pressure_features(df)
        else:
            return np.zeros(len(df))

    n_samples = len(df)

    # 1. Hydraulic Gradient & Differentials (spatial kink detection)
    grad_cols = [c for c in df.columns if c.startswith("grad_st")]
    if grad_cols:
        grads = df[grad_cols].values
        # Gradient spread across sections: normal = low dispersion, leak = kinked/spread
        grad_spread = np.std(grads, axis=1)
        s_grad = 1.0 / (1.0 + np.exp(-120.0 * (grad_spread - gradient_tolerance_bar_per_km)))
        # Suppress baseline floor when well within normal tolerance
        s_grad = np.where(grad_spread < gradient_tolerance_bar_per_km * 0.70, s_grad * 0.15, s_grad)
    else:
        s_grad = np.zeros(n_samples)

    # 2. Pressure Slopes / dP/dt (decompression rate)
    dp_dt_cols = [c for c in df.columns if c.startswith("P_st") and c.endswith("_dp_dt")]
    if dp_dt_cols:
        dp_dts = df[dp_dt_cols].rolling(5, min_periods=1).mean().values if len(df) > 1 else df[dp_dt_cols].values
        # Most negative pressure slope across stations
        min_dp_dt = np.min(dp_dts, axis=1)
        # Higher evidence when negative slope exceeds threshold
        s_slope = 1.0 / (1.0 + np.exp(-8.0 * (-min_dp_dt - abs(dp_dt_threshold_bar_s))))
        s_slope = np.where(-min_dp_dt < abs(dp_dt_threshold_bar_s) * 0.7, s_slope * 0.25, s_slope)
    else:
        s_slope = np.zeros(n_samples)

    # 3. Rolling Pressure Variance (dynamic instability)
    var_cols = [c for c in df.columns if c.startswith("P_st") and c.endswith("_var")]
    if var_cols:
        vars_arr = df[var_cols].values
        max_var = np.max(vars_arr, axis=1)
        s_var = 1.0 / (1.0 + np.exp(-30.0 * (max_var - var_threshold_bar2)))
        # Suppress baseline variance floor
        s_var = np.where(max_var < var_threshold_bar2 * 0.7, s_var * 0.25, s_var)
    else:
        s_var = np.zeros(n_samples)

    # 4. Synthesize overall pressure evidence
    raw_pressure_score = (w_grad * s_grad) + (w_slope * s_slope) + (w_var * s_var)
    clipped_scores = np.clip(raw_pressure_score, 0.0, 1.0)

    return clipped_scores[0] if is_single else clipped_scores


