"""
Feature Engineering & Line-Pack Correction Module.

Transforms raw 10 Hz pressure & 1 Hz flow data into ML-ready diagnostic features.

Features calculated:
  - Station pressures (P_st1 .. P_st6)
  - Pressure rates of change (dP/dt per station)
  - Short-window pressure rolling variance (5s rolling window)
  - Adjacent station pressure differentials (P1-P2, P2-P3, etc.)
  - Section hydraulic gradients (bar/km) and total pipeline gradient
  - Raw flow imbalance (flow_in - flow_out)
  - Line-pack rate-of-change storage estimation
  - Line-pack corrected flow imbalance

NOTE ON LINE-PACK CORRECTION:
  This implementation uses a simplified prototype approximation:
    Corrected Imbalance = (Flow_in - Flow_out) - K_linepack * (d P_avg / dt)
  In a production industrial pipeline system, line-pack dynamic storage modeling
  requires precise calibration using exact pipe geometry, fluid density, compressibility,
  temperature profiles, and steady-state hydraulic equations.
"""

import os
import numpy as np
import pandas as pd

# Default line-pack coefficient: m3/h per (bar/s)
DEFAULT_LINEPACK_COEFF = 15.0


def extract_pipeline_features(
    df_raw: pd.DataFrame,
    linepack_coeff: float = DEFAULT_LINEPACK_COEFF,
    dt_sec: float = 0.1,  # 10 Hz sampling
    variance_window_sec: float = 5.0
) -> pd.DataFrame:
    """
    Computes diagnostic features for ML anomaly detection and flow evidence.

    Parameters:
      df_raw: Input raw DataFrame with columns ['timestamp', 'operating_mode',
              'event_ground_truth', 'flow_in', 'flow_out', 'P_st1'..'P_st6']
      linepack_coeff: Line-pack storage coefficient K_lp
      dt_sec: Time interval between samples (0.1 s for 10 Hz)
      variance_window_sec: Window length for pressure variance calculation (5.0 s)

    Returns:
      DataFrame enriched with engineered features.
    """
    from src.features.pressure_features import extract_pressure_features
    from src.features.flow_features import extract_flow_features

    # 1-4. Extract pressure derivatives, rolling variance, differentials, and gradients
    df_with_pressure = extract_pressure_features(
        df_raw,
        dt_sec=dt_sec,
        variance_window_sec=variance_window_sec
    )

    # 5-6. Extract raw flow imbalance, line-pack rate, and corrected flow imbalance
    df_complete = extract_flow_features(
        df_with_pressure,
        linepack_coeff=linepack_coeff
    )

    return df_complete



def main():
    """Reads raw data, extracts features, and saves to data/processed/."""
    raw_path = os.path.join(os.path.dirname(__file__), "../../data/raw/synthetic_pipeline_data.csv")
    proc_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    os.makedirs(proc_dir, exist_ok=True)
    out_path = os.path.join(proc_dir, "pipeline_features.csv")

    if not os.path.exists(raw_path):
        from src.data_generation.generate_data import generate_pipeline_dataset
        df_raw = generate_pipeline_dataset()
    else:
        df_raw = pd.read_csv(raw_path)

    df_proc = extract_pipeline_features(df_raw)
    df_proc.to_csv(out_path, index=False)
    print(f"[PREPROCESSING] Extracted features saved to '{out_path}'. Shape: {df_proc.shape}")


if __name__ == "__main__":
    main()
