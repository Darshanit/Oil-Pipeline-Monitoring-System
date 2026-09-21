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
    df = df_raw.copy()
    window_samples = int(max(1, variance_window_sec / dt_sec))

    station_cols = [c for c in df.columns if c.startswith("P_st")]
    n_stations = len(station_cols)

    # 1. dP/dt for each station (bar/s) using central numerical gradient
    dp_dt_cols = []
    for col in station_cols:
        dp_dt_name = f"{col}_dp_dt"
        # 2-point gradient divided by dt
        df[dp_dt_name] = np.gradient(df[col].values, dt_sec)
        dp_dt_cols.append(dp_dt_name)

    # 2. Short-window pressure variance (5s rolling)
    for col in station_cols:
        var_name = f"{col}_var"
        df[var_name] = df[col].rolling(window=window_samples, min_periods=1).var().fillna(0.0)

    # 3. Adjacent Station Differentials (bar)
    diff_cols = []
    for i in range(n_stations - 1):
        c1, c2 = station_cols[i], station_cols[i + 1]
        diff_name = f"diff_st{i+1}_st{i+2}"
        df[diff_name] = df[c1] - df[c2]
        diff_cols.append(diff_name)

    # 4. Section Hydraulic Gradients (bar/km assuming 20 km station spacing)
    station_spacing_km = 20.0
    grad_cols = []
    for i in range(n_stations - 1):
        diff_name = f"diff_st{i+1}_st{i+2}"
        grad_name = f"grad_st{i+1}_st{i+2}"
        df[grad_name] = df[diff_name] / station_spacing_km
        grad_cols.append(grad_name)

    # Total Overall Hydraulic Gradient (bar/km over 100 km)
    total_length_km = 100.0
    df["grad_total"] = (df[station_cols[0]] - df[station_cols[-1]]) / total_length_km

    # 5. Raw Flow Imbalance (m3/h)
    df["raw_flow_imbalance"] = df["flow_in"] - df["flow_out"]

    # 6. Line-Pack Storage Rate Estimation
    # Mean dP/dt across all 6 monitoring stations (bar/s)
    df["mean_dp_dt"] = df[dp_dt_cols].mean(axis=1)

    # Line-pack rate = K_lp * (d P_avg / dt)
    # When pressure rises (d P_avg / dt > 0), oil is accumulating in pipe (inlet > outlet naturally)
    df["linepack_rate"] = linepack_coeff * df["mean_dp_dt"]

    # Corrected Flow Imbalance = Raw Imbalance - Linepack Storage Rate
    df["corrected_flow_imbalance"] = df["raw_flow_imbalance"] - df["linepack_rate"]

    return df


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
