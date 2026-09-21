"""
Pressure Feature Engineering Module.

Computes time-series pressure derivatives, short-window rolling variances,
adjacent station differentials, and section hydraulic gradients.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_dp_dt(pressures: np.ndarray, dt_sec: float = 0.1) -> np.ndarray:
    """Computes rate of change of pressure (bar/s) using central numerical gradient."""
    return np.gradient(pressures, dt_sec)


def compute_rolling_variance(pressures: np.ndarray, window_samples: int = 50) -> np.ndarray:
    """Computes rolling variance over a sliding sample window."""
    s = pd.Series(pressures)
    return s.rolling(window=max(1, window_samples), min_periods=1).var().fillna(0.0).values


def compute_station_differentials(
    pressures: np.ndarray
) -> np.ndarray:
    """
    Computes adjacent station pressure differentials (P_i - P_{i+1}).
    pressures: shape (n_samples, n_stations)
    Returns: shape (n_samples, n_stations - 1)
    """
    return pressures[:, :-1] - pressures[:, 1:]


def compute_hydraulic_gradients(
    differentials: np.ndarray,
    station_spacing_km: float = 20.0
) -> np.ndarray:
    """Computes section hydraulic gradients (bar/km)."""
    return differentials / station_spacing_km


def extract_pressure_features(
    df: pd.DataFrame,
    dt_sec: float = 0.1,
    variance_window_sec: float = 5.0,
    station_spacing_km: float = 20.0,
    total_length_km: float = 100.0
) -> pd.DataFrame:
    """
    Extracts all pressure diagnostic features from a dataframe containing P_st* columns.
    Returns a dataframe enriched with pressure feature columns.
    """
    res = df.copy()
    station_cols = [c for c in res.columns if c.startswith("P_st") and not c.endswith(("_dp_dt", "_var"))]
    window_samples = int(max(1, variance_window_sec / dt_sec))

    dp_dt_cols = []
    for col in station_cols:
        dp_dt_name = f"{col}_dp_dt"
        res[dp_dt_name] = compute_dp_dt(res[col].values, dt_sec)
        dp_dt_cols.append(dp_dt_name)

        var_name = f"{col}_var"
        res[var_name] = compute_rolling_variance(res[col].values, window_samples)

    for i in range(len(station_cols) - 1):
        c1, c2 = station_cols[i], station_cols[i + 1]
        diff_name = f"diff_st{i+1}_st{i+2}"
        grad_name = f"grad_st{i+1}_st{i+2}"
        res[diff_name] = res[c1] - res[c2]
        res[grad_name] = res[diff_name] / station_spacing_km

    res["grad_total"] = (res[station_cols[0]] - res[station_cols[-1]]) / total_length_km
    res["mean_dp_dt"] = res[dp_dt_cols].mean(axis=1)

    return res

