"""
Flow Feature Engineering & Line-Pack Correction Module.

Computes raw flow mass balance, line-pack storage rate of change,
and corrected dynamic flow imbalance.
"""

from typing import Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG


def compute_raw_flow_imbalance(
    flow_in: Union[np.ndarray, pd.Series],
    flow_out: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """Computes raw inlet minus outlet flow imbalance (m3/h)."""
    return np.asarray(flow_in) - np.asarray(flow_out)


def compute_linepack_rate(
    mean_dp_dt: Union[np.ndarray, pd.Series],
    linepack_coeff: float = DEFAULT_CONFIG.default_linepack_coeff
) -> np.ndarray:
    """
    Estimates rate of fluid accumulation (line-pack storage change) in m3/h:
        Linepack Rate = K_lp * (d P_avg / dt)
    """
    return linepack_coeff * np.asarray(mean_dp_dt)


def compute_corrected_flow_imbalance(
    raw_imbalance: Union[np.ndarray, pd.Series],
    linepack_rate: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Computes line-pack corrected flow imbalance:
        Corrected Imbalance = Raw Imbalance - Linepack Rate
    """
    return np.asarray(raw_imbalance) - np.asarray(linepack_rate)


def extract_flow_features(
    df: pd.DataFrame,
    linepack_coeff: float = DEFAULT_CONFIG.default_linepack_coeff
) -> pd.DataFrame:
    """Enriches dataframe with raw and linepack-corrected flow imbalance features."""
    res = df.copy()
    raw_imb = compute_raw_flow_imbalance(res["flow_in"].values, res["flow_out"].values)
    res["raw_flow_imbalance"] = raw_imb

    if "mean_dp_dt" in res.columns:
        lp_rate = compute_linepack_rate(res["mean_dp_dt"].values, linepack_coeff)
    else:
        # Fallback if mean_dp_dt not yet computed
        station_cols = [c for c in res.columns if c.startswith("P_st") and not c.endswith(("_dp_dt", "_var"))]
        dp_dts = [np.gradient(res[col].values, DEFAULT_CONFIG.dt_sec) for col in station_cols]
        mean_dp = np.mean(dp_dts, axis=0)
        res["mean_dp_dt"] = mean_dp
        lp_rate = compute_linepack_rate(mean_dp, linepack_coeff)

    res["linepack_rate"] = lp_rate
    res["corrected_flow_imbalance"] = compute_corrected_flow_imbalance(raw_imb, lp_rate)
    return res

