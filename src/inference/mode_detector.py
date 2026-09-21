"""
Operating Mode Detector Module.

Classifies pipeline operating state into:
  - 'Shut-in': Pumps stopped, negligible flow (< 10 m3/h)
  - 'Ramping': Flow rate dynamically accelerating/decelerating (|dQ/dt| > threshold)
  - 'Flowing': Steady-state commercial operation
"""

from typing import Union
import numpy as np
import pandas as pd


def detect_operating_mode(
    flow_in: float,
    flow_out: float,
    flow_in_dt: float = 0.0,
    flow_threshold_low: float = 10.0,
    ramp_rate_threshold: float = 0.8  # m3/h per second
) -> str:
    """
    Classifies operating mode for a single time step based on flow dynamics.
    """
    avg_flow = (flow_in + flow_out) / 2.0

    # 1. Shut-in mode check
    if avg_flow < flow_threshold_low:
        return "Shut-in"

    # 2. Ramping mode check
    if abs(flow_in_dt) > ramp_rate_threshold:
        return "Ramping"

    # 3. Steady Flowing mode
    return "Flowing"


def classify_operating_modes_batch(
    df: pd.DataFrame,
    dt_sec: float = 0.1
) -> pd.Series:
    """
    Assigns operating mode across a dataframe if 'operating_mode' not pre-labeled.
    """
    if "operating_mode" in df.columns:
        return df["operating_mode"]

    flow_in = df["flow_in"].values
    flow_out = df["flow_out"].values
    flow_dt = np.gradient(flow_in, dt_sec)

    modes = []
    for q_in, q_out, dq in zip(flow_in, flow_out, flow_dt):
        modes.append(detect_operating_mode(q_in, q_out, dq))

    return pd.Series(modes, index=df.index, name="operating_mode")

