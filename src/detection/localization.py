"""
Leak Localization Module.

Estimates physical leak location (km) along pipeline using:
  1. Acoustic NPW Arrival Timing: x_L = (x_A + x_B - v_wave * (t_B - t_A)) / 2
  2. Hydraulic Pressure Gradient Deflection Profile.

Explicitly accounts for physical sampling resolution limits and outputs uncertainty margins.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG


def localize_leak_from_npw(
    arrival_times: Dict[str, float],
    wave_speed_km_s: float = DEFAULT_CONFIG.wave_speed_km_s,
    stations_km: tuple = DEFAULT_CONFIG.station_positions_km
) -> Optional[Dict]:
    """
    Computes leak location using acoustic NPW arrival time differences between station pairs.
    """
    if len(arrival_times) < 2:
        return None

    # Parse station indices and arrival times
    st_indices = []
    for st_name in arrival_times.keys():
        if "P_st" in st_name:
            st_num = int(st_name.replace("P_st", "")) - 1
            if 0 <= st_num < len(stations_km):
                st_indices.append((st_num, arrival_times[st_name]))

    if len(st_indices) < 2:
        return None

    st_indices.sort(key=lambda x: x[0])

    candidates = []
    for idx in range(len(st_indices) - 1):
        st_a_num, t_a = st_indices[idx]
        st_b_num, t_b = st_indices[idx + 1]

        x_a = stations_km[st_a_num]
        x_b = stations_km[st_b_num]

        dt = t_b - t_a

        # Acoustic wave front localization formula:
        # x_L = (x_A + x_B - v_wave * (t_B - t_A)) / 2
        x_leak = (x_a + x_b - wave_speed_km_s * dt) / 2.0

        if x_a - 10.0 <= x_leak <= x_b + 10.0:
            candidates.append(np.clip(x_leak, 0.0, DEFAULT_CONFIG.pipeline_length_km))

    if not candidates:
        return None

    est_km = float(np.mean(candidates))
    nearest_st_idx = int(np.argmin([abs(est_km - s) for s in stations_km]))
    nearest_st_name = f"Station {nearest_st_idx + 1} ({int(stations_km[nearest_st_idx])} km)"

    margin = DEFAULT_CONFIG.empirical_localization_margin_km
    search_start = max(0.0, round(est_km - margin, 1))
    search_end = min(DEFAULT_CONFIG.pipeline_length_km, round(est_km + margin, 1))

    return {
        "estimated_km": round(est_km, 1),
        "nearest_station": nearest_st_name,
        "search_region": f"{search_start} – {search_end} km",
        "method": "Negative Pressure Wave Timing",
        "margin_km": margin,
        "is_estimate": True
    }


def localize_leak_from_hydraulic_drop(
    df_row: Union[pd.Series, Dict],
    baseline_row: Optional[Union[pd.Series, Dict]] = None,
    stations_km: tuple = DEFAULT_CONFIG.station_positions_km
) -> Dict:
    """
    Estimates leak location from hydraulic gradient deflection across stations.
    """
    station_cols = [f"P_st{i+1}" for i in range(len(stations_km))]
    current_pressures = np.array([float(df_row[col]) for col in station_cols])

    if baseline_row is not None:
        base_pressures = np.array([float(baseline_row[col]) for col in station_cols])
        p_drops = base_pressures - current_pressures
    else:
        linear_baseline = np.linspace(current_pressures[0], current_pressures[-1], len(stations_km))
        p_drops = linear_baseline - current_pressures

    drop_diffs = np.diff(p_drops)
    max_seg_idx = int(np.argmax(drop_diffs))
    x_a = stations_km[max_seg_idx]
    x_b = stations_km[max_seg_idx + 1]

    y_a = p_drops[max_seg_idx]
    y_b = p_drops[max_seg_idx + 1]
    total_y = y_a + y_b

    if total_y > 0:
        ratio = y_b / total_y
        est_km = x_a + ratio * (x_b - x_a)
    else:
        est_km = (x_a + x_b) / 2.0

    est_km = float(np.clip(est_km, 0.0, DEFAULT_CONFIG.pipeline_length_km))
    nearest_st_idx = int(np.argmin([abs(est_km - s) for s in stations_km]))
    nearest_st_name = f"Station {nearest_st_idx + 1} ({int(stations_km[nearest_st_idx])} km)"

    margin = 3.5
    search_start = max(0.0, round(est_km - margin, 1))
    search_end = min(DEFAULT_CONFIG.pipeline_length_km, round(est_km + margin, 1))

    return {
        "estimated_km": round(est_km, 1),
        "nearest_station": nearest_st_name,
        "search_region": f"{search_start} – {search_end} km",
        "method": "Hydraulic Gradient Drop Profile",
        "margin_km": margin,
        "is_estimate": True
    }


def estimate_leak_location(
    df_row: Union[pd.Series, Dict],
    npw_event: Optional[Dict] = None,
    baseline_row: Optional[Union[pd.Series, Dict]] = None
) -> Dict:
    """
    Main localization entry point: uses acoustic NPW arrival timing if available,
    otherwise falls back to hydraulic gradient deflection.
    """
    if npw_event is not None and "arrival_times" in npw_event:
        npw_loc = localize_leak_from_npw(npw_event["arrival_times"])
        if npw_loc is not None:
            return npw_loc

    return localize_leak_from_hydraulic_drop(df_row, baseline_row)

