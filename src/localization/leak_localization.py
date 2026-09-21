"""
Leak Localization Module.

Estimates the physical location (in km) of a detected oil pipeline leak.

Supports two complementary localization techniques:
  1. Acoustic NPW Arrival Time Difference: Uses wave arrival timestamps t_A, t_B and wave speed v_wave.
     Exact Formula: x_leak = (x_A + x_B - v_wave * (t_B - t_A)) / 2
  2. Hydraulic Pressure Gradient Deflection: Uses station pressure drops and hydraulic gradient kinks.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

STATIONS_KM = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
WAVE_SPEED_KM_S = 1.0  # 1000 m/s


def localize_leak_from_npw(
    arrival_times: Dict[str, float],
    wave_speed_km_s: float = WAVE_SPEED_KM_S
) -> Optional[Dict]:
    """
    Computes leak location using NPW arrival time differences between station pairs.
    """
    if len(arrival_times) < 2:
        return None

    # Parse station numbers and sort by location
    st_indices = []
    for st_name in arrival_times.keys():
        if "P_st" in st_name:
            st_num = int(st_name.replace("P_st", "")) - 1
            if 0 <= st_num < len(STATIONS_KM):
                st_indices.append((st_num, arrival_times[st_name]))

    if len(st_indices) < 2:
        return None

    st_indices.sort(key=lambda x: x[0])

    candidates = []
    for idx in range(len(st_indices) - 1):
        st_a_num, t_a = st_indices[idx]
        st_b_num, t_b = st_indices[idx + 1]

        x_a = STATIONS_KM[st_a_num]
        x_b = STATIONS_KM[st_b_num]

        dt = t_b - t_a  # t_B - t_A

        # Correct acoustic wave localization formula:
        # x_L = (x_A + x_B - v_wave * (t_B - t_A)) / 2
        x_leak = (x_a + x_b - wave_speed_km_s * dt) / 2.0

        if x_a - 10.0 <= x_leak <= x_b + 10.0:
            candidates.append(np.clip(x_leak, 0.0, 100.0))

    if not candidates:
        return None

    est_km = float(np.mean(candidates))
    nearest_st_idx = int(np.argmin([abs(est_km - s) for s in STATIONS_KM]))
    nearest_st_name = f"Station {nearest_st_idx + 1} ({int(STATIONS_KM[nearest_st_idx])} km)"

    search_start = max(0.0, round(est_km - 3.0, 1))
    search_end = min(100.0, round(est_km + 3.0, 1))

    return {
        "estimated_km": round(est_km, 1),
        "nearest_station": nearest_st_name,
        "search_region": f"{search_start} – {search_end} km",
        "method": "Negative Pressure Wave Timing",
        "margin_km": 2.5
    }


def localize_leak_from_hydraulic_drop(
    df_row: pd.Series,
    baseline_row: Optional[pd.Series] = None
) -> Dict:
    """
    Estimates leak location from hydraulic gradient deflection across stations.
    """
    station_cols = [f"P_st{i+1}" for i in range(len(STATIONS_KM))]
    current_pressures = df_row[station_cols].values.astype(float)

    if baseline_row is not None:
        base_pressures = baseline_row[station_cols].values.astype(float)
        p_drops = base_pressures - current_pressures
    else:
        linear_baseline = np.linspace(current_pressures[0], current_pressures[-1], len(STATIONS_KM))
        p_drops = linear_baseline - current_pressures

    # Calculate station-to-station drop increments
    # A leak at x_L causes pressure drop downstream to jump
    drop_diffs = np.diff(p_drops)  # len 5

    # Max positive drop diff indicates station segment where leak is present
    max_seg_idx = int(np.argmax(drop_diffs))
    x_a = STATIONS_KM[max_seg_idx]
    x_b = STATIONS_KM[max_seg_idx + 1]

    # Interpolate location within segment (x_a, x_b) based on drop ratio
    y_a = p_drops[max_seg_idx]
    y_b = p_drops[max_seg_idx + 1]
    total_y = y_a + y_b

    if total_y > 0:
        ratio = y_b / total_y
        est_km = x_a + ratio * (x_b - x_a)
    else:
        est_km = (x_a + x_b) / 2.0

    est_km = float(np.clip(est_km, 0.0, 100.0))
    nearest_st_idx = int(np.argmin([abs(est_km - s) for s in STATIONS_KM]))
    nearest_st_name = f"Station {nearest_st_idx + 1} ({int(STATIONS_KM[nearest_st_idx])} km)"

    search_start = max(0.0, round(est_km - 4.0, 1))
    search_end = min(100.0, round(est_km + 4.0, 1))

    return {
        "estimated_km": round(est_km, 1),
        "nearest_station": nearest_st_name,
        "search_region": f"{search_start} – {search_end} km",
        "method": "Hydraulic Gradient Drop Profile",
        "margin_km": 3.5
    }


def estimate_leak_location(
    df_row: pd.Series,
    npw_event: Optional[Dict] = None,
    baseline_row: Optional[pd.Series] = None
) -> Dict:
    """
    Main localization entry point: combines NPW timing (if available) with hydraulic drop analysis.
    """
    if npw_event is not None and "arrival_times" in npw_event:
        npw_loc = localize_leak_from_npw(npw_event["arrival_times"])
        if npw_loc is not None:
            return npw_loc

    return localize_leak_from_hydraulic_drop(df_row, baseline_row)


def main():
    arrival_times = {"P_st4": 538.0, "P_st5": 522.0}
    res = localize_leak_from_npw(arrival_times)
    print(f"[LOCALIZATION TEST] NPW estimated leak location: {res}")


if __name__ == "__main__":
    main()
