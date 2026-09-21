"""
Negative Pressure Wave (NPW) Detection Module.

Monitors high-frequency 10 Hz pressure rates of change (dP/dt) across all pipeline stations.
Identifies rapid acoustic rarefaction wave fronts caused by sudden rupture or large leaks.
Extracts:
  - Sharp negative pressure slopes (dP/dt)
  - Event time
  - Station arrival times
  - Affected stations
  - Wave propagation order
  - Estimated acoustic wave speed
  - NPW evidence score in [0.0, 1.0]

PROTOTYPE LIMITATION NOTICE:
----------------------------
The current dataset sampling interval (dt = 0.1s at 10 Hz) inherently limits acoustic arrival
time resolution and wave localization precision:
  - Acoustic wave travel speed in crude oil: ~1.0 km/s (1000 m/s).
  - Spatial quantization limit: dx = (v * dt) / 2 = (1.0 km/s * 0.1s) / 2 = 0.05 km (50 meters).
  - Moving-average filtering (0.5s window) introduces an empirical temporal smearing of +/-0.25s,
    yielding a prototype localization uncertainty of +/-0.25 to 0.50 km.
In industrial field installations, edge microcontrollers with high-frequency acquisition
(1 kHz to 10 kHz) and sub-millisecond GPS/PTP timestamp synchronization are required
for meter-accurate acoustic rupture localization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG

PROTOTYPE_LIMITATION = (
    "PROTOTYPE LIMITATION: Digital sampling at 10 Hz (dt=0.1s) introduces acoustic spatial "
    "quantization of dx = (v_wave * dt) / 2 = 50 m. Arrival smearing from finite filter response "
    "yields +/-0.25–0.5 km uncertainty. Field deployment requires GPS-synchronized 1-10 kHz DAQ."
)


def estimate_wave_speed_from_arrivals(
    arrival_times: Dict[str, float],
    stations_km: Tuple[float, ...] = DEFAULT_CONFIG.station_positions_km,
    nominal_speed_km_s: float = DEFAULT_CONFIG.wave_speed_km_s
) -> float:
    """
    Estimates acoustic wave speed (km/s) from directional arrival time differences.
    """
    if len(arrival_times) < 2:
        return nominal_speed_km_s

    # Map station names to distance
    st_positions = {}
    for st_name, t_arr in arrival_times.items():
        if "P_st" in st_name:
            try:
                idx = int(st_name.replace("P_st", "")) - 1
                if 0 <= idx < len(stations_km):
                    st_positions[st_name] = (stations_km[idx], t_arr)
            except ValueError:
                continue

    if len(st_positions) < 2:
        return nominal_speed_km_s

    # Sort stations by arrival time
    sorted_by_arrival = sorted(st_positions.items(), key=lambda x: x[1][1])
    speed_candidates = []

    # Calculate speeds between sequentially triggered stations propagating in the same direction
    for i in range(len(sorted_by_arrival) - 1):
        st_a, (x_a, t_a) = sorted_by_arrival[i]
        st_b, (x_b, t_b) = sorted_by_arrival[i + 1]
        dt = abs(t_b - t_a)
        dx = abs(x_b - x_a)
        if dt > 0.05 and dx > 0.0:  # avoid divide by zero / simultaneous
            speed = dx / dt
            # Physical validity check for acoustic wave in liquid (0.6 - 1.6 km/s)
            if 0.5 <= speed <= 2.0:
                speed_candidates.append(speed)

    if speed_candidates:
        return float(np.round(np.median(speed_candidates), 2))

    return float(nominal_speed_km_s)


def detect_negative_pressure_waves(
    df_features: pd.DataFrame,
    dp_dt_threshold: float = DEFAULT_CONFIG.npw_dp_dt_threshold,
    min_event_separation_sec: float = DEFAULT_CONFIG.npw_min_event_separation_sec,
    dt_sec: float = DEFAULT_CONFIG.dt_sec,
    smoothing_window: int = DEFAULT_CONFIG.npw_smoothing_window,
    evidence_duration_sec: float = DEFAULT_CONFIG.npw_evidence_duration_sec,
    stations_km: Tuple[float, ...] = DEFAULT_CONFIG.station_positions_km
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Scans dataset for Negative Pressure Wave (NPW) occurrences across stations.

    Returns:
      Tuple of:
        - npw_evidence_scores: Array of continuous evidence scores in [0.0, 1.0] for each row
        - npw_events: List of detected NPW event dictionaries with:
            * event_time: Earliest detection timestamp
            * start_time: Same as event_time (backward compatibility)
            * affected_stations: List of stations triggered by the wave
            * arrival_times: Dict mapping station to arrival timestamp
            * wave_propagation_order: Stations sorted chronologically by wave arrival
            * estimated_wave_speed: Measured acoustic velocity (km/s)
            * drop_magnitudes: Dict mapping station to peak dP/dt drop rate
            * peak_score / npw_evidence: Continuous evidence in [0.0, 1.0]
            * prototype_limitation: Technical note on digital sampling quantization
    """
    timestamps = df_features["timestamp"].values
    n_samples = len(timestamps)

    station_cols = [c for c in df_features.columns if c.startswith("P_st") and not c.endswith(("_dp_dt", "_var"))]
    n_stations = len(station_cols)

    if n_stations == 0:
        return np.zeros(n_samples), []

    # Compute rolling mean smoothed dP/dt across stations
    smoothed_dp_dt = np.zeros((n_samples, n_stations))
    for idx, col in enumerate(station_cols):
        p_smooth = df_features[col].rolling(window=smoothing_window, min_periods=1, center=True).mean().values
        smoothed_dp_dt[:, idx] = np.gradient(p_smooth, dt_sec)

    npw_evidence_scores = np.zeros(n_samples)
    npw_events = []

    trigger_matrix = smoothed_dp_dt < dp_dt_threshold

    i = 0
    while i < n_samples:
        if np.any(trigger_matrix[i, :]):
            event_start_idx = i
            # Lookahead window of 25s to collect arrivals across all stations (stations spaced 20km apart at 1.0 km/s)
            window_end = min(n_samples, i + int(25.0 / dt_sec))

            arrival_times: Dict[str, float] = {}
            drop_magnitudes: Dict[str, float] = {}

            for st_idx in range(n_stations):
                st_triggers = trigger_matrix[event_start_idx:window_end, st_idx]
                if np.any(st_triggers):
                    first_trigger_offset = int(np.argmax(st_triggers))
                    arr_idx = event_start_idx + first_trigger_offset
                    st_name = station_cols[st_idx]
                    arrival_times[st_name] = float(timestamps[arr_idx])
                    drop_magnitudes[st_name] = float(smoothed_dp_dt[arr_idx, st_idx])

            # A valid NPW requires wavefront confirmation at 2 or more stations
            if len(arrival_times) >= 2:
                evidence_end_idx = min(n_samples, event_start_idx + int(evidence_duration_sec / dt_sec))

                # Earliest event arrival timestamp
                earliest_t = min(arrival_times.values())

                # Sort stations in chronological wave propagation order
                wave_propagation_order = sorted(arrival_times.keys(), key=lambda s: arrival_times[s])

                # Estimate wave propagation speed
                est_speed = estimate_wave_speed_from_arrivals(arrival_times, stations_km)

                # Normalized evidence score: proportional to drop magnitude above threshold
                max_drop = min(drop_magnitudes.values())
                # clip to [0.75, 1.0] for confirmed multi-station NPW
                event_score = float(np.clip(abs(max_drop) / 2.0, 0.75, 1.0))

                npw_evidence_scores[event_start_idx:evidence_end_idx] = event_score

                npw_events.append({
                    "event_time": earliest_t,
                    "start_time": float(timestamps[event_start_idx]),
                    "affected_stations": list(arrival_times.keys()),
                    "arrival_times": arrival_times,
                    "wave_propagation_order": wave_propagation_order,
                    "estimated_wave_speed": est_speed,
                    "drop_magnitudes": drop_magnitudes,
                    "peak_score": event_score,
                    "npw_evidence": event_score,
                    "prototype_limitation": PROTOTYPE_LIMITATION
                })

                i += int(min_event_separation_sec / dt_sec)
                continue
        i += 1

    return npw_evidence_scores, npw_events


