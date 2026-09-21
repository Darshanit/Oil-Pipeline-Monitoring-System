"""
Negative Pressure Wave (NPW) Detection Module.

Monitors high-frequency 10 Hz pressure rates of change (dP/dt) across all monitoring stations.
Applies noise suppression filtering to distinguish true transient wave fronts from background sensor noise.
Detects sharp pressure drop waves caused by sudden pipe wall ruptures or large leaks.
Extracts wave arrival timestamps per station for acoustic localization.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def detect_negative_pressure_waves(
    df_features: pd.DataFrame,
    dp_dt_threshold: float = -1.20,  # bar/s drop threshold for real NPW front
    min_event_separation_sec: float = 30.0,
    dt_sec: float = 0.1
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Scans dataset for Negative Pressure Wave (NPW) occurrences.

    Parameters:
      df_features: DataFrame containing timestamp and 'P_stX' station pressure columns
      dp_dt_threshold: Threshold below which dP/dt is considered a sharp pressure drop
      min_event_separation_sec: Cooldown window between separate NPW events
      dt_sec: Time step (0.1 s for 10 Hz)

    Returns:
      Tuple of:
        - npw_evidence_score: Array of continuous evidence scores in [0, 1] for each row
        - npw_events: List of detected NPW event dictionaries with arrival details
    """
    timestamps = df_features["timestamp"].values
    n_samples = len(timestamps)

    station_cols = [c for c in df_features.columns if c.startswith("P_st") and not c.endswith(("_dp_dt", "_var"))]
    n_stations = len(station_cols)

    # Compute 5-sample (0.5s) rolling mean smoothed dP/dt across stations
    smoothed_dp_dt = np.zeros((n_samples, n_stations))
    for idx, col in enumerate(station_cols):
        p_smooth = df_features[col].rolling(window=5, min_periods=1, center=True).mean().values
        smoothed_dp_dt[:, idx] = np.gradient(p_smooth, dt_sec)

    npw_evidence_scores = np.zeros(n_samples)
    npw_events = []

    trigger_matrix = smoothed_dp_dt < dp_dt_threshold

    i = 0
    while i < n_samples:
        if np.any(trigger_matrix[i, :]):
            event_start_idx = i
            window_end = min(n_samples, i + int(10.0 / dt_sec))

            arrival_times = {}
            drop_magnitudes = {}

            for st_idx in range(n_stations):
                st_triggers = trigger_matrix[event_start_idx:window_end, st_idx]
                if np.any(st_triggers):
                    first_trigger_offset = np.argmax(st_triggers)
                    arr_idx = event_start_idx + first_trigger_offset
                    st_name = station_cols[st_idx]
                    arrival_times[st_name] = float(timestamps[arr_idx])
                    drop_magnitudes[st_name] = float(smoothed_dp_dt[arr_idx, st_idx])

            if len(arrival_times) >= 2:
                # Valid NPW event detected across multiple stations
                evidence_end_idx = min(n_samples, event_start_idx + int(25.0 / dt_sec))

                max_drop = min(drop_magnitudes.values())
                event_score = float(np.clip(abs(max_drop) / 2.0, 0.75, 1.0))

                npw_evidence_scores[event_start_idx:evidence_end_idx] = event_score

                npw_events.append({
                    "start_time": float(timestamps[event_start_idx]),
                    "affected_stations": list(arrival_times.keys()),
                    "arrival_times": arrival_times,
                    "drop_magnitudes": drop_magnitudes,
                    "peak_score": event_score
                })

                i += int(min_event_separation_sec / dt_sec)
                continue
        i += 1

    return npw_evidence_scores, npw_events


def main():
    import os
    proc_path = os.path.join(os.path.dirname(__file__), "../../data/processed/pipeline_features.csv")
    if os.path.exists(proc_path):
        df_proc = pd.read_csv(proc_path)
        scores, events = detect_negative_pressure_waves(df_proc)
        print(f"[NPW DETECT] Detected {len(events)} true NPW events.")
        for ev in events:
            print(f"  Event at t={ev['start_time']}s | Stations: {ev['affected_stations']} | Arrivals: {ev['arrival_times']}")


if __name__ == "__main__":
    main()
