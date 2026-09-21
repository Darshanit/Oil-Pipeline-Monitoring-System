"""
Negative Pressure Wave (NPW) Detection Module (Backward Compatibility Facade).

Re-exports NPW detection from src.detection.npw.
"""

from src.detection.npw import detect_negative_pressure_waves

__all__ = ["detect_negative_pressure_waves"]


def main():
    import os
    import pandas as pd
    proc_path = os.path.join(os.path.dirname(__file__), "../../data/processed/pipeline_features.csv")
    if os.path.exists(proc_path):
        df_proc = pd.read_csv(proc_path)
        scores, events = detect_negative_pressure_waves(df_proc)
        print(f"[NPW DETECT] Detected {len(events)} true NPW events.")
        for ev in events:
            print(f"  Event at t={ev['start_time']}s | Stations: {ev['affected_stations']} | Arrivals: {ev['arrival_times']}")


if __name__ == "__main__":
    main()
