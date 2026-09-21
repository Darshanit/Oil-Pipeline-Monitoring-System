"""
Unit tests for Synthetic Data Generation module.
"""

import pytest
from src.data_generation import generate_pipeline_dataset, STATIONS_KM, STATION_NAMES


def test_data_generation_shape_and_columns():
    df = generate_pipeline_dataset(duration_sec=60.0, dt_pressure=0.1)

    # 60 seconds at 10 Hz = 600 samples
    assert len(df) == 600
    assert "timestamp" in df.columns
    assert "operating_mode" in df.columns
    assert "event_ground_truth" in df.columns
    assert "flow_in" in df.columns
    assert "flow_out" in df.columns

    for col in STATION_NAMES:
        assert col in df.columns


def test_station_count_and_locations():
    assert len(STATIONS_KM) == 6
    assert STATIONS_KM[0] == 0.0
    assert STATIONS_KM[-1] == 100.0
