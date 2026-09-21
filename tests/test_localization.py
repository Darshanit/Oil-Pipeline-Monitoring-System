"""
Unit tests for Leak Localization math.
"""

import pytest
import numpy as np
import pandas as pd
from src.localization import (
    localize_leak_from_npw,
    localize_leak_from_hydraulic_drop,
    estimate_leak_location
)


def test_npw_localization_math():
    # Ground truth leak at 78.0 km
    # Station 4 (60km) arrival t4 = 538.0 s (distance 18 km)
    # Station 6 (100km) arrival t6 = 542.0 s (distance 22 km)
    # arrival time diff t6 - t4 = 4.0 s
    arrival_times = {"P_st4": 538.0, "P_st6": 542.0}

    loc_info = localize_leak_from_npw(arrival_times, wave_speed_km_s=1.0)
    assert loc_info is not None
    assert loc_info["estimated_km"] == 78.0
    assert loc_info["method"] == "Negative Pressure Wave Timing"


def test_hydraulic_drop_localization():
    row_data = {
        "P_st1": 50.0,
        "P_st2": 42.0,
        "P_st3": 34.0,
        "P_st4": 26.0,
        "P_st5": 14.0,  # drop at 80 km
        "P_st6": 6.0    # drop at 100 km
    }
    df_row = pd.Series(row_data)

    loc_info = localize_leak_from_hydraulic_drop(df_row)
    assert "estimated_km" in loc_info
    assert "nearest_station" in loc_info
    assert 50.0 <= loc_info["estimated_km"] <= 90.0
