"""
Deterministic Scenario Generator Module.

Provides deterministic simulation data generation for 5 canonical demonstration scenarios:
  1. NORMAL: Steady nominal flowing state (no anomalies)
  2. SMALL CHRONIC LEAK: Pinhole chronic leak (~16 m3/h at 57 km)
  3. LARGE LEAK: Rupture leak with acoustic NPW propagation (~85 m3/h at 78 km)
  4. PUMP TRANSIENT: Secondary pump start surge oscillation
  5. VALVE EVENT: Valve throttling maneuver transient

Adheres strictly to the real 11-column dataset schema and does NOT modify the original raw file.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG
from src.simulation.scenarios import (
    ScenarioType,
    SmallLeakScenario,
    LargeLeakScenario,
    PumpTransientScenario,
    ValveEventScenario
)
from src.simulation.injector import (
    inject_small_leak,
    inject_large_leak_npw,
    inject_pump_transient,
    inject_valve_event
)

STATIONS_KM = list(DEFAULT_CONFIG.station_positions_km)
STATION_NAMES = [f"P_st{i+1}" for i in range(len(STATIONS_KM))]


def list_available_scenarios() -> List[str]:
    """Returns list of canonical demonstration scenario names."""
    return [s.value for s in ScenarioType]


def generate_scenario_data(
    scenario_type: Union[ScenarioType, str] = ScenarioType.NORMAL,
    duration_sec: float = 120.0,
    dt_pressure: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a deterministic synthetic dataset for the specified scenario.

    Parameters:
      scenario_type: ScenarioType enum or string matching scenario name.
      duration_sec: Total duration in seconds (default 120.0s = 1200 samples at 10 Hz).
      dt_pressure: Sampling interval (default 0.1s = 10 Hz).
      random_seed: Fixed seed for deterministic noise and repeatability.

    Returns:
      DataFrame matching the real-format 11-column schema:
        ['timestamp', 'operating_mode', 'event_ground_truth', 'flow_in', 'flow_out',
         'P_st1', 'P_st2', 'P_st3', 'P_st4', 'P_st5', 'P_st6']
    """
    if isinstance(scenario_type, str):
        normalized = scenario_type.strip().upper()
        matched = None
        for st in ScenarioType:
            if st.value.upper() == normalized or st.name.upper() == normalized:
                matched = st
                break
        if matched is None:
            raise ValueError(f"Unknown scenario '{scenario_type}'. Available: {list_available_scenarios()}")
        scenario_type = matched

    np.random.seed(random_seed)
    timestamps = np.arange(0.0, duration_sec, dt_pressure)
    n_samples = len(timestamps)

    # Base operational baseline: Steady nominal flowing state
    operating_mode = np.array(["Flowing"] * n_samples, dtype=object)
    event_ground_truth = np.array(["NORMAL"] * n_samples, dtype=object)

    base_inlet_flow = 500.0  # m3/h
    inlet_flow = np.full(n_samples, base_inlet_flow)
    outlet_flow = np.full(n_samples, base_inlet_flow)

    # Hydraulic static profile: St1=50 bar, 0.4 bar/km gradient
    base_p1 = 50.0
    nominal_dp_per_km = 0.4
    pressures = np.zeros((n_samples, len(STATIONS_KM)))
    for i, dist in enumerate(STATIONS_KM):
        pressures[:, i] = base_p1 - nominal_dp_per_km * dist

    # Inject scenario anomaly signature
    if scenario_type == ScenarioType.NORMAL:
        pass

    elif scenario_type == ScenarioType.SMALL_CHRONIC_LEAK:
        scen = SmallLeakScenario(
            location_km=57.0,
            start_time=30.0,
            end_time=duration_sec,
            leak_rate_m3h=16.0,
            ramp_duration_sec=15.0
        )
        mask, outlet_flow, pressures = inject_small_leak(
            timestamps=timestamps,
            outlet_flow=outlet_flow,
            pressures=pressures,
            scenario=scen,
            stations_km=DEFAULT_CONFIG.station_positions_km
        )
        event_ground_truth[mask] = "SMALL_LEAK"

    elif scenario_type == ScenarioType.LARGE_LEAK:
        scen = LargeLeakScenario(
            location_km=78.0,
            start_time=35.0,
            leak_rate_m3h=85.0,
            npw_drop_bar=-1.8,
            wave_speed_km_s=1.0
        )
        mask, outlet_flow, pressures = inject_large_leak_npw(
            timestamps=timestamps,
            outlet_flow=outlet_flow,
            pressures=pressures,
            scenario=scen,
            stations_km=DEFAULT_CONFIG.station_positions_km
        )
        event_ground_truth[mask] = "LARGE_LEAK"

    elif scenario_type == ScenarioType.PUMP_TRANSIENT:
        scen = PumpTransientScenario(
            start_time=30.0,
            duration_sec=15.0,
            surge_pressure_bar=4.0,
            flow_kick_m3h=35.0
        )
        mask, inlet_flow, pressures = inject_pump_transient(
            timestamps=timestamps,
            inlet_flow=inlet_flow,
            pressures=pressures,
            scenario=scen,
            stations_km=DEFAULT_CONFIG.station_positions_km
        )
        event_ground_truth[mask] = "PUMP_START"

    elif scenario_type == ScenarioType.VALVE_EVENT:
        scen = ValveEventScenario(
            start_time=30.0,
            duration_sec=15.0,
            upstream_surge_bar=3.0,
            downstream_drop_bar=-2.5,
            flow_dip_m3h=25.0
        )
        mask, outlet_flow, pressures = inject_valve_event(
            timestamps=timestamps,
            outlet_flow=outlet_flow,
            pressures=pressures,
            scenario=scen,
            stations_km=DEFAULT_CONFIG.station_positions_km
        )
        event_ground_truth[mask] = "VALVE_MANEUVER"

    # Add realistic sensor measurement noise using fixed seed
    p_noise = np.random.normal(0.0, 0.05, size=pressures.shape)
    q_noise_in = np.random.normal(0.0, 0.8, size=n_samples)
    q_noise_out = np.random.normal(0.0, 0.8, size=n_samples)

    pressures += p_noise
    inlet_flow += q_noise_in
    outlet_flow += q_noise_out

    df_dict = {
        "timestamp": np.round(timestamps, 2),
        "operating_mode": operating_mode,
        "event_ground_truth": event_ground_truth,
        "flow_in": np.round(inlet_flow, 2),
        "flow_out": np.round(outlet_flow, 2),
    }
    for i, col in enumerate(STATION_NAMES):
        df_dict[col] = np.round(pressures[:, i], 3)

    return pd.DataFrame(df_dict)
