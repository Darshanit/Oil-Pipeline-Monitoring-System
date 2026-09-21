"""
Synthetic Data Generation Module for Oil Pipeline Pressure & Flow Simulation.

Simulates 6 monitoring stations along a 100 km oil pipeline over 600 seconds.
Includes operational modes (Flowing, Ramping) and realistic event anomalies:
  1. Normal operation
  2. Pump flow ramping event (line-pack accumulation)
  3. Pump start/stop pressure transient
  4. Small chronic leak (57 km, t=330s, ~18 m3/h)
  5. Large leak with Negative Pressure Wave propagation (78 km, t=520s, ~85 m3/h, v=1.0 km/s)
"""

import os
import numpy as np
import pandas as pd


STATIONS_KM = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
STATION_NAMES = [f"P_st{i+1}" for i in range(len(STATIONS_KM))]
WAVE_SPEED_KM_S = 1.0  # 1000 m/s wave propagation speed in oil


def generate_pipeline_dataset(
    duration_sec: float = 600.0,
    dt_pressure: float = 0.1,  # 10 Hz
    dt_flow: float = 1.0,      # 1 Hz (aligned into 10 Hz grid)
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic 10 Hz multi-station pressure and flow dataset.
    """
    np.random.seed(random_seed)
    timestamps = np.arange(0.0, duration_sec, dt_pressure)
    n_samples = len(timestamps)

    # Base array initializations
    operating_mode = np.array(["Flowing"] * n_samples, dtype=object)
    event_ground_truth = np.array(["NORMAL"] * n_samples, dtype=object)

    base_inlet_flow = 500.0  # m3/h
    inlet_flow = np.full(n_samples, base_inlet_flow)
    outlet_flow = np.full(n_samples, base_inlet_flow)

    # Station pressure initializations (bar)
    base_p1 = 50.0
    nominal_dp_per_km = 0.4  # bar/km at 500 m3/h -> St1=50, St6=10 bar
    pressures = np.zeros((n_samples, len(STATIONS_KM)))

    # Compute baseline static hydraulic profile
    for i, dist in enumerate(STATIONS_KM):
        pressures[:, i] = base_p1 - nominal_dp_per_km * dist

    # -------------------------------------------------------------
    # 1. Ramping Event (t = 150s to 240s)
    # -------------------------------------------------------------
    ramp_mask = (timestamps >= 150.0) & (timestamps <= 240.0)
    operating_mode[ramp_mask] = "Ramping"
    event_ground_truth[ramp_mask] = "RAMP"

    # Smooth sigmoid flow increase from 500 to 580 m3/h
    ramp_progress = (timestamps[ramp_mask] - 150.0) / (240.0 - 150.0)
    smooth_ramp = 1.0 / (1.0 + np.exp(-10.0 * (ramp_progress - 0.5)))
    flow_delta = 80.0 * smooth_ramp

    inlet_flow[ramp_mask] += flow_delta
    # Line-pack effect: outlet flow lags inlet flow during ramp up
    outlet_flow[ramp_mask] += flow_delta * 0.70

    # Pressure changes due to higher flow friction (dP proportional to Q^1.8)
    q_ratio = (base_inlet_flow + flow_delta) / base_inlet_flow
    dp_factor = q_ratio ** 1.8
    for i, dist in enumerate(STATIONS_KM):
        pressures[ramp_mask, i] = base_p1 - (nominal_dp_per_km * dist * dp_factor)

    # Post-ramp steady state (t = 240s to 300s)
    post_ramp = (timestamps > 240.0) & (timestamps < 300.0)
    inlet_flow[post_ramp] = 580.0
    outlet_flow[post_ramp] = 580.0
    dp_factor_post = (580.0 / 500.0) ** 1.8
    for i, dist in enumerate(STATIONS_KM):
        pressures[post_ramp, i] = base_p1 - (nominal_dp_per_km * dist * dp_factor_post)

    # -------------------------------------------------------------
    # 2. Pump Start/Stop Transient (t = 260s to 275s)
    # -------------------------------------------------------------
    transient_mask = (timestamps >= 260.0) & (timestamps <= 275.0)
    event_ground_truth[transient_mask] = "TRANSIENT"
    t_transient = timestamps[transient_mask] - 260.0
    # Damped sine pulse representing valve/pump transient disturbance
    transient_pulse = 2.5 * np.exp(-0.4 * t_transient) * np.sin(2.0 * np.pi * 0.3 * t_transient)
    for i, dist in enumerate(STATIONS_KM):
        # Propagation delay along pipeline
        delay = dist / WAVE_SPEED_KM_S
        t_delayed = np.maximum(0, t_transient - delay)
        pulse = 2.5 * np.exp(-0.4 * t_delayed) * np.sin(2.0 * np.pi * 0.3 * t_delayed)
        pressures[transient_mask, i] += pulse

    # -------------------------------------------------------------
    # 3. Small Chronic Leak (Location = 57 km, t = 330s onwards)
    # -------------------------------------------------------------
    small_leak_mask = (timestamps >= 330.0) & (timestamps < 520.0)
    event_ground_truth[small_leak_mask] = "SMALL_LEAK"
    leak_start_t = 330.0
    leak_ramp_duration = 15.0  # 15s gradual leak onset
    small_leak_x = 57.0  # km

    t_leak = np.maximum(0.0, timestamps[small_leak_mask] - leak_start_t)
    small_leak_rate = 16.0 * np.minimum(1.0, t_leak / leak_ramp_duration)  # m3/h

    # Outlet flow decreases by leak rate
    outlet_flow[small_leak_mask] -= small_leak_rate

    # Pressure drop kink around 57 km
    for i, dist in enumerate(STATIONS_KM):
        if dist < small_leak_x:
            # Minor upstream drop
            drop = 0.15 * (small_leak_rate / 16.0)
        else:
            # Significant downstream drop
            drop = 0.75 * (small_leak_rate / 16.0) + 0.01 * (dist - small_leak_x)
        pressures[small_leak_mask, i] -= drop

    # -------------------------------------------------------------
    # 4. Large Leak & Negative Pressure Wave (Location = 78 km, t = 520s onwards)
    # -------------------------------------------------------------
    large_leak_mask = timestamps >= 520.0
    event_ground_truth[large_leak_mask] = "LARGE_LEAK"
    large_leak_x = 78.0  # km
    large_leak_t = 520.0  # s
    large_leak_rate = 85.0  # m3/h

    outlet_flow[large_leak_mask] -= (small_leak_rate[-1] if len(small_leak_rate) > 0 else 16.0) + large_leak_rate

    # NPW transient drop step (-1.8 bar) propagating outward from 78 km
    for i, dist in enumerate(STATIONS_KM):
        distance_to_leak = abs(dist - large_leak_x)
        arrival_time = large_leak_t + (distance_to_leak / WAVE_SPEED_KM_S)

        # Baseline leak drop post-steady state
        if dist < large_leak_x:
            steady_drop = 0.5 + 0.02 * dist
        else:
            steady_drop = 2.2 + 0.03 * (dist - large_leak_x)

        # Sharp wave front transition
        sample_mask = timestamps >= arrival_time
        wave_front = -1.8 * (1.0 - np.exp(-1.5 * (timestamps[sample_mask] - arrival_time)))
        pressures[sample_mask, i] += wave_front - steady_drop

    # Add realistic sensor measurement noise
    p_noise = np.random.normal(0.0, 0.08, size=pressures.shape)
    q_noise_in = np.random.normal(0.0, 1.2, size=n_samples)
    q_noise_out = np.random.normal(0.0, 1.2, size=n_samples)

    pressures += p_noise
    inlet_flow += q_noise_in
    outlet_flow += q_noise_out

    # Construct final dataframe
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


def generate_shut_in_dataset(
    duration_sec: float = 120.0,
    dt_pressure: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic dataset representing a pipeline in SHUT-IN mode:
    Zero flow throughput, static locked-in head pressure (~35 bar) across all stations.
    """
    np.random.seed(random_seed)
    timestamps = np.arange(0.0, duration_sec, dt_pressure)
    n_samples = len(timestamps)

    operating_mode = np.array(["Shut-in"] * n_samples, dtype=object)
    event_ground_truth = np.array(["NORMAL"] * n_samples, dtype=object)

    inlet_flow = np.zeros(n_samples) + np.random.normal(0.0, 0.1, size=n_samples)
    outlet_flow = np.zeros(n_samples) + np.random.normal(0.0, 0.1, size=n_samples)

    # Static line-pack pressure (~35 bar) across all stations
    pressures = np.full((n_samples, len(STATIONS_KM)), 35.0)
    p_noise = np.random.normal(0.0, 0.03, size=pressures.shape)
    pressures += p_noise

    df_dict = {
        "timestamp": np.round(timestamps, 2),
        "operating_mode": operating_mode,
        "event_ground_truth": event_ground_truth,
        "flow_in": np.round(np.maximum(0.0, inlet_flow), 2),
        "flow_out": np.round(np.maximum(0.0, outlet_flow), 2),
    }
    for i, col in enumerate(STATION_NAMES):
        df_dict[col] = np.round(pressures[:, i], 3)

    return pd.DataFrame(df_dict)


def generate_pump_transient_dataset(
    duration_sec: float = 120.0,
    dt_pressure: float = 0.1,
    transient_start_sec: float = 30.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic dataset with a prominent pump start transient event at t=transient_start_sec.
    """
    np.random.seed(random_seed)
    timestamps = np.arange(0.0, duration_sec, dt_pressure)
    n_samples = len(timestamps)

    operating_mode = np.array(["Flowing"] * n_samples, dtype=object)
    event_ground_truth = np.array(["NORMAL"] * n_samples, dtype=object)

    inlet_flow = np.full(n_samples, 500.0)
    outlet_flow = np.full(n_samples, 500.0)

    base_p1 = 50.0
    nominal_dp_per_km = 0.4
    pressures = np.zeros((n_samples, len(STATIONS_KM)))
    for i, dist in enumerate(STATIONS_KM):
        pressures[:, i] = base_p1 - nominal_dp_per_km * dist

    # Pump start transient: sudden surge pulse propagating down the line
    transient_mask = (timestamps >= transient_start_sec) & (timestamps <= transient_start_sec + 15.0)
    event_ground_truth[transient_mask] = "PUMP_START"
    t_trans = timestamps[transient_mask] - transient_start_sec

    # Damped pressure surge oscillation
    for i, dist in enumerate(STATIONS_KM):
        delay = dist / WAVE_SPEED_KM_S
        t_delayed = np.maximum(0, t_trans - delay)
        pulse = 4.0 * np.exp(-0.35 * t_delayed) * np.sin(2.0 * np.pi * 0.4 * t_delayed)
        pressures[transient_mask, i] += pulse

    # Momentary flow kick
    inlet_flow[transient_mask] += 35.0 * np.exp(-0.3 * t_trans)

    p_noise = np.random.normal(0.0, 0.05, size=pressures.shape)
    pressures += p_noise
    inlet_flow += np.random.normal(0.0, 0.8, size=n_samples)
    outlet_flow += np.random.normal(0.0, 0.8, size=n_samples)

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


def generate_valve_event_dataset(
    duration_sec: float = 120.0,
    dt_pressure: float = 0.1,
    valve_event_start_sec: float = 30.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic dataset with a valve maneuver transient at t=valve_event_start_sec.
    """
    np.random.seed(random_seed)
    timestamps = np.arange(0.0, duration_sec, dt_pressure)
    n_samples = len(timestamps)

    operating_mode = np.array(["Flowing"] * n_samples, dtype=object)
    event_ground_truth = np.array(["NORMAL"] * n_samples, dtype=object)

    inlet_flow = np.full(n_samples, 500.0)
    outlet_flow = np.full(n_samples, 500.0)

    base_p1 = 50.0
    nominal_dp_per_km = 0.4
    pressures = np.zeros((n_samples, len(STATIONS_KM)))
    for i, dist in enumerate(STATIONS_KM):
        pressures[:, i] = base_p1 - nominal_dp_per_km * dist

    valve_mask = (timestamps >= valve_event_start_sec) & (timestamps <= valve_event_start_sec + 15.0)
    event_ground_truth[valve_mask] = "VALVE_MANEUVER"
    t_v = timestamps[valve_mask] - valve_event_start_sec

    # Upstream pressure rises, downstream pressure drops temporarily during valve throttle
    for i, dist in enumerate(STATIONS_KM):
        delay = dist / WAVE_SPEED_KM_S
        t_delayed = np.maximum(0, t_v - delay)
        if dist < 50.0:
            pulse = 3.0 * np.exp(-0.3 * t_delayed) * (1.0 - np.cos(2.0 * np.pi * 0.2 * t_delayed))
        else:
            pulse = -2.5 * np.exp(-0.3 * t_delayed) * (1.0 - np.cos(2.0 * np.pi * 0.2 * t_delayed))
        pressures[valve_mask, i] += pulse

    outlet_flow[valve_mask] -= 25.0 * np.exp(-0.3 * t_v)

    p_noise = np.random.normal(0.0, 0.05, size=pressures.shape)
    pressures += p_noise
    inlet_flow += np.random.normal(0.0, 0.8, size=n_samples)
    outlet_flow += np.random.normal(0.0, 0.8, size=n_samples)

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



def main():
    """Generates synthetic dataset and saves to data/raw/."""
    output_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "synthetic_pipeline_data.csv")

    df = generate_pipeline_dataset()
    df.to_csv(filepath, index=False)
    print(f"[DATA GENERATION] Saved synthetic dataset to '{filepath}'. Shape: {df.shape}")


if __name__ == "__main__":
    main()
