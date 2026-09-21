"""
Signal Anomaly Injector Module.

Provides modular signal injection algorithms to apply operational transitions,
hydraulic leaks, acoustic waves, and sensor noise to pipeline signals.
"""

from typing import Tuple
import numpy as np
from src.config import DEFAULT_CONFIG
from src.simulation.scenarios import (
    RampScenario,
    TransientScenario,
    SmallLeakScenario,
    LargeLeakScenario
)


def inject_ramp_event(
    timestamps: np.ndarray,
    inlet_flow: np.ndarray,
    outlet_flow: np.ndarray,
    pressures: np.ndarray,
    scenario: RampScenario = RampScenario(),
    stations_km: tuple = DEFAULT_CONFIG.station_positions_km,
    base_inlet_flow: float = 500.0,
    nominal_dp_per_km: float = 0.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Injects pump flow ramping and line-pack dynamic charging effect."""
    ramp_mask = (timestamps >= scenario.start_time) & (timestamps <= scenario.end_time)
    ramp_progress = (timestamps[ramp_mask] - scenario.start_time) / (scenario.end_time - scenario.start_time)
    smooth_ramp = 1.0 / (1.0 + np.exp(-10.0 * (ramp_progress - 0.5)))
    flow_delta = (scenario.target_flow - scenario.initial_flow) * smooth_ramp

    inlet_flow[ramp_mask] += flow_delta
    outlet_flow[ramp_mask] += flow_delta * scenario.linepack_lag_factor

    q_ratio = (base_inlet_flow + flow_delta) / base_inlet_flow
    dp_factor = q_ratio ** 1.8
    for i, dist in enumerate(stations_km):
        pressures[ramp_mask, i] = 50.0 - (nominal_dp_per_km * dist * dp_factor)

    # Post-ramp steady state
    post_ramp = (timestamps > scenario.end_time) & (timestamps < 300.0)
    inlet_flow[post_ramp] = scenario.target_flow
    outlet_flow[post_ramp] = scenario.target_flow
    dp_factor_post = (scenario.target_flow / base_inlet_flow) ** 1.8
    for i, dist in enumerate(stations_km):
        pressures[post_ramp, i] = 50.0 - (nominal_dp_per_km * dist * dp_factor_post)

    return ramp_mask, inlet_flow, outlet_flow, pressures


def inject_transient_pulse(
    timestamps: np.ndarray,
    pressures: np.ndarray,
    scenario: TransientScenario = TransientScenario(),
    stations_km: tuple = DEFAULT_CONFIG.station_positions_km,
    wave_speed_km_s: float = DEFAULT_CONFIG.wave_speed_km_s
) -> Tuple[np.ndarray, np.ndarray]:
    """Injects damped sine transient wave propagating across pipeline."""
    mask = (timestamps >= scenario.start_time) & (timestamps <= scenario.end_time)
    t_transient = timestamps[mask] - scenario.start_time
    for i, dist in enumerate(stations_km):
        delay = dist / wave_speed_km_s
        t_delayed = np.maximum(0, t_transient - delay)
        pulse = scenario.pulse_amplitude_bar * np.exp(-scenario.decay_rate * t_delayed) * np.sin(
            2.0 * np.pi * scenario.frequency_hz * t_delayed
        )
        pressures[mask, i] += pulse
    return mask, pressures


def inject_small_leak(
    timestamps: np.ndarray,
    outlet_flow: np.ndarray,
    pressures: np.ndarray,
    scenario: SmallLeakScenario = SmallLeakScenario(),
    stations_km: tuple = DEFAULT_CONFIG.station_positions_km
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Injects small chronic leak with gradual onset and hydraulic drop profile."""
    mask = (timestamps >= scenario.start_time) & (timestamps < scenario.end_time)
    t_leak = np.maximum(0.0, timestamps[mask] - scenario.start_time)
    small_leak_rate = scenario.leak_rate_m3h * np.minimum(1.0, t_leak / scenario.ramp_duration_sec)

    outlet_flow[mask] -= small_leak_rate

    for i, dist in enumerate(stations_km):
        if dist < scenario.location_km:
            drop = 0.15 * (small_leak_rate / scenario.leak_rate_m3h)
        else:
            drop = 0.75 * (small_leak_rate / scenario.leak_rate_m3h) + 0.01 * (dist - scenario.location_km)
        pressures[mask, i] -= drop

    return mask, outlet_flow, pressures


def inject_large_leak_npw(
    timestamps: np.ndarray,
    outlet_flow: np.ndarray,
    pressures: np.ndarray,
    scenario: LargeLeakScenario = LargeLeakScenario(),
    stations_km: tuple = DEFAULT_CONFIG.station_positions_km
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Injects large rupture leak with Negative Pressure Wave outward propagation."""
    mask = timestamps >= scenario.start_time
    outlet_flow[mask] -= scenario.leak_rate_m3h

    for i, dist in enumerate(stations_km):
        distance_to_leak = abs(dist - scenario.location_km)
        arrival_time = scenario.start_time + (distance_to_leak / scenario.wave_speed_km_s)

        if dist < scenario.location_km:
            steady_drop = 0.5 + 0.02 * dist
        else:
            steady_drop = 2.2 + 0.03 * (dist - scenario.location_km)

        sample_mask = timestamps >= arrival_time
        wave_front = scenario.npw_drop_bar * (1.0 - np.exp(-1.5 * (timestamps[sample_mask] - arrival_time)))
        pressures[sample_mask, i] += wave_front - steady_drop

    return mask, outlet_flow, pressures

