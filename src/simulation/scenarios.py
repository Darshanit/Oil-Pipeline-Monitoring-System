"""
Simulation Scenarios Module.

Defines parameters and profiles for pipeline operational events:
  1. Steady Nominal Flow
  2. Pump Flow Ramping (Line-Pack Accumulation)
  3. Pump Start/Stop Transient Pulse
  4. Small Chronic Leak (Pinhole)
  5. Large Rupture Leak with Negative Pressure Wave (NPW)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RampScenario:
    start_time: float = 150.0
    end_time: float = 240.0
    initial_flow: float = 500.0
    target_flow: float = 580.0
    linepack_lag_factor: float = 0.70


@dataclass(frozen=True)
class TransientScenario:
    start_time: float = 260.0
    end_time: float = 275.0
    pulse_amplitude_bar: float = 2.5
    decay_rate: float = 0.4
    frequency_hz: float = 0.3


@dataclass(frozen=True)
class SmallLeakScenario:
    location_km: float = 57.0
    start_time: float = 330.0
    end_time: float = 520.0
    leak_rate_m3h: float = 16.0
    ramp_duration_sec: float = 15.0


@dataclass(frozen=True)
class LargeLeakScenario:
    location_km: float = 78.0
    start_time: float = 520.0
    leak_rate_m3h: float = 85.0
    npw_drop_bar: float = -1.8
    wave_speed_km_s: float = 1.0

