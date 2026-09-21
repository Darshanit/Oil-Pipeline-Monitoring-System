"""
Simulation subpackage init.

Provides scenario definitions and signal injection algorithms.
"""

from src.simulation.scenarios import (
    RampScenario,
    TransientScenario,
    SmallLeakScenario,
    LargeLeakScenario,
)
from src.simulation.injector import (
    inject_ramp_event,
    inject_transient_pulse,
    inject_small_leak,
    inject_large_leak_npw,
)

__all__ = [
    "RampScenario",
    "TransientScenario",
    "SmallLeakScenario",
    "LargeLeakScenario",
    "inject_ramp_event",
    "inject_transient_pulse",
    "inject_small_leak",
    "inject_large_leak_npw",
]

