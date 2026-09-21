"""
Simulation subpackage init.

Provides scenario definitions and signal injection algorithms.
"""

from src.simulation.scenarios import (
    ScenarioType,
    RampScenario,
    TransientScenario,
    SmallLeakScenario,
    LargeLeakScenario,
    PumpTransientScenario,
    ValveEventScenario,
)
from src.simulation.injector import (
    inject_ramp_event,
    inject_transient_pulse,
    inject_small_leak,
    inject_large_leak_npw,
    inject_pump_transient,
    inject_valve_event,
)
from src.simulation.simulator import (
    generate_scenario_data,
    list_available_scenarios,
)

__all__ = [
    "ScenarioType",
    "RampScenario",
    "TransientScenario",
    "SmallLeakScenario",
    "LargeLeakScenario",
    "PumpTransientScenario",
    "ValveEventScenario",
    "inject_ramp_event",
    "inject_transient_pulse",
    "inject_small_leak",
    "inject_large_leak_npw",
    "inject_pump_transient",
    "inject_valve_event",
    "generate_scenario_data",
    "list_available_scenarios",
]

