"""
Fusion subpackage init.

Provides evidence fusion functions and alert state machine.
"""

from src.fusion.fusion import (
    fuse_evidence_signals,
    fuse_single_step,
    get_recommended_action,
)
from src.fusion.alert_state import AlertStateMachine, AlertState

__all__ = [
    "fuse_evidence_signals",
    "fuse_single_step",
    "get_recommended_action",
    "AlertStateMachine",
    "AlertState",
]

