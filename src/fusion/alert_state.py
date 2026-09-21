"""
Alert State Machine & Hysteresis Engine.

Manages state transitions between NORMAL, SUSPECTED ANOMALY, and LEAK DETECTED.
Implements:
  - M-of-N persistence filtering
  - Dual-threshold hysteresis to prevent alarm chatter
  - Dynamic false alarm suppression reasoning (e.g., pump ramp transients)
"""

from collections import deque
from enum import Enum
from typing import Dict, Optional, Tuple
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.contract import AlertState


class AlertStateMachine:
    """
    Stateful engine evaluating alert conditions with persistence and hysteresis.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.current_state: AlertState = AlertState.NORMAL
        self.history: deque = deque(maxlen=config.persistence_n_samples)
        self.consecutive_leak_count: int = 0
        self.consecutive_suspect_count: int = 0

    def reset(self) -> None:
        """Resets state machine to initial clean conditions."""
        self.current_state = AlertState.NORMAL
        self.history.clear()
        self.consecutive_leak_count = 0
        self.consecutive_suspect_count = 0

    def check_agreement(self, signals: Optional[Dict[str, float]]) -> Tuple[bool, int]:
        """
        Evaluates whether at least `min_agreement_signals` independent evidence channels
        exceed their configured individual detection thresholds.
        """
        if signals is None:
            return True, 0
        triggers = sum(
            1 for sig, thresh in self.config.signal_thresholds.items()
            if signals.get(sig, 0.0) >= thresh
        )
        return triggers >= self.config.min_agreement_signals, triggers

    def update(
        self,
        raw_score: float,
        mode: str = "Flowing",
        flow_imbalance: float = 0.0,
        npw_active: bool = False,
        signals: Optional[Dict[str, float]] = None
    ) -> Tuple[AlertState, int, Optional[str]]:
        """
        Updates state with a new time-step score.

        Parameters:
          raw_score: Instantaneous fused evidence score [0, 1]
          mode: Current operating mode
          flow_imbalance: Linepack-corrected flow imbalance
          npw_active: Whether an NPW wave front was recently detected
          signals: Optional dict of independent signal scores {"ml": ..., "flow": ..., "npw": ..., "pressure": ...}

        Returns:
          Tuple of (current_state, persistence_count, suppression_reason)
        """
        self.history.append(raw_score)
        persistence_len = len(self.history)
        rolling_mean = sum(self.history) / persistence_len if persistence_len > 0 else raw_score

        suppression_reason: Optional[str] = None

        # Check pump ramp line-pack transient suppression
        if mode == "Ramping" and not npw_active and flow_imbalance < self.config.flow_imbalance_threshold_m3h:
            suppression_reason = "Transient line-pack charging during flow ramp; leak alert suppressed."
            self.current_state = AlertState.NORMAL
            return self.current_state, 0, suppression_reason

        # Check Agreement Rule: at least min_agreement_signals must exceed individual thresholds
        if signals is not None:
            agrees, n_agree = self.check_agreement(signals)
            if not agrees and self.current_state == AlertState.NORMAL:
                if raw_score >= self.config.suspect_confidence_threshold:
                    suppression_reason = (
                        f"Suppressed by agreement rule: fewer than {self.config.min_agreement_signals} "
                        f"independent signals agree ({n_agree} active)."
                    )
                    return self.current_state, 0, suppression_reason

        # Count samples exceeding thresholds within the N-sample window
        n_leak_triggers = sum(1 for s in self.history if s >= self.config.leak_confidence_threshold)
        n_suspect_triggers = sum(1 for s in self.history if s >= self.config.suspect_confidence_threshold)

        # State transition logic with Hysteresis & M-of-N persistence
        if self.current_state == AlertState.LEAK_DETECTED:
            # Check hysteresis drop to exit LEAK DETECTED
            if rolling_mean < self.config.hysteresis_leak_recovery:
                if rolling_mean >= self.config.suspect_confidence_threshold:
                    self.current_state = AlertState.SUSPECTED_ANOMALY
                else:
                    self.current_state = AlertState.NORMAL
        elif self.current_state == AlertState.SUSPECTED_ANOMALY:
            # Check promotion to LEAK DETECTED
            if n_leak_triggers >= self.config.persistence_m_triggers or rolling_mean >= self.config.leak_confidence_threshold:
                self.current_state = AlertState.LEAK_DETECTED
            # Check hysteresis drop to exit SUSPECTED ANOMALY
            elif rolling_mean < self.config.hysteresis_suspect_recovery:
                self.current_state = AlertState.NORMAL
        else:  # NORMAL
            if n_leak_triggers >= self.config.persistence_m_triggers or (rolling_mean >= self.config.leak_confidence_threshold and persistence_len >= 10):
                self.current_state = AlertState.LEAK_DETECTED
            elif n_suspect_triggers >= (self.config.persistence_m_triggers // 2) or (rolling_mean >= self.config.suspect_confidence_threshold and persistence_len >= 5):
                self.current_state = AlertState.SUSPECTED_ANOMALY

        # Record suppression reason if raw score is high but persistence not met
        if self.current_state == AlertState.NORMAL and raw_score >= self.config.suspect_confidence_threshold:
            suppression_reason = (
                f"Transient anomaly ({raw_score:.2f}) suppressed by persistence filter "
                f"({n_suspect_triggers}/{self.config.persistence_m_triggers} triggers)."
            )

        persistence_val = n_leak_triggers if self.current_state == AlertState.LEAK_DETECTED else n_suspect_triggers
        return self.current_state, persistence_val, suppression_reason

