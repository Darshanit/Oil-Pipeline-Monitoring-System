"""
Operating Mode Gating & Transient Suppression Module.

Provides operational state gating across pipeline modes:
  - 'FLOWING'
  - 'RAMPING'
  - 'SHUT-IN'

Known operational maneuvers (pump starts/stops, valve closures/openings, and ramping onset)
induce hydraulic transients (water hammer, pressure surge reflections, line-pack shifts).
During configurable settling periods following these known transitions:
  1. Leak evidence is suppressed or scaled down.
  2. A descriptive suppression_reason is attached (e.g. "Suppressed: pump-start transient").
  3. Fused confidence is capped below the leak threshold to guarantee that a pump or valve
     transient CANNOT automatically become a CRITICAL (LEAK DETECTED) alert.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG
from src.contract import AlertState


class OperatingModeGating:
    """
    Manages operational gating and transient suppression windows for pipeline monitoring.
    """

    def __init__(
        self,
        pump_settling_sec: float = DEFAULT_CONFIG.transient_settling_sec,
        valve_settling_sec: float = DEFAULT_CONFIG.valve_transient_settling_sec,
        ramp_settling_sec: float = 20.0,
        leak_threshold: float = DEFAULT_CONFIG.leak_confidence_threshold,
        suspect_threshold: float = DEFAULT_CONFIG.suspect_confidence_threshold
    ):
        self.pump_settling_sec = pump_settling_sec
        self.valve_settling_sec = valve_settling_sec
        self.ramp_settling_sec = ramp_settling_sec
        self.leak_threshold = leak_threshold
        self.suspect_threshold = suspect_threshold

        # Active transient tracking: list of (start_time, end_time, reason)
        self.known_transients: List[Tuple[float, float, str]] = []

    def register_transient(self, start_time: float, transient_type: str, duration_sec: Optional[float] = None) -> None:
        """
        Registers a known operational transition event and its settling window.
        """
        t_type = str(transient_type).strip().upper()
        if duration_sec is None:
            if "VALVE" in t_type:
                duration_sec = self.valve_settling_sec
            elif "RAMP" in t_type:
                duration_sec = self.ramp_settling_sec
            else:
                duration_sec = self.pump_settling_sec

        # Standardized suppression reason string formatting
        if "START" in t_type:
            reason = "Suppressed: pump-start transient"
        elif "STOP" in t_type:
            reason = "Suppressed: pump-stop transient"
        elif "VALVE" in t_type:
            reason = "Suppressed: valve maneuver transient"
        elif "RAMP" in t_type:
            reason = "Suppressed: operational ramping transient"
        else:
            reason = f"Suppressed: {transient_type.lower()} transient"

        self.known_transients.append((start_time, start_time + duration_sec, reason))

    def get_suppression(self, timestamp: float, mode: str = "FLOWING") -> Tuple[bool, Optional[str]]:
        """
        Checks whether a given timestamp falls within an active transient settling window.
        Returns: (is_suppressed, suppression_reason)
        """
        canon_mode = str(mode).strip().upper().replace("-", "_")

        for start_t, end_t, reason in self.known_transients:
            if start_t <= timestamp <= end_t:
                return True, reason

        return False, None

    def apply_gating_single(
        self,
        raw_confidence: float,
        timestamp: float,
        mode: str = "FLOWING",
        known_event: Optional[str] = None
    ) -> Tuple[float, str, Optional[str]]:
        """
        Applies gating to a single time step.
        Returns: (gated_confidence, alert_status, suppression_reason)
        """
        if known_event:
            self.register_transient(timestamp, known_event)

        is_suppressed, reason = self.get_suppression(timestamp, mode)

        if is_suppressed:
            # Suppress/attenuate confidence: cap strictly below leak threshold
            # A transient must NEVER automatically become CRITICAL (LEAK DETECTED)
            capped_conf = min(raw_confidence * 0.35, self.leak_threshold * 0.70)
            if capped_conf >= self.suspect_threshold:
                status = AlertState.SUSPECTED_ANOMALY.value
            else:
                status = AlertState.NORMAL.value
            return capped_conf, status, reason

        # Normal unsuppressed behavior
        if raw_confidence >= self.leak_threshold:
            status = AlertState.LEAK_DETECTED.value
        elif raw_confidence >= self.suspect_threshold:
            status = AlertState.SUSPECTED_ANOMALY.value
        else:
            status = AlertState.NORMAL.value

        return raw_confidence, status, None

    def apply_gating_batch(
        self,
        df_results: pd.DataFrame,
        timestamps: Optional[np.ndarray] = None,
        modes: Optional[Union[pd.Series, np.ndarray]] = None,
        event_ground_truth: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Applies operational gating across an entire results DataFrame.
        Enforces transient suppression and prevents false CRITICAL alarms.
        """
        res = df_results.copy()
        n = len(res)

        ts = timestamps if timestamps is not None else res.get("timestamp", np.arange(n, dtype=float)).values
        mode_series = modes if modes is not None else res.get("operating_mode", pd.Series(["Flowing"] * n))

        events = None
        if event_ground_truth is not None:
            events = event_ground_truth
        elif "event_ground_truth" in res.columns:
            events = res["event_ground_truth"]

        # Pre-register any known events from the event column
        if events is not None:
            prev_ev = None
            for t, ev in zip(ts, events):
                ev_str = str(ev).strip().upper()
                if ev_str in ["TRANSIENT", "PUMP_START", "PUMP_STOP", "VALVE_EVENT", "VALVE_MANEUVER", "RAMP"]:
                    if ev_str != prev_ev:
                        self.register_transient(float(t), ev_str)
                prev_ev = ev_str

        suppression_reasons = []
        is_suppressed_list = []
        adjusted_conf = np.array(res["confidence_pct"], copy=True, dtype=float) if "confidence_pct" in res.columns else np.zeros(n)
        adjusted_status = list(res["status"]) if "status" in res.columns else [AlertState.NORMAL.value] * n

        leak_thresh_pct = self.leak_threshold * 100.0
        suspect_thresh_pct = self.suspect_threshold * 100.0

        for i in range(n):
            t = float(ts[i])
            m = str(mode_series.iloc[i] if hasattr(mode_series, "iloc") else mode_series[i])
            is_supp, reason = self.get_suppression(t, m)

            is_suppressed_list.append(is_supp)
            suppression_reasons.append(reason)

            if is_supp:
                # Transient suppression: scale down confidence and strictly cap below leak threshold
                # A pump or valve transient must NEVER become CRITICAL
                curr_conf = adjusted_conf[i]
                capped = min(curr_conf * 0.35, leak_thresh_pct * 0.70)
                adjusted_conf[i] = round(capped, 1)

                if capped >= suspect_thresh_pct:
                    adjusted_status[i] = AlertState.SUSPECTED_ANOMALY.value
                else:
                    adjusted_status[i] = AlertState.NORMAL.value

        res["is_suppressed"] = is_suppressed_list
        res["suppression_reason"] = suppression_reasons
        res["confidence_pct"] = adjusted_conf
        res["status"] = adjusted_status

        return res


def apply_operating_mode_gating(
    df_results: pd.DataFrame,
    timestamps: Optional[np.ndarray] = None,
    modes: Optional[Union[pd.Series, np.ndarray]] = None,
    events: Optional[Union[pd.Series, np.ndarray]] = None,
    pump_settling_sec: float = DEFAULT_CONFIG.transient_settling_sec,
    valve_settling_sec: float = DEFAULT_CONFIG.valve_transient_settling_sec
) -> pd.DataFrame:
    """
    Functional entry point for applying operating mode gating and transient suppression.
    """
    gating = OperatingModeGating(
        pump_settling_sec=pump_settling_sec,
        valve_settling_sec=valve_settling_sec
    )
    return gating.apply_gating_batch(
        df_results=df_results,
        timestamps=timestamps,
        modes=modes,
        event_ground_truth=events
    )
