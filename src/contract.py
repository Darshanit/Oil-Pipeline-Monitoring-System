"""
Detection Result Contract Module.

Defines the immutable DetectionResult dataclass and AlertState enumeration
representing the single source of truth for pipeline monitoring telemetry,
evidence scoring, fusion confidence, and localization.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


class AlertState(str, Enum):
    """Enumeration of operational pipeline alert states."""
    NORMAL = "NORMAL"
    MONITORING = "MONITORING"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    # Backward compatibility aliases
    SUSPECTED_ANOMALY = "SUSPECTED ANOMALY"
    LEAK_DETECTED = "LEAK DETECTED"


@dataclass(frozen=True)
class DetectionResult:
    """
    Immutable data contract representing detection, evidence fusion,
    and localization results at a single point in time.
    """
    timestamp: float
    mode: str
    pressures: Dict[str, float]
    flow_in: float
    flow_out: float
    flow_imbalance_raw: float
    flow_imbalance_corrected: float
    anomaly_score: float
    flow_evidence: float
    npw_evidence: float
    pressure_evidence: float
    fusion_confidence: float
    contributions: Dict[str, float]
    alert_state: str
    persistence: int
    leak_km: Optional[float] = None
    segment: Optional[str] = None
    npw_information: Optional[Dict[str, Any]] = None
    is_estimate: bool = False
    label: str = "PROTOTYPE ESTIMATE"
    is_acknowledged: bool = False
    alert_duration: float = 0.0
    attention_required: bool = False
    suppression_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a flat dictionary suitable for logging or tabular storage."""
        data = asdict(self)
        # Unpack nested pressures into top-level columns if needed
        return data

    @classmethod
    def from_row(
        cls,
        row_dict: Dict[str, Any],
        station_names: Optional[list] = None
    ) -> "DetectionResult":
        """Constructs a DetectionResult instance from a raw or processed dictionary/series."""
        if station_names is None:
            station_names = [f"P_st{i+1}" for i in range(6)]

        pressures = {st: float(row_dict[st]) for st in station_names if st in row_dict}

        contributions = {
            "ml": float(row_dict.get("ml_evidence_score", 0.0)),
            "flow": float(row_dict.get("flow_evidence_score", 0.0)),
            "npw": float(row_dict.get("npw_evidence_score", 0.0)),
            "pressure": float(row_dict.get("pressure_evidence_score", 0.0))
        }

        est_km = row_dict.get("estimated_leak_km")
        if est_km is not None and (isinstance(est_km, float) and (est_km != est_km)):  # check NaN
            est_km = None

        return cls(
            timestamp=float(row_dict["timestamp"]),
            mode=str(row_dict.get("operating_mode", "Flowing")),
            pressures=pressures,
            flow_in=float(row_dict["flow_in"]),
            flow_out=float(row_dict["flow_out"]),
            flow_imbalance_raw=float(row_dict.get("raw_flow_imbalance", row_dict["flow_in"] - row_dict["flow_out"])),
            flow_imbalance_corrected=float(row_dict.get("corrected_flow_imbalance", 0.0)),
            anomaly_score=float(row_dict.get("ml_evidence_score", 0.0)),
            flow_evidence=float(row_dict.get("flow_evidence_score", 0.0)),
            npw_evidence=float(row_dict.get("npw_evidence_score", 0.0)),
            pressure_evidence=float(row_dict.get("pressure_evidence_score", 0.0)),
            fusion_confidence=float(row_dict.get("confidence_pct", 0.0)),
            contributions=contributions,
            alert_state=str(row_dict.get("status", AlertState.NORMAL.value)),
            persistence=int(row_dict.get("persistence_count", 0)),
            leak_km=est_km,
            segment=row_dict.get("nearest_station"),
            npw_information=row_dict.get("npw_info"),
            is_estimate=bool(est_km is not None),
            suppression_reason=row_dict.get("suppression_reason")
        )

