"""
Evidence Fusion & Persistence Engine Module (Backward Compatibility Facade).

Re-exports fusion logic from src.fusion.fusion.
"""

from src.fusion.fusion import (
    DEFAULT_CONFIG,
    fuse_evidence_signals,
    fuse_single_step,
    get_recommended_action,
)

DEFAULT_WEIGHTS = DEFAULT_CONFIG.fusion_weights

__all__ = [
    "DEFAULT_WEIGHTS",
    "fuse_evidence_signals",
    "fuse_single_step",
    "get_recommended_action",
]


def main():
    import numpy as np
    n = 100
    ml = np.full(n, 0.2)
    flow = np.full(n, 0.1)
    npw = np.full(n, 0.0)
    ml[50:] = 0.85
    flow[50:] = 0.90
    npw[50:70] = 0.95

    df_res, _ = fuse_evidence_signals(ml, flow, npw)
    print(f"[FUSION TEST] Baseline confidence: {df_res['confidence_pct'].iloc[10]}%")
    print(f"[FUSION TEST] Leak confidence: {df_res['confidence_pct'].iloc[60]}%")


if __name__ == "__main__":
    main()
