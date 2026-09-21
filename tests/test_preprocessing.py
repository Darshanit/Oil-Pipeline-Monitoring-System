"""
Unit tests for Feature Engineering & Line-Pack Correction.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generation import generate_pipeline_dataset
from src.preprocessing import extract_pipeline_features, DEFAULT_LINEPACK_COEFF


def test_feature_extraction_columns():
    df_raw = generate_pipeline_dataset(duration_sec=30.0, dt_pressure=0.1)
    df_features = extract_pipeline_features(df_raw, linepack_coeff=DEFAULT_LINEPACK_COEFF)

    assert "raw_flow_imbalance" in df_features.columns
    assert "mean_dp_dt" in df_features.columns
    assert "linepack_rate" in df_features.columns
    assert "corrected_flow_imbalance" in df_features.columns
    assert "grad_total" in df_features.columns

    # Test linepack formula: corrected = raw - linepack_rate
    raw_imb = df_features["raw_flow_imbalance"].values
    lp_rate = df_features["linepack_rate"].values
    corr_imb = df_features["corrected_flow_imbalance"].values

    np.testing.assert_allclose(corr_imb, raw_imb - lp_rate, atol=1e-5)
