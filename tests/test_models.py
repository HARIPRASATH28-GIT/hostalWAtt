"""Unit tests for anomaly detection models and baseline outputs."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def scored_test_data():
    """Load pre-scored evaluation dataset."""
    path = Path("data/processed/test_evaluation_scored.parquet")
    if not path.exists():
        from src.models import train_and_save_all_models
        df_scored, _ = train_and_save_all_models()
        return df_scored
    return pd.read_parquet(path)


def test_model_predictions_binary(scored_test_data):
    """Verify that all models output binary 0/1 anomaly predictions."""
    for col in ["b1_pred", "b2_pred", "m1_pred", "m2_pred"]:
        assert col in scored_test_data.columns, f"Missing prediction column {col}"
        unique_vals = set(scored_test_data[col].unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary prediction values in {col}: {unique_vals}"


def test_severity_scores_bounded(scored_test_data):
    """Verify that severity scores are bounded within [0, 100]."""
    for col in ["b1_severity", "b2_severity", "m1_severity", "m2_severity"]:
        assert col in scored_test_data.columns, f"Missing severity column {col}"
        assert (scored_test_data[col] >= 0.0).all(), f"Negative severity in {col}"
        assert (scored_test_data[col] <= 100.0).all(), f"Severity exceeding 100 in {col}"


def test_expected_kwh_non_negative(scored_test_data):
    """Verify that baseline expected kWh estimations are non-negative."""
    for col in ["b1_expected", "b2_expected", "m1_expected", "m2_expected"]:
        assert col in scored_test_data.columns, f"Missing expected column {col}"
        assert (scored_test_data[col] >= 0.0).all(), f"Negative expected kWh in {col}"


def test_model_artifacts_saved():
    """Verify that models and metadata yaml exist on disk."""
    proc_dir = Path("data/processed")
    assert (proc_dir / "m2_lightgbm_model.pkl").exists(), "LightGBM model artifact missing"
    assert (proc_dir / "m1_iforest_model.pkl").exists(), "Isolation Forest model artifact missing"
    assert (proc_dir / "model_metadata.yaml").exists(), "Model metadata YAML missing"
