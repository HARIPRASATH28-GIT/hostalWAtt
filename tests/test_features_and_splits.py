"""Unit tests for feature engineering pipeline and temporal split leakage prevention."""

import pytest
import pandas as pd
import numpy as np
from src.features import build_feature_pipeline
from src.models import time_based_train_test_split


@pytest.fixture(scope="module")
def sample_data():
    """Load sample synthetic block data."""
    df = pd.read_parquet("data/synthetic/hostel_blocks.parquet")
    return df


def test_time_based_split_no_leakage(sample_data):
    """Ensure temporal split has strict boundary with zero future leakage."""
    split_date = "2025-10-01 00:00:00"
    train_df, test_df = time_based_train_test_split(sample_data, split_date=split_date)

    max_train_time = train_df["date_time"].max()
    min_test_time = test_df["date_time"].min()

    assert max_train_time < pd.Timestamp(split_date), "Train data contains dates past split boundary"
    assert min_test_time >= pd.Timestamp(split_date), "Test data contains dates before split boundary"
    assert len(train_df) + len(test_df) == len(sample_data), "Row count discrepancy in temporal split"


def test_feature_pipeline_columns_and_nans(sample_data):
    """Verify that feature engineering produces all required columns with zero NaNs."""
    df_feat, feature_cols = build_feature_pipeline(sample_data)

    for col in feature_cols:
        assert col in df_feat.columns, f"Missing engineered feature: {col}"
        assert df_feat[col].isna().sum() == 0, f"NaN values detected in feature: {col}"


def test_lag_zero_future_leakage(sample_data):
    """Verify that lag_1 strictly matches the previous hour's energy consumption."""
    df_feat, _ = build_feature_pipeline(sample_data)

    # For Block_A, check hours 5 to 50
    block_a = df_feat[df_feat["hostel_block"] == "Block_A"].sort_values("date_time").reset_index(drop=True)
    for idx in range(5, 50):
        prev_actual = block_a.loc[idx - 1, "energy_kwh"]
        curr_lag1 = block_a.loc[idx, "lag_1"]
        assert np.isclose(prev_actual, curr_lag1, atol=1e-3), (
            f"Lag leakage at index {idx}: prev actual={prev_actual}, lag_1={curr_lag1}"
        )
