"""Unit tests for LEAD dataset loader and fallback mechanism."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_lead import load_lead_dataset, generate_lead_fallback


def test_lead_fallback_generation(tmp_path):
    """Test generating LEAD fallback file."""
    test_file = tmp_path / "test_lead.csv"
    df = generate_lead_fallback(test_file, num_days=10, seed=123)

    assert test_file.exists()
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "meter_reading" in df.columns
    assert "anomaly" in df.columns


def test_lead_loader_alignment():
    """Test lead dataset loading, timestamp alignment, and missing value handling."""
    df_lead, is_fallback = load_lead_dataset()

    assert isinstance(df_lead, pd.DataFrame)
    assert len(df_lead) > 0
    assert "building_id" in df_lead.columns
    assert "meter_reading" in df_lead.columns
    assert "timestamp" in df_lead.columns
    assert df_lead["meter_reading"].isna().sum() == 0
    assert (df_lead["meter_reading"] >= 0).all()
