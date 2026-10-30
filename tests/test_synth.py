"""Unit tests for synthetic data generator and hierarchy conservation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.synth import generate_hostel_dataset, load_config


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate or load synthetic datasets once for module tests."""
    df_blocks, df_rooms = generate_hostel_dataset()
    return df_blocks, df_rooms


def test_block_data_schema(synthetic_data):
    """Verify block dataset schema and non-null constraints."""
    df_blocks, _ = synthetic_data
    required_cols = [
        "hostel_block", "date_time", "energy_kwh", "occupancy",
        "room_count", "day_type", "outside_temp_c",
        "is_anomaly", "anomaly_type", "injected_extra_kwh"
    ]
    for col in required_cols:
        assert col in df_blocks.columns, f"Missing required column: {col}"

    assert df_blocks.isna().sum().sum() == 0, "Null values detected in synthetic block dataset"


def test_block_counts_and_duration(synthetic_data):
    """Ensure exactly 5 blocks and 8760 hours per block."""
    df_blocks, _ = synthetic_data
    blocks = df_blocks["hostel_block"].unique()
    assert len(blocks) == 5, f"Expected 5 blocks, got {len(blocks)}"

    for b in blocks:
        b_df = df_blocks[df_blocks["hostel_block"] == b]
        assert len(b_df) == 8760, f"Block {b} has {len(b_df)} hours instead of 8760"


def test_anomaly_types_present(synthetic_data):
    """Verify all 5 requested anomaly types exist in the dataset."""
    df_blocks, _ = synthetic_data
    expected_types = {
        "appliance_spike", "left_on", "wiring_drift",
        "intermittent_fault", "meter_fault"
    }
    actual_types = set(df_blocks["anomaly_type"].unique()) - {"none"}
    assert expected_types.issubset(actual_types), f"Missing anomaly types: {expected_types - actual_types}"


def test_non_negative_energy(synthetic_data):
    """Ensure energy_kwh is never negative."""
    df_blocks, df_rooms = synthetic_data
    assert (df_blocks["energy_kwh"] >= 0).all(), "Negative energy consumption found in block data"
    assert (df_rooms["room_energy_kwh"] >= 0).all(), "Negative energy consumption found in room data"


def test_room_sum_conservation(synthetic_data):
    """Verify physical law: sum of room loads must equal block load at every hour."""
    df_blocks, df_rooms = synthetic_data
    demo_block_id = "Block_A"

    demo_b = df_blocks[df_blocks["hostel_block"] == demo_block_id].set_index("date_time")["energy_kwh"]
    room_sum = df_rooms.groupby("date_time")["room_energy_kwh"].sum()

    diff = np.abs(demo_b - room_sum)
    max_error = diff.max()
    assert max_error < 1e-3, f"Conservation law violated! Max discrepancy: {max_error} kWh"
