"""Unit tests for subset-sum appliance matching and room localization ranking."""

import pytest
import pandas as pd
from pathlib import Path
from src.room_localization import (
    match_appliance_subset_sum, rank_rooms_for_event,
    evaluate_room_localization_accuracy
)


def test_subset_sum_matching():
    """Verify subset-sum matches known appliance combinations within tolerance band."""
    # Target: 1.35 kW (electric kettle)
    matches_kettle = match_appliance_subset_sum(1.35, tolerance_pct=10.0)
    assert len(matches_kettle) > 0
    top_match = matches_kettle[0]
    assert "Electric Kettle" in top_match["appliances"]
    assert top_match["error_kw"] < 0.1

    # Target: ~1.575 kW (1.5 kW room heater + 0.075 kW ceiling fan)
    matches_combo = match_appliance_subset_sum(1.575, tolerance_pct=5.0)
    assert len(matches_combo) > 0
    combo_names = [m["appliances"] for m in matches_combo]
    assert any("Room Convector Heater" in c for c in combo_names)


def test_room_ranking_hierarchy():
    """Verify room ranking produces sorted candidates for demo block."""
    rooms_file = Path("data/synthetic/demo_block_rooms.parquet")
    if not rooms_file.exists():
        pytest.skip("Demo block room dataset not yet generated")

    df_rooms = pd.read_parquet(rooms_file)
    first_dt = df_rooms["date_time"].iloc[100]

    floor, ranked_df = rank_rooms_for_event(df_rooms, first_dt, target_block="Block_A")
    assert floor in ["Floor_1", "Floor_2", "Floor_3"]
    assert len(ranked_df) == 24  # 3 floors * 8 rooms
    assert "room_id" in ranked_df.columns
    assert "candidate_score" in ranked_df.columns
    # Check rank ordering
    assert ranked_df["candidate_score"].is_monotonic_decreasing


def test_room_localization_accuracy_bounds():
    """Verify localization accuracy evaluation returns valid percentage bounds [0, 100]."""
    rooms_file = Path("data/synthetic/demo_block_rooms.parquet")
    if not rooms_file.exists():
        pytest.skip("Demo block room dataset not yet generated")

    df_rooms = pd.read_parquet(rooms_file)
    eval_res = evaluate_room_localization_accuracy(df_rooms, target_block="Block_A")

    assert 0.0 <= eval_res["top_1_accuracy"] <= 100.0
    assert 0.0 <= eval_res["top_3_accuracy"] <= 100.0
    assert eval_res["top_3_accuracy"] >= eval_res["top_1_accuracy"]
