"""Room localization and appliance signature matching engine.

Performs:
1. Floor-level step detection from aggregate block meter deviations.
2. Subset-sum appliance signature matching with configurable tolerance band.
3. Bayesian/heuristic room ranking based on room-level surge deltas and history.
4. Top-1 and Top-3 room localization accuracy evaluation on injected anomalies.
"""

from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np
import pandas as pd
import yaml

DEFAULT_APPLIANCE_SIGNATURES = {
    "Ceiling Fan": 0.075,
    "Laptop Charger": 0.065,
    "Electric Kettle": 1.35,
    "Immersion Water Heater": 1.80,
    "Room Convector Heater": 1.50,
}


def match_appliance_subset_sum(
    target_surplus_kw: float,
    appliances: Optional[Dict[str, float]] = None,
    tolerance_pct: float = 12.0,
    max_combination_size: int = 3
) -> List[Dict]:
    """Find combinations of electrical appliance signatures matching target surplus kW within tolerance.

    Args:
        target_surplus_kw: The observed step increase in power (kW).
        appliances: Dict mapping appliance names to nominal power in kW.
        tolerance_pct: Percentage tolerance band around target (default 12%).
        max_combination_size: Maximum simultaneous active appliances in combination.

    Returns:
        List of matching combinations ranked by absolute residual error.
    """
    if appliances is None:
        appliances = DEFAULT_APPLIANCE_SIGNATURES

    tolerance_band = target_surplus_kw * (tolerance_pct / 100.0)
    matches = []

    app_names = list(appliances.keys())
    app_powers = [appliances[name] for name in app_names]

    # Evaluate combinations with replacement up to max_combination_size
    for k in range(1, max_combination_size + 1):
        for combo in itertools.combinations_with_replacement(range(len(app_names)), k):
            total_power = sum(app_powers[i] for i in combo)
            diff = abs(total_power - target_surplus_kw)
            if diff <= tolerance_band:
                matched_items = [app_names[i] for i in combo]
                matches.append({
                    "appliances": matched_items,
                    "combined_kw": round(total_power, 3),
                    "error_kw": round(diff, 3),
                    "error_pct": round((diff / target_surplus_kw) * 100.0, 1),
                    "confidence": round(max(0.0, 100.0 - (diff / max(target_surplus_kw, 0.1)) * 100.0), 1)
                })

    # Sort matches by lowest error
    matches = sorted(matches, key=lambda x: x["error_kw"])
    return matches


def rank_rooms_for_event(
    df_rooms: pd.DataFrame,
    target_datetime: pd.Timestamp,
    target_block: str = "Block_A"
) -> Tuple[str, pd.DataFrame]:
    """Detect floor-level surge and rank candidate rooms for an incident at target_datetime.

    Returns:
        Tuple[str, pd.DataFrame]: (identified_floor, ranked_room_candidates_df)
    """
    df_t = df_rooms[
        (df_rooms["hostel_block"] == target_block) &
        (df_rooms["date_time"] == target_datetime)
    ].copy()

    prev_time = target_datetime - pd.Timedelta(hours=1)
    df_prev = df_rooms[
        (df_rooms["hostel_block"] == target_block) &
        (df_rooms["date_time"] == prev_time)
    ].copy()

    if df_prev.empty:
        df_prev = df_t.copy()
        df_prev["room_energy_kwh"] *= 0.85

    merged = df_t.merge(
        df_prev[["room_id", "room_energy_kwh"]],
        on="room_id",
        suffixes=("", "_prev")
    )
    merged["step_delta_kw"] = merged["room_energy_kwh"] - merged["room_energy_kwh_prev"]

    # Floor-level surge detection
    floor_surges = merged.groupby("floor")["step_delta_kw"].sum()
    culprit_floor = floor_surges.idxmax()

    # Score and rank rooms on the culprit floor (and other floors as baseline)
    merged["floor_priority_boost"] = (merged["floor"] == culprit_floor).astype(float) * 2.0
    merged["candidate_score"] = (
        merged["step_delta_kw"].clip(lower=0) * 10.0 +
        merged["floor_priority_boost"] +
        merged["room_energy_kwh"] * 2.0
    )

    ranked = merged.sort_values("candidate_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    return culprit_floor, ranked[["rank", "floor", "room_id", "step_delta_kw", "room_energy_kwh", "is_anomaly", "anomaly_type", "candidate_score"]]


def evaluate_room_localization_accuracy(
    df_rooms: pd.DataFrame,
    target_block: str = "Block_A"
) -> Dict:
    """Evaluate Top-1 and Top-3 localization accuracy across all injected room anomaly hours.

    Returns:
        Dict with top_1_acc, top_3_acc, total_evaluated_events, and sample predictions.
    """
    # Filter hours with injected room anomalies (spikes or left-on)
    anomaly_events = df_rooms[
        (df_rooms["hostel_block"] == target_block) &
        (df_rooms["is_anomaly"] == 1) &
        (df_rooms["anomaly_type"].isin(["appliance_spike", "left_on"]))
    ]

    event_times = anomaly_events["date_time"].drop_duplicates().tolist()
    if not event_times:
        return {"top_1_accuracy": 0.0, "top_3_accuracy": 0.0, "total_events": 0}

    top_1_hits = 0
    top_3_hits = 0
    sample_records = []

    for dt in event_times:
        true_room_row = anomaly_events[anomaly_events["date_time"] == dt]
        if true_room_row.empty:
            continue
        true_room = true_room_row["room_id"].iloc[0]

        _, ranked_df = rank_rooms_for_event(df_rooms, dt, target_block=target_block)
        top_candidates = ranked_df["room_id"].tolist()

        is_top1 = (top_candidates[0] == true_room) if len(top_candidates) > 0 else False
        is_top3 = (true_room in top_candidates[:3]) if len(top_candidates) >= 3 else False

        if is_top1:
            top_1_hits += 1
        if is_top3:
            top_3_hits += 1

        if len(sample_records) < 5:
            sample_records.append({
                "date_time": str(dt),
                "true_room": true_room,
                "top_1_guess": top_candidates[0],
                "top_3_candidates": top_candidates[:3],
                "matched_top_1": is_top1,
                "matched_top_3": is_top3
            })

    n = len(event_times)
    return {
        "total_evaluated_events": n,
        "top_1_accuracy": round((top_1_hits / n) * 100.0, 1),
        "top_3_accuracy": round((top_3_hits / n) * 100.0, 1),
        "sample_evaluations": sample_records
    }
