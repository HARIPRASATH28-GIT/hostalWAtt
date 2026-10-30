"""Intervention planner and anomaly typology classifier.

Classifies electrical anomaly events by duration, shape, and time of day,
maps them to specific facilities actions, and prioritizes hostel blocks
based on wasted energy, financial payoff, persistence, and recurrence.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


ACTION_CATALOG = {
    "appliance_spike": {
        "action": "Immediate Room-by-Room Inspection",
        "description": "Deploy floor proctor to inspect for unauthorized high-wattage appliances (immersion heating rods, induction cooktops, heavy electric kettles).",
        "protocol": "Confiscate unauthorized heating coils; verify circuit breaker rating.",
        "urgency": "HIGH",
        "expected_payoff_multiplier": 0.85
    },
    "left_on": {
        "action": "Warden Night Round & Common-Area Shutdown",
        "description": "Verify common-room high-bay fixtures, corridor floodlights, laundry geysers, or exhaust fans running continuously unattended.",
        "protocol": "Implement automatic daylight/occupancy sensor or evening switch-off protocol.",
        "urgency": "MEDIUM",
        "expected_payoff_multiplier": 0.90
    },
    "wiring_drift": {
        "action": "Licensed Electrician Insulation & Grounding Audit",
        "description": "Persistent rising baseline with elevated night floor indicates progressive dielectric breakdown, moisture ingress, or sub-surface ground leakage.",
        "protocol": "Perform Megger insulation resistance test on distribution feeder and balance phase loads.",
        "urgency": "CRITICAL",
        "expected_payoff_multiplier": 0.95
    },
    "intermittent_fault": {
        "action": "Distribution Busbar & Switchgear Contact Inspection",
        "description": "Rapid oscillatory surges at odd hours suggest loose contactor terminals, arcing busbars, or cycling faulty refrigerator compressor.",
        "protocol": "Thermal imaging (thermography) scan of panel board and retightening of lugs.",
        "urgency": "HIGH",
        "expected_payoff_multiplier": 0.70
    },
    "meter_fault": {
        "action": "Submeter Calibration & RS-485 Communication Check",
        "description": "Zero readings or completely frozen flatline values indicate CT disconnection, tripped meter MCB, or frozen IoT gateway firmware.",
        "protocol": "Power-cycle IoT gateway, test CT polarity, and inspect RS-485 Modbus loop.",
        "urgency": "CRITICAL",
        "expected_payoff_multiplier": 0.50
    }
}


def classify_event_signature(
    duration_hours: int,
    mean_excess_kw: float,
    start_hour: int,
    is_holiday: bool,
    is_zero: bool
) -> str:
    """Heuristic rule-based anomaly classifier using duration, magnitude, and calendar context."""
    if is_zero or mean_excess_kw < -3.0:
        return "meter_fault"

    # Multi-week persistent drift (> 48 hours continuous)
    if duration_hours >= 48:
        return "wiring_drift"

    # Overnight or holiday flat load (6 to 18 hours)
    if 6 <= duration_hours <= 18 and ((start_hour >= 21 or start_hour <= 4) or is_holiday):
        return "left_on"

    # Short sudden heavy surge (1 to 5 hours, delta > 0.9 kW)
    if duration_hours <= 5 and mean_excess_kw >= 0.9:
        return "appliance_spike"

    # Intermittent bursting at odd hours
    if duration_hours <= 6 and (1 <= start_hour <= 5):
        return "intermittent_fault"

    # Fallback default
    return "appliance_spike" if mean_excess_kw > 1.0 else "left_on"


def plan_block_interventions(
    df_waste: pd.DataFrame,
    tariff_inr: float = 7.50
) -> pd.DataFrame:
    """Rank hostel blocks and generate prioritized operational intervention plans.

    Priority Score is computed via weighted multi-criteria ranking:
    Score = 0.35 * Waste_kWh + 0.35 * Cost_INR + 0.15 * Persistence + 0.15 * Recurrence
    """
    df = df_waste.copy()
    blocks = df["hostel_block"].unique()

    intervention_rows = []

    for b in blocks:
        b_df = df[df["hostel_block"] == b].sort_values("date_time").reset_index(drop=True)
        flagged = b_df[b_df["wasted_kwh"] > 0]

        total_waste_kwh = float(flagged["wasted_kwh"].sum())
        total_waste_inr = round(total_waste_kwh * tariff_inr, 0)
        total_flagged_hours = len(flagged)

        if total_flagged_hours == 0:
            continue

        # Count occurrences of detected anomaly types
        if "anomaly_type" in flagged.columns:
            pred_types = flagged["anomaly_type"].value_counts().to_dict()
            primary_type = flagged["anomaly_type"].mode()[0] if not flagged["anomaly_type"].empty else "appliance_spike"
            if primary_type == "none":
                # Find second most common
                candidates = [k for k in pred_types.keys() if k != "none"]
                primary_type = candidates[0] if candidates else "appliance_spike"
        else:
            primary_type = "appliance_spike"

        action_meta = ACTION_CATALOG.get(primary_type, ACTION_CATALOG["appliance_spike"])

        # Persistence: average continuous consecutive run of flagged hours
        is_flag_seq = (b_df["wasted_kwh"] > 0).astype(int)
        runs = (is_flag_seq != is_flag_seq.shift()).cumsum()
        flagged_runs = b_df[b_df["wasted_kwh"] > 0].groupby(runs).size()
        avg_persistence = float(flagged_runs.mean()) if not flagged_runs.empty else 1.0
        max_persistence = int(flagged_runs.max()) if not flagged_runs.empty else 1

        # Recurrence rate: percentage of days with at least one incident
        days_with_anomaly = b_df[b_df["wasted_kwh"] > 0]["date_time"].dt.date.nunique()
        total_days = max(1, b_df["date_time"].dt.date.nunique())
        recurrence_pct = round((days_with_anomaly / total_days) * 100.0, 1)

        # Estimated financial payoff = cost * action efficiency multiplier
        est_payoff_inr = round(total_waste_inr * action_meta["expected_payoff_multiplier"], 0)

        # Priority calculation
        priority_raw = (
            0.40 * (total_waste_kwh / 100.0) +
            0.30 * (total_waste_inr / 1000.0) +
            0.15 * (avg_persistence * 2.0) +
            0.15 * (recurrence_pct / 5.0)
        )
        priority_score = round(min(100.0, priority_raw * 10.0), 1)

        intervention_rows.append({
            "Block": b,
            "Priority Score": priority_score,
            "Urgency": action_meta["urgency"],
            "Primary Anomaly": primary_type,
            "Recommended Action": action_meta["action"],
            "Action Protocol": action_meta["protocol"],
            "Wasted Energy (kWh)": round(total_waste_kwh, 1),
            "Avoidable Loss (₹)": total_waste_inr,
            "Estimated Payoff (₹)": est_payoff_inr,
            "Avg Persistence (hrs)": round(avg_persistence, 1),
            "Max Consecutive (hrs)": max_persistence,
            "Recurrence Rate (%)": recurrence_pct
        })

    plan_df = pd.DataFrame(intervention_rows).sort_values("Priority Score", ascending=False).reset_index(drop=True)
    return plan_df
