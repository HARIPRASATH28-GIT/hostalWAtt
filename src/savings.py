"""Savings estimation and payback sensitivity calculator.

Calculates wasted kWh, financial loss in INR, temporal aggregations,
and sensor hardware payback periods under conservative, expected, and optimistic scenarios.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load system configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_hourly_waste(
    df: pd.DataFrame,
    actual_col: str = "energy_kwh",
    expected_col: str = "m2_expected",
    flag_col: str = "m2_pred",
    tariff_inr: float = 7.50
) -> pd.DataFrame:
    """Calculate hourly wasted energy (kWh) and associated financial loss (INR).

    wasted_kwh = max(actual - expected, 0) over flagged hours.
    cost_inr = wasted_kwh * tariff_inr.
    """
    df = df.copy()
    actual = df[actual_col].values
    expected = df[expected_col].values
    flags = df[flag_col].values

    # Avoidable excess load on flagged hours
    wasted = np.where(flags == 1, np.maximum(actual - expected, 0.0), 0.0)

    df["wasted_kwh"] = np.round(wasted, 3)
    df["cost_inr"] = np.round(wasted * tariff_inr, 2)
    return df


def aggregate_savings(
    df_waste: pd.DataFrame,
    tariff_inr: float = 7.50
) -> Dict:
    """Compute multi-level aggregations: block, week, month, and projected annual savings."""
    df = df_waste.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    # 1. Block-level breakdown
    block_agg = (
        df.groupby("hostel_block")
        .agg(
            total_kwh=("energy_kwh", "sum"),
            flagged_hours=("wasted_kwh", lambda x: (x > 0).sum()),
            wasted_kwh=("wasted_kwh", "sum"),
            wasted_inr=("cost_inr", "sum")
        )
        .round(2)
        .reset_index()
    )

    # 2. Monthly breakdown
    df["year_month"] = df["date_time"].dt.to_period("M").astype(str)
    month_agg = (
        df.groupby("year_month")
        .agg(
            wasted_kwh=("wasted_kwh", "sum"),
            wasted_inr=("cost_inr", "sum"),
            anomaly_hours=("wasted_kwh", lambda x: (x > 0).sum())
        )
        .round(2)
        .reset_index()
    )

    # 3. Weekly breakdown (last 12 weeks)
    df["year_week"] = df["date_time"].dt.to_period("W").astype(str)
    week_agg = (
        df.groupby("year_week")
        .agg(
            wasted_kwh=("wasted_kwh", "sum"),
            wasted_inr=("cost_inr", "sum")
        )
        .round(2)
        .reset_index()
        .tail(12)
    )

    # 4. Projected Annual Savings
    total_hours_in_data = (df["date_time"].max() - df["date_time"].min()).total_seconds() / 3600.0
    scale_to_annual = 8760.0 / max(total_hours_in_data, 1.0)
    actual_waste_kwh = df["wasted_kwh"].sum()
    projected_annual_kwh = round(actual_waste_kwh * scale_to_annual, 1)
    projected_annual_inr = round(projected_annual_kwh * tariff_inr, 0)

    return {
        "block_summary": block_agg,
        "monthly_summary": month_agg,
        "weekly_summary": week_agg,
        "total_wasted_kwh": round(actual_waste_kwh, 1),
        "total_cost_inr": round(df["cost_inr"].sum(), 0),
        "projected_annual_kwh": projected_annual_kwh,
        "projected_annual_inr": projected_annual_inr,
        "tariff_applied": tariff_inr
    }


def compute_payback_sensitivity(
    annual_savings_inr: float,
    num_devices: int = 5,
    sensor_cost_per_device: float = 8500.0,
    install_cost_per_device: float = 1500.0,
    annual_maintenance_pct: float = 0.05
) -> pd.DataFrame:
    """Calculate payback period (in months) and 3-year ROI under 3 sensitivity scenarios.

    Scenarios:
    - Conservative: 45% anomaly elimination rate, +15% Capex
    - Expected: 75% anomaly elimination rate, base Capex
    - Optimistic: 90% anomaly elimination rate, -10% Capex
    """
    total_capex = num_devices * (sensor_cost_per_device + install_cost_per_device)
    annual_opex = total_capex * annual_maintenance_pct

    scenarios = [
        {"name": "Conservative", "capture_rate": 0.45, "capex_mult": 1.15},
        {"name": "Expected", "capture_rate": 0.75, "capex_mult": 1.00},
        {"name": "Optimistic", "capture_rate": 0.90, "capex_mult": 0.90},
    ]

    records = []
    for sc in scenarios:
        adjusted_capex = total_capex * sc["capex_mult"]
        annual_realized_savings = (annual_savings_inr * sc["capture_rate"]) - annual_opex
        monthly_net_savings = annual_realized_savings / 12.0

        if monthly_net_savings > 0:
            payback_months = adjusted_capex / monthly_net_savings
        else:
            payback_months = float("inf")

        # 3-year Net ROI %
        three_year_net = (annual_realized_savings * 3.0) - adjusted_capex
        roi_3yr = (three_year_net / adjusted_capex) * 100.0

        records.append({
            "Scenario": sc["name"],
            "Capture Rate (%)": int(sc["capture_rate"] * 100),
            "Adjusted Capex (₹)": round(adjusted_capex, 0),
            "Annual Net Savings (₹)": round(max(annual_realized_savings, 0), 0),
            "Payback Period (Months)": round(payback_months, 1) if payback_months != float("inf") else "> 60",
            "3-Year Net ROI (%)": round(roi_3yr, 1)
        })

    return pd.DataFrame(records)
