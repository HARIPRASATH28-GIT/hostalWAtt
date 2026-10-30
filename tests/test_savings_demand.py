"""Unit tests for savings, demand, and payback calculations."""

import pytest
import pandas as pd
import numpy as np
from src.savings import calculate_hourly_waste, aggregate_savings, compute_payback_sensitivity
from src.demand import calculate_system_maximum_demand


@pytest.fixture(scope="module")
def scored_sample():
    """Load evaluation scored dataset."""
    df = pd.read_parquet("data/processed/test_evaluation_scored.parquet")
    return df


def test_savings_non_negative_waste(scored_sample):
    """Ensure wasted energy and cost are never negative."""
    df_waste = calculate_hourly_waste(scored_sample, tariff_inr=7.50)
    assert (df_waste["wasted_kwh"] >= 0.0).all(), "Negative wasted kWh found"
    assert (df_waste["cost_inr"] >= 0.0).all(), "Negative cost in INR found"

    # Wasted kWh should only be positive where m2_pred == 1
    flagged = df_waste[df_waste["m2_pred"] == 1]
    unflagged = df_waste[df_waste["m2_pred"] == 0]
    assert (unflagged["wasted_kwh"] == 0.0).all(), "Unflagged rows contain non-zero wasted kWh"


def test_savings_aggregations(scored_sample):
    """Verify block, monthly, weekly, and annual savings aggregations."""
    df_waste = calculate_hourly_waste(scored_sample, tariff_inr=7.50)
    summary = aggregate_savings(df_waste, tariff_inr=7.50)

    assert "block_summary" in summary
    assert "monthly_summary" in summary
    assert "projected_annual_kwh" in summary
    assert summary["projected_annual_kwh"] > 0
    assert summary["projected_annual_inr"] > 0


def test_payback_sensitivity():
    """Verify sensor payback period and ROI sensitivity table."""
    df_sens = compute_payback_sensitivity(annual_savings_inr=150000.0, num_devices=5)

    assert len(df_sens) == 3
    assert set(df_sens["Scenario"].values) == {"Conservative", "Expected", "Optimistic"}
    assert "Payback Period (Months)" in df_sens.columns
    assert "3-Year Net ROI (%)" in df_sens.columns


def test_demand_calculation(scored_sample):
    """Verify coincident maximum demand, block contributions, and lower bound disclaimer."""
    demand_res = calculate_system_maximum_demand(scored_sample, power_factor=0.90)

    assert demand_res["system_peak_kw"] > 0
    assert demand_res["system_peak_kva"] > demand_res["system_peak_kw"]
    assert "Hourly integration provides a lower-bound" in demand_res["disclaimer"]

    # Coincident contributions sum to total system peak within 0.1 kW
    coincident_sum = demand_res["coincident_contributions"]["peak_kw"].sum()
    assert np.isclose(coincident_sum, demand_res["system_peak_kw"], atol=0.2)
