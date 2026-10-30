"""Page 3: Savings & Financial ROI Calculator.

Quantifies avoidable energy losses, temporal waste aggregations,
and features an interactive sensor submetering payback & ROI calculator
with conservative, expected, and optimistic sensitivity scenarios.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yaml

from app.ui_utils import apply_custom_css, render_header, render_metric_card, get_plotly_theme
from src.savings import calculate_hourly_waste, aggregate_savings, compute_payback_sensitivity

st.set_page_config(page_title="Savings & Payback - HostelWatt", page_icon="💰", layout="wide")
apply_custom_css()
render_header(
    title="Energy Savings & Sensor ROI Calculator",
    subtitle="Evaluate avoidable energy expenditures, projected annual savings, and IoT submetering capital payback",
    icon="💰"
)

# Load config
with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tariff_inr = config["tariff"]["energy_rate_inr_per_kwh"]
citation = config["tariff"]["citation"]

# Load scored data
@st.cache_data
def get_waste_data():
    proc_file = ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet"
    if proc_file.exists():
        df = pd.read_parquet(proc_file)
    else:
        df = pd.read_parquet(ROOT_DIR / "data" / "synthetic" / "hostel_blocks.parquet")
        df["m2_expected"] = df["energy_kwh"] * 0.95
        df["m2_pred"] = df["is_anomaly"]
    return calculate_hourly_waste(df, tariff_inr=tariff_inr)

df_waste = get_waste_data()
savings_meta = aggregate_savings(df_waste, tariff_inr=tariff_inr)

# Top KPI Summary Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card("Total Avoidable Waste", f"{savings_meta['total_wasted_kwh']:,.1f} kWh", "Observed test period")
with k2:
    render_metric_card("Total Financial Loss", f"₹ {savings_meta['total_cost_inr']:,.0f}", f"@ ₹{tariff_inr:.2f} / kWh (TNEB LT-IIB)", status="critical")
with k3:
    render_metric_card("Projected Annual Waste", f"{savings_meta['projected_annual_kwh']:,.0f} kWh", "Annualized campus forecast")
with k4:
    render_metric_card("Projected Annual Savings", f"₹ {savings_meta['projected_annual_inr']:,.0f}", "Annualized avoidable loss", status="warning")

st.markdown(
    f"""
    <div class="disclaimer-banner">
        ⚖️ <strong>Regulatory Citation:</strong> Energy rates computed in accordance with <em>{citation}</em>.
        All figures represent baseline model estimates of avoidable surplus consumption.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Aggregations: Block Breakdown & Monthly Trend
col_left, col_right = st.columns([6, 6])

with col_left:
    st.markdown("#### 🏢 Waste & Cost Breakdown by Block")
    block_df = savings_meta["block_summary"].sort_values("wasted_inr", ascending=False)
    fig_block = px.bar(
        block_df,
        x="hostel_block",
        y="wasted_inr",
        text="wasted_inr",
        labels={"hostel_block": "Hostel Block", "wasted_inr": "Avoidable Cost (₹)"},
        color="wasted_inr",
        color_continuous_scale="Reds",
        template="plotly_dark"
    )
    theme = get_plotly_theme()
    fig_block.update_layout(**theme["layout"])
    fig_block.update_layout(height=320, coloraxis_showscale=False)
    fig_block.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_block, use_container_width=True)

with col_right:
    st.markdown("#### 📅 Monthly Avoidable Expenditure Trend")
    month_df = savings_meta["monthly_summary"]
    fig_month = px.line(
        month_df,
        x="year_month",
        y="wasted_inr",
        markers=True,
        labels={"year_month": "Month", "wasted_inr": "Avoidable Cost (₹)"},
        template="plotly_dark",
        color_discrete_sequence=["#f87171"]
    )
    fig_month.update_layout(**theme["layout"])
    fig_month.update_layout(height=320)
    st.plotly_chart(fig_month, use_container_width=True)

st.markdown("---")

# Interactive IoT Submetering Payback & ROI Simulator
st.markdown("### 🧮 IoT Submetering Payback & Sensitivity Simulator")
st.markdown(
    """
    Model the capital expenditure (Capex), operational maintenance (Opex), and payback period for deploying
    smart IoT energy meters across hostel wings to eliminate electrical waste.
    """
)

sc1, sc2, sc3 = st.columns(3)
with sc1:
    sensor_cost = st.number_input("Hardware Cost per Meter (₹)", min_value=2000.0, max_value=30000.0, value=8500.0, step=500.0)
with sc2:
    install_cost = st.number_input("Installation & Wiring per Meter (₹)", min_value=500.0, max_value=10000.0, value=1500.0, step=250.0)
with sc3:
    num_devices = st.slider("Number of Submeter Points", min_value=1, max_value=25, value=5, step=1)

# Calculate sensitivity
annual_target = savings_meta["projected_annual_inr"]
df_sensitivity = compute_payback_sensitivity(
    annual_savings_inr=annual_target,
    num_devices=num_devices,
    sensor_cost_per_device=sensor_cost,
    install_cost_per_device=install_cost
)

st.markdown("#### 📊 Sensitivity Matrix Across Adoption Scenarios")
st.dataframe(
    df_sensitivity.style.format({
        "Adjusted Capex (₹)": "₹{:,.0f}",
        "Annual Net Savings (₹)": "₹{:,.0f}",
        "3-Year Net ROI (%)": "{:.1f}%"
    }),
    use_container_width=True
)

st.markdown(
    """
    - **Conservative Scenario**: 45% anomaly mitigation efficiency, with +15% Capex contingency for cabling.
    - **Expected Scenario**: 75% anomaly mitigation efficiency at nominal equipment and installation costs.
    - **Optimistic Scenario**: 90% anomaly mitigation efficiency through rapid automated push alerts and student compliance.
    """
)
