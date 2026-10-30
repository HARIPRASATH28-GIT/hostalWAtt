"""Page 5: Maximum Demand (MD) Safeguard.

Tracks system-level coincident maximum demand, individual block contributions,
90% capacity alerts, and apparent power (kVA) power-factor modeling.
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
import plotly.graph_objects as go
import yaml

from app.ui_utils import apply_custom_css, render_header, render_metric_card, get_plotly_theme
from src.demand import calculate_system_maximum_demand

st.set_page_config(page_title="Maximum Demand - HostelWatt", page_icon="⚡", layout="wide")
apply_custom_css()
render_header(
    title="Maximum Demand (MD) Safeguard",
    subtitle="Monitor campus coincident peak demand, block-level coincident shares, and 90% capacity threshold alerts",
    icon="⚡"
)

# Load data
@st.cache_data
def get_data():
    file = ROOT_DIR / "data" / "synthetic" / "hostel_blocks.parquet"
    if not file.exists():
        file = ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet"
    return pd.read_parquet(file)

df_all = get_data()

# Interactive Power Factor & Warning threshold controls
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    pf_val = st.slider("Campus Power Factor (Lagging)", min_value=0.75, max_value=0.98, value=0.90, step=0.01)
with col_ctrl2:
    thresh_pct = st.slider("MD Warning Threshold (% of Monthly Peak)", min_value=80.0, max_value=98.0, value=90.0, step=1.0)

# Calculate Demand Engine metrics
demand_res = calculate_system_maximum_demand(df_all, power_factor=pf_val, warning_threshold_pct=thresh_pct)

# Top KPI Tiles
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card("Recorded System Peak", f"{demand_res['system_peak_kw']:.1f} kW", f"At {demand_res['coincident_peak_time']:%d %b %H:00}")
with k2:
    render_metric_card("Apparent Peak (kVA)", f"{demand_res['system_peak_kva']:.1f} kVA", f"@ {pf_val:.2f} PF", status="warning")
with k3:
    monthly_peak = demand_res["monthly_summary"]["peak_kw"].iloc[-1]
    render_metric_card("Current Month Peak", f"{monthly_peak:.1f} kW", "Latest 30-day window")
with k4:
    alert_hours_count = demand_res["monthly_summary"]["alert_hours"].sum()
    render_metric_card(f"Hours > {thresh_pct:.0f}% of MD", f"{alert_hours_count} hrs", "High penalty exposure risk", status="critical")

st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Engineering Lower-Bound Notice:</strong> All values are integrated over 1-hour meter intervals.
        Because electric utilities (e.g., TNEB / TANGEDCO) bill Maximum Demand based on <strong>15-minute or 30-minute sliding windows</strong>,
        hourly average power smooths intra-hour spikes and constitutes a strict <strong>engineering lower bound</strong> of billing demand.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Visualizations: System Load Curve and Block Coincident Shares
col_chart1, col_chart2 = st.columns([7, 5])

with col_chart1:
    st.markdown("#### 📈 Campus Coincident Hourly Load Profile")
    sys_df = demand_res["hourly_system_df"].tail(168 * 2)  # Last 4 weeks
    fig_sys = go.Figure()

    fig_sys.add_trace(go.Scatter(
        x=sys_df["date_time"],
        y=sys_df["system_total_kw"],
        mode="lines",
        name="Coincident System Power (kW)",
        line=dict(color="#38bdf8", width=2)
    ))

    fig_sys.add_trace(go.Scatter(
        x=sys_df["date_time"],
        y=sys_df["warning_threshold_kw"],
        mode="lines",
        name=f"{thresh_pct:.0f}% Warning Threshold",
        line=dict(color="#f87171", width=1.5, dash="dash")
    ))

    theme = get_plotly_theme()
    fig_sys.update_layout(**theme["layout"])
    fig_sys.update_layout(height=350, hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig_sys, use_container_width=True)

with col_chart2:
    st.markdown("#### 🥧 Block Contribution at Coincident Peak")
    coinc_df = demand_res["coincident_contributions"]
    fig_pie = px.pie(
        coinc_df,
        names="hostel_block",
        values="peak_kw",
        hole=0.45,
        template="plotly_dark",
        color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c"]
    )
    fig_pie.update_layout(**theme["layout"])
    fig_pie.update_layout(height=350, margin=dict(t=20, b=20, l=10, r=10))
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.markdown("### 📊 Block Coincident vs. Non-Coincident Demand Comparison")

comp_df = coinc_df.merge(demand_res["non_coincident_peaks"], on="hostel_block", suffixes=("_coincident", "_individual"))
comp_df["diversity_factor_contribution"] = (comp_df["max_kw"] / comp_df["peak_kw"]).round(2)

st.dataframe(
    comp_df[[
        "hostel_block", "peak_kw", "peak_kva", "share_pct", "max_kw", "max_kva", "peak_time"
    ]].rename(columns={
        "hostel_block": "Block",
        "peak_kw": "Coincident kW",
        "peak_kva": "Coincident kVA",
        "share_pct": "Share at Peak (%)",
        "max_kw": "Individual Peak kW",
        "max_kva": "Individual Peak kVA",
        "peak_time": "Time of Individual Peak"
    }),
    use_container_width=True
)
