"""Page 1: Block Explorer.

Interactive drill-down into block-level electricity time series:
- Actual vs. Model Expected consumption with highlighted anomaly flags
- Rich hover tooltips displaying root cause, severity, delta, and top contributing features
- Consumption Heatmap across Hour of Day vs. Day of Year / Week
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

from app.ui_utils import apply_custom_css, render_header, get_plotly_theme
from src.savings import calculate_hourly_waste

st.set_page_config(page_title="Block Explorer - HostelWatt", page_icon="📈", layout="wide")
apply_custom_css()
render_header(
    title="Block Energy Explorer",
    subtitle="Inspect actual vs. expected baseline load curves, anomaly timestamps, and diurnal consumption heatmaps",
    icon="📈"
)

# Load data
@st.cache_data
def load_data():
    proc_file = ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet"
    if proc_file.exists():
        df = pd.read_parquet(proc_file)
    else:
        df = pd.read_parquet(ROOT_DIR / "data" / "synthetic" / "hostel_blocks.parquet")
        df["m2_expected"] = df["energy_kwh"] * 0.95
        df["m2_pred"] = df["is_anomaly"]
        df["m2_severity"] = 50.0
    return calculate_hourly_waste(df)

df_all = load_data()
df_all["date_time"] = pd.to_datetime(df_all["date_time"])

# Filters
col_b, col_d1, col_d2 = st.columns([3, 3, 3])
blocks = sorted(df_all["hostel_block"].unique())

with col_b:
    selected_block = st.selectbox("Select Hostel Block", options=blocks, index=0)

min_date = df_all["date_time"].min().date()
max_date = df_all["date_time"].max().date()
default_start = max(min_date, max_date - pd.Timedelta(days=21))

with col_d1:
    start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
with col_d2:
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Filter dataframe
mask = (
    (df_all["hostel_block"] == selected_block) &
    (df_all["date_time"].dt.date >= start_date) &
    (df_all["date_time"].dt.date <= end_date)
)
df_block = df_all[mask].sort_values("date_time").reset_index(drop=True)

if df_block.empty:
    st.warning("No data found for the selected date range.")
    st.stop()

# Time series actual vs expected with anomalies
st.markdown("### ⚡ Actual Load vs. Model Baseline (Expected kWh)")

fig = go.Figure()

# 1. Expected Baseline Curve
fig.add_trace(go.Scatter(
    x=df_block["date_time"],
    y=df_block["m2_expected"],
    mode="lines",
    name="Expected Baseline (LightGBM)",
    line=dict(color="#38bdf8", width=2, dash="dot"),
    hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Expected: %{y:.2f} kWh<extra></extra>"
))

# 2. Actual Consumption
fig.add_trace(go.Scatter(
    x=df_block["date_time"],
    y=df_block["energy_kwh"],
    mode="lines",
    name="Actual Measured Load",
    line=dict(color="#818cf8", width=2),
    hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Actual: %{y:.2f} kWh<extra></extra>"
))

# 3. Anomaly markers
anom_pts = df_block[df_block["m2_pred"] == 1]
if not anom_pts.empty:
    hover_texts = [
        f"<b>🚨 ANOMALY DETECTED</b><br>"
        f"Time: {row['date_time']:%Y-%m-%d %H:%M}<br>"
        f"Actual: {row['energy_kwh']:.2f} kWh | Expected: {row['m2_expected']:.2f} kWh<br>"
        f"Waste Delta: +{row['wasted_kwh']:.2f} kWh (₹{row['cost_inr']:.1f})<br>"
        f"Severity: {row['m2_severity']:.0f}/100<br>"
        f"Classification: {row.get('anomaly_type', 'Unclassified')}<br>"
        f"Day Type: {row.get('day_type', 'regular')} | Temp: {row.get('outside_temp_c', 28)}°C"
        for _, row in anom_pts.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=anom_pts["date_time"],
        y=anom_pts["energy_kwh"],
        mode="markers",
        name="Flagged Anomaly",
        marker=dict(
            symbol="circle",
            size=9,
            color=anom_pts["m2_severity"],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Severity", x=1.02, len=0.8)
        ),
        text=hover_texts,
        hoverinfo="text"
    ))

theme = get_plotly_theme()
fig.update_layout(**theme["layout"])
fig.update_layout(
    height=420,
    hovermode="closest",
    legend=dict(orientation="h", y=1.1, x=0),
    xaxis_title="Timestamp",
    yaxis_title="Energy Consumption (kWh = avg kW)"
)
st.plotly_chart(fig, use_container_width=True)

# Hour x Day Heatmap
st.markdown("---")
st.markdown("### 🌡️ Diurnal Consumption Heatmap (Hour of Day vs. Date)")

df_block["date_str"] = df_block["date_time"].dt.strftime("%Y-%m-%d")
df_block["hour_int"] = df_block["date_time"].dt.hour

pivot_heat = df_block.pivot_table(index="hour_int", columns="date_str", values="energy_kwh", aggfunc="mean")

fig_heat = px.imshow(
    pivot_heat,
    labels=dict(x="Date", y="Hour of Day (0-23)", color="kWh"),
    color_continuous_scale="Plasma",
    aspect="auto",
    template="plotly_dark"
)
fig_heat.update_layout(**theme["layout"])
fig_heat.update_layout(
    height=340,
    xaxis_title="Date",
    yaxis_title="Hour of Day",
    yaxis=dict(tickmode="linear", tick0=0, dtick=2)
)
st.plotly_chart(fig_heat, use_container_width=True)

# Explainability Section
st.markdown("---")
st.markdown("### 🔍 Model Explainability & Feature Contribution Breakdown")
st.markdown(
    """
    For every flagged incident, **HostelWatt** compares observed load against contextual drivers (ambient temperature, active student occupancy, academic schedules, and historical lags).
    """
)

if not anom_pts.empty:
    sample_anom = anom_pts.iloc[-1]
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.metric("Latest Flagged Time", f"{sample_anom['date_time']:%d %b %H:00}")
        st.metric("Surplus Delta", f"+{sample_anom['wasted_kwh']:.2f} kWh", delta=f"₹{sample_anom['cost_inr']:.0f} wasted")
    with ec2:
        st.metric("Severity Score", f"{sample_anom['m2_severity']:.1f} / 100")
        st.metric("Root Cause Type", f"{sample_anom.get('anomaly_type', 'appliance_spike')}")
    with ec3:
        st.markdown("**Top Contextual Drivers:**")
        st.write(f"- **Occupancy**: {sample_anom.get('occupancy', 'N/A')} residents")
        st.write(f"- **Academic Status**: `{sample_anom.get('day_type', 'regular')}`")
        st.write(f"- **Ambient Temp**: {sample_anom.get('outside_temp_c', 'N/A')} °C")
        st.write(f"- **Lag t-24**: {sample_anom.get('lag_24', 'N/A')} kWh")
else:
    st.success("No anomalies detected in the selected timeframe.")
