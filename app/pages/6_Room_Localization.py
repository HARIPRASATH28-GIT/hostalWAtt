"""Page 6: Room Localization & Appliance Disaggregation (Demo Block).

Demonstrates:
- Hierarchical drilldown from Block to Floor to Room
- Floor-level step surge detection
- Subset-sum appliance electrical signature matching (Fan 75W, Kettle 1.35kW, Heater 1.8kW)
- Ranked room culprit shortlist
- Top-1 and Top-3 localization accuracy metrics across all injected events
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
from src.room_localization import (
    match_appliance_subset_sum, rank_rooms_for_event,
    evaluate_room_localization_accuracy, DEFAULT_APPLIANCE_SIGNATURES
)

st.set_page_config(page_title="Room Localization - HostelWatt", page_icon="🔍", layout="wide")
apply_custom_css()
render_header(
    title="Room-Level Anomaly Localization (Demo Block)",
    subtitle="Isolate unauthorized high-wattage appliances down to specific hostel rooms using subset-sum signature matching",
    icon="🔍"
)

# Load room-level data for Block A
@st.cache_data
def get_room_data():
    file = ROOT_DIR / "data" / "synthetic" / "demo_block_rooms.parquet"
    if not file.exists():
        from src.synth import generate_hostel_dataset
        _, df_r = generate_hostel_dataset(str(ROOT_DIR / "config.yaml"))
        return df_r
    return pd.read_parquet(file)

df_rooms = get_room_data()
df_rooms["date_time"] = pd.to_datetime(df_rooms["date_time"])

# Evaluation Metrics Banner
eval_metrics = evaluate_room_localization_accuracy(df_rooms, target_block="Block_A")

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card("Top-1 Localization", f"{eval_metrics['top_1_accuracy']:.1f}%", "Exact room pinpointed on 1st rank")
with k2:
    render_metric_card("Top-3 Shortlist Accuracy", f"{eval_metrics['top_3_accuracy']:.1f}%", "Room captured within top 3", status="warning")
with k3:
    render_metric_card("Evaluated Injected Events", f"{eval_metrics['total_evaluated_events']}", "Spikes & continuous loads")
with k4:
    render_metric_card("Hierarchy Resolution", "24 Rooms", "Block A (3 Floors × 8 Rooms)")

st.markdown("---")

# Incident Inspector
st.markdown("### 🎯 Live Incident Appliance Disaggregation Inspector")

# Filter incidents
anom_events = df_rooms[
    (df_rooms["is_anomaly"] == 1) &
    (df_rooms["anomaly_type"].isin(["appliance_spike", "left_on"]))
].drop_duplicates("date_time").sort_values("date_time", ascending=False)

event_dates = anom_events["date_time"].tolist()
if not event_dates:
    event_dates = [df_rooms["date_time"].max()]

sel_time = st.selectbox(
    "Select Incident Timestamp to Inspect",
    options=event_dates,
    format_func=lambda x: f"{x:%Y-%m-%d %H:%M} — {anom_events[anom_events['date_time'] == x]['anomaly_type'].iloc[0]}"
)

# Run Floor Detection & Room Ranking
culprit_floor, ranked_df = rank_rooms_for_event(df_rooms, sel_time, target_block="Block_A")

# Calculate observed floor surplus
curr_floor_sum = df_rooms[(df_rooms["date_time"] == sel_time) & (df_rooms["floor"] == culprit_floor)]["room_energy_kwh"].sum()
prev_floor_sum = df_rooms[(df_rooms["date_time"] == sel_time - pd.Timedelta(hours=1)) & (df_rooms["floor"] == culprit_floor)]["room_energy_kwh"].sum()
surplus_delta = max(0.1, curr_floor_sum - prev_floor_sum)

col_res1, col_res2 = st.columns([5, 7])

with col_res1:
    st.markdown("#### 🔬 Appliance Signature Matching (Subset-Sum)")
    st.write(f"**Observed Floor Surplus:** `{surplus_delta:.3f} kW` on **{culprit_floor}**")

    matches = match_appliance_subset_sum(surplus_delta, tolerance_pct=15.0)

    if matches:
        st.markdown("**Matched Electrical Signatures:**")
        for m in matches[:3]:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                    <div style="color: #60a5fa; font-weight: 700;">🔌 {' + '.join(m['appliances'])}</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">
                        Combined: <strong>{m['combined_kw']:.2f} kW</strong> |
                        Confidence: <strong>{m['confidence']:.0f}%</strong> (Delta Error: {m['error_kw']:.3f} kW)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No single canonical appliance signature match within 15% tolerance.")

with col_res2:
    st.markdown("#### 🏆 Ranked Culprit Room Shortlist")
    st.dataframe(
        ranked_df.head(6)[[
            "rank", "floor", "room_id", "step_delta_kw", "room_energy_kwh", "is_anomaly", "anomaly_type"
        ]].rename(columns={
            "rank": "Rank",
            "floor": "Floor",
            "room_id": "Room",
            "step_delta_kw": "Surge Delta (kW)",
            "room_energy_kwh": "Active Load (kWh)",
            "is_anomaly": "Ground Truth",
            "anomaly_type": "Label"
        }),
        use_container_width=True
    )

st.markdown("---")

# Visual: Floor and Room Load Bar Chart at Incident
st.markdown("#### 🏢 Room Consumption Distribution Across All Floors")
incident_rooms = df_rooms[df_rooms["date_time"] == sel_time].copy()
fig_rooms = px.bar(
    incident_rooms,
    x="room_id",
    y="room_energy_kwh",
    color="floor",
    labels={"room_id": "Room Number", "room_energy_kwh": "Measured Load (kWh)", "floor": "Floor"},
    template="plotly_dark",
    color_discrete_sequence=["#38bdf8", "#f472b6", "#a78bfa"]
)
theme = get_plotly_theme()
fig_rooms.update_layout(**theme["layout"])
fig_rooms.update_layout(height=340, xaxis_tickangle=-45)
st.plotly_chart(fig_rooms, use_container_width=True)
