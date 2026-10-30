"""HostelWatt: AI-Powered Hostel Electricity Monitoring & Anomaly Detection.

Minimal, clean, and professional homepage dashboard.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.savings import calculate_hourly_waste
from src.synth import generate_hostel_dataset

# 1. Page Configuration
st.set_page_config(
    page_title="HostelWatt",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Clean, Minimal CSS (Theme-adaptive for Dark and Light mode)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Compact container padding to fit on standard laptop screens */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px;
    }

    /* Minimal Header */
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.85rem;
        opacity: 0.75;
        margin-top: 2px;
        margin-bottom: 0.75rem;
    }

    /* KPI Cards */
    .kpi-container {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border-left: 3px solid #3b82f6;
    }
    .kpi-container.alert {
        border-left: 3px solid #ef4444;
    }
    .kpi-container.warning {
        border-left: 3px solid #f59e0b;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.75;
        margin-bottom: 0.2rem;
    }
    .kpi-val {
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.74rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }

    /* Section Headings */
    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 3. Data & Configuration Loading
@st.cache_data
def load_project_data():
    """Load evaluation scored data or synthetic fallback, with calculated waste."""
    with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tariff = cfg.get("tariff", {}).get("energy_rate_inr_per_kwh", 7.50)
    proc_file = ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet"
    synth_file = ROOT_DIR / "data" / "synthetic" / "hostel_blocks.parquet"

    if proc_file.exists():
        df = pd.read_parquet(proc_file)
    elif synth_file.exists():
        df = pd.read_parquet(synth_file)
        df["m2_expected"] = df["energy_kwh"] * 0.95
        df["m2_pred"] = df.get("is_anomaly", 0)
    else:
        df, _ = generate_hostel_dataset(str(ROOT_DIR / "config.yaml"))
        df["m2_expected"] = df["energy_kwh"] * 0.95
        df["m2_pred"] = df.get("is_anomaly", 0)

    df["date_time"] = pd.to_datetime(df["date_time"])
    df = calculate_hourly_waste(df, tariff_inr=tariff)
    return df, tariff, cfg


df_all, tariff_inr, config = load_project_data()

block_name_map = {b["id"]: b["name"] for b in config.get("blocks", [])}

# 4. Header & Date-Range Filter
min_dt = df_all["date_time"].min().date()
max_dt = df_all["date_time"].max().date()
# Default to last 14 days of data to maintain a clean, compact view
default_start = max(min_dt, max_dt - pd.Timedelta(days=14))

header_col, filter_col = st.columns([6, 4])
with header_col:
    st.markdown(
        """
        <div class="app-title">HostelWatt</div>
        <div class="app-subtitle">AI-Powered Hostel Electricity Monitoring & Anomaly Detection</div>
        """,
        unsafe_allow_html=True,
    )

with filter_col:
    date_selection = st.date_input(
        "Date Range",
        value=(default_start, max_dt),
        min_value=min_dt,
        max_value=max_dt,
        label_visibility="collapsed",
    )

# Parse date filter safely
if isinstance(date_selection, (tuple, list)):
    if len(date_selection) == 2:
        start_date, end_date = date_selection
    elif len(date_selection) == 1:
        start_date = end_date = date_selection[0]
    else:
        start_date, end_date = default_start, max_dt
else:
    start_date = end_date = date_selection

# Filter dataset
mask = (df_all["date_time"].dt.date >= start_date) & (df_all["date_time"].dt.date <= end_date)
df_filtered = df_all[mask].copy()


# Helper function to render a clean KPI card
def render_kpi(label: str, value: str, subtext: str, card_type: str = "normal"):
    cls_suffix = f" {card_type}" if card_type in ["alert", "warning"] else ""
    st.markdown(
        f"""
        <div class="kpi-container{cls_suffix}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 5. Component A: Four Small KPI Cards
c1, c2, c3, c4 = st.columns(4)

if not df_filtered.empty:
    total_kwh = float(df_filtered["energy_kwh"].sum())
    anom_count = int(df_filtered["m2_pred"].sum())
    waste_kwh = float(df_filtered["wasted_kwh"].sum())
    waste_cost_inr = waste_kwh * tariff_inr

    hourly_system = df_filtered.groupby("date_time")["energy_kwh"].sum()
    peak_demand_kw = float(hourly_system.max())
    peak_time_str = f"Peak @ {hourly_system.idxmax():%d %b %H:00}"

    with c1:
        render_kpi("Total Energy Consumption", f"{total_kwh:,.1f} kWh", f"₹ {total_kwh * tariff_inr:,.0f} estimated cost")
    with c2:
        render_kpi(
            "Detected Anomalies",
            f"{anom_count:,}",
            "Flagged abnormal hours",
            card_type="alert" if anom_count > 0 else "normal",
        )
    with c3:
        render_kpi(
            "Estimated Energy Waste",
            f"{waste_kwh:,.1f} kWh",
            f"₹ {waste_cost_inr:,.0f} avoidable cost",
            card_type="warning" if waste_kwh > 0 else "normal",
        )
    with c4:
        render_kpi("Peak Electricity Demand", f"{peak_demand_kw:.1f} kW", peak_time_str)
else:
    with c1:
        render_kpi("Total Energy Consumption", "N/A", "No data in range")
    with c2:
        render_kpi("Detected Anomalies", "N/A", "No data in range")
    with c3:
        render_kpi("Estimated Energy Waste", "N/A", "No data in range")
    with c4:
        render_kpi("Peak Electricity Demand", "N/A", "No data in range")


# 6. Component B: One Main Electricity Consumption Chart
chart_head_col, chart_scope_col = st.columns([7, 3])
with chart_head_col:
    st.markdown(
        "<div class=\"section-title\">Electricity Consumption & Baseline Comparison</div>",
        unsafe_allow_html=True,
    )
with chart_scope_col:
    scope_options = ["All Blocks (Total)"] + [b["name"] for b in config.get("blocks", [])]
    selected_scope = st.selectbox("View Scope", options=scope_options, index=0, label_visibility="collapsed")

if not df_filtered.empty:
    if selected_scope == "All Blocks (Total)":
        chart_df = df_filtered.groupby("date_time", as_index=False).agg({
            "energy_kwh": "sum",
            "m2_expected": "sum",
            "m2_pred": lambda x: int((x == 1).any()),
        }).sort_values("date_time")
    else:
        chosen_id = next((b["id"] for b in config.get("blocks", []) if b["name"] == selected_scope), selected_scope)
        chart_df = df_filtered[df_filtered["hostel_block"] == chosen_id].sort_values("date_time")

    fig = go.Figure()

    # Baseline Expected Curve
    fig.add_trace(go.Scatter(
        x=chart_df["date_time"],
        y=chart_df["m2_expected"],
        mode="lines",
        name="Expected Baseline",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate="Expected: %{y:.2f} kWh<extra></extra>",
    ))

    # Actual Consumption Curve
    fig.add_trace(go.Scatter(
        x=chart_df["date_time"],
        y=chart_df["energy_kwh"],
        mode="lines",
        name="Actual Consumption",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="Actual: %{y:.2f} kWh<extra></extra>",
    ))

    # Anomaly points highlighted clearly
    anom_points = chart_df[chart_df["m2_pred"] == 1]
    if not anom_points.empty:
        fig.add_trace(go.Scatter(
            x=anom_points["date_time"],
            y=anom_points["energy_kwh"],
            mode="markers",
            name="Anomaly Detected",
            marker=dict(color="#ef4444", size=6, symbol="circle"),
            hovertemplate="🚨 Anomaly: %{y:.2f} kWh<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=45, r=15, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Load (kWh)",
            title_font=dict(size=11),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(size=10),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No consumption data available for the selected date range.")


# 7. Components C & D: Compact Hostel Block Status Table & Latest Anomalies Table
col_tbl_left, col_tbl_right = st.columns([5, 5])

# C. One Compact Hostel Block Status Table
with col_tbl_left:
    st.markdown("<div class=\"section-title\">Hostel Block Status</div>", unsafe_allow_html=True)

    if not df_filtered.empty:
        block_rows = []
        for b in config.get("blocks", []):
            b_id = b["id"]
            b_name = b["name"]
            b_df = df_filtered[df_filtered["hostel_block"] == b_id]

            if not b_df.empty:
                b_energy = float(b_df["energy_kwh"].sum())
                b_anoms = int(b_df["m2_pred"].sum())
                b_waste_kwh = float(b_df["wasted_kwh"].sum())
                b_waste_cost = float(b_df["cost_inr"].sum())

                # Existing project classification rule
                if b_anoms > 40 or b_waste_kwh > 200:
                    status_text = "Critical"
                elif b_anoms > 15 or b_waste_kwh > 80:
                    status_text = "Warning"
                else:
                    status_text = "Normal"
            else:
                b_energy = 0.0
                b_anoms = 0
                b_waste_cost = 0.0
                status_text = "Normal"

            block_rows.append({
                "Hostel Block": b_name,
                "Energy Consumption (kWh)": round(b_energy, 1),
                "Anomaly Count": b_anoms,
                "Estimated Waste Cost (₹)": round(b_waste_cost, 0),
                "Status": status_text,
            })

        df_blocks = pd.DataFrame(block_rows)
        st.dataframe(
            df_blocks,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hostel Block": st.column_config.TextColumn("Hostel Block"),
                "Energy Consumption (kWh)": st.column_config.NumberColumn("Consumption", format="%.1f kWh"),
                "Anomaly Count": st.column_config.NumberColumn("Anomalies", format="%d"),
                "Estimated Waste Cost (₹)": st.column_config.NumberColumn("Waste Cost", format="₹ %d"),
                "Status": st.column_config.TextColumn("Status"),
            },
        )
    else:
        st.info("No block records available for the selected period.")


# D. Latest Anomalies Table (5 most recent incidents)
with col_tbl_right:
    st.markdown("<div class=\"section-title\">Latest Anomalies</div>", unsafe_allow_html=True)

    if not df_filtered.empty:
        anom_subset = df_filtered[df_filtered["m2_pred"] == 1].sort_values("date_time", ascending=False).head(5)

        if anom_subset.empty:
            st.info("No anomalies detected in the selected period.")
        else:
            anom_rows = []
            for _, r in anom_subset.iterrows():
                b_label = block_name_map.get(r["hostel_block"], r["hostel_block"])
                # Extract simple short name like "Block A"
                b_short = b_label.split(" (")[0] if " (" in b_label else b_label

                atype = r.get("anomaly_type", "none")
                if atype == "none":
                    atype = "Surge Spike" if r["energy_kwh"] > r["m2_expected"] else "Load Dip"
                else:
                    atype = atype.replace("_", " ").title()

                actual_val = float(r["energy_kwh"])
                expected_val = float(r["m2_expected"])
                excess_val = max(0.0, actual_val - expected_val)

                anom_rows.append({
                    "Timestamp": r["date_time"].strftime("%Y-%m-%d %H:%M"),
                    "Hostel Block": b_short,
                    "Anomaly Type": atype,
                    "Actual Consumption": round(actual_val, 2),
                    "Expected Consumption": round(expected_val, 2),
                    "Estimated Excess Energy": round(excess_val, 2),
                })

            df_anoms = pd.DataFrame(anom_rows)
            st.dataframe(
                df_anoms,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn("Timestamp"),
                    "Hostel Block": st.column_config.TextColumn("Block"),
                    "Anomaly Type": st.column_config.TextColumn("Type"),
                    "Actual Consumption": st.column_config.NumberColumn("Actual", format="%.2f kWh"),
                    "Expected Consumption": st.column_config.NumberColumn("Expected", format="%.2f kWh"),
                    "Estimated Excess Energy": st.column_config.NumberColumn("Excess", format="+%.2f kWh"),
                },
            )
    else:
        st.info("No anomaly data available for the selected period.")
