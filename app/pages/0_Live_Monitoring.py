"""Page 0: Real-Time Continuous Electricity Monitoring.

Provides:
- Independent backend daemon monitoring status & controls (60s demo vs 3600s production)
- Prominent labeling of simulated meter telemetry
- Auto-refreshing alert queue and telemetry metrics (every 15s via st.fragment)
- Interactive anomaly spike injector for live testing & demonstrations
- Alert acknowledgment and resolution workflow stored in persistent SQLite
- Historical telemetry audit log
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
import requests

from app.ui_utils import apply_custom_css, render_header, render_metric_card, get_plotly_theme
from backend.database import (
    get_latest_readings, get_historical_readings,
    get_live_alerts, acknowledge_alert, resolve_alert, get_monitoring_stats
)
from backend.scheduler import scheduler_instance

st.set_page_config(page_title="Live Monitoring - HostelWatt", page_icon="🔴", layout="wide")
apply_custom_css()

render_header(
    title="Real-Time Continuous Electricity Monitoring",
    subtitle="Independent backend daemon polling simulated meter feeds, detecting anomalous load spikes, and tracking live alerts",
    icon="🔴"
)

# Load configuration
config_path = ROOT_DIR / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tariff_inr = config["tariff"]["energy_rate_inr_per_kwh"]

# Top prominent disclaimer and simulated feed label
st.markdown(
    """
    <div style="background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(99, 102, 241, 0.15)); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 10px; padding: 0.85rem 1.25rem; margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="background: #ef4444; color: #fff; font-size: 0.75rem; font-weight: 800; padding: 0.2rem 0.55rem; border-radius: 4px; letter-spacing: 0.05em; margin-right: 0.6rem;">
                    [SIMULATED METER DATA]
                </span>
                <span style="color: #f1f5f9; font-weight: 600; font-size: 0.95rem;">
                    Backend Telemetry Ingestion Active
                </span>
                <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 0.25rem;">
                    Telemetry is generated and evaluated independently in the background on a dedicated schedule. Webpage automatically polls database state every 15 seconds.
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: #38bdf8; font-size: 0.85rem; font-family: monospace;">
                    ⚡ Ready for Modbus/MQTT Smart Meters
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Backend Control Bar
st.markdown("### 🎛️ Backend Monitoring Service Controls")
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2.5, 3, 3, 3.5])

scheduler_status = scheduler_instance.get_status()
is_running = scheduler_status["is_running"]
current_interval = scheduler_status["interval_seconds"]
current_mode = scheduler_status["mode"]

with ctrl_c1:
    if is_running:
        if st.button("⏹ Stop Daemon", type="secondary"):
            scheduler_instance.stop()
            st.rerun()
    else:
        if st.button("▶ Start Daemon", type="primary"):
            scheduler_instance.start()
            st.rerun()

with ctrl_c2:
    mode_choice = st.selectbox(
        "Operating Frequency",
        options=["Demo Mode (60 Seconds)", "Production Mode (1 Hour / 3600s)"],
        index=0 if current_interval <= 60 else 1
    )
    desired_interval = 60 if "60" in mode_choice else 3600
    desired_mode = "demo" if desired_interval == 60 else "production"
    if desired_interval != current_interval:
        scheduler_instance.set_interval(desired_interval, mode=desired_mode)
        st.success(f"Updated backend interval to {desired_interval}s!")

with ctrl_c3:
    if st.button("🔄 Poll / Trigger Tick Now"):
        tick_res = scheduler_instance.trigger_tick()
        st.success(f"Processed {tick_res.get('blocks_processed', 5)} blocks immediately!")
        st.rerun()

with ctrl_c4:
    last_run = scheduler_status["last_run_timestamp"] or "Just started"
    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: #94a3b8; padding-top: 0.5rem;">
            <strong>Status:</strong> <span style="color: {'#4ade80' if is_running else '#f87171'}; font-weight: 700;">{'RUNNING (Background)' if is_running else 'STOPPED'}</span><br>
            <strong>Last Tick:</strong> <code>{last_run}</code><br>
            <strong>Total Cycles:</strong> <code>{scheduler_status['total_ticks']}</code>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Auto-Refreshing Fragment (polls every 15s without generating extra readings)
@st.fragment(run_every="15s")
def render_live_monitoring_view():
    stats = get_monitoring_stats()
    latest_readings = get_latest_readings(limit=40)
    active_alerts = get_live_alerts(status_filter="NEW", limit=25)
    all_alerts = get_live_alerts(limit=50)

    # Top KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Live Ingested Readings", f"{stats['total_readings']:,}", "Persistent SQLite records")
    with k2:
        render_metric_card("Active Unresolved Alerts", f"{stats['new_alerts']}", "Require Warden action", status="critical" if stats['new_alerts'] > 0 else "normal")
    with k3:
        render_metric_card("Total Avoidable Excess", f"{stats['total_excess_kwh']:,.1f} kWh", "Avoidable surplus detected")
    with k4:
        render_metric_card("Estimated Excess Cost", f"₹ {stats['total_excess_cost_inr']:,.0f}", f"@ ₹{tariff_inr:.2f} / kWh", status="warning" if stats['total_excess_cost_inr'] > 0 else "normal")

    st.markdown(
        f"<div style='text-align: right; font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;'>Auto-refreshed: {datetime.now().strftime('%H:%M:%S')} (Interval: 15s)</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Interactive Spike Injection Panel (For Project Reviews & Demo)
    st.markdown("### ⚡ Live Demonstration: Inject Test Electrical Spike")
    st.markdown("Demonstrate real-time anomaly detection by queuing an abnormal electrical load onto any hostel block.")

    inj_col1, inj_col2, inj_col3, inj_col4 = st.columns([3, 3, 3, 3])
    with inj_col1:
        inj_block = st.selectbox("Target Hostel Block", options=["Block_A", "Block_B", "Block_C", "Block_D", "Block_E"], index=0, key="inj_b")
    with inj_col2:
        inj_type = st.selectbox(
            "Anomaly Pattern",
            options=["appliance_spike", "left_on", "wiring_drift", "meter_fault"],
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
            key="inj_t"
        )
    with inj_col3:
        inj_kw = st.number_input("Spike Magnitude (kW)", min_value=1.0, max_value=10.0, value=3.2, step=0.5, key="inj_kw")
    with inj_col4:
        st.write("")
        st.write("")
        if st.button("🚀 Inject Spike & Evaluate Now", type="primary", key="btn_inj"):
            scheduler_instance.inject_spike(inj_block, extra_kwh=inj_kw, anomaly_type=inj_type)
            tick_out = scheduler_instance.trigger_tick()
            st.success(f"Injected +{inj_kw} kW on {inj_block}! Anomaly flagged and alert recorded in SQLite.")
            st.rerun()

    st.markdown("---")

    # Real-Time Live Alert Queue
    st.markdown(f"### 🚨 Real-Time Anomaly Alert Queue ({len(active_alerts)} Active Alerts)")

    if not active_alerts:
        st.success("🟢 No active unacknowledged electrical alerts! All block telemetry operating within normal expected thresholds.")
    else:
        for alert in active_alerts:
            a_id = alert["id"]
            severity = alert["severity_score"]
            sev_color = "#f87171" if severity >= 65 else "#fbbf24"
            badge_class = "status-critical" if severity >= 65 else "status-warning"

            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 0.75rem; border-left: 4px solid {sev_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: #f8fafc;">
                            🏢 {alert['hostel_block']} — {alert['anomaly_type'].replace('_', ' ').title()}
                        </span>
                        <div>
                            <span class="status-badge {badge_class}">Severity: {severity:.0f}/100</span>
                            <span style="color: #ef4444; font-weight: 700; margin-left: 0.5rem;">● NEW ALERT</span>
                        </div>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin: 0.35rem 0;">
                        <strong>Timestamp:</strong> {alert['timestamp']} |
                        <strong>Actual:</strong> {alert['energy_kwh']:.2f} kW |
                        <strong>Expected Baseline:</strong> {alert['expected_kwh']:.2f} kW |
                        <strong>Excess:</strong> <span style="color: #f87171; font-weight: 700;">+{alert['excess_kwh']:.2f} kWh</span> |
                        <strong>Avoidable Cost:</strong> <span style="color: #f87171; font-weight: 700;">₹{alert['excess_cost_inr']:.0f}</span>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.85rem; color: #cbd5e1;">
                        <strong>Recommended Protocol:</strong> {alert['recommended_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Interactive Acknowledge / Resolve action buttons
            btn_col1, btn_col2, _ = st.columns([2, 2, 8])
            with btn_col1:
                if st.button(f"Acknowledge #{a_id}", key=f"live_ack_{a_id}"):
                    acknowledge_alert(a_id, user="Warden_Live")
                    st.success(f"Alert #{a_id} Acknowledged!")
                    st.rerun()
            with btn_col2:
                if st.button(f"Resolve #{a_id}", key=f"live_res_{a_id}"):
                    resolve_alert(a_id, user="Warden_Live")
                    st.success(f"Alert #{a_id} Resolved!")
                    st.rerun()

    st.markdown("---")

    # Real-Time Telemetry Curve (Actual vs Expected)
    st.markdown("### 📈 Live Telemetry Stream: Actual vs. Model Expected Load")

    if latest_readings:
        df_readings = pd.DataFrame(latest_readings)
        df_readings["timestamp"] = pd.to_datetime(df_readings["timestamp"])
        df_readings = df_readings.sort_values("timestamp")

        # Block selector for chart
        chart_block = st.selectbox(
            "Filter Telemetry Curve by Block",
            options=sorted(df_readings["hostel_block"].unique()),
            index=0,
            key="chart_blk"
        )
        df_sub = df_readings[df_readings["hostel_block"] == chart_block]

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=df_sub["timestamp"],
            y=df_sub["expected_kwh"],
            mode="lines",
            name="Expected Baseline (LightGBM)",
            line=dict(color="#38bdf8", dash="dot", width=2)
        ))
        fig_live.add_trace(go.Scatter(
            x=df_sub["timestamp"],
            y=df_sub["energy_kwh"],
            mode="lines+markers",
            name="Measured Load (kWh)",
            line=dict(color="#818cf8", width=2)
        ))

        anom_rows = df_sub[df_sub["is_anomaly"] == 1]
        if not anom_rows.empty:
            fig_live.add_trace(go.Scatter(
                x=anom_rows["timestamp"],
                y=anom_rows["energy_kwh"],
                mode="markers",
                name="Flagged Spike",
                marker=dict(color="#f87171", size=11, symbol="triangle-up")
            ))

        theme = get_plotly_theme()
        fig_live.update_layout(**theme["layout"])
        fig_live.update_layout(
            height=340,
            hovermode="x unified",
            xaxis_title="Reading Time",
            yaxis_title="Power Demand (kW = kWh/h)",
            legend=dict(orientation="h", y=1.12, x=0)
        )
        st.plotly_chart(fig_live, use_container_width=True)

    # Historical Telemetry Table
    st.markdown("---")
    st.markdown("### 🗄️ Historical Monitoring Telemetry Log (Persistent Database)")
    df_hist = get_historical_readings(limit=100)
    if not df_hist.empty:
        st.dataframe(
            df_hist[[
                "id", "timestamp", "hostel_block", "energy_kwh", "expected_kwh",
                "excess_kwh", "excess_cost_inr", "is_anomaly", "severity_score",
                "anomaly_type", "source"
            ]].style.format({
                "energy_kwh": "{:.2f}",
                "expected_kwh": "{:.2f}",
                "excess_kwh": "{:.2f}",
                "excess_cost_inr": "₹{:.2f}",
                "severity_score": "{:.0f}"
            }),
            use_container_width=True
        )


# Execute auto-refreshing fragment
render_live_monitoring_view()
