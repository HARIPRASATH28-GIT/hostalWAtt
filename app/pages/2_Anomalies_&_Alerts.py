"""Page 2: Anomalies & Alerts Feed.

Live operational alert management center:
- Ranked incident feed with severity badges
- Block, escalation, and resolution status filters
- Interactive Acknowledge and Resolve workflow stored in persistent SQLite
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import yaml

from app.ui_utils import apply_custom_css, render_header
from src.alerts import fetch_alerts as fetch_batch_alerts, acknowledge_alert as ack_batch, resolve_alert as res_batch
from backend.database import get_live_alerts, acknowledge_alert as ack_live, resolve_alert as res_live

st.set_page_config(page_title="Anomalies & Alerts - HostelWatt", page_icon="🚨", layout="wide")
apply_custom_css()
render_header(
    title="Anomalies & Alerts Management Center",
    subtitle="Audit, track, acknowledge, and resolve real-time live alerts and historical facility incident logs",
    icon="🚨"
)

# Tab selection for Live Stream Alerts vs Historical Incident Log
tab_live, tab_hist = st.tabs([
    "🔴 Real-Time Continuous Monitoring Alerts (Live Feed)",
    "🗄️ Historical Semester Batch Incident Log (Audit Archive)"
])

with tab_live:
    st.markdown("#### 🚨 Real-Time Continuous Alerts (From Backend Daemon)")
    live_records = get_live_alerts(limit=100)
    if not live_records:
        st.info("No real-time alerts currently in live queue. Trigger a test spike from the Live Monitoring page to observe real-time alerting!")
    else:
        df_live = pd.DataFrame(live_records)
        c_l1, c_l2 = st.columns(2)
        with c_l1:
            l_status = st.selectbox("Live Alert Status", options=["ALL", "NEW", "ACKNOWLEDGED", "RESOLVED"], index=0, key="l_stat")
        with c_l2:
            l_block = st.selectbox("Live Alert Block", options=["ALL"] + sorted(df_live["hostel_block"].unique().tolist()), index=0, key="l_blk")

        if l_status != "ALL":
            df_live = df_live[df_live["status"] == l_status]
        if l_block != "ALL":
            df_live = df_live[df_live["hostel_block"] == l_block]

        for _, l_row in df_live.head(20).iterrows():
            a_id = l_row["id"]
            stat = l_row["status"]
            sev = l_row["severity_score"]
            badge = "status-critical" if sev >= 65 else "status-warning"

            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                            🏢 {l_row['hostel_block']} — {l_row['anomaly_type'].replace('_', ' ').title()}
                        </span>
                        <div>
                            <span class="status-badge {badge}">Severity: {sev:.0f}/100</span>
                            <span style="font-weight: 700; margin-left: 0.5rem; color: {'#f87171' if stat=='NEW' else ('#fbbf24' if stat=='ACKNOWLEDGED' else '#4ade80')};">
                                ● {stat}
                            </span>
                        </div>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin: 0.35rem 0;">
                        <strong>Time:</strong> {l_row['timestamp']} |
                        <strong>Actual:</strong> {l_row['energy_kwh']:.2f} kW |
                        <strong>Expected:</strong> {l_row['expected_kwh']:.2f} kW |
                        <strong>Surplus:</strong> <strong style="color: #f87171;">+{l_row['excess_kwh']:.2f} kWh (₹{l_row['excess_cost_inr']:.0f})</strong>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.85rem; color: #cbd5e1;">
                        <strong>Action Protocol:</strong> {l_row['recommended_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            col_b1, col_b2, _ = st.columns([2, 2, 8])
            with col_b1:
                if stat == "NEW":
                    if st.button(f"Acknowledge #{a_id}", key=f"ack_l_{a_id}"):
                        ack_live(a_id, user="Warden")
                        st.success(f"Live Alert #{a_id} Acknowledged!")
                        st.rerun()
                else:
                    st.caption(f"Ack: {l_row.get('acknowledged_by', 'Warden')}")
            with col_b2:
                if stat != "RESOLVED":
                    if st.button(f"Resolve #{a_id}", key=f"res_l_{a_id}"):
                        res_live(a_id, user="Warden")
                        st.success(f"Live Alert #{a_id} Resolved!")
                        st.rerun()
                else:
                    st.caption(f"Resolved @ {l_row.get('resolved_at', 'Completed')}")

with tab_hist:
    # Filter controls for historical archive
    col_f1, col_f2, col_f3 = st.columns(3)

    alerts_df = fetch_batch_alerts(str(ROOT_DIR / "data" / "alerts.db"))


with col_f1:
    all_blocks = ["ALL"] + sorted(alerts_df["hostel_block"].unique().tolist()) if not alerts_df.empty else ["ALL"]
    sel_block = st.selectbox("Filter by Block", options=all_blocks, index=0)

with col_f2:
    status_options = ["ALL", "NEW", "ACKNOWLEDGED", "RESOLVED"]
    sel_status = st.selectbox("Filter by Status", options=status_options, index=0)

with col_f3:
    escalation_options = ["ALL", "ESCALATED (Warden/Maintenance)", "ACTIVE (Investigate)", "LOG (Monitor)"]
    sel_esc = st.selectbox("Filter by Severity Tier", options=escalation_options, index=0)

# Filter dataframe
filtered_df = alerts_df.copy()
if not filtered_df.empty:
    if sel_block != "ALL":
        filtered_df = filtered_df[filtered_df["hostel_block"] == sel_block]
    if sel_status != "ALL":
        filtered_df = filtered_df[filtered_df["status"] == sel_status]
    if sel_esc != "ALL":
        filtered_df = filtered_df[filtered_df["escalation_level"] == sel_esc]

# Metrics summary bar
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
total_active = len(alerts_df[alerts_df["status"] != "RESOLVED"]) if not alerts_df.empty else 0
total_escalated = len(alerts_df[alerts_df["escalation_level"].str.contains("ESCALATED", na=False)]) if not alerts_df.empty else 0
total_waste_inr = alerts_df["cost_inr"].sum() if not alerts_df.empty else 0

with m1:
    st.metric("Total Incident Records", f"{len(alerts_df)}")
with m2:
    st.metric("Active / Pending Action", f"{total_active}", delta="Unresolved")
with m3:
    st.metric("Critical Escalations", f"{total_escalated}", delta="Warden alert")
with m4:
    st.metric("Total Flagged Waste (Est.)", f"₹ {total_waste_inr:,.0f}")

st.markdown("---")
st.markdown(f"### 📋 Active Incident Queue ({len(filtered_df)} Matching Records)")

if filtered_df.empty:
    st.info("No alert records matching current filter criteria.")
else:
    # Render interactive cards with acknowledge/resolve buttons
    for idx, row in filtered_df.head(25).iterrows():
        alert_id = row["id"]
        status = row["status"]
        esc = row["escalation_level"]

        if "ESCALATED" in esc:
            badge_class = "status-critical"
        elif "ACTIVE" in esc:
            badge_class = "status-warning"
        else:
            badge_class = "status-normal"

        status_pill = {
            "NEW": '<span style="color: #f87171; font-weight: 700;">● NEW</span>',
            "ACKNOWLEDGED": '<span style="color: #fbbf24; font-weight: 700;">◐ ACKNOWLEDGED</span>',
            "RESOLVED": '<span style="color: #4ade80; font-weight: 700;">✔ RESOLVED</span>'
        }.get(status, status)

        with st.container():
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                            🏢 {row['hostel_block']} — {row['anomaly_type'].replace('_', ' ').title()}
                        </span>
                        <div>
                            <span class="status-badge {badge_class}" style="margin-right: 0.5rem;">{esc}</span>
                            {status_pill}
                        </div>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                        <strong>Window:</strong> {row['start_time']} to {row['end_time']} ({row['duration_hours']} hrs) |
                        <strong>Severity:</strong> {row['severity_score']:.0f}/100 |
                        <strong>Est. Waste:</strong> {row['wasted_kwh']:.1f} kWh (<strong style="color: #f87171;">₹{row['cost_inr']:.0f}</strong>)
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 0.6rem 0.8rem; border-radius: 8px; border-left: 3px solid #6366f1; color: #cbd5e1; font-size: 0.85rem;">
                        <strong>Action Protocol:</strong> {row['suggested_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Action buttons row
            b_c1, b_c2, b_c3 = st.columns([2, 2, 8])
            with b_c1:
                if status == "NEW":
                    if st.button(f"Acknowledge #{alert_id}", key=f"ack_{alert_id}"):
                        acknowledge_alert(alert_id, user="Warden", db_path=str(ROOT_DIR / "data" / "alerts.db"))
                        st.success(f"Alert #{alert_id} Acknowledged!")
                        st.rerun()
                else:
                    st.caption(f"Ack by: {row.get('acknowledged_by', 'Warden')}")
            with b_c2:
                if status != "RESOLVED":
                    if st.button(f"Resolve #{alert_id}", key=f"res_{alert_id}"):
                        resolve_alert(alert_id, notes="Inspection complete & issue mitigated", db_path=str(ROOT_DIR / "data" / "alerts.db"))
                        st.success(f"Alert #{alert_id} Resolved!")
                        st.rerun()
                else:
                    st.caption(f"Resolved @ {row.get('resolved_at', 'Completed')}")
            with b_c3:
                st.empty()
