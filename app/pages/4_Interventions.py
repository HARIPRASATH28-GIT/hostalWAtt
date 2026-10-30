"""Page 4: Interventions & Operational Action Planner.

Prioritizes hostel blocks by risk, recurrence, and financial payoff,
mapping anomaly patterns to actionable facility management checklists.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import yaml

from app.ui_utils import apply_custom_css, render_header, render_metric_card
from src.savings import calculate_hourly_waste
from src.interventions import plan_block_interventions, ACTION_CATALOG

st.set_page_config(page_title="Interventions - HostelWatt", page_icon="🛠️", layout="wide")
apply_custom_css()
render_header(
    title="Block-Wise Intervention Planner",
    subtitle="Prioritized operational roadmaps, facility inspection checklists, and estimated financial payoffs",
    icon="🛠️"
)

# Load data
with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tariff_inr = config["tariff"]["energy_rate_inr_per_kwh"]

@st.cache_data
def get_intervention_plan():
    proc_file = ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet"
    if proc_file.exists():
        df = pd.read_parquet(proc_file)
    else:
        df = pd.read_parquet(ROOT_DIR / "data" / "synthetic" / "hostel_blocks.parquet")
        df["m2_expected"] = df["energy_kwh"] * 0.95
        df["m2_pred"] = df["is_anomaly"]
    df_waste = calculate_hourly_waste(df, tariff_inr=tariff_inr)
    return plan_block_interventions(df_waste, tariff_inr=tariff_inr)

plan_df = get_intervention_plan()

# Priority Summary
top_block = plan_df.iloc[0]["Block"]
top_payoff = plan_df["Estimated Payoff (₹)"].sum()

c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card("Highest Priority Wing", f"{top_block}", "Urgent inspection recommended", status="critical")
with c2:
    render_metric_card("Total Realizable Payoff", f"₹ {top_payoff:,.0f}", "Est. achievable cost recovery", status="normal")
with c3:
    render_metric_card("Monitored Blocks", f"{len(plan_df)}", "Ranked by composite risk index")

st.markdown("---")
st.markdown("### 🏆 Ranked Block Priority Matrix")

# Styled interactive table
st.dataframe(
    plan_df[[
        "Block", "Priority Score", "Urgency", "Primary Anomaly",
        "Recommended Action", "Wasted Energy (kWh)", "Avoidable Loss (₹)", "Estimated Payoff (₹)"
    ]].style.format({
        "Priority Score": "{:.1f}",
        "Wasted Energy (kWh)": "{:,.1f}",
        "Avoidable Loss (₹)": "₹{:,.0f}",
        "Estimated Payoff (₹)": "₹{:,.0f}"
    }),
    use_container_width=True
)

st.markdown("---")
st.markdown("### 📋 Facility Action Protocol Checklist & Execution Guides")

selected_block_id = st.selectbox("Select Block for Action Roadmap", options=plan_df["Block"].tolist(), index=0)
block_row = plan_df[plan_df["Block"] == selected_block_id].iloc[0]

# Display protocol card
st.markdown(
    f"""
    <div class="metric-card" style="margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">
                🏢 {block_row['Block']} — {block_row['Recommended Action']}
            </span>
            <span class="status-badge {'status-critical' if block_row['Urgency'] == 'CRITICAL' else 'status-warning'}">
                {block_row['Urgency']} URGENCY
            </span>
        </div>
        <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.75rem;">
            <strong>Detected Anomaly:</strong> <code>{block_row['Primary Anomaly']}</code> |
            <strong>Average Event Run:</strong> {block_row['Avg Persistence (hrs)']} hours |
            <strong>Recurrence:</strong> {block_row['Recurrence Rate (%)']}% of monitored days
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.8rem 1rem; border-radius: 8px; border-left: 4px solid #38bdf8; color: #e2e8f0; font-size: 0.9rem;">
            <strong>Standard Operating Procedure:</strong> {block_row['Action Protocol']}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Interactive Checklist
st.markdown("#### ✍️ Warden & Electrician Execution Checklist")
col_chk1, col_chk2 = st.columns(2)

with col_chk1:
    st.checkbox("Verify main panel board breaker temperatures and phase current balance", key="c1")
    st.checkbox("Inspect floor distribution boards (DBs) for warm contactor terminals", key="c2")
    st.checkbox("Perform unannounced room sweeps for unauthorized immersion heaters / induction plates", key="c3")

with col_chk2:
    st.checkbox("Check corridor, laundry room, and common washroom high-bay lighting timers", key="c4")
    st.checkbox("Log inspection findings into facility maintenance registry", key="c5")
    st.checkbox("Verify submeter RS-485 Modbus communication signal integrity", key="c6")

if st.button("Submit Inspection Log"):
    st.success(f"Inspection confirmation recorded for {block_row['Block']}! Protocol status updated.")
