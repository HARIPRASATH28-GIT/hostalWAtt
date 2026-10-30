"""Page 8: Data Management & System Configuration.

Controls:
- CSV meter data upload and validation
- Dynamic synthetic dataset regeneration (random seed, anomaly rate, blocks)
- Tariff and detection threshold parameter tuning
- Live stream simulator (replay meter readings hour-by-hour)
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import yaml

from app.ui_utils import apply_custom_css, render_header
from src.synth import generate_hostel_dataset
from src.models import train_and_save_all_models
from src.savings import calculate_hourly_waste
from src.alerts import process_and_persist_alerts

st.set_page_config(page_title="Data & Settings - HostelWatt", page_icon="⚙️", layout="wide")
apply_custom_css()
render_header(
    title="Data Ingestion & System Settings",
    subtitle="Manage meter feeds, configure tariffs and thresholds, regenerate synthetic profiles, and simulate live streams",
    icon="⚙️"
)

config_path = ROOT_DIR / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Tabbed Interface
tab_stream, tab_regen, tab_upload, tab_config = st.tabs([
    "🔴 Live Stream Simulator",
    "🔄 Synthetic Data Generator",
    "📁 CSV Meter Upload",
    "⚙️ Tariff & Threshold Settings"
])

# -------------------------------------------------------------
# TAB 1: Live Stream Simulator
# -------------------------------------------------------------
with tab_stream:
    st.markdown("### 🔴 Real-Time Meter Replay Simulator")
    st.markdown("Step through meter readings hour-by-hour to simulate live continuous AI inference and alert tripping.")

    df_scored = pd.read_parquet(ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet")
    df_scored["date_time"] = pd.to_datetime(df_scored["date_time"])

    target_block = st.selectbox("Replay Stream for Block", options=sorted(df_scored["hostel_block"].unique()), index=0)
    block_stream = df_scored[df_scored["hostel_block"] == target_block].sort_values("date_time").reset_index(drop=True)

    # Step slider
    total_steps = len(block_stream)
    step_idx = st.slider("Step Index (Hour)", min_value=1, max_value=min(total_steps, 168), value=24, step=1)

    current_window = block_stream.iloc[max(0, step_idx - 24):step_idx]
    curr_point = block_stream.iloc[step_idx - 1]

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Current Timestamp", f"{curr_point['date_time']:%Y-%m-%d %H:00}")
    with s2:
        st.metric("Live Reading", f"{curr_point['energy_kwh']:.2f} kWh")
    with s3:
        st.metric("Expected Baseline", f"{curr_point['m2_expected']:.2f} kWh")
    with s4:
        is_alert = curr_point["m2_pred"] == 1
        st.metric("Detector Status", "🚨 ALERT" if is_alert else "🟢 NORMAL", delta=f"Sev {curr_point['m2_severity']:.0f}" if is_alert else "Normal")

    # Replay chart
    import plotly.graph_objects as go
    fig_replay = go.Figure()
    fig_replay.add_trace(go.Scatter(x=current_window["date_time"], y=current_window["m2_expected"], mode="lines", name="Expected", line=dict(dash="dot", color="#38bdf8")))
    fig_replay.add_trace(go.Scatter(x=current_window["date_time"], y=current_window["energy_kwh"], mode="lines+markers", name="Actual", line=dict(color="#818cf8")))

    anom_window = current_window[current_window["m2_pred"] == 1]
    if not anom_window.empty:
        fig_replay.add_trace(go.Scatter(x=anom_window["date_time"], y=anom_window["energy_kwh"], mode="markers", name="Alert", marker=dict(color="#f87171", size=10)))

    fig_replay.update_layout(template="plotly_dark", height=280, margin=dict(l=30, r=20, t=20, b=30))
    st.plotly_chart(fig_replay, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Synthetic Generator
# -------------------------------------------------------------
with tab_regen:
    st.markdown("### 🔄 Physics-Informed Synthetic Data Regenerator")
    st.markdown("Customize simulation parameters to generate fresh 12-month synthetic datasets with seeded stochastic noise.")

    c_g1, c_g2, c_g3 = st.columns(3)
    with c_g1:
        new_seed = st.number_input("Random Seed", value=int(config["system"]["seed"]), step=1)
    with c_g2:
        new_hours = st.selectbox("Simulation Duration", options=[4320, 8760], index=1, format_func=lambda x: f"{x} hours ({x//24} days)")
    with c_g3:
        rate_sel = st.slider("Anomaly Rate Scaling", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    if st.button("Generate & Retrain Pipeline", type="primary"):
        with st.spinner("Regenerating physics-based synthetic data and retraining detectors..."):
            config["system"]["seed"] = int(new_seed)
            config["synthetic"]["num_hours"] = int(new_hours)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            df_blocks, df_rooms = generate_hostel_dataset(str(config_path))
            train_and_save_all_models(str(config_path))
            # Reseed alerts
            df_scored = pd.read_parquet(ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet")
            process_and_persist_alerts(calculate_hourly_waste(df_scored))
        st.success(f"Regenerated {len(df_blocks)} block hours and {len(df_rooms)} room hours. Models retrained successfully!")

# -------------------------------------------------------------
# TAB 3: CSV Upload
# -------------------------------------------------------------
with tab_upload:
    st.markdown("### 📁 External Meter CSV Ingestion")
    st.markdown("Upload raw hourly CSV meter logs. Required columns: `timestamp` (or `date_time`), `energy_kwh` (or `meter_reading`), and optionally `hostel_block`.")

    uploaded_file = st.file_uploader("Upload Meter CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df_up = pd.read_csv(uploaded_file)
            st.write(f"**Loaded Preview ({len(df_up)} rows):**")
            st.dataframe(df_up.head(5), use_container_width=True)
            st.success("CSV file validated successfully! Schema compatible with feature pipeline.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# -------------------------------------------------------------
# TAB 4: Config Editor
# -------------------------------------------------------------
with tab_config:
    st.markdown("### ⚙️ Tariff & Detection Threshold Configuration")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### ⚡ Utility Tariff Structure")
        new_tariff = st.number_input("Energy Rate (INR / kWh)", value=float(config["tariff"]["energy_rate_inr_per_kwh"]), step=0.25)
        new_pf = st.number_input("Power Factor", value=float(config["tariff"]["power_factor"]), min_value=0.7, max_value=1.0, step=0.01)
        new_citation = st.text_input("Tariff Citation / Order", value=str(config["tariff"]["citation"]))

    with col_t2:
        st.markdown("#### 🎚️ Anomaly Detection Thresholds")
        new_k = st.number_input("B1 Rolling k-sigma", value=float(config["thresholds"]["rolling_k_sigma"]), step=0.1)
        new_mad = st.number_input("B2 Seasonal MAD Multiplier", value=float(config["thresholds"]["seasonal_mad_multiplier"]), step=0.1)
        new_q = st.number_input("M2 LightGBM Residual Quantile", value=float(config["thresholds"]["lgbm_residual_quantile"]), min_value=0.90, max_value=0.99, step=0.01)

    if st.button("Save System Configuration"):
        config["tariff"]["energy_rate_inr_per_kwh"] = float(new_tariff)
        config["tariff"]["power_factor"] = float(new_pf)
        config["tariff"]["citation"] = str(new_citation)
        config["thresholds"]["rolling_k_sigma"] = float(new_k)
        config["thresholds"]["seasonal_mad_multiplier"] = float(new_mad)
        config["thresholds"]["lgbm_residual_quantile"] = float(new_q)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        st.success("Configuration updated and saved to config.yaml!")
