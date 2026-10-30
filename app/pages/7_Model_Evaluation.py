"""Page 7: Model Evaluation & Benchmark Metrics.

Comprehensive evaluation suite:
- Precision, Recall, F1, PR-AUC, and Confusion Matrix
- Model Comparison Table (B1, B2, M1, M2)
- Detection Delay distribution
- Recall breakdown per anomaly type
- Context Feature Ablation showing False Positive reduction
- LEAD 1.0 vs. Residential Hostel domain gap analysis
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
from src.evaluate import (
    compare_all_models, compute_detection_delay_and_type_recall,
    compute_core_metrics, run_context_ablation
)

st.set_page_config(page_title="Model Evaluation - HostelWatt", page_icon="🧪", layout="wide")
apply_custom_css()
render_header(
    title="Model Benchmarks & Performance Metrics",
    subtitle="Comparative evaluation of B1, B2, M1, M2 detectors, detection delays, ablation impact, and domain gap",
    icon="🧪"
)

# Load scored test dataset
@st.cache_data
def load_eval_data():
    proc_file = ROOT_DIR / "data" / "processed" / "test_evaluation_scored.parquet"
    if not proc_file.exists():
        from src.models import train_and_save_all_models
        df_scored, _ = train_and_save_all_models()
        return df_scored
    return pd.read_parquet(proc_file)

df_eval = load_eval_data()

# Model Comparison Table
st.markdown("### 📊 Model Benchmark Comparison (Out-of-Time Test Set)")
comparison_df = compare_all_models(df_eval)

st.dataframe(
    comparison_df.style.format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "PR-AUC": "{:.3f}",
        "Avg Detection Delay (h)": "{:.2f}",
        "False Alerts / Block-Wk": "{:.1f}"
    }).highlight_max(subset=["Precision", "Recall", "F1 Score", "PR-AUC"], color="#1e3a5f"),
    use_container_width=True
)

st.markdown("---")

# Row: Confusion Matrix & Recall Per Anomaly Type
col_cm, col_rec = st.columns([5, 7])

with col_cm:
    st.markdown("#### 🎯 Confusion Matrix (M2 LightGBM)")
    y_true = df_eval["is_anomaly"].values
    y_pred = df_eval["m2_pred"].values
    m2_metrics = compute_core_metrics(y_true, y_pred)

    cm = m2_metrics["confusion_matrix"]
    cm_labels = ["Normal", "Anomaly"]
    fig_cm = px.imshow(
        cm,
        x=cm_labels,
        y=cm_labels,
        text_auto=True,
        labels=dict(x="Predicted Label", y="Ground Truth Label", color="Count"),
        color_continuous_scale="Blues",
        template="plotly_dark"
    )
    theme = get_plotly_theme()
    fig_cm.update_layout(**theme["layout"])
    fig_cm.update_layout(height=320)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_rec:
    st.markdown("#### 🏷️ Recall by Anomaly Type (M2 LightGBM)")
    type_recalls, avg_delay, delay_dist = compute_detection_delay_and_type_recall(df_eval, "m2_pred")

    rec_df = pd.DataFrame([
        {"Anomaly Type": k.replace("_", " ").title(), "Recall (%)": round(v * 100.0, 1)}
        for k, v in type_recalls.items()
    ]).sort_values("Recall (%)", ascending=False)

    fig_rec = px.bar(
        rec_df,
        x="Recall (%)",
        y="Anomaly Type",
        orientation="h",
        text="Recall (%)",
        template="plotly_dark",
        color="Recall (%)",
        color_continuous_scale="Viridis"
    )
    fig_rec.update_layout(**theme["layout"])
    fig_rec.update_layout(height=320, coloraxis_showscale=False)
    fig_rec.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    st.plotly_chart(fig_rec, use_container_width=True)

st.markdown("---")

# Feature Ablation Analysis
st.markdown("### 🔬 Feature Ablation Study: Impact of Context Features")
st.markdown(
    """
    To evaluate whether contextual metadata (occupancy, academic day types, outside temperature) reduces false alarms,
    we ablate context features and evaluate a **Lag-Only Baseline Model**.
    """
)

# Load metadata for feature list
with open(ROOT_DIR / "data" / "processed" / "model_metadata.yaml", "r", encoding="utf-8") as f:
    meta_dict = yaml.safe_load(f)

# Sample train/test for ablation demonstration
ablation_results = meta_dict.get("ablation_results", None)
if ablation_results is None:
    train_df = pd.read_parquet(ROOT_DIR / "data" / "synthetic" / "hostel_blocks.parquet")
    from src.features import build_feature_pipeline
    df_feat, f_cols = build_feature_pipeline(train_df)
    from src.models import time_based_train_test_split
    tr_df, ts_df = time_based_train_test_split(df_feat)
    ablation_results = run_context_ablation(tr_df, ts_df, f_cols)

ab1, ab2 = st.columns([4, 8])
with ab1:
    render_metric_card(
        "False Alarm Reduction",
        f"-{ablation_results['false_positive_reduction_pct']:.1f}%",
        "Context vs. Lag-Only",
        status="normal"
    )
    st.metric("Full Model False Positives", ablation_results["full_model"]["false_positives"])
    st.metric("Lag-Only False Positives", ablation_results["lag_only_model"]["false_positives"])

with ab2:
    st.info(f"💡 **Analysis Findings:**\n\n{ablation_results['explanation']}")

st.markdown("---")

# LEAD 1.0 vs Synthetic Domain Gap
st.markdown("### 🏢 Dataset Domain Gap: Commercial LEAD 1.0 vs. Residential Hostels")
lead_meta = meta_dict.get("lead_gap", {})

c_g1, c_g2, c_g3 = st.columns(3)
with c_g1:
    render_metric_card("LEAD 1.0 Precision", f"{lead_meta.get('lead_precision', 0.57):.2f}", "Commercial Office Benchmark")
with c_g2:
    render_metric_card("LEAD 1.0 Recall", f"{lead_meta.get('lead_recall', 1.0):.2f}", "High recall on chiller surges")
with c_g3:
    render_metric_card("LEAD 1.0 F1 Score", f"{lead_meta.get('lead_f1', 0.73):.2f}", "Commercial building domain")

st.markdown(
    f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1rem 1.25rem; margin-top: 1rem; color: #cbd5e1; font-size: 0.9rem;">
        <strong>Honest Domain Gap Assessment:</strong><br>
        {lead_meta.get('domain_observations', 'Commercial buildings follow rigid centralized schedules. Hostels are decentralized and driven by irregular student habits.')}
    </div>
    """,
    unsafe_allow_html=True
)
