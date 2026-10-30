"""Comprehensive evaluation module for anomaly detectors.

Calculates:
- Precision, Recall, F1, PR-AUC, Confusion Matrix
- Detection Delay (hours from anomaly onset to first flag)
- Recall per anomaly type
- False alerts per block per week
- Feature ablation (context vs. lag-only) demonstrating false positive reduction
- Model benchmark summary table (B1, B2, M1, M2)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix
)
import lightgbm as lgb


def compute_core_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict:
    """Compute precision, recall, F1, confusion matrix, and PR-AUC."""
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    pr_auc_val = 0.0
    if y_scores is not None and len(np.unique(y_true)) > 1:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_scores)
        pr_auc_val = float(auc(rec_arr, prec_arr))

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc_val, 4),
        "confusion_matrix": cm,
        "true_negatives": cm[0][0],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_positives": cm[1][1]
    }


def compute_detection_delay_and_type_recall(
    df_eval: pd.DataFrame,
    pred_col: str = "m2_pred"
) -> Tuple[Dict[str, float], float, List[float]]:
    """Compute recall per anomaly type and distribution of detection delay hours."""
    df = df_eval.copy().sort_values(["hostel_block", "date_time"]).reset_index(drop=True)

    # 1. Recall per anomaly type
    types = [t for t in df["anomaly_type"].unique() if t != "none"]
    type_recalls = {}
    for t in types:
        sub = df[df["anomaly_type"] == t]
        if len(sub) > 0:
            rec = float(recall_score(sub["is_anomaly"], sub[pred_col], zero_division=0))
            type_recalls[t] = round(rec, 4)

    # 2. Detection delay: identify continuous anomaly events
    delays = []
    blocks = df["hostel_block"].unique()

    for b in blocks:
        b_df = df[df["hostel_block"] == b].reset_index(drop=True)
        is_anom = (b_df["is_anomaly"] == 1).astype(int)
        runs = (is_anom != is_anom.shift()).cumsum()

        for run_id, run_group in b_df[b_df["is_anomaly"] == 1].groupby(runs):
            flags = run_group[pred_col].values
            detected_indices = np.where(flags == 1)[0]
            if len(detected_indices) > 0:
                # Delay = index of first positive detection (0 hours = detected in first hour)
                delays.append(float(detected_indices[0]))
            else:
                # Missed event: penalty equals event duration
                delays.append(float(len(flags)))

    avg_delay = round(float(np.mean(delays)), 2) if delays else 0.0
    return type_recalls, avg_delay, delays


def compute_false_alerts_per_block_week(
    df_eval: pd.DataFrame,
    pred_col: str = "m2_pred"
) -> float:
    """Calculate average number of false alarm hours per block per week."""
    total_hours = len(df_eval)
    num_blocks = df_eval["hostel_block"].nunique()
    total_weeks = (total_hours / (num_blocks * 168.0)) if num_blocks > 0 else 1.0

    # False positives
    fps = ((df_eval["is_anomaly"] == 0) & (df_eval[pred_col] == 1)).sum()
    fp_per_block_week = (fps / (num_blocks * max(total_weeks, 0.1)))
    return round(float(fp_per_block_week), 2)


def run_context_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_features: List[str],
    seed: int = 42
) -> Dict:
    """Compare Full Model (Context + Lags) vs. Lag-Only Model.

    Demonstrates that context features (occupancy, calendar, weather)
    significantly reduce false positive alerts during exams, holidays, and weather shifts.
    """
    # Lag only features
    lag_features = ["lag_1", "lag_24", "lag_168", "rolling_mean_6h", "rolling_mean_24h"]

    clean_train = train_df[train_df["is_anomaly"] == 0] if "is_anomaly" in train_df.columns else train_df

    # 1. Full Model
    full_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=seed, verbose=-1)
    full_model.fit(clean_train[all_features], clean_train["energy_kwh"])
    full_res = np.abs(test_df["energy_kwh"].values - full_model.predict(test_df[all_features]))
    full_thresh = np.quantile(np.abs(clean_train["energy_kwh"].values - full_model.predict(clean_train[all_features])), 0.97)
    full_preds = (full_res > full_thresh).astype(int)

    # 2. Lag Only Model (Ablated)
    lag_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=seed, verbose=-1)
    lag_model.fit(clean_train[lag_features], clean_train["energy_kwh"])
    lag_res = np.abs(test_df["energy_kwh"].values - lag_model.predict(test_df[lag_features]))
    lag_thresh = np.quantile(np.abs(clean_train["energy_kwh"].values - lag_model.predict(clean_train[lag_features])), 0.97)
    lag_preds = (lag_res > lag_thresh).astype(int)

    y_true = test_df["is_anomaly"].values

    m_full = compute_core_metrics(y_true, full_preds, full_res)
    m_lag = compute_core_metrics(y_true, lag_preds, lag_res)

    fp_full = m_full["false_positives"]
    fp_lag = m_lag["false_positives"]
    fp_reduction_pct = round(((fp_lag - fp_full) / max(fp_lag, 1)) * 100.0, 1)

    return {
        "full_model": m_full,
        "lag_only_model": m_lag,
        "false_positive_reduction_pct": fp_reduction_pct,
        "explanation": (
            f"Context features (occupancy, academic day_type, ambient temperature) reduced false alarms by "
            f"{fp_reduction_pct}% (from {fp_lag} to {fp_full} false alarms in test set). Without context, "
            f"legitimate study hours during exams and vacant holiday hours are misclassified as anomalies."
        )
    }


def compare_all_models(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Generate comparative benchmark table across B1, B2, M1, and M2."""
    models = [
        ("B1: Rolling Mean ± k*σ", "b1_pred", "b1_severity"),
        ("B2: Seasonal Median + MAD", "b2_pred", "b2_severity"),
        ("M1: Isolation Forest", "m1_pred", "m1_severity"),
        ("M2: LightGBM Residual", "m2_pred", "m2_severity"),
    ]

    records = []
    y_true = df_eval["is_anomaly"].values

    for name, pred_col, score_col in models:
        y_pred = df_eval[pred_col].values
        y_score = df_eval[score_col].values if score_col in df_eval.columns else y_pred
        metrics = compute_core_metrics(y_true, y_pred, y_score)
        _, avg_delay, _ = compute_detection_delay_and_type_recall(df_eval, pred_col)
        fp_rate = compute_false_alerts_per_block_week(df_eval, pred_col)

        records.append({
            "Model": name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
            "PR-AUC": metrics["pr_auc"],
            "Avg Detection Delay (h)": avg_delay,
            "False Alerts / Block-Wk": fp_rate
        })

    return pd.DataFrame(records)
