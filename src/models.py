"""Anomaly Detection Models for Hostel Energy Consumption.

Implements and evaluates:
- B1: Rolling mean +/- k*sigma
- B2: Seasonal profile (hour x weekday x day_type median) + MAD threshold
- M1: Isolation Forest on contextual features
- M2: Forecast-residual LightGBM regressor predicting expected kWh and scoring severity
- LEAD 1.0 benchmark training & domain gap comparison
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Optional, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

from src.features import build_feature_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_based_train_test_split(
    df: pd.DataFrame,
    split_date: str = "2025-10-01 00:00:00",
    time_col: str = "date_time"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data strictly by time boundary (never random) to prevent future leakage.

    Default: Jan-Sep (9 months, 75%) for training, Oct-Dec (3 months, 25%) for testing.
    """
    ts_split = pd.Timestamp(split_date)
    train_df = df[df[time_col] < ts_split].copy()
    test_df = df[df[time_col] >= ts_split].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# B1: Rolling Mean +/- k*sigma Baseline
# ---------------------------------------------------------------------------
def fit_predict_b1_rolling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_sigma: float = 2.8,
    target_col: str = "energy_kwh",
    group_col: str = "hostel_block"
) -> pd.DataFrame:
    """B1 Detector: Flag points deviating beyond rolling 24h mean +/- k * rolling std."""
    combined = pd.concat([train_df, test_df], ignore_index=True).sort_values([group_col, "date_time"]).reset_index(drop=True)

    rolling_mean = combined.groupby(group_col)[target_col].transform(
        lambda s: s.shift(1).rolling(24, min_periods=3).mean()
    ).bfill()

    rolling_std = combined.groupby(group_col)[target_col].transform(
        lambda s: s.shift(1).rolling(24, min_periods=3).std()
    ).fillna(0.5).bfill()

    upper_bound = rolling_mean + k_sigma * rolling_std
    lower_bound = np.maximum(0.0, rolling_mean - k_sigma * rolling_std)

    actual = combined[target_col].values
    deviation = np.abs(actual - rolling_mean.values)
    flags = ((actual > upper_bound.values) | (actual < lower_bound.values)).astype(int)

    # Severity score (0 to 100)
    z_scores = deviation / (rolling_std.values + 1e-4)
    severity = np.clip((z_scores / 5.0) * 100.0, 0.0, 100.0).round(1)

    combined["b1_pred"] = flags
    combined["b1_severity"] = severity
    combined["b1_expected"] = rolling_mean.round(3)

    return combined[combined["date_time"] >= test_df["date_time"].min()].copy()



# ---------------------------------------------------------------------------
# B2: Seasonal Profile (Hour x Weekday x Day_Type) + MAD Baseline
# ---------------------------------------------------------------------------
def fit_predict_b2_seasonal(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mad_multiplier: float = 3.2,
    target_col: str = "energy_kwh",
    group_col: str = "hostel_block"
) -> pd.DataFrame:
    """B2 Detector: Flag deviations from (hour x weekday x day_type) median + MAD threshold."""
    # Learn seasonal profile strictly on training set
    train_copy = train_df.copy()
    train_copy["hour"] = train_copy["date_time"].dt.hour
    train_copy["weekday"] = train_copy["date_time"].dt.dayofweek

    # Calculate median profile
    profile = (
        train_copy.groupby([group_col, "hour", "weekday", "day_type"])[target_col]
        .median()
        .rename("seasonal_median")
        .reset_index()
    )

    # Calculate MAD (Median Absolute Deviation) per group
    train_merged = train_copy.merge(profile, on=[group_col, "hour", "weekday", "day_type"], how="left")
    train_merged["abs_dev"] = (train_merged[target_col] - train_merged["seasonal_median"]).abs()
    mad_profile = (
        train_merged.groupby([group_col, "hour", "weekday", "day_type"])["abs_dev"]
        .median()
        .rename("seasonal_mad")
        .reset_index()
    )
    profile = profile.merge(mad_profile, on=[group_col, "hour", "weekday", "day_type"], how="left")
    profile["seasonal_mad"] = profile["seasonal_mad"].replace(0.0, 0.5).fillna(0.5)

    # Score test data
    test_copy = test_df.copy()
    test_copy["hour"] = test_copy["date_time"].dt.hour
    test_copy["weekday"] = test_copy["date_time"].dt.dayofweek

    test_scored = test_copy.merge(profile, on=[group_col, "hour", "weekday", "day_type"], how="left")
    test_scored["seasonal_median"] = test_scored["seasonal_median"].fillna(test_scored[target_col].median())
    test_scored["seasonal_mad"] = test_scored["seasonal_mad"].fillna(0.5)

    residuals = (test_scored[target_col] - test_scored["seasonal_median"]).abs()
    threshold = mad_multiplier * test_scored["seasonal_mad"]
    flags = (residuals > threshold).astype(int)

    # Severity score 0 to 100
    mad_ratio = residuals / (test_scored["seasonal_mad"] + 1e-4)
    severity = np.clip((mad_ratio / 6.0) * 100.0, 0.0, 100.0).round(1)

    test_scored["b2_pred"] = flags
    test_scored["b2_severity"] = severity
    test_scored["b2_expected"] = test_scored["seasonal_median"].round(3)
    return test_scored


# ---------------------------------------------------------------------------
# M1: Isolation Forest Contextual Detector
# ---------------------------------------------------------------------------
def fit_predict_m1_iforest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    contamination: float = 0.03,
    seed: int = 42
) -> Tuple[pd.DataFrame, IsolationForest]:
    """M1 Detector: Isolation Forest on multi-dimensional contextual features."""
    X_train = train_df[feature_cols].fillna(0.0)
    X_test = test_df[feature_cols].fillna(0.0)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train)

    raw_preds = model.predict(X_test)  # -1 for anomaly, 1 for normal
    scores = -model.score_samples(X_test)  # Higher score = more anomalous

    # Normalize anomaly score to 0 - 100 severity
    min_s, max_s = np.percentile(scores, 5), np.percentile(scores, 99)
    severity = np.clip((scores - min_s) / (max_s - min_s + 1e-5) * 100.0, 0.0, 100.0).round(1)
    flags = (raw_preds == -1).astype(int)

    test_copy = test_df.copy()
    test_copy["m1_pred"] = flags
    test_copy["m1_severity"] = severity
    # Isolation forest provides anomaly score, use rolling/seasonal as proxy expected
    test_copy["m1_expected"] = test_copy["rolling_mean_24h"].round(3)

    return test_copy, model


# ---------------------------------------------------------------------------
# M2: Forecast-Residual Model (LightGBM)
# ---------------------------------------------------------------------------
def fit_predict_m2_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "energy_kwh",
    residual_quantile: float = 0.97,
    seed: int = 42
) -> Tuple[pd.DataFrame, lgb.LGBMRegressor, Dict]:
    """M2 Detector: Supervised LightGBM regressor predicting expected kWh from context.

    Calculates dynamic residual threshold to flag anomalies and provide
    the foundation for energy waste (kWh) and cost (INR) calculation.
    """
    # Exclude clean-up of extreme injected anomalies during training to prevent poisoning
    clean_train = train_df[train_df["is_anomaly"] == 0].copy() if "is_anomaly" in train_df.columns else train_df

    X_train = clean_train[feature_cols]
    y_train = clean_train[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    train_residuals = np.abs(y_train.values - y_pred_train)
    threshold = float(np.quantile(train_residuals, residual_quantile))

    y_pred_test = model.predict(X_test)
    test_residuals = np.abs(y_test.values - y_pred_test)

    flags = (test_residuals > threshold).astype(int)

    # Normalize severity from 0 to 100 based on multiples of residual threshold
    severity = np.clip((test_residuals / (threshold * 2.5 + 1e-4)) * 100.0, 0.0, 100.0).round(1)

    test_copy = test_df.copy()
    test_copy["m2_pred"] = flags
    test_copy["m2_severity"] = severity
    test_copy["m2_expected"] = np.clip(y_pred_test, 0.0, None).round(3)
    test_copy["m2_residual"] = test_residuals.round(3)

    # Feature importances
    feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    metadata = {
        "threshold": threshold,
        "feature_importances": sorted_importance,
        "train_mae": float(np.mean(train_residuals)),
        "test_mae": float(np.mean(test_residuals))
    }

    return test_copy, model, metadata


# ---------------------------------------------------------------------------
# LEAD 1.0 Benchmark & Domain Gap Analysis
# ---------------------------------------------------------------------------
def evaluate_lead_domain_gap(
    df_lead: pd.DataFrame,
    seed: int = 42
) -> Dict:
    """Train LightGBM on commercial LEAD 1.0 dataset and quantify domain gap metrics.

    Highlights structural differences between commercial office schedules
    and decentralized residential student hostels.
    """
    df_feat = df_lead.copy().sort_values("timestamp").reset_index(drop=True)
    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat["weekday"] = df_feat["timestamp"].dt.dayofweek
    df_feat["is_weekend"] = (df_feat["weekday"] >= 5).astype(int)
    df_feat["sin_hour"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["cos_hour"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)

    # Lag features
    shifted = df_feat["meter_reading"].shift(1)
    df_feat["lag_1"] = shifted
    df_feat["lag_24"] = df_feat["meter_reading"].shift(24)
    df_feat["rolling_mean_24h"] = shifted.rolling(24, min_periods=1).mean()
    df_feat = df_feat.bfill()

    feature_cols = ["hour", "weekday", "is_weekend", "sin_hour", "cos_hour", "lag_1", "lag_24", "rolling_mean_24h"]

    # 75/25 time split
    n_train = int(len(df_feat) * 0.75)
    train_lead = df_feat.iloc[:n_train]
    test_lead = df_feat.iloc[n_train:]

    X_train, y_train = train_lead[feature_cols], train_lead["meter_reading"]
    X_test, y_test = test_lead[feature_cols], test_lead["meter_reading"]

    lead_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=seed, verbose=-1)
    lead_model.fit(X_train, y_train)

    preds_test = lead_model.predict(X_test)
    residuals = np.abs(y_test.values - preds_test)
    thresh = np.quantile(residuals, 0.95)
    lead_flags = (residuals > thresh).astype(int)
    y_true_lead = test_lead["anomaly"].values

    from sklearn.metrics import precision_score, recall_score, f1_score

    p = float(precision_score(y_true_lead, lead_flags, zero_division=0))
    r = float(recall_score(y_true_lead, lead_flags, zero_division=0))
    f1 = float(f1_score(y_true_lead, lead_flags, zero_division=0))

    return {
        "lead_precision": round(p, 3),
        "lead_recall": round(r, 3),
        "lead_f1": round(f1, 3),
        "lead_test_hours": len(test_lead),
        "domain_observations": (
            "Commercial LEAD 1.0 displays strict weekday 08:00-18:00 HVAC schedules with predictable weekend troughs. "
            "Hostel living exhibits decentralized, student-driven night owl hours, exam study surges (00:00-04:00), "
            "and unauthorized heating appliances requiring occupancy and calendar context to avoid high false alarms."
        )
    }


def train_and_save_all_models(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, Dict]:
    """Train all models, score test set, and persist artifacts."""
    config = load_config(config_path)
    synth_file = Path(config["system"]["synthetic_dir"]) / "hostel_blocks.parquet"
    if not synth_file.exists():
        from src.synth import generate_hostel_dataset
        df_blocks, _ = generate_hostel_dataset(config_path)
    else:
        df_blocks = pd.read_parquet(synth_file)

    logger.info("Building feature pipeline for %d rows...", len(df_blocks))
    df_feat, feature_cols = build_feature_pipeline(df_blocks)

    train_df, test_df = time_based_train_test_split(df_feat, split_date="2025-10-01 00:00:00")
    logger.info("Train rows: %d (Jan-Sep), Test rows: %d (Oct-Dec)", len(train_df), len(test_df))

    # B1
    logger.info("Running B1 Rolling Mean Detector...")
    test_scored = fit_predict_b1_rolling(
        train_df, test_df, k_sigma=config["thresholds"]["rolling_k_sigma"]
    )

    # B2
    logger.info("Running B2 Seasonal Profile Detector...")
    b2_res = fit_predict_b2_seasonal(
        train_df, test_df, mad_multiplier=config["thresholds"]["seasonal_mad_multiplier"]
    )
    test_scored["b2_pred"] = b2_res["b2_pred"].values
    test_scored["b2_severity"] = b2_res["b2_severity"].values
    test_scored["b2_expected"] = b2_res["b2_expected"].values

    # M1
    logger.info("Running M1 Isolation Forest Detector...")
    m1_res, m1_model = fit_predict_m1_iforest(
        train_df, test_df, feature_cols,
        contamination=config["thresholds"]["isolation_forest_contamination"],
        seed=config["system"]["seed"]
    )
    test_scored["m1_pred"] = m1_res["m1_pred"].values
    test_scored["m1_severity"] = m1_res["m1_severity"].values
    test_scored["m1_expected"] = m1_res["m1_expected"].values

    # M2
    logger.info("Running M2 LightGBM Forecast-Residual Detector...")
    m2_res, m2_model, m2_meta = fit_predict_m2_lgbm(
        train_df, test_df, feature_cols,
        residual_quantile=config["thresholds"]["lgbm_residual_quantile"],
        seed=config["system"]["seed"]
    )
    test_scored["m2_pred"] = m2_res["m2_pred"].values
    test_scored["m2_severity"] = m2_res["m2_severity"].values
    test_scored["m2_expected"] = m2_res["m2_expected"].values
    test_scored["m2_residual"] = m2_res["m2_residual"].values

    # Also evaluate LEAD domain gap
    from src.data_lead import load_lead_dataset
    df_lead, _ = load_lead_dataset(config_path)
    lead_gap = evaluate_lead_domain_gap(df_lead, seed=config["system"]["seed"])

    # Persist processed evaluation predictions & model artifacts
    proc_dir = Path(config["system"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    test_scored.to_parquet(proc_dir / "test_evaluation_scored.parquet", index=False)
    joblib.dump(m2_model, proc_dir / "m2_lightgbm_model.pkl")
    joblib.dump(m1_model, proc_dir / "m1_iforest_model.pkl")
    with open(proc_dir / "model_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"m2_meta": m2_meta, "lead_gap": lead_gap, "feature_cols": feature_cols}, f)

    logger.info("Successfully trained and saved all models to %s", proc_dir)
    return test_scored, {"m2_meta": m2_meta, "lead_gap": lead_gap}


if __name__ == "__main__":
    test_scored, meta = train_and_save_all_models()
    print("M2 Top 5 Features:", list(meta["m2_meta"]["feature_importances"].keys())[:5])
    print("LEAD Domain Gap Metrics:", meta["lead_gap"])
