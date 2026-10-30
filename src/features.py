"""Feature engineering module for hostel energy anomaly detection.

Builds temporal, calendar, contextual, lag, and rolling statistics features.
Guarantees zero data leakage by shifting historical lags and rolling windows.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


def add_cyclical_features(df: pd.DataFrame, time_col: str = "date_time") -> pd.DataFrame:
    """Add sine and cosine cyclical encodings for hour of day and day of week."""
    df = df.copy()
    hour = df[time_col].dt.hour
    dayofweek = df[time_col].dt.dayofweek

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["hour"] = hour
    df["weekday"] = dayofweek
    df["is_weekend"] = (dayofweek >= 5).astype(int)

    return df


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode occupancy, cooling degree impact, and academic day types."""
    df = df.copy()

    # Day type one-hot encoding
    df["day_type_regular"] = (df["day_type"] == "regular").astype(int)
    df["day_type_exam"] = (df["day_type"] == "exam").astype(int)
    df["day_type_holiday"] = (df["day_type"] == "holiday").astype(int)

    # Cooling degree impact (cooling load increases above 26°C)
    df["cooling_degree_c"] = np.maximum(df["outside_temp_c"] - 26.0, 0.0)

    # Occupancy per room ratio
    if "occupancy" in df.columns and "room_count" in df.columns:
        df["occ_per_room"] = df["occupancy"] / np.maximum(df["room_count"], 1.0)
    else:
        df["occ_per_room"] = 1.0

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "energy_kwh",
    group_col: str = "hostel_block"
) -> pd.DataFrame:
    """Compute lag and rolling window features strictly on past observations (shift 1).

    Features added:
    - Lag t-1, t-24, t-168 (1 hour, 1 day, 1 week ago)
    - Rolling 6h mean and std
    - Rolling 24h mean and std
    - Rolling 168h mean
    """
    df = df.copy().sort_values([group_col, "date_time"]).reset_index(drop=True)

    # Shifted values to prevent any data leakage
    shifted = df.groupby(group_col)[target_col].shift(1)
    df["lag_1"] = shifted
    df["lag_24"] = df.groupby(group_col)[target_col].shift(24)
    df["lag_168"] = df.groupby(group_col)[target_col].shift(168)

    # Rolling statistics strictly on shifted history
    df["rolling_mean_6h"] = df.groupby(group_col)["lag_1"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["rolling_std_6h"] = df.groupby(group_col)["lag_1"].transform(lambda s: s.rolling(6, min_periods=1).std()).fillna(0.0)
    df["rolling_mean_24h"] = df.groupby(group_col)["lag_1"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["rolling_std_24h"] = df.groupby(group_col)["lag_1"].transform(lambda s: s.rolling(24, min_periods=1).std()).fillna(0.0)
    df["rolling_mean_168h"] = df.groupby(group_col)["lag_1"].transform(lambda s: s.rolling(168, min_periods=1).mean())

    # Backward fill the initial lag window per block using available observations
    feature_cols = [
        "lag_1", "lag_24", "lag_168",
        "rolling_mean_6h", "rolling_std_6h",
        "rolling_mean_24h", "rolling_std_24h",
        "rolling_mean_168h"
    ]
    for col in feature_cols:
        df[col] = df.groupby(group_col)[col].bfill()

    return df



def build_feature_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Execute complete feature engineering pipeline.

    Returns:
        Tuple[pd.DataFrame, List[str]]: (processed_dataframe, feature_column_names)
    """
    df_feat = add_cyclical_features(df, time_col="date_time")
    df_feat = add_contextual_features(df_feat)
    df_feat = add_lag_and_rolling_features(df_feat)

    feature_cols = [
        "hour", "weekday", "is_weekend",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "outside_temp_c", "cooling_degree_c", "occupancy", "room_count", "occ_per_room",
        "day_type_regular", "day_type_exam", "day_type_holiday",
        "lag_1", "lag_24", "lag_168",
        "rolling_mean_6h", "rolling_std_6h",
        "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h"
    ]

    return df_feat, feature_cols
