"""LEAD 1.0 Energy Anomaly Dataset Loader and Preprocessor.

Loads hourly meter data from local files or falls back gracefully to a synthetic
LEAD-like benchmark dataset if the raw files are not downloaded yet.
Handles missing timestamps, duplicates, and anomaly labels.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load system configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_lead_fallback(
    output_path: Path,
    num_days: int = 180,
    seed: int = 42
) -> pd.DataFrame:
    """Generate a synthetic LEAD-like commercial building energy dataset.

    Used when the raw LEAD 1.0 dataset is not downloaded locally.
    Simulates commercial building profiles (HVAC schedule, weekday/weekend)
    with labeled anomalies.
    """
    logger.info("Generating LEAD-like benchmark dataset fallback at: %s", output_path)
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp("2024-01-01 00:00:00")
    hours = num_days * 24
    dates = pd.date_range(start_date, periods=hours, freq="h")

    # Commercial building parameters
    base_load = 45.0  # kW baseline
    working_hours_boost = 35.0  # 08:00 to 18:00 HVAC & lighting
    weekend_factor = 0.35

    readings = []
    anomalies = []

    for dt in dates:
        is_weekend = dt.weekday() >= 5
        hour = dt.hour

        # Commercial diurnal profile
        if 8 <= hour <= 18:
            profile = working_hours_boost * np.sin(np.pi * (hour - 8) / 10)
        else:
            profile = 0.0

        if is_weekend:
            profile *= weekend_factor

        noise = np.random.normal(0, 1.8)
        val = base_load + profile + noise

        # Anomaly probability ~2.5%
        is_anom = 0
        if np.random.rand() < 0.025:
            is_anom = 1
            anom_type = np.random.choice(["spike", "night_overuse", "chiller_surge"])
            if anom_type == "spike":
                val += np.random.uniform(20.0, 45.0)
            elif anom_type == "night_overuse":
                val += np.random.uniform(15.0, 25.0)
            else:
                val += np.random.uniform(25.0, 50.0)

        readings.append(max(0.0, round(val, 3)))
        anomalies.append(is_anom)

    df = pd.DataFrame({
        "building_id": "COMM_LEAD_01",
        "timestamp": dates,
        "meter_reading": readings,
        "anomaly": anomalies
    })

    # Introduce a few realistic missing timestamps and duplicates to test cleaning
    mask_drop = np.random.choice(len(df), size=max(1, int(len(df) * 0.002)), replace=False)
    df_clean = df.drop(index=mask_drop).reset_index(drop=True)

    df_clean.to_csv(output_path, index=False)
    logger.info("Saved %d rows of LEAD benchmark data to %s", len(df_clean), output_path)
    return df_clean


def load_lead_dataset(
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, bool]:
    """Load and preprocess the LEAD dataset.

    Returns:
        Tuple[pd.DataFrame, bool]: (cleaned_df, is_fallback)
        is_fallback is True if the local download was absent and fallback was used.
    """
    config = load_config(config_path)
    raw_dir = Path(config["system"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Search for user-provided LEAD files
    csv_candidates = list(raw_dir.glob("*.csv"))
    # Exclude generated fallback if actual datasets are present
    real_candidates = [p for p in csv_candidates if "fallback" not in p.name.lower()]

    if real_candidates:
        target_file = real_candidates[0]
        logger.info("Found LEAD raw dataset file: %s", target_file)
        df_raw = pd.read_csv(target_file)
        is_fallback = False
    else:
        print("\n" + "=" * 70)
        print("NOTICE: LEAD 1.0 dataset not detected in data/raw/.")
        print("To download the authentic LEAD 1.0 dataset, please follow instructions at:")
        print("  --> https://github.com/samy101/lead-dataset")
        print("Place downloaded CSVs inside the 'data/raw/' directory.")
        print("Falling back to pre-packaged LEAD-like commercial benchmark data for offline demo.")
        print("=" * 70 + "\n")
        fallback_file = raw_dir / "lead_fallback_sample.csv"
        if not fallback_file.exists():
            df_raw = generate_lead_fallback(fallback_file, seed=config["system"]["seed"])
        else:
            df_raw = pd.read_csv(fallback_file)
        is_fallback = True

    # Standardize column naming
    col_map = {}
    for c in df_raw.columns:
        c_lower = c.strip().lower()
        if "time" in c_lower or "date" in c_lower:
            col_map[c] = "timestamp"
        elif "meter" in c_lower or "kwh" in c_lower or "energy" in c_lower or "reading" in c_lower:
            col_map[c] = "meter_reading"
        elif "anom" in c_lower or "label" in c_lower:
            col_map[c] = "anomaly"
        elif "build" in c_lower or "block" in c_lower or "id" in c_lower:
            col_map[c] = "building_id"

    df = df_raw.rename(columns=col_map)
    if "building_id" not in df.columns:
        df["building_id"] = "LEAD_BUILDING_1"
    if "anomaly" not in df.columns:
        df["anomaly"] = 0

    # Ensure timestamp is parsed and clean
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Remove duplicates
    df = df.drop_duplicates(subset=["building_id", "timestamp"], keep="first")

    # Reindex to regular hourly intervals to handle gaps
    buildings = df["building_id"].unique()
    dfs_aligned = []

    for b_id in buildings:
        b_df = df[df["building_id"] == b_id].set_index("timestamp")
        full_idx = pd.date_range(b_df.index.min(), b_df.index.max(), freq="h")
        b_reindexed = b_df.reindex(full_idx)
        b_reindexed["building_id"] = b_id

        # Forward fill up to 3 hours for missing meter readings, then fill with rolling mean
        b_reindexed["meter_reading"] = (
            b_reindexed["meter_reading"]
            .ffill(limit=3)
            .bfill(limit=3)
            .fillna(b_reindexed["meter_reading"].median() if not b_reindexed["meter_reading"].empty else 20.0)
        )
        b_reindexed["anomaly"] = b_reindexed["anomaly"].fillna(0).astype(int)
        b_reindexed = b_reindexed.reset_index().rename(columns={"index": "timestamp"})
        dfs_aligned.append(b_reindexed)

    cleaned_df = pd.concat(dfs_aligned, ignore_index=True)

    # Save cleaned version to data/processed
    processed_dir = Path(config["system"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(processed_dir / "lead_cleaned.parquet", index=False)

    return cleaned_df, is_fallback


if __name__ == "__main__":
    df_lead, fallback_status = load_lead_dataset()
    print(f"Loaded LEAD dataset (Fallback={fallback_status}): {len(df_lead)} rows.")
    print(df_lead.head())
