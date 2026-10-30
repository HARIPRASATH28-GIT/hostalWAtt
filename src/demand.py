"""Maximum Demand (MD) estimation, coincident peak analysis, and capacity alerts.

NOTE: All hourly energy readings (kWh/h) reflect average hourly active power (kW).
Actual utility billing demands are recorded on 15-minute or 30-minute sliding windows,
meaning hourly values represent an ENGINEERING LOWER BOUND of instantaneous peak demand.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_system_maximum_demand(
    df: pd.DataFrame,
    power_factor: float = 0.90,
    warning_threshold_pct: float = 90.0
) -> Dict:
    """Calculate system-level coincident maximum demand and individual block contributions.

    Args:
        df: Hourly block readings (date_time, hostel_block, energy_kwh).
        power_factor: Assumed displacement power factor (default 0.90 lagging).
        warning_threshold_pct: Alert threshold as % of monthly recorded MD (default 90.0).

    Returns:
        Dict containing coincident peak timestamp, system peak kW/kVA, block contributions,
        and hours breaching 90% of monthly recorded MD.
    """
    df_calc = df.copy()
    df_calc["date_time"] = pd.to_datetime(df_calc["date_time"])

    # Hourly system total power (kW) = sum across all blocks
    hourly_sys = df_calc.pivot_table(index="date_time", columns="hostel_block", values="energy_kwh", aggfunc="sum").fillna(0.0)
    hourly_sys["system_total_kw"] = hourly_sys.sum(axis=1)
    hourly_sys["system_total_kva"] = hourly_sys["system_total_kw"] / power_factor

    # Identify Monthly Recorded Maximum Demand
    hourly_sys["year_month"] = hourly_sys.index.to_period("M").astype(str)
    monthly_md = hourly_sys.groupby("year_month")["system_total_kw"].transform("max")
    hourly_sys["monthly_md_kw"] = monthly_md
    hourly_sys["warning_threshold_kw"] = monthly_md * (warning_threshold_pct / 100.0)
    hourly_sys["is_near_peak_alert"] = hourly_sys["system_total_kw"] >= hourly_sys["warning_threshold_kw"]

    # Overall system peak hour
    peak_idx = hourly_sys["system_total_kw"].idxmax()
    peak_record = hourly_sys.loc[peak_idx]

    # Block contributions at system coincident peak hour
    block_cols = [c for c in hourly_sys.columns if c not in ["system_total_kw", "system_total_kva", "year_month", "monthly_md_kw", "warning_threshold_kw", "is_near_peak_alert"]]
    coincident_contributions = []
    total_at_peak = peak_record["system_total_kw"]

    for b in block_cols:
        b_kw = float(peak_record[b])
        share_pct = (b_kw / total_at_peak * 100.0) if total_at_peak > 0 else 0.0
        coincident_contributions.append({
            "hostel_block": b,
            "peak_kw": round(b_kw, 2),
            "peak_kva": round(b_kw / power_factor, 2),
            "share_pct": round(share_pct, 1)
        })

    # Individual non-coincident block peaks
    non_coincident_peaks = []
    for b in block_cols:
        b_max_kw = float(hourly_sys[b].max())
        b_max_time = hourly_sys[b].idxmax()
        non_coincident_peaks.append({
            "hostel_block": b,
            "max_kw": round(b_max_kw, 2),
            "max_kva": round(b_max_kw / power_factor, 2),
            "peak_time": b_max_time
        })

    # Monthly summary table
    monthly_summary = (
        hourly_sys.groupby("year_month")
        .agg(
            peak_kw=("system_total_kw", "max"),
            avg_kw=("system_total_kw", "mean"),
            alert_hours=("is_near_peak_alert", "sum")
        )
        .round(2)
        .reset_index()
    )
    monthly_summary["peak_kva"] = (monthly_summary["peak_kw"] / power_factor).round(2)

    return {
        "coincident_peak_time": peak_idx,
        "system_peak_kw": round(float(total_at_peak), 2),
        "system_peak_kva": round(float(total_at_peak / power_factor), 2),
        "power_factor": power_factor,
        "coincident_contributions": pd.DataFrame(coincident_contributions),
        "non_coincident_peaks": pd.DataFrame(non_coincident_peaks),
        "monthly_summary": monthly_summary,
        "hourly_system_df": hourly_sys.reset_index(),
        "warning_threshold_pct": warning_threshold_pct,
        "disclaimer": "Hourly integration provides a lower-bound estimate; actual utility 15-min MD is higher."
    }
