"""Alert generation, deduplication, cooldown, escalation, and SQLite persistence.

Connects to the unified HostelWatt database (data/hostelwatt.db) while preserving
complete backward compatibility for previous batch evaluation pipelines and tests.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime
import sqlite3
import pandas as pd
import yaml

from backend.database import (
    init_db, insert_alert_record, fetch_alerts_dataframe,
    acknowledge_alert as ack_db, resolve_alert as res_db,
    DB_PATH_DEFAULT
)


def init_alert_db(db_path: str = DB_PATH_DEFAULT):
    """Initialize SQLite database table for alert logs."""
    init_db(db_path)


def process_and_persist_alerts(
    df_waste: pd.DataFrame,
    db_path: str = DB_PATH_DEFAULT,
    cooldown_hours: int = 4,
    escalation_hours: int = 6,
    tariff_inr: float = 7.50
) -> int:
    """Group continuous anomaly runs per block into discrete alerts, apply dedupe, cooldown, escalation, and save to unified database."""
    init_db(db_path)
    df = df_waste.copy().sort_values(["hostel_block", "date_time"]).reset_index(drop=True)

    alerts_created = 0
    blocks = df["hostel_block"].unique()

    for b in blocks:
        b_df = df[df["hostel_block"] == b].reset_index(drop=True)
        is_anom = (b_df["wasted_kwh"] > 0).astype(int)
        runs = (is_anom != is_anom.shift()).cumsum()

        for run_id, run_group in b_df[b_df["wasted_kwh"] > 0].groupby(runs):
            duration = len(run_group)
            start_dt = run_group["date_time"].min()
            end_dt = run_group["date_time"].max()
            wasted_kwh = float(run_group["wasted_kwh"].sum())
            cost_inr = round(wasted_kwh * tariff_inr, 2)
            severity = float(run_group["m2_severity"].max()) if "m2_severity" in run_group.columns else 50.0

            if "anomaly_type" in run_group.columns:
                types = run_group["anomaly_type"].value_counts()
                top_type = types.index[0] if types.index[0] != "none" else (types.index[1] if len(types) > 1 else "appliance_spike")
            else:
                top_type = "appliance_spike"

            if duration >= escalation_hours:
                escalation_level = "ESCALATED (Warden/Maintenance)"
                suggested_action = f"Immediate physical inspection on {b}: Persistent continuous surge of {duration}h. Check circuit load & notify facility supervisor."
            elif duration >= 3:
                escalation_level = "ACTIVE (Investigate)"
                suggested_action = f"Floor check on {b}: Abnormal surge active for {duration}h. Inspect common areas and room corridors."
            else:
                escalation_level = "LOG (Monitor)"
                suggested_action = f"Automated log: Short blip ({duration}h) on {b}. System under auto-observation."

            alert_key = f"{b}_{start_dt.strftime('%Y%m%d%H')}_{top_type}"
            actual_kw = float(run_group["energy_kwh"].max()) if "energy_kwh" in run_group.columns else wasted_kwh
            expected_kw = float(run_group["m2_expected"].mean()) if "m2_expected" in run_group.columns else max(0.0, actual_kw - wasted_kwh)

            inserted, _ = insert_alert_record(
                alert_key=alert_key,
                hostel_block=b,
                timestamp=str(start_dt),
                start_time=str(start_dt),
                end_time=str(end_dt),
                duration_hours=float(duration),
                energy_kwh=actual_kw,
                expected_kwh=expected_kw,
                excess_kwh=wasted_kwh,
                excess_cost_inr=cost_inr,
                severity_score=severity,
                anomaly_type=top_type,
                recommended_action=suggested_action,
                escalation_level=escalation_level,
                source="HISTORICAL_BATCH",
                db_path=db_path
            )
            if inserted:
                alerts_created += 1

    return alerts_created


def fetch_alerts(
    db_path: str = DB_PATH_DEFAULT,
    status_filter: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve all alert records from unified SQLite with optional status filtering."""
    init_db(db_path)
    return fetch_alerts_dataframe(status_filter=status_filter, db_path=db_path)


def acknowledge_alert(alert_id: int, user: str = "Warden", db_path: str = DB_PATH_DEFAULT) -> bool:
    """Mark an alert as ACKNOWLEDGED."""
    return ack_db(alert_id=alert_id, user=user, db_path=db_path)


def resolve_alert(alert_id: int, notes: str = "Resolved by facility staff", db_path: str = DB_PATH_DEFAULT) -> bool:
    """Mark an alert as RESOLVED."""
    return res_db(alert_id=alert_id, user="Warden", notes=notes, db_path=db_path)
