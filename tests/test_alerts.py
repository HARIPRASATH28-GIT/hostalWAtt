"""Unit tests for alert deduplication, cooldown, escalation, and SQLite persistence."""

import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from src.alerts import (
    init_alert_db, process_and_persist_alerts, fetch_alerts,
    acknowledge_alert, resolve_alert
)


@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database path for isolated testing."""
    db_file = tmp_path / "test_alerts.db"
    return str(db_file)


def test_alert_processing_and_escalation(temp_db):
    """Test continuous anomaly grouping, escalation level assignment, and deduplication."""
    # Create synthetic series with a 1-hour blip and an 8-hour persistent surge
    dates = pd.date_range("2025-10-01 00:00:00", periods=20, freq="h")
    wasted = [0.0] * 20
    # 1-hour blip at hour 2
    wasted[2] = 1.5
    # 8-hour persistent surge from hour 10 to 17
    for i in range(10, 18):
        wasted[i] = 2.0

    df_test = pd.DataFrame({
        "hostel_block": "Block_A",
        "date_time": dates,
        "wasted_kwh": wasted,
        "m2_severity": [60.0] * 20,
        "anomaly_type": ["appliance_spike"] * 20
    })

    created = process_and_persist_alerts(df_test, db_path=temp_db, escalation_hours=6)
    assert created == 2, f"Expected 2 alerts generated, got {created}"

    # Fetch alerts
    alerts_df = fetch_alerts(db_path=temp_db)
    assert len(alerts_df) == 2

    # Check that 8-hour event is escalated
    persistent_alert = alerts_df[alerts_df["duration_hours"] == 8]
    assert len(persistent_alert) == 1
    assert "ESCALATED" in persistent_alert["escalation_level"].iloc[0]

    # Deduplication test: re-processing the same data should create 0 new alerts
    created_again = process_and_persist_alerts(df_test, db_path=temp_db)
    assert created_again == 0, "Duplicate alerts created despite existing unique keys"


def test_alert_acknowledge_and_resolve(temp_db):
    """Test state transitions from NEW -> ACKNOWLEDGED -> RESOLVED."""
    dates = pd.date_range("2025-10-01 00:00:00", periods=5, freq="h")
    df_test = pd.DataFrame({
        "hostel_block": "Block_B",
        "date_time": dates,
        "wasted_kwh": [1.5] * 5,
        "m2_severity": [70.0] * 5,
        "anomaly_type": ["left_on"] * 5
    })

    process_and_persist_alerts(df_test, db_path=temp_db)
    alerts = fetch_alerts(db_path=temp_db)
    alert_id = int(alerts["id"].iloc[0])

    # Acknowledge
    ack_res = acknowledge_alert(alert_id, user="Warden_John", db_path=temp_db)
    assert ack_res is True

    alerts_ack = fetch_alerts(db_path=temp_db)
    assert alerts_ack.loc[alerts_ack["id"] == alert_id, "status"].iloc[0] == "ACKNOWLEDGED"
    assert alerts_ack.loc[alerts_ack["id"] == alert_id, "acknowledged_by"].iloc[0] == "Warden_John"

    # Resolve
    res_res = resolve_alert(alert_id, notes="Switched off corridor lamps", db_path=temp_db)
    assert res_res is True

    alerts_res = fetch_alerts(db_path=temp_db)
    assert alerts_res.loc[alerts_res["id"] == alert_id, "status"].iloc[0] == "RESOLVED"
    assert alerts_res.loc[alerts_res["id"] == alert_id, "resolution_notes"].iloc[0] == "Switched off corridor lamps"
