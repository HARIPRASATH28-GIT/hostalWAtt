"""Unit and integration tests for the continuous electricity monitoring backend."""

import pytest
import datetime
from pathlib import Path
from backend.database import (
    init_db, insert_reading, insert_live_alert,
    get_latest_readings, get_historical_readings, get_live_alerts,
    acknowledge_alert, resolve_alert, get_monitoring_stats
)
from backend.meter_interface import SimulatedMeterSource, MeterPayload
from backend.detector import RealtimeDetector
from backend.scheduler import MonitoringScheduler


@pytest.fixture
def test_db(tmp_path):
    """Isolated test database."""
    db_file = tmp_path / "test_monitor.db"
    init_db(str(db_file))
    return str(db_file)


def test_database_reading_crud(test_db):
    """Test inserting and querying real-time meter readings."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r_id = insert_reading(
        timestamp=now_str,
        hostel_block="Block_A",
        energy_kwh=16.5,
        expected_kwh=13.0,
        residual_kwh=3.5,
        is_anomaly=1,
        severity_score=75.0,
        excess_kwh=3.5,
        excess_cost_inr=26.25,
        anomaly_type="appliance_spike",
        source="SIMULATED",
        db_path=test_db
    )
    assert r_id > 0

    readings = get_latest_readings(limit=10, db_path=test_db)
    assert len(readings) == 1
    assert readings[0]["hostel_block"] == "Block_A"
    assert readings[0]["source"] == "SIMULATED"
    assert readings[0]["is_anomaly"] == 1


def test_database_alert_deduplication_and_lifecycle(test_db):
    """Test alert insertion, unique alert_key deduplication, and state transitions."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_key = "Block_B_202610291530_appliance_spike"

    # 1. First insertion
    inserted, a_id = insert_live_alert(
        alert_key=alert_key,
        hostel_block="Block_B",
        timestamp=now_str,
        energy_kwh=18.0,
        expected_kwh=14.0,
        excess_kwh=4.0,
        excess_cost_inr=30.0,
        severity_score=80.0,
        anomaly_type="appliance_spike",
        recommended_action="Inspect rooms",
        source="SIMULATED",
        db_path=test_db
    )
    assert inserted is True
    assert a_id > 0

    # 2. Duplicate insertion with exact same key must not create duplicate
    dup_inserted, dup_id = insert_live_alert(
        alert_key=alert_key,
        hostel_block="Block_B",
        timestamp=now_str,
        energy_kwh=18.0,
        expected_kwh=14.0,
        excess_kwh=4.0,
        excess_cost_inr=30.0,
        severity_score=80.0,
        anomaly_type="appliance_spike",
        recommended_action="Inspect rooms",
        source="SIMULATED",
        db_path=test_db
    )
    assert dup_inserted is False
    assert dup_id == a_id

    # Verify only 1 alert in DB
    alerts = get_live_alerts(db_path=test_db)
    assert len(alerts) == 1
    assert alerts[0]["status"] == "NEW"

    # 3. Acknowledge alert
    ack_res = acknowledge_alert(a_id, user="Warden_Tester", db_path=test_db)
    assert ack_res is True

    alerts_ack = get_live_alerts(db_path=test_db)
    assert alerts_ack[0]["status"] == "ACKNOWLEDGED"
    assert alerts_ack[0]["acknowledged_by"] == "Warden_Tester"

    # 4. Resolve alert
    res_res = resolve_alert(a_id, user="Warden_Tester", db_path=test_db)
    assert res_res is True

    alerts_res = get_live_alerts(db_path=test_db)
    assert alerts_res[0]["status"] == "RESOLVED"


def test_simulated_meter_source_labels_and_spikes():
    """Verify that simulated readings are clearly labeled and on-demand spike injection works."""
    source = SimulatedMeterSource()
    readings = source.read_all_blocks()

    assert len(readings) == 5
    for r in readings:
        assert r.source == "SIMULATED"
        assert r.is_simulated is True
        assert "[SIMULATED METER FEED]" in r.display_label
        assert r.energy_kwh > 0

    # Inject test spike on Block_C
    base_reading = source.read_block("Block_C")
    source.inject_spike(block_id="Block_C", extra_kwh=4.0, anomaly_type="appliance_spike")
    spiked_reading = source.read_block("Block_C")

    assert spiked_reading.injected_anomaly_type == "appliance_spike"
    assert spiked_reading.injected_extra_kwh == 4.0
    assert spiked_reading.energy_kwh >= base_reading.energy_kwh + 3.0



def test_detector_evaluation_normal_vs_spike():
    """Verify detector identifies normal vs spiked readings and calculates financial impact."""
    detector = RealtimeDetector()

    # 1. Normal reading
    normal_payload = MeterPayload(
        block_id="Block_A",
        timestamp=datetime.datetime(2026, 10, 29, 14, 0),
        energy_kwh=13.5,
        occupancy=140,
        room_count=72,
        day_type="regular",
        outside_temp_c=28.0
    )
    res_norm = detector.evaluate_reading(normal_payload)
    assert res_norm.expected_kwh > 0
    assert res_norm.excess_kwh >= 0

    # 2. Heavy Spike reading
    spiked_payload = MeterPayload(
        block_id="Block_A",
        timestamp=datetime.datetime(2026, 10, 29, 14, 0),
        energy_kwh=22.0,
        occupancy=140,
        room_count=72,
        day_type="regular",
        outside_temp_c=28.0,
        injected_anomaly_type="appliance_spike",
        injected_extra_kwh=5.0
    )
    res_spike = detector.evaluate_reading(spiked_payload)
    assert res_spike.is_anomaly == 1
    assert res_spike.anomaly_type == "appliance_spike"
    assert res_spike.excess_kwh > 2.0
    assert res_spike.excess_cost_inr > 0
    assert res_spike.severity_score >= 50.0
    assert "Inspection" in res_spike.recommended_action


def test_scheduler_trigger_tick_and_status():
    """Verify scheduler triggers execution without errors and updates status metrics."""
    scheduler = MonitoringScheduler()
    init_status = scheduler.get_status()
    assert "is_running" in init_status
    assert "interval_seconds" in init_status

    # Trigger single synchronous tick
    tick_out = scheduler.trigger_tick()
    assert tick_out["status"] == "success"
    assert tick_out["blocks_processed"] == 5

    # Check updated stats
    new_status = scheduler.get_status()
    assert new_status["total_ticks"] >= 1
