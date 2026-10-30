"""Unified persistent database manager for HostelWatt.

Consolidates all historical batch alerts, real-time continuous meter readings,
live spike alerts, and system state into a single SQLite database (data/hostelwatt.db).
Eliminates duplicate database files and provides thread-safe access with WAL mode.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import datetime
import sqlite3
import pandas as pd
import yaml

DB_PATH_DEFAULT = "data/hostelwatt.db"


def get_db_path(config_path: str = "config.yaml") -> str:
    """Get configured unified database path."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("monitoring", {}).get("database_path", DB_PATH_DEFAULT)
    except Exception:
        return DB_PATH_DEFAULT


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a configured SQLite connection with row_factory and WAL mode."""
    path = Path(db_path or DB_PATH_DEFAULT)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Optional[str] = None):
    """Create unified database tables if they do not exist."""
    conn = get_connection(db_path)
    with conn:
        # 1. Telemetry readings table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meter_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                hostel_block TEXT NOT NULL,
                energy_kwh REAL NOT NULL,
                expected_kwh REAL NOT NULL,
                residual_kwh REAL NOT NULL,
                is_anomaly INTEGER NOT NULL,
                severity_score REAL NOT NULL,
                excess_kwh REAL NOT NULL,
                excess_cost_inr REAL NOT NULL,
                anomaly_type TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'SIMULATED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_readings_time ON meter_readings (timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_readings_block ON meter_readings (hostel_block);")

        # 2. Unified alerts table (covers live stream spikes and historical incident batches)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_key TEXT UNIQUE,
                hostel_block TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                duration_hours REAL DEFAULT 1.0,
                energy_kwh REAL NOT NULL,
                expected_kwh REAL NOT NULL,
                excess_kwh REAL NOT NULL,
                excess_cost_inr REAL NOT NULL,
                severity_score REAL NOT NULL,
                anomaly_type TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                escalation_level TEXT NOT NULL DEFAULT 'ACTIVE (Investigate)',
                status TEXT NOT NULL DEFAULT 'NEW',
                acknowledged_at TEXT,
                acknowledged_by TEXT,
                resolved_at TEXT,
                resolved_by TEXT,
                resolution_notes TEXT,
                source TEXT NOT NULL DEFAULT 'LIVE_STREAM',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alert_records (status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_block ON alert_records (hostel_block);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_source ON alert_records (source);")

        # 3. System settings table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    conn.close()


def insert_reading(
    timestamp: str,
    hostel_block: str,
    energy_kwh: float,
    expected_kwh: float,
    residual_kwh: float,
    is_anomaly: int,
    severity_score: float,
    excess_kwh: float,
    excess_cost_inr: float,
    anomaly_type: str = "none",
    source: str = "SIMULATED",
    db_path: Optional[str] = None
) -> int:
    """Insert a new meter reading record."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO meter_readings (
                timestamp, hostel_block, energy_kwh, expected_kwh, residual_kwh,
                is_anomaly, severity_score, excess_kwh, excess_cost_inr,
                anomaly_type, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp, hostel_block, round(energy_kwh, 3), round(expected_kwh, 3),
                round(residual_kwh, 3), int(is_anomaly), round(severity_score, 1),
                round(excess_kwh, 3), round(excess_cost_inr, 2), anomaly_type, source
            )
        )
        new_id = cursor.lastrowid
    conn.close()
    return new_id


def insert_alert_record(
    alert_key: str,
    hostel_block: str,
    timestamp: str,
    energy_kwh: float,
    expected_kwh: float,
    excess_kwh: float,
    excess_cost_inr: float,
    severity_score: float,
    anomaly_type: str,
    recommended_action: str,
    escalation_level: str = "ACTIVE (Investigate)",
    duration_hours: float = 1.0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    source: str = "LIVE_STREAM",
    db_path: Optional[str] = None
) -> Tuple[bool, int]:
    """Insert an alert record with unique alert_key deduplication."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO alert_records (
                    alert_key, hostel_block, timestamp, start_time, end_time,
                    duration_hours, energy_kwh, expected_kwh, excess_kwh,
                    excess_cost_inr, severity_score, anomaly_type,
                    recommended_action, escalation_level, status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW', ?)
                """,
                (
                    alert_key, hostel_block, timestamp,
                    start_time or timestamp, end_time or timestamp,
                    float(duration_hours), round(energy_kwh, 3),
                    round(expected_kwh, 3), round(excess_kwh, 3),
                    round(excess_cost_inr, 2), round(severity_score, 1),
                    anomaly_type, recommended_action, escalation_level, source
                )
            )
            inserted = True
            new_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM alert_records WHERE alert_key = ?", (alert_key,))
            row = cursor.fetchone()
            inserted = False
            new_id = row["id"] if row else -1
    conn.close()
    return inserted, new_id


# Alias for backward compatibility
insert_live_alert = insert_alert_record


def get_latest_readings(limit: int = 50, db_path: Optional[str] = None) -> List[Dict]:
    """Fetch latest readings across blocks."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM meter_readings
        ORDER BY id DESC LIMIT ?
        """,
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_historical_readings(
    block: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve historical readings dataframe."""
    conn = get_connection(db_path)
    query = "SELECT * FROM meter_readings"
    params = []
    if block and block != "ALL":
        query += " WHERE hostel_block = ?"
        params.append(block)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_unified_alerts(
    status_filter: Optional[str] = None,
    block_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str] = None
) -> List[Dict]:
    """Retrieve alerts with multi-criteria filtering."""
    conn = get_connection(db_path)
    query = "SELECT * FROM alert_records WHERE 1=1"
    params = []

    if status_filter and status_filter != "ALL":
        query += " AND status = ?"
        params.append(status_filter)
    if block_filter and block_filter != "ALL":
        query += " AND hostel_block = ?"
        params.append(block_filter)
    if source_filter and source_filter != "ALL":
        query += " AND source = ?"
        params.append(source_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# Backward compatibility aliases
get_live_alerts = get_unified_alerts


def fetch_alerts_dataframe(
    status_filter: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve alerts as a Pandas DataFrame (matching legacy fetch_alerts signature)."""
    conn = get_connection(db_path)
    query = "SELECT * FROM alert_records"
    params = []
    if status_filter and status_filter != "ALL":
        query += " WHERE status = ?"
        params.append(status_filter)
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Ensure compatibility column aliases exist
    if not df.empty:
        if "wasted_kwh" not in df.columns and "excess_kwh" in df.columns:
            df["wasted_kwh"] = df["excess_kwh"]
        if "cost_inr" not in df.columns and "excess_cost_inr" in df.columns:
            df["cost_inr"] = df["excess_cost_inr"]
        if "suggested_action" not in df.columns and "recommended_action" in df.columns:
            df["suggested_action"] = df["recommended_action"]
    return df


def acknowledge_alert(alert_id: int, user: str = "Warden", db_path: Optional[str] = None) -> bool:
    """Acknowledge a pending alert."""
    conn = get_connection(db_path)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE alert_records
            SET status = 'ACKNOWLEDGED', acknowledged_at = ?, acknowledged_by = ?
            WHERE id = ? AND status = 'NEW'
            """,
            (now_str, user, alert_id)
        )
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def resolve_alert(alert_id: int, user: str = "Electrician", notes: str = "Resolved by facility staff", db_path: Optional[str] = None) -> bool:
    """Resolve an active or acknowledged alert."""
    conn = get_connection(db_path)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE alert_records
            SET status = 'RESOLVED', resolved_at = ?, resolved_by = ?, resolution_notes = ?
            WHERE id = ?
            """,
            (now_str, user, notes, alert_id)
        )
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_monitoring_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute dashboard summary KPIs from unified database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS c FROM meter_readings;")
    total_readings = cursor.fetchone()["c"]

    cursor.execute("SELECT COUNT(*) AS c FROM alert_records WHERE status = 'NEW';")
    new_alerts = cursor.fetchone()["c"]

    cursor.execute("SELECT COUNT(*) AS c FROM alert_records WHERE status != 'RESOLVED';")
    active_alerts = cursor.fetchone()["c"]

    cursor.execute("SELECT SUM(excess_kwh) AS kwh, SUM(excess_cost_inr) AS inr FROM meter_readings WHERE is_anomaly = 1;")
    row_sum = cursor.fetchone()
    total_excess_kwh = row_sum["kwh"] or 0.0
    total_excess_inr = row_sum["inr"] or 0.0

    cursor.execute("SELECT MAX(energy_kwh) AS peak_kw FROM meter_readings;")
    peak_row = cursor.fetchone()
    peak_reading = peak_row["peak_kw"] or 0.0

    conn.close()
    return {
        "total_readings": total_readings,
        "new_alerts": new_alerts,
        "active_alerts": active_alerts,
        "total_excess_kwh": round(total_excess_kwh, 2),
        "total_excess_cost_inr": round(total_excess_inr, 2),
        "peak_reading_kw": round(peak_reading, 2)
    }


def set_setting(key: str, value: str, db_path: Optional[str] = None):
    """Store system setting key-value pair."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO system_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP);",
            (key, value)
        )
    conn.close()


def get_setting(key: str, default: Optional[str] = None, db_path: Optional[str] = None) -> Optional[str]:
    """Retrieve system setting value."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM system_settings WHERE key = ?;", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else default
