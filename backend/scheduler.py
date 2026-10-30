"""Independent background monitoring scheduler.

Executes continuous meter polling and anomaly detection independently of whether
any browser or dashboard window is currently open.
Prevents duplicate jobs, deduplicates alerts, and supports dynamic interval switching
(60-second demo mode vs. 3600-second hourly production mode).
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging
import threading
import time
import yaml

from backend.database import init_db, insert_reading, insert_live_alert, set_setting, get_setting
from backend.meter_interface import SimulatedMeterSource, MeterPayload
from backend.detector import RealtimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Thread-safe background scheduler for real-time electricity telemetry processing."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MonitoringScheduler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if getattr(self, "_initialized", False):
            return

        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        m_cfg = self.config.get("monitoring", {})
        self.mode = m_cfg.get("mode", "demo")
        self.interval_seconds = int(
            m_cfg.get("demo_interval_seconds", 60) if self.mode == "demo" else m_cfg.get("prod_interval_seconds", 3600)
        )

        self.meter_source = SimulatedMeterSource(config_path)
        self.detector = RealtimeDetector(config_path)

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._job_lock = threading.Lock()

        self.last_run_timestamp: Optional[datetime] = None
        self.total_ticks = 0
        self.total_alerts_generated = 0

        # Initialize SQLite schema
        init_db()

        self._initialized = True
        logger.info("Initialized MonitoringScheduler in '%s' mode (interval: %ds).", self.mode, self.interval_seconds)

    def start(self):
        """Start the background monitoring worker thread if not already active."""
        with self._job_lock:
            if self.is_running:
                logger.info("MonitoringScheduler is already running.")
                return

            self.is_running = True
            self._stop_event.clear()
            set_setting("monitoring_status", "RUNNING")
            set_setting("monitoring_mode", self.mode)
            set_setting("interval_seconds", str(self.interval_seconds))

            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="HostelWattMonitor")
            self._thread.start()
            logger.info("Started background monitoring thread.")

    def stop(self):
        """Signal background worker thread to terminate."""
        with self._job_lock:
            if not self.is_running:
                return

            self.is_running = False
            self._stop_event.set()
            set_setting("monitoring_status", "STOPPED")
            logger.info("Stopped background monitoring thread.")

    def set_interval(self, interval_seconds: int, mode: Optional[str] = None):
        """Dynamically update tick interval (e.g., 60s for demo, 3600s for production)."""
        with self._job_lock:
            self.interval_seconds = int(interval_seconds)
            if mode:
                self.mode = mode
            set_setting("interval_seconds", str(self.interval_seconds))
            set_setting("monitoring_mode", self.mode)
            logger.info("Updated monitoring interval to %ds (Mode: %s).", self.interval_seconds, self.mode)

    def trigger_tick(self) -> Dict:
        """Execute one evaluation cycle across all hostel blocks synchronously."""
        # Non-blocking lock to guarantee no duplicate overlapping executions
        if not self._job_lock.acquire(blocking=False):
            return {"status": "busy", "message": "Previous tick cycle still processing"}

        try:
            now = datetime.now()
            readings = self.meter_source.read_all_blocks(now)
            processed_results = []
            alerts_in_tick = 0

            for payload in readings:
                # 1. Detect anomalies & expected baseline
                res = self.detector.evaluate_reading(payload)

                # 2. Persist reading to database
                insert_reading(
                    timestamp=res.timestamp,
                    hostel_block=res.hostel_block,
                    energy_kwh=res.energy_kwh,
                    expected_kwh=res.expected_kwh,
                    residual_kwh=res.residual_kwh,
                    is_anomaly=res.is_anomaly,
                    severity_score=res.severity_score,
                    excess_kwh=res.excess_kwh,
                    excess_cost_inr=res.excess_cost_inr,
                    anomaly_type=res.anomaly_type,
                    source=res.source
                )

                # 3. If anomaly flagged, generate deduplicated alert
                if res.is_anomaly == 1:
                    # Deduplication key based on block + timestamp hour + type
                    time_bucket = payload.timestamp.strftime("%Y%m%d%H%M")
                    alert_key = f"{res.hostel_block}_{time_bucket}_{res.anomaly_type}"

                    inserted, _ = insert_live_alert(
                        alert_key=alert_key,
                        hostel_block=res.hostel_block,
                        timestamp=res.timestamp,
                        energy_kwh=res.energy_kwh,
                        expected_kwh=res.expected_kwh,
                        excess_kwh=res.excess_kwh,
                        excess_cost_inr=res.excess_cost_inr,
                        severity_score=res.severity_score,
                        anomaly_type=res.anomaly_type,
                        recommended_action=res.recommended_action,
                        source=res.source
                    )
                    if inserted:
                        alerts_in_tick += 1

                processed_results.append(res.to_dict())

            self.last_run_timestamp = now
            self.total_ticks += 1
            self.total_alerts_generated += alerts_in_tick
            set_setting("last_run_timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))

            return {
                "status": "success",
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "blocks_processed": len(processed_results),
                "alerts_created": alerts_in_tick,
                "readings": processed_results
            }

        finally:
            self._job_lock.release()

    def inject_spike(
        self,
        block_id: str,
        extra_kwh: float = 2.4,
        anomaly_type: str = "appliance_spike"
    ):
        """Queue an abnormal electrical spike on next reading for testing."""
        self.meter_source.inject_spike(block_id, extra_kwh, anomaly_type)
        logger.info("Queued test spike of +%.2f kWh (%s) on block %s.", extra_kwh, anomaly_type, block_id)

    def _worker_loop(self):
        """Loop running in background thread."""
        logger.info("Monitoring background loop active.")
        while not self._stop_event.is_set():
            try:
                self.trigger_tick()
            except Exception as e:
                logger.error("Error during scheduled monitoring tick: %s", e)

            # Sleep in 1-second chunks to allow rapid graceful termination
            for _ in range(self.interval_seconds):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("Monitoring background loop exited.")

    def get_status(self) -> Dict:
        """Return operational health status dictionary."""
        return {
            "is_running": self.is_running,
            "mode": self.mode,
            "interval_seconds": self.interval_seconds,
            "last_run_timestamp": self.last_run_timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.last_run_timestamp else None,
            "total_ticks": self.total_ticks,
            "total_alerts_generated": self.total_alerts_generated
        }


# Global instance
scheduler_instance = MonitoringScheduler()
