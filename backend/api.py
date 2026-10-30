"""FastAPI REST API server for continuous hostel electricity monitoring.

Exposes endpoints for:
- Monitoring lifecycle: start, stop, interval configuration (60s demo vs 3600s prod)
- Telemetry: latest readings, historical data queries, and statistics
- Alerts: live alert feed, acknowledgment, and resolution workflows
- Testing: manual tick triggering and on-demand spike injection for reviews
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from backend.database import (
    init_db, get_latest_readings, get_historical_readings,
    get_live_alerts, acknowledge_alert, resolve_alert, get_monitoring_stats
)
from backend.scheduler import scheduler_instance

app = FastAPI(
    title="HostelWatt Continuous Electricity Monitoring API",
    description="REST backend orchestrating simulated/hardware meter polling, real-time ML anomaly detection, and operational alerts.",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class ConfigUpdateRequest(BaseModel):
    interval_seconds: int = Field(..., ge=1, le=86400, description="Polling interval in seconds (e.g. 60 or 3600)")
    mode: Optional[str] = Field("demo", description="'demo' (60s) or 'production' (3600s)")
    anomaly_residual_threshold_kw: Optional[float] = Field(None, ge=0.1, le=50.0)


class SpikeInjectionRequest(BaseModel):
    hostel_block: str = Field(..., description="Target block ID (e.g. 'Block_A')")
    extra_kwh: float = Field(2.5, ge=0.5, le=20.0, description="Surge delta in kW")
    anomaly_type: str = Field("appliance_spike", description="Type (appliance_spike, left_on, wiring_drift, meter_fault)")


class AlertActionRequest(BaseModel):
    user: str = Field("Warden", description="Name/role of operator")
    notes: Optional[str] = Field("Action completed", description="Resolution notes")


@app.on_event("startup")
def startup_event():
    """Initialize database and start background scheduler on API boot."""
    init_db()
    # Auto-start scheduler in demo 60s mode
    scheduler_instance.start()


@app.get("/")
def root():
    return {
        "service": "HostelWatt Continuous Electricity Monitoring API",
        "status": "online",
        "docs_url": "/docs",
        "ui_url": "http://localhost:8501"
    }


@app.get("/api/status")
def get_status():
    """Retrieve current background scheduler status."""
    return scheduler_instance.get_status()


@app.post("/api/monitor/start")
def start_monitor():
    """Start continuous monitoring background thread."""
    scheduler_instance.start()
    return {"message": "Continuous monitoring started", "status": scheduler_instance.get_status()}


@app.post("/api/monitor/stop")
def stop_monitor():
    """Stop continuous monitoring background thread."""
    scheduler_instance.stop()
    return {"message": "Continuous monitoring stopped", "status": scheduler_instance.get_status()}


@app.post("/api/monitor/config")
def update_config(req: ConfigUpdateRequest):
    """Update monitoring interval and mode."""
    scheduler_instance.set_interval(req.interval_seconds, req.mode)
    if req.anomaly_residual_threshold_kw is not None:
        scheduler_instance.detector.residual_threshold = req.anomaly_residual_threshold_kw
    return {"message": "Configuration updated", "status": scheduler_instance.get_status()}


@app.post("/api/readings/tick")
def trigger_reading_tick():
    """Manually trigger an immediate telemetry reading and evaluation across all blocks."""
    res = scheduler_instance.trigger_tick()
    return res


@app.post("/api/readings/inject-spike")
def inject_test_spike(req: SpikeInjectionRequest):
    """Queue a simulated spike for demonstration and testing."""
    scheduler_instance.inject_spike(
        block_id=req.hostel_block,
        extra_kwh=req.extra_kwh,
        anomaly_type=req.anomaly_type
    )
    # Immediately trigger a tick so the spike is evaluated right away
    res = scheduler_instance.trigger_tick()
    return {
        "message": f"Queued and executed spike on {req.hostel_block}",
        "tick_result": res
    }


@app.get("/api/readings/latest")
def fetch_latest_readings(limit: int = Query(50, ge=1, le=500)):
    """Fetch recent meter readings from persistent database."""
    readings = get_latest_readings(limit=limit)
    return {"count": len(readings), "readings": readings}


@app.get("/api/readings/history")
def fetch_historical_readings(
    block: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Query historical readings table."""
    df = get_historical_readings(block=block, limit=limit)
    return {"count": len(df), "readings": df.to_dict(orient="records")}


@app.get("/api/alerts")
def fetch_alerts(
    status: Optional[str] = Query(None, description="'NEW', 'ACKNOWLEDGED', 'RESOLVED', or 'ALL'"),
    block: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500)
):
    """Query live alerts with filtering."""
    alerts = get_live_alerts(status_filter=status, block_filter=block, limit=limit)
    return {"count": len(alerts), "alerts": alerts}


@app.post("/api/alerts/{alert_id}/acknowledge")
def ack_alert(alert_id: int, req: AlertActionRequest):
    """Acknowledge an alert."""
    success = acknowledge_alert(alert_id, user=req.user)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged/resolved")
    return {"message": f"Alert {alert_id} acknowledged by {req.user}", "alert_id": alert_id}


@app.post("/api/alerts/{alert_id}/resolve")
def res_alert(alert_id: int, req: AlertActionRequest):
    """Resolve an alert."""
    success = resolve_alert(alert_id, user=req.user)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": f"Alert {alert_id} resolved by {req.user}", "alert_id": alert_id}


@app.get("/api/stats")
def fetch_stats():
    """Retrieve real-time summary KPIs."""
    stats = get_monitoring_stats()
    scheduler_status = scheduler_instance.get_status()
    return {**stats, "scheduler": scheduler_status}


if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
