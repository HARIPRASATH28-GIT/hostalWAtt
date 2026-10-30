"""Real-time anomaly detector and expected baseline inference engine.

Applies the trained LightGBM forecast-residual model to compute expected kWh,
detect unusual spikes beyond configurable thresholds, calculate financial excess (INR),
score severity (0-100), and recommend targeted operational actions.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import datetime
import joblib
import numpy as np
import pandas as pd
import yaml

from backend.meter_interface import MeterPayload
from backend.database import get_connection
from src.interventions import ACTION_CATALOG


@dataclass
class DetectionResult:
    """Detection analysis output for a meter reading."""
    hostel_block: str
    timestamp: str
    energy_kwh: float
    expected_kwh: float
    residual_kwh: float
    is_anomaly: int
    severity_score: float
    excess_kwh: float
    excess_cost_inr: float
    anomaly_type: str
    recommended_action: str
    action_protocol: str
    urgency: str
    source: str = "SIMULATED"
    display_label: str = "[SIMULATED METER FEED]"

    def to_dict(self) -> Dict:
        return asdict(self)


class RealtimeDetector:
    """Inference engine evaluating live meter telemetry."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.tariff_inr = float(self.config["tariff"]["energy_rate_inr_per_kwh"])
        self.residual_threshold = float(
            self.config.get("monitoring", {}).get("anomaly_residual_threshold_kw", 1.80)
        )

        # Load trained LightGBM model if present
        model_path = Path("data/processed/m2_lightgbm_model.pkl")
        self.model = None
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None

    def _get_historical_context(self, block_id: str) -> Dict[str, float]:
        """Fetch past readings for block from database to compute lags and rolling window."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT energy_kwh FROM meter_readings
            WHERE hostel_block = ?
            ORDER BY id DESC LIMIT 168
            """,
            (block_id,)
        )
        rows = [r["energy_kwh"] for r in cursor.fetchall()]
        conn.close()

        if not rows:
            # Baseline proxy
            base_kw = 14.0
            return {
                "lag_1": base_kw,
                "lag_24": base_kw,
                "lag_168": base_kw,
                "rolling_mean_6h": base_kw,
                "rolling_std_6h": 0.4,
                "rolling_mean_24h": base_kw,
                "rolling_std_24h": 0.5,
                "rolling_mean_168h": base_kw
            }

        lag_1 = rows[0]
        lag_24 = rows[23] if len(rows) > 23 else lag_1
        lag_168 = rows[167] if len(rows) > 167 else lag_1

        win_6 = rows[:6]
        win_24 = rows[:24]

        return {
            "lag_1": lag_1,
            "lag_24": lag_24,
            "lag_168": lag_168,
            "rolling_mean_6h": float(np.mean(win_6)),
            "rolling_std_6h": float(np.std(win_6)) if len(win_6) > 1 else 0.4,
            "rolling_mean_24h": float(np.mean(win_24)),
            "rolling_std_24h": float(np.std(win_24)) if len(win_24) > 1 else 0.5,
            "rolling_mean_168h": float(np.mean(rows))
        }

    def predict_expected(self, payload: MeterPayload) -> float:
        """Predict expected electricity consumption (kWh) using ML model or physics fallback."""
        dt = payload.timestamp
        hist = self._get_historical_context(payload.block_id)

        cooling_deg = max(payload.outside_temp_c - 26.0, 0.0)
        occ_per_room = payload.occupancy / max(payload.room_count, 1.0)

        feature_row = {
            "hour": dt.hour,
            "weekday": dt.weekday(),
            "is_weekend": int(dt.weekday() >= 5),
            "sin_hour": np.sin(2 * np.pi * dt.hour / 24.0),
            "cos_hour": np.cos(2 * np.pi * dt.hour / 24.0),
            "sin_dow": np.sin(2 * np.pi * dt.weekday() / 7.0),
            "cos_dow": np.cos(2 * np.pi * dt.weekday() / 7.0),
            "outside_temp_c": payload.outside_temp_c,
            "cooling_degree_c": cooling_deg,
            "occupancy": payload.occupancy,
            "room_count": payload.room_count,
            "occ_per_room": occ_per_room,
            "day_type_regular": int(payload.day_type == "regular"),
            "day_type_exam": int(payload.day_type == "exam"),
            "day_type_holiday": int(payload.day_type == "holiday"),
            **hist
        }

        if self.model is not None:
            try:
                feature_names = self.model.feature_name_
                x_vec = pd.DataFrame([feature_row])[feature_names]
                pred = float(self.model.predict(x_vec)[0])
                return max(0.8, round(pred, 3))
            except Exception:
                pass

        # Heuristic fallback based on rolling historical mean + occupancy
        expected = hist["rolling_mean_24h"] * (0.8 + 0.2 * (payload.occupancy / 150.0))
        return max(0.8, round(expected, 3))

    def evaluate_reading(self, payload: MeterPayload) -> DetectionResult:
        """Analyze a meter reading packet, compute deviation, and generate alert metadata."""
        expected = self.predict_expected(payload)
        actual = payload.energy_kwh

        residual = round(abs(actual - expected), 3)
        excess_kwh = max(0.0, round(actual - expected, 3))
        excess_cost = round(excess_kwh * self.tariff_inr, 2)

        # Anomaly determination
        is_anom = 0
        anom_type = "none"

        # Check for meter zero fault
        if actual <= 0.05:
            is_anom = 1
            anom_type = "meter_fault"
        elif payload.injected_anomaly_type != "none":
            # Direct injected test anomaly
            is_anom = 1
            anom_type = payload.injected_anomaly_type
        elif excess_kwh >= self.residual_threshold:
            is_anom = 1
            if excess_kwh >= 2.0:
                anom_type = "appliance_spike"
            else:
                anom_type = "left_on" if (payload.timestamp.hour >= 22 or payload.timestamp.hour <= 5) else "appliance_spike"

        # Severity score (0 to 100)
        if is_anom:
            sev_raw = (residual / max(self.residual_threshold * 2.2, 0.5)) * 100.0
            severity = float(np.clip(sev_raw, 25.0, 100.0))
        else:
            severity = float(np.clip((residual / max(self.residual_threshold, 1.0)) * 20.0, 0.0, 24.0))

        # Recommend Action
        action_meta = ACTION_CATALOG.get(anom_type, ACTION_CATALOG["appliance_spike"])

        return DetectionResult(
            hostel_block=payload.block_id,
            timestamp=payload.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            energy_kwh=actual,
            expected_kwh=expected,
            residual_kwh=residual,
            is_anomaly=is_anom,
            severity_score=round(severity, 1),
            excess_kwh=excess_kwh,
            excess_cost_inr=excess_cost,
            anomaly_type=anom_type,
            recommended_action=action_meta["action"],
            action_protocol=action_meta["protocol"],
            urgency=action_meta["urgency"] if is_anom else "NORMAL",
            source=payload.source,
            display_label=payload.display_label
        )
