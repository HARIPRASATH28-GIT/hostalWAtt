"""Meter hardware interface and physics-based simulated reading generator.

Provides:
- BaseMeterSource abstraction ready for real IoT smart meter integration (Modbus / MQTT)
- SimulatedMeterSource generating realistic continuous readings across all blocks
- Explicit labeling of all simulated feeds to differentiate from hardware meters
- On-demand spike injection for testing and live demonstration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import yaml

from src.synth import assign_day_type, generate_diurnal_multiplier


@dataclass
class MeterPayload:
    """Standardized meter telemetry packet."""
    block_id: str
    timestamp: datetime
    energy_kwh: float
    occupancy: int
    room_count: int
    day_type: str
    outside_temp_c: float
    source: str = "SIMULATED"
    is_simulated: bool = True
    display_label: str = "[SIMULATED METER FEED]"
    injected_anomaly_type: str = "none"
    injected_extra_kwh: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return d


class BaseMeterSource(ABC):
    """Abstract base class for electricity meter telemetry."""

    @abstractmethod
    def read_block(self, block_id: str, timestamp: Optional[datetime] = None) -> MeterPayload:
        """Fetch current telemetry for a specific hostel block."""
        pass

    @abstractmethod
    def read_all_blocks(self, timestamp: Optional[datetime] = None) -> List[MeterPayload]:
        """Fetch current telemetry across all monitored hostel blocks."""
        pass


class SimulatedMeterSource(BaseMeterSource):
    """Physics-informed simulator for real-time hostel block electricity telemetry."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.blocks_cfg = {b["id"]: b for b in self.config["blocks"]}
        self.rng = np.random.default_rng(self.config["system"].get("seed", 42))

        # Queue for manual / programmatic spike injections for demo testing
        self._injected_spikes: Dict[str, Dict] = {}

    def inject_spike(
        self,
        block_id: str,
        extra_kwh: float = 2.2,
        anomaly_type: str = "appliance_spike"
    ):
        """Queue a deliberate anomaly spike for the next reading of a given block."""
        self._injected_spikes[block_id] = {
            "extra_kwh": float(extra_kwh),
            "anomaly_type": anomaly_type
        }

    def _estimate_ambient_temp(self, dt: datetime) -> float:
        """Simulate realistic ambient temperature (°C) for given timestamp."""
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        seasonal_mean = 27.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 75) / 365.0)
        diurnal = 5.0 * np.sin(2 * np.pi * (hour - 8.5) / 24.0)
        return float(np.clip(seasonal_mean + diurnal + self.rng.normal(0, 0.4), 18.0, 42.0))

    def read_block(self, block_id: str, timestamp: Optional[datetime] = None) -> MeterPayload:
        """Generate a simulated electrical reading for a single block."""
        if block_id not in self.blocks_cfg:
            raise ValueError(f"Unknown hostel block: {block_id}")

        dt = timestamp or datetime.now()
        b_cfg = self.blocks_cfg[block_id]
        room_count = b_cfg["room_count"]
        base_occ = b_cfg["base_occupancy"]
        base_kw = b_cfg["base_kw"]
        noise_sigma = b_cfg.get("noise_sigma", 0.35)

        day_type = assign_day_type(dt)
        is_weekend = dt.weekday() >= 5
        hour = dt.hour

        # Dynamic occupancy
        if day_type == "holiday":
            occ_ratio = float(self.rng.uniform(0.12, 0.22))
        elif day_type == "exam":
            occ_ratio = float(self.rng.uniform(0.94, 1.0))
        else:
            occ_ratio = float(self.rng.uniform(0.82, 0.94) if not is_weekend else self.rng.uniform(0.75, 0.88))

        active_occ = int(round(base_occ * occ_ratio))
        temp_c = self._estimate_ambient_temp(dt)

        # Diurnal base load
        diurnal_mult = generate_diurnal_multiplier(hour, day_type, is_weekend)
        load = base_kw * diurnal_mult * (0.4 + 0.6 * (active_occ / max(base_occ, 1)))

        # Cooling / Fan demand
        if temp_c > 26.0:
            load += 0.08 * ((temp_c - 26.0) ** 1.15) * (room_count / 50.0)

        # Baseline noise
        load += float(self.rng.normal(0, noise_sigma))
        load = max(0.8, load)

        # Check for queued injected spikes (for interactive demonstration)
        anom_type = "none"
        injected_extra = 0.0
        if block_id in self._injected_spikes:
            spike_info = self._injected_spikes.pop(block_id)
            injected_extra = spike_info["extra_kwh"]
            anom_type = spike_info["anomaly_type"]
            load += injected_extra

        return MeterPayload(
            block_id=block_id,
            timestamp=dt,
            energy_kwh=round(load, 3),
            occupancy=active_occ,
            room_count=room_count,
            day_type=day_type,
            outside_temp_c=round(temp_c, 1),
            source="SIMULATED",
            is_simulated=True,
            display_label="[SIMULATED METER FEED]",
            injected_anomaly_type=anom_type,
            injected_extra_kwh=round(injected_extra, 3)
        )

    def read_all_blocks(self, timestamp: Optional[datetime] = None) -> List[MeterPayload]:
        """Generate telemetry for all configured hostel blocks."""
        dt = timestamp or datetime.now()
        return [self.read_block(b_id, dt) for b_id in self.blocks_cfg.keys()]


class HardwareModbusMeterSource(BaseMeterSource):
    """Production driver stub for RS-485 Modbus / MQTT / IoT smart electricity meters.

    Allows seamless drop-in replacement of the simulation engine with physical hardware
    without changing detection or alerting architecture.
    """

    def __init__(self, host: str = "192.168.1.100", port: int = 502):
        self.host = host
        self.port = port

    def read_block(self, block_id: str, timestamp: Optional[datetime] = None) -> MeterPayload:
        raise NotImplementedError(
            "Hardware smart meter driver is configured for future deployment. "
            "Use SimulatedMeterSource for current operation."
        )

    def read_all_blocks(self, timestamp: Optional[datetime] = None) -> List[MeterPayload]:
        raise NotImplementedError("Hardware smart meter driver is configured for future deployment.")
