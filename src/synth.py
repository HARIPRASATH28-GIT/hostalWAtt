"""Physics-informed synthetic hostel electricity data generator.

Simulates 12 months of hourly meter readings for multiple hostel blocks with
realistic diurnal curves, academic schedules (regular/exam/holiday), weather response,
and labeled injection of 5 standard anomaly types.
Generates hierarchical floor- and room-level meter data for a demo block.
"""

from pathlib import Path
from typing import Dict, List, Tuple
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


def assign_day_type(dt: pd.Timestamp) -> str:
    """Classify date into regular, exam, or holiday based on academic calendar."""
    month = dt.month
    day = dt.day

    # Summer break (June 1 - July 15) & Winter break (Dec 20 - Dec 31)
    if (month == 6) or (month == 7 and day <= 15) or (month == 12 and day >= 20):
        return "holiday"
    # Specific festival holidays
    if (month == 1 and day in [14, 15, 26]) or (month == 8 and day == 15) or (month == 10 and day in [2, 24, 25]):
        return "holiday"

    # Exam periods: Mid-term in March (Mar 15-26), End-sem in May (May 2-18), Mid-term in Oct (Oct 10-22), End-sem in Dec (Dec 1-14)
    if (month == 3 and 15 <= day <= 26) or (month == 5 and 2 <= day <= 18) or \
       (month == 10 and 10 <= day <= 22) or (month == 12 and 1 <= day <= 14):
        return "exam"

    return "regular"


def simulate_outside_temp(timestamps: pd.DatetimeIndex, seed: int = 42) -> np.ndarray:
    """Simulate realistic ambient temperature (°C) with seasonal & diurnal cycle."""
    np.random.seed(seed)
    day_of_year = timestamps.dayofyear.values
    hour = timestamps.hour.values

    # Annual seasonal temperature curve (Tropical/South India: peaking in May ~36°C, winter Jan ~24°C)
    seasonal_mean = 27.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 75) / 365.0)

    # Diurnal cycle: minimum at 05:00, maximum at 14:00 (range ~8-12°C)
    diurnal_variation = 5.0 * np.sin(2 * np.pi * (hour - 8.5) / 24.0)

    # Ambient weather noise
    noise = np.random.normal(0, 0.8, size=len(timestamps))

    temps = seasonal_mean + diurnal_variation + noise
    return np.round(np.clip(temps, 18.0, 42.5), 1)


def generate_diurnal_multiplier(hour: int, day_type: str, is_weekend: bool) -> float:
    """Calculate hourly diurnal base demand multiplier based on schedule."""
    # Base profile across 24 hours:
    # Hours 0-5: Night floor
    # Hours 6-9: Morning peak (showers, geysers, prep)
    # Hours 10-16: Academic class hours (hostel mostly vacant)
    # Hours 17-23: Evening peak (study, laptops, fans, common rooms)
    if 0 <= hour <= 5:
        if day_type == "exam":
            # Elevated study load past midnight
            base = 0.55 if hour <= 3 else 0.35
        else:
            base = 0.22 if hour <= 4 else 0.32
    elif 6 <= hour <= 9:
        base = 0.95 + 0.15 * np.sin(np.pi * (hour - 6) / 3)
    elif 10 <= hour <= 16:
        if is_weekend or day_type == "holiday":
            base = 0.65
        else:
            base = 0.38  # In class
    else:  # 17 to 23
        base = 0.85 + 0.25 * np.sin(np.pi * (hour - 17) / 6)

    if is_weekend:
        base *= 1.08  # Slightly more continuous presence

    return base


def inject_anomalies_to_block(
    block_id: str,
    df_block: pd.DataFrame,
    rng: np.random.Generator
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Inject 5 labeled anomaly types with ground truth annotations.

    a) appliance_spike: +1.0 to +2.0 kW for 1 to 4 hours (heater/induction)
    b) left_on: flat extra load 8 to 12 hours, mostly overnight/holiday
    c) wiring_drift: slow baseline rise (~+0.5% per day for 2 to 4 weeks) with higher night-time floor
    d) intermittent_fault: random short bursts at odd hours
    e) meter_fault: zero or flatline for several hours
    """
    n_hours = len(df_block)
    is_anomaly = np.zeros(n_hours, dtype=int)
    anomaly_type = np.array(["none"] * n_hours, dtype=object)
    injected_extra = np.zeros(n_hours, dtype=float)
    events_log = []

    # 1. Inject wiring_drift (1 event per block over a 3-week window in second half of year)
    drift_start = rng.integers(int(n_hours * 0.45), int(n_hours * 0.75))
    drift_duration = rng.integers(14 * 24, 25 * 24)  # 14 to 25 days
    drift_end = min(n_hours, drift_start + drift_duration)

    for i, idx in enumerate(range(drift_start, drift_end)):
        # Gradual drift rate +0.5% per day
        drift_kw = 0.005 * (i / 24.0) * df_block.loc[idx, "energy_kwh"]
        # Ensure higher night-time floor
        hour = df_block.loc[idx, "date_time"].hour
        if hour <= 5:
            drift_kw += 0.8

        is_anomaly[idx] = 1
        anomaly_type[idx] = "wiring_drift"
        injected_extra[idx] = round(drift_kw, 3)

    events_log.append({
        "block": block_id,
        "type": "wiring_drift",
        "start": df_block.loc[drift_start, "date_time"],
        "end": df_block.loc[drift_end - 1, "date_time"],
        "duration_hours": drift_duration
    })

    # Available candidate hours for episodic anomalies (avoid overlap with drift)
    candidate_indices = [
        i for i in range(24, n_hours - 24)
        if not (drift_start <= i < drift_end)
    ]
    rng.shuffle(candidate_indices)

    # 2. Inject appliance_spike: +1.0 to +2.0 kW for 1 to 4 hours
    num_spikes = rng.integers(14, 22)
    spike_count = 0
    used_indices = set(range(drift_start, drift_end))

    for idx in candidate_indices:
        if spike_count >= num_spikes:
            break
        duration = rng.integers(1, 5)
        indices = list(range(idx, idx + duration))
        if any(i in used_indices for i in indices) or max(indices) >= n_hours:
            continue

        extra_kw = rng.uniform(1.1, 2.1)
        for i in indices:
            is_anomaly[i] = 1
            anomaly_type[i] = "appliance_spike"
            injected_extra[i] = round(extra_kw + rng.normal(0, 0.05), 3)
            used_indices.add(i)

        events_log.append({
            "block": block_id,
            "type": "appliance_spike",
            "start": df_block.loc[indices[0], "date_time"],
            "end": df_block.loc[indices[-1], "date_time"],
            "extra_kw": round(extra_kw, 2)
        })
        spike_count += 1

    # 3. Inject left_on: flat extra load 8 to 12 hours, mostly overnight/holiday
    num_left_on = rng.integers(8, 14)
    left_on_count = 0
    for idx in candidate_indices:
        if left_on_count >= num_left_on:
            break
        dt = df_block.loc[idx, "date_time"]
        is_overnight_or_hol = (dt.hour >= 21 or dt.hour <= 2) or (df_block.loc[idx, "day_type"] == "holiday")
        if not is_overnight_or_hol:
            continue

        duration = rng.integers(8, 13)
        indices = list(range(idx, idx + duration))
        if any(i in used_indices for i in indices) or max(indices) >= n_hours:
            continue

        flat_kw = rng.uniform(0.9, 1.6)
        for i in indices:
            is_anomaly[i] = 1
            anomaly_type[i] = "left_on"
            injected_extra[i] = round(flat_kw + rng.normal(0, 0.03), 3)
            used_indices.add(i)

        events_log.append({
            "block": block_id,
            "type": "left_on",
            "start": df_block.loc[indices[0], "date_time"],
            "end": df_block.loc[indices[-1], "date_time"],
            "flat_kw": round(flat_kw, 2)
        })
        left_on_count += 1

    # 4. Inject intermittent_fault: random short bursts at odd hours (01:00 to 04:00)
    num_intermittent = rng.integers(7, 12)
    intermittent_count = 0
    for idx in candidate_indices:
        if intermittent_count >= num_intermittent:
            break
        dt = df_block.loc[idx, "date_time"]
        if not (1 <= dt.hour <= 4):
            continue

        duration = rng.integers(2, 5)
        indices = list(range(idx, idx + duration))
        if any(i in used_indices for i in indices) or max(indices) >= n_hours:
            continue

        for i in indices:
            burst_kw = rng.uniform(1.2, 1.9) if rng.random() > 0.25 else 0.2
            is_anomaly[i] = 1
            anomaly_type[i] = "intermittent_fault"
            injected_extra[i] = round(burst_kw, 3)
            used_indices.add(i)

        events_log.append({
            "block": block_id,
            "type": "intermittent_fault",
            "start": df_block.loc[indices[0], "date_time"],
            "end": df_block.loc[indices[-1], "date_time"]
        })
        intermittent_count += 1

    # 5. Inject meter_fault: zero or flatline for 4 to 8 hours
    num_meter_faults = rng.integers(5, 9)
    meter_fault_count = 0
    for idx in candidate_indices:
        if meter_fault_count >= num_meter_faults:
            break
        duration = rng.integers(4, 9)
        indices = list(range(idx, idx + duration))
        if any(i in used_indices for i in indices) or max(indices) >= n_hours:
            continue

        fault_mode = rng.choice(["zero", "flatline"])
        flat_val = max(0.5, round(df_block.loc[indices[0], "energy_kwh"] * 0.4, 2))

        for i in indices:
            orig = df_block.loc[i, "energy_kwh"]
            new_val = 0.0 if fault_mode == "zero" else flat_val
            is_anomaly[i] = 1
            anomaly_type[i] = "meter_fault"
            injected_extra[i] = round(new_val - orig, 3)
            used_indices.add(i)

        events_log.append({
            "block": block_id,
            "type": "meter_fault",
            "mode": fault_mode,
            "start": df_block.loc[indices[0], "date_time"],
            "end": df_block.loc[indices[-1], "date_time"]
        })
        meter_fault_count += 1

    df_block["is_anomaly"] = is_anomaly
    df_block["anomaly_type"] = anomaly_type
    df_block["injected_extra_kwh"] = injected_extra
    # Apply injected extra load to energy_kwh
    df_block["energy_kwh"] = np.clip(df_block["energy_kwh"] + injected_extra, 0.0, None).round(3)

    return df_block, events_log


def generate_hostel_dataset(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 12 months hourly dataset for all configured blocks and hierarchical demo block data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (blocks_df, demo_rooms_df)
    """
    config = load_config(config_path)
    seed = config["system"]["seed"]
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp(config["synthetic"]["start_date"])
    num_hours = config["synthetic"]["num_hours"]
    timestamps = pd.date_range(start_date, periods=num_hours, freq="h")

    # Outside temperature vector (common ambient climate across hostel campus)
    outside_temp = simulate_outside_temp(timestamps, seed=seed)

    # Academic day type calendar
    day_types = [assign_day_type(ts) for ts in timestamps]

    blocks_data = []
    blocks_cfg = config["blocks"]

    for b_cfg in blocks_cfg:
        b_id = b_cfg["id"]
        room_count = b_cfg["room_count"]
        base_occ = b_cfg["base_occupancy"]
        base_kw = b_cfg["base_kw"]
        noise_sigma = b_cfg["noise_sigma"]

        # Simulate dynamic occupancy based on day type and weekday
        occupancies = []
        for dt, d_type in zip(timestamps, day_types):
            is_weekend = dt.weekday() >= 5
            if d_type == "holiday":
                occ_factor = rng.uniform(0.12, 0.22)
            elif d_type == "exam":
                occ_factor = rng.uniform(0.94, 1.0)
            else:
                occ_factor = rng.uniform(0.82, 0.94) if not is_weekend else rng.uniform(0.75, 0.88)
            occupancies.append(int(round(base_occ * occ_factor)))

        occupancies = np.array(occupancies)

        # Baseline energy calculation with physics components:
        # 1. Diurnal baseline * occupancy scaling
        # 2. Weather response: cooling demand for temp > 26°C (fans & AC)
        # 3. Block specific random Gaussian noise
        energy_readings = []
        for i, dt in enumerate(timestamps):
            hour = dt.hour
            is_weekend = dt.weekday() >= 5
            d_type = day_types[i]
            temp = outside_temp[i]

            diurnal_mult = generate_diurnal_multiplier(hour, d_type, is_weekend)
            occ_ratio = occupancies[i] / float(base_occ)

            # Base electrical load
            block_load = base_kw * diurnal_mult * (0.4 + 0.6 * occ_ratio)

            # Temperature sensitivity: fan and cooling load increases above 26°C
            if temp > 26.0:
                cooling_demand = 0.08 * ((temp - 26.0) ** 1.15) * (room_count / 50.0)
                block_load += cooling_demand

            # Natural noise
            block_load += rng.normal(0, noise_sigma)
            energy_readings.append(max(1.0, round(block_load, 3)))

        df_b = pd.DataFrame({
            "hostel_block": b_id,
            "date_time": timestamps,
            "energy_kwh": energy_readings,
            "occupancy": occupancies,
            "room_count": room_count,
            "day_type": day_types,
            "outside_temp_c": outside_temp
        })

        # Inject anomalies with ground truth labels
        df_b, _ = inject_anomalies_to_block(b_id, df_b, rng)
        blocks_data.append(df_b)

    all_blocks_df = pd.concat(blocks_data, ignore_index=True)

    # Generate floor-level and room-level breakdown for the demo block (Block_A)
    demo_block_id = config["demo_block"]["block_id"]
    demo_df = all_blocks_df[all_blocks_df["hostel_block"] == demo_block_id].copy().reset_index(drop=True)

    num_floors = config["demo_block"]["floors"]
    rooms_per_floor = config["demo_block"]["rooms_per_floor"]
    total_rooms = num_floors * rooms_per_floor

    room_ids = []
    floor_mapping = {}
    for f in range(1, num_floors + 1):
        for r in range(1, rooms_per_floor + 1):
            r_id = f"Room_{f}0{r}" if r < 10 else f"Room_{f}{r}"
            room_ids.append(r_id)
            floor_mapping[r_id] = f"Floor_{f}"

    # Allocate baseline room distribution weights that sum to 1.0
    raw_weights = rng.uniform(0.8, 1.2, size=total_rooms)
    base_weights = raw_weights / raw_weights.sum()

    room_records = []
    # Identify target room for appliance anomalies in demo block (e.g., Room_204)
    target_appliance_room = "Room_204"

    for idx, row in demo_df.iterrows():
        total_kwh = row["energy_kwh"]
        anom_type = row["anomaly_type"]
        injected = row["injected_extra_kwh"]

        # Base load without injected anomaly for regular distribution
        clean_kwh = max(0.1, total_kwh - injected)

        # Room proportions with micro-fluctuations
        step_weights = base_weights * rng.uniform(0.92, 1.08, size=total_rooms)
        step_weights /= step_weights.sum()

        room_kwhs = clean_kwh * step_weights

        # If there's an appliance spike or left-on anomaly, route it to the specific room!
        if anom_type in ["appliance_spike", "left_on"] and injected > 0:
            target_idx = room_ids.index(target_appliance_room)
            room_kwhs[target_idx] += injected
        elif anom_type == "wiring_drift" and injected > 0:
            # Distributed wiring drift across Floor 2 rooms
            f2_indices = [i for i, r in enumerate(room_ids) if floor_mapping[r] == "Floor_2"]
            extra_per_room = injected / len(f2_indices)
            for fi in f2_indices:
                room_kwhs[fi] += extra_per_room
        elif anom_type == "meter_fault":
            # Scale down proportionally if meter fault occurred at block meter
            pass
        else:
            # Distribute residual evenly to ensure exact mathematical sum
            pass

        # Exact adjustment so room loads sum precisely to total_kwh
        diff = total_kwh - room_kwhs.sum()
        room_kwhs += diff / total_rooms
        room_kwhs = np.clip(room_kwhs, 0.0, None)
        # Final normalization to ensure sum == total_kwh
        room_kwhs = (room_kwhs / room_kwhs.sum()) * total_kwh

        for r_idx, r_id in enumerate(room_ids):
            room_records.append({
                "hostel_block": demo_block_id,
                "floor": floor_mapping[r_id],
                "room_id": r_id,
                "date_time": row["date_time"],
                "room_energy_kwh": round(room_kwhs[r_idx], 4),
                "is_anomaly": row["is_anomaly"],
                "anomaly_type": row["anomaly_type"] if (r_id == target_appliance_room and anom_type in ["appliance_spike", "left_on"]) else ("wiring_drift" if (floor_mapping[r_id] == "Floor_2" and anom_type == "wiring_drift") else "none")
            })

    demo_rooms_df = pd.DataFrame(room_records)

    # Save to disk
    synth_dir = Path(config["system"]["synthetic_dir"])
    synth_dir.mkdir(parents=True, exist_ok=True)

    all_blocks_df.to_parquet(synth_dir / "hostel_blocks.parquet", index=False)
    all_blocks_df.to_csv(synth_dir / "hostel_blocks.csv", index=False)
    demo_rooms_df.to_parquet(synth_dir / "demo_block_rooms.parquet", index=False)

    logger.info("Successfully generated %d rows of block data and %d rows of room data.",
                len(all_blocks_df), len(demo_rooms_df))

    return all_blocks_df, demo_rooms_df


if __name__ == "__main__":
    df_blocks, df_rooms = generate_hostel_dataset()
    print("Block Data Summary:")
    print(df_blocks.groupby(["hostel_block", "anomaly_type"]).size())
    print("\nRoom Data Preview:")
    print(df_rooms.head())
