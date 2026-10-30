# Implementation Plan: "HostelWatt" AI-Powered Electricity Anomaly Detection Dashboard

## 1. Project Overview & Scope
**HostelWatt** is an end-to-end AI system designed for institutional hostel facilities to detect block-wise electrical anomalies, estimate avoidable energy waste (kWh) and financial loss (INR), plan targeted operational interventions, and track alert lifecycles.

---

## 2. Directory Structure
```
hostelwatt/
├── config.yaml                     # Config for tariffs, thresholds, seed, blocks, demo settings
├── requirements.txt                # Fixed versions for reproducibility
├── README.md                       # Comprehensive guide, setup, architecture, screenshots
├── docs/
│   └── ASSUMPTIONS.md              # Explicit operational, engineering & economic assumptions
├── data/
│   ├── raw/                        # LEAD 1.0 raw dataset (with automated fallback generator)
│   ├── processed/                  # Cleaned & aligned features
│   ├── synthetic/                  # Synthetic hostel & floor/room level datasets
│   └── alerts.db                   # SQLite persistent database for alert management
├── src/
│   ├── __init__.py
│   ├── data_lead.py                # LEAD dataset loader, missing value handling, alignment & fallback
│   ├── synth.py                    # 5-block synthetic generator + floor/room hierarchy for demo block
│   ├── features.py                 # Time, calendar, weather, occupancy, lag & rolling window features
│   ├── models.py                   # B1 (Rolling), B2 (Seasonal), M1 (Isolation Forest), M2 (LightGBM)
│   ├── savings.py                  # Expected vs actual savings, TNEB tariff modeling, payback calculator
│   ├── demand.py                   # Block/System Maximum Demand, coincident peak, 90% threshold alert
│   ├── interventions.py            # Heuristic rule-based anomaly typing & block priority ranking
│   ├── alerts.py                   # Deduplication, cooldown, escalation engine, SQLite alert store
│   └── evaluate.py                 # Time-split evaluation, PR-AUC, detection delay, ablation, domain gap
├── app/
│   ├── Home.py                     # Main Overview: KPI cards, Block status tiles, system overview
│   └── pages/
│       ├── 1_Block_Explorer.py     # Time-series actual vs expected, anomaly flags, hour x day heatmap
│       ├── 2_Anomalies_&_Alerts.py # Ranked alert feed, acknowledge/resolve SQLite workflow
│       ├── 3_Savings_Calculator.py # Waste breakdown, annual forecast, interactive ROI/payback calculator
│       ├── 4_Interventions.py      # Prioritized block action matrix, checklists, payoff estimates
│       ├── 5_Maximum_Demand.py     # System peak, coincident peak, 90% MD alert, PF kVA simulator
│       ├── 6_Room_Localization.py  # Floor/room drilldown, subset-sum appliance matching, Top-K accuracy
│       ├── 7_Model_Evaluation.py   # PR-AUC, Confusion Matrix, Detection delay, Ablation, LEAD domain gap
│       ├── 8_Data_&_Settings.py    # CSV upload, synthetic regenerator, live stream simulator, config editor
│       └── 9_About_&_Limitations.py# Methodology, engineering assumptions, disclaimer on estimates
└── tests/
    ├── __init__.py
    ├── test_synth.py               # Data schema, diurnal curves, anomaly labels, room sum integrity
    ├── test_features_and_splits.py # Temporal split boundaries, no future leakage, feature lag integrity
    ├── test_models.py              # Baseline & LightGBM output shapes, bounds, severity 0-100
    ├── test_savings_demand.py      # Savings math, TNEB tariff math, system MD lower-bound logic
    ├── test_alerts.py              # Deduplication, cooldown window, 6h escalation, SQLite persistence
    └── test_room_localization.py   # Subset-sum appliance matching with tolerance band
```

---

## 3. Detailed Phase Breakdown

### Phase 1: Data Engine
- **LEAD 1.0 Loader (`src/data_lead.py`)**:
  - Checks `data/raw/` for LEAD dataset. If missing, prints clear download instructions from `github.com/samy101/lead-dataset` and generates a lightweight synthetic LEAD-like dataset so the pipeline runs out-of-the-box offline.
  - Handles missing timestamps, forward-fills gaps up to 3h, aligns hourly frequency.
- **Synthetic Generator (`src/synth.py`)**:
  - Simulates 5 hostel blocks (`Block_A` to `Block_E`), 12 months (8,760 hours/block).
  - Diurnal profile: Morning peak (06:00–09:00), afternoon trough, evening peak (18:00–23:00), night floor.
  - Day types: Regular, Exam (midnight–04:00 extended study load), Holiday (occupancy drops to 10-20%).
  - Physics factors: Scaling with occupancy, room count, and outside temperature (cooling/fans quadratic response).
  - 5 Anomaly Types injected (1-5% hours):
    1. `appliance_spike`: +1.0 to +2.0 kW for 1-4 hours (immersion rod, electric kettle).
    2. `left_on`: Flat extra load for 8-12 hours overnight or during holidays.
    3. `wiring_drift`: Baseline drift +0.5%/day over 2-4 weeks with elevated night floor.
    4. `intermittent_fault`: Random short bursts at odd hours (loose contact, cycling compressor).
    5. `meter_fault`: Frozen reading / flatline or zero values for 4-8 hours.
  - Hierarchy for 1 Demo Block: 3 floors, 8 rooms per floor (24 rooms). Room-level loads sum exactly to the block load.

### Phase 2: Feature Engineering & Anomaly Detection Models
- **Feature Engineering (`src/features.py`)**:
  - Temporal: hour, day of week, month, day of year, cyclical sine/cosine encodings.
  - Contextual: occupancy ratio, normalized room count, outside temperature, day type one-hot.
  - Lag & Rolling: lags (t-1, t-24, t-168), rolling statistics (mean, std over 6h, 24h, 168h).
  - Strict temporal ordering to prevent data leakage.
- **Models (`src/models.py`)**:
  - **B1 (Rolling baseline)**: Rolling 24h mean $\pm k \cdot \sigma$.
  - **B2 (Seasonal baseline)**: (hour $\times$ weekday $\times$ day_type) median + Median Absolute Deviation (MAD).
  - **M1 (Isolation Forest)**: Unsupervised context outlier detection.
  - **M2 (Forecast-Residual LightGBM / XGBoost fallback)**: Supervised regressor predicting expected kWh. Residual $r_t = |y_t - \hat{y}_t|$. Dynamic quantile/residual threshold flag; severity normalized to 0–100.
  - Temporal Train/Test split: First 9 months train, last 3 months test (strictly no random splitting).
  - LEAD 1.0 benchmark and domain gap analysis (commercial office profile vs student living profile).

### Phase 3: Savings, Demand, Interventions & Alert Lifecycle
- **Savings Engine (`src/savings.py`)**:
  - $Wasted\_kWh = \max(Actual - Expected, 0)$ on flagged hours.
  - $Cost\_INR = Wasted\_kWh \times Tariff$ (Configurable in `config.yaml`, default ₹7.50/kWh based on TNEB LT-IIB).
  - Aggregations: Event, block, weekly, monthly, annual projection.
  - Sensor ROI / Payback sensitivity analysis (conservative, expected, optimistic).
- **Maximum Demand (`src/demand.py`)**:
  - Hourly kWh = average kW.
  - System Maximum Demand: coincident sum across all blocks per hour.
  - 90% threshold alert on recorded monthly MD. Lower-bound disclaimer for hourly sampling.
  - Apparent power (kVA) estimation with configurable power factor (0.85-0.95).
- **Intervention Engine (`src/interventions.py`)**:
  - Rule-based classifier mapping anomaly shapes & timestamps to categories.
  - Action mapping: Room inspection, Night rounds, Electrician wiring audit, Holiday shutdown checklist, Meter maintenance.
  - Block Priority Index = $w_1 \cdot \text{waste} + w_2 \cdot \text{cost} + w_3 \cdot \text{persistence} + w_4 \cdot \text{recurrence}$.
- **Alert Lifecycle (`src/alerts.py`)**:
  - Cooldown window (prevents duplicate alert spam for ongoing anomaly).
  - Escalation logic: 1h anomaly $\rightarrow$ Log; $\ge 6$h anomaly $\rightarrow$ Escalate to Warden/Admin.
  - Persistent SQLite store (`data/alerts.db`) with Acknowledge and Resolve state transitions.

### Phase 4: Streamlit Dashboard (10 Specialized Pages)
1. **Home (Overview)**: KPI cards (Active alerts, total waste kWh/INR this month, Peak MD kW), Block status cards (Normal/Warning/Critical), quick system status.
2. **Block Explorer**: Time series plot of actual vs expected with anomaly highlights, interactive tooltips (features & root causes), hour $\times$ day consumption heatmap.
3. **Anomalies & Alerts**: Ranked alert feed with filtering, severity pills, Acknowledge/Resolve buttons updating SQLite.
4. **Savings Calculator**: Event/Block/Month breakdowns, annual savings projections, interactive payback/ROI calculator with slider inputs.
5. **Interventions**: Actionable block-wise roadmap, checklists, urgency badges, estimated payoffs.
6. **Maximum Demand**: System load curve, block contributions at coincident peak, 90% threshold tracker, power factor kVA simulator.
7. **Room Localization (Demo Block)**: Drill-down floor and room load curves, Subset-Sum appliance matching algorithm, Top-1 and Top-3 room localization accuracy.
8. **Model Evaluation**: Metrics table (Precision, Recall, F1, PR-AUC), confusion matrix, detection delay distribution, per-type recall, feature ablation impact, LEAD vs Synthetic domain gap.
9. **Data & Settings**: CSV file uploader, synthetic data regenerator (seed, blocks, noise, anomaly rate), live stream simulator (hour-by-hour step replay), tariff and threshold config editor.
10. **About & Limitations**: Engineering assumptions, LEAD dataset citations, TNEB tariff order references, sensor resolution limitations.

---

## 4. Key Engineering Assumptions (Summary for `docs/ASSUMPTIONS.md`)
1. **Demand Lower-Bound**: Hourly average power (kWh/h = kW) smooths short-duration peaks; actual instantaneous 15-minute maximum demand is $\ge$ hourly average.
2. **Tariff Structure**: Single-tier LT-IIB institutional energy tariff (default ₹7.50/kWh). Demand charges modeled separately.
3. **Appliance Signatures**: Immersion heater: 1.5–2.0 kW; Electric kettle: 1.2–1.5 kW; Ceiling fan: ~75 W; Laptop: ~65 W.
4. **Label Leakage Prevention**: Features for time $t$ only utilize historical data $t' < t$ (lags and past rolling windows).
5. **Alert Deduplication**: Alerts for the same block within a 4-hour window are grouped into a continuous event.

---

## 5. Verification Plan
- **Unit & Integration Tests (`pytest tests/`)**:
  - `test_synth.py`: Validates synthetic schema, 8760 hourly steps per block, non-negative kWh, demo room sums equal block total.
  - `test_features_and_splits.py`: Validates temporal split ordering, lag calculation without future data leakage.
  - `test_models.py`: Validates model training, expected kWh generation, severity bounded [0, 100].
  - `test_savings_demand.py`: Validates waste calculation $\ge 0$, tariff multiplication, MD coincident peak logic.
  - `test_alerts.py`: Validates SQLite CRUD, deduplication, cooldown, 6h escalation triggers.
  - `test_room_localization.py`: Validates subset-sum appliance matching within tolerance band.
- **Visual Validation**:
  - Automated or subagent browser verification of Streamlit dashboard pages.
