# Operational & Engineering Assumptions: HostelWatt

This document outlines the explicit assumptions, data limitations, and physics approximations underlying the **HostelWatt** electricity anomaly detection and demand management system.

---

## 1. Electrical & Metering Assumptions

### 1.1 Sampling Frequency & Maximum Demand (MD) Lower-Bound
- **Assumption**: All meter data is recorded as hourly cumulative energy ($\text{kWh}$).
- **Engineering Principle**: Because $1 \text{ kWh}$ consumed over $1 \text{ hour}$ equals an average power of $1 \text{ kW}$, the hourly consumption directly represents average hourly active power ($P_{\text{avg}}$).
- **Critical Limitation**: Real-world Maximum Demand (MD) billings by state utilities (such as TNEB / TANGEDCO) are computed using a **15-minute or 30-minute sliding window integration**. Hourly averaging dampens transient surges and intra-hour peaks. Therefore:
  $$\text{Hourly MD} \le \text{True 15-minute Utility MD}$$
  All MD values computed in this dashboard are explicitly marked as **engineering lower bounds**.

### 1.2 Power Factor (PF) & Apparent Power (kVA)
- **Assumption**: Submeters measure active energy ($\text{kWh}$). The power factor across residential hostel blocks is assumed to be **0.90 lagging** (typical for mixed linear lighting, fan inductive loads, and switched-mode electronics).
- **kVA Computation**: $\text{kVA} = \frac{\text{kW}}{\text{Power Factor}} = \frac{\text{kW}}{0.90}$.
- **Configurability**: The PF parameter can be adjusted dynamically in `config.yaml` or through the Settings dashboard.

---

## 2. Tariff & Financial Savings Assumptions

### 2.1 Tariff Classification
- **Reference**: Tamil Nadu Electricity Regulatory Commission (TNERC) Tariff Order for **TNEB LT Tariff II-B** (applicable to recognized educational institutions and student hostels).
- **Energy Rate**: Set to a base of **₹7.50 per kWh**.
- **Demand Charges**: Set to **₹350.00 per kVA per month** above contracted capacity.
- **Estimated Savings**: Wasted energy is computed as:
  $$\text{Wasted kWh}_t = \max(y_t - \hat{y}_t, 0) \quad \text{for flagged hours } t$$
  $$\text{Financial Loss (INR)} = \text{Wasted kWh} \times \text{Tariff Rate (₹7.50)}$$
  All savings figures represent **estimated avoidable costs**, subject to actual tariff structure and seasonal slab variations.

---

## 3. Behavioral & Environmental Load Dynamics

### 3.1 Diurnal Patterns
- **Morning Peak (06:00 – 09:00)**: Student wake-up, geysers/water heaters, room lighting, personal grooming appliances.
- **Academic Hours (09:00 – 17:00)**: Hostels operate at base/standby load (~20–30% of peak) with empty rooms.
- **Evening Peak (18:00 – 23:00)**: Students return, laptop charging, ceiling fans, lighting, common room entertainment.
- **Night Floor (00:00 – 06:00)**: Dormant sleep state, baseline ventilation and emergency corridor lighting.

### 3.2 Calendar & Day-Type Regimes
- **Regular Days**: Standard academic weekday/weekend cycle.
- **Exam Periods**: Elevated load between 00:00 and 04:00 due to late-night study lamps, laptops, and group sessions.
- **Holidays / Vacations**: Occupancy drops to ~10–20% (foreign/research students remaining). Load drops proportionally.

### 3.3 Weather Response
- Outside ambient temperature is modeled from 20°C (winter night) to 39°C (summer afternoon).
- Cooling demand (ceiling fans, occasional coolers/ventilation) scales non-linearly with cooling degree hours above 26°C.

---

## 4. Anomaly Characterization & Detection

| Anomaly Type | Electrical Signature | Primary Duration | Root Cause / Real-World Analogue |
| :--- | :--- | :--- | :--- |
| **`appliance_spike`** | $+1.0\text{ to }+2.0\text{ kW}$ sudden step-up | $1\text{ to }4\text{ hours}$ | Unauthorized high-wattage device (electric kettle, immersion rod, hot plate) |
| **`left_on`** | $+0.8\text{ to }+1.5\text{ kW}$ flat continuous addition | $8\text{ to }12\text{ hours}$ | Corridor/common room high-bay lights or exhaust left on during holiday or overnight |
| **`wiring_drift`** | $+0.5\%/\text{day}$ gradual baseline increase | $2\text{ to }4\text{ weeks}$ | Deteriorating insulation, sub-surface leakage, partial neutral grounding fault |
| **`intermittent_fault`** | Oscillating rapid spikes ($\pm 1.2\text{ kW}$) | Burst cycles | Loose busbar connection, arcing switchgear, chattering contactor |
| **`meter_fault`** | Zero reading ($0.0\text{ kW}$) or stuck flatline | $4\text{ to }12\text{ hours}$ | CT/PT disconnection, RS-485 bus failure, frozen microcontroller firmware |

---

## 5. Room Localization & Subset-Sum Algorithm

### 5.1 Hierarchical Metering Topography
- The demo block (`Block_A`) contains **3 floors** and **8 rooms per floor** (total 24 rooms).
- **Physical Law of Conservation**:
  $$\sum_{r=1}^{8} \text{Room}_{f, r}(t) = \text{Floor}_f(t), \quad \sum_{f=1}^{3} \text{Floor}_f(t) = \text{Block}(t)$$

### 5.2 Subset-Sum Matching with Tolerance Band
- When an abnormal surplus $\Delta P$ is observed at the floor or block level, the localization engine evaluates subsets of room-level step deviations against known appliance signatures:
  - Ceiling Fan: $\approx 75\text{ W}$ ($0.075\text{ kW}$)
  - Electric Kettle: $\approx 1.35\text{ kW}$
  - Room / Water Heater: $\approx 1.50 - 1.80\text{ kW}$
- A tolerance band of $\pm 12\%$ is permitted to account for voltage fluctuations ($210\text{V} - 240\text{V}$).

---

## 6. Domain Gap: LEAD 1.0 vs. Residential Hostels

- **LEAD 1.0 Dataset**: Collected primarily from **commercial and academic buildings** (offices, computer laboratories, lecture halls) in the Global North. Characteristics include strict centralized HVAC schedules, low weekend occupancy, and industrial 3-phase machinery.
- **Hostel Reality**: Highly decentralized, occupant-driven, irregular sleep cycles (especially during exams), seasonal monsoon/summer shifts, and unauthorized personal resistive heating loads.
- **Application**: Models trained on commercial data exhibit higher false positive rates when applied to student living quarters without contextual adjustment features.
