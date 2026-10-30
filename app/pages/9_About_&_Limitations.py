"""Page 9: About, Methodologies & Operational Limitations.

Documents:
- System design rationale and physics foundations
- Transparent reporting of the domain gap between commercial buildings and residential hostels
- Distinction between direct measurements and machine learning baseline estimates
- Regulatory citations and data acknowledgments
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import yaml

from app.ui_utils import apply_custom_css, render_header

st.set_page_config(page_title="About & Limitations - HostelWatt", page_icon="📖", layout="wide")
apply_custom_css()
render_header(
    title="System Assumptions & Engineering Disclaimers",
    subtitle="Documentation of mathematical models, sensor resolution limits, domain gap considerations, and citations",
    icon="📖"
)

with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

st.markdown(
    """
    ### 🔬 1. Estimates vs. Direct Measurements

    It is vital for facilities engineers, hostel wardens, and administrative authorities to differentiate
    between **physically metered observations** and **inferred algorithmic estimates**:

    - **Measured Parameters**:
      - Total hourly energy consumption ($\text{kWh}$) at block submeters.
      - Ambient campus outside dry-bulb temperature ($^\circ\text{C}$).
      - Academic calendar date and scheduled day types (`regular`, `exam`, `holiday`).
    - **Estimated / Modeled Parameters**:
      - **Expected Baseline Load ($\hat{y}_t$)**: Supervised regression forecast derived from LightGBM contextual feature trees.
      - **Avoidable Energy Waste**: $\max(y_t - \hat{y}_t, 0)$ on flagged hours.
      - **Financial Avoidable Loss (INR)**: Product of wasted kWh and configured single-tier tariff rate.
      - **Maximum Demand (MD)**: Lower-bound estimation based on 60-minute averaging.

    ---

    ### ⚠️ 2. Maximum Demand Hourly Integration Disclaimer

    State electric distribution utilities (such as **TNEB / TANGEDCO**) calculate commercial Maximum Demand penalties
    using **15-minute or 30-minute sliding window integration**:
    $$\text{True Utility MD}_{15\text{min}} \ge \text{HostelWatt MD}_{60\text{min}}$$
    Short transient spikes (e.g., motor inrush, synchronized geyser switching) that last 10–20 minutes
    are smoothed out in hourly data. Therefore, all Maximum Demand values reported in this dashboard represent an
    **engineering lower bound**.

    ---

    ### 🏢 3. Domain Gap: Commercial LEAD 1.0 vs. Residential Hostels

    When evaluating algorithms trained on public benchmarks such as **LEAD 1.0**:
    1. **LEAD 1.0 Characteristics**:
       - Sourced primarily from **commercial offices, computer centers, and institutional administration buildings** in the Global North.
       - Dominated by centralized building management systems (BMS), strict 08:00–18:00 weekday occupancy schedules, and low weekend consumption.
    2. **Hostel Characteristics**:
       - Decentralized, occupant-driven consumption with non-linear sleep cycles.
       - Late-night study surges (00:00–04:00) during mid-term and final examination periods.
       - Summer vacation vacancy with ~15% residual research occupancy.
       - High prevalence of unauthorized personal heating coils (immersion rods, electric kettles) with fast step profiles.
    3. **Operational Consequence**:
       - Models trained on commercial datasets without contextual adjustment factors suffer from unacceptable false alarm rates during exams and student holidays.

    ---

    ### 📚 4. Citations & References

    1. **LEAD 1.0 Energy Anomaly Dataset**:
       - *GitHub Repository*: [https://github.com/samy101/lead-dataset](https://github.com/samy101/lead-dataset)
       - Citation: Large-scale Energy Anomaly Detection dataset with labeled hourly commercial smart meter readings.
    2. **Electricity Tariff Structure**:
       - *Tamil Nadu Electricity Regulatory Commission (TNERC)* Comprehensive Tariff Order.
       - Schedule: **LT Tariff II-B** (Recognized Educational Institutions, Student Hostels).
       - Energy Charge: Base ₹7.50 / kWh (configurable in `config.yaml`).
    3. **Appliance Electrical Signatures**:
       - Immersion Water Heating Rod: $1.5\text{ to }2.0\text{ kW}$ (purely resistive).
       - Electric Tea Kettle: $1.2\text{ to }1.5\text{ kW}$ (resistive heating element).
       - BLDC / Induction Ceiling Fan: $65\text{ to }80\text{ W}$ ($0.065\text{ to }0.08\text{ kW}$).
       - Laptop Switched-Mode Power Supply: $50\text{ to }85\text{ W}$.
    """
)
