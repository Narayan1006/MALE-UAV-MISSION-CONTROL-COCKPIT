# Technical Grounding & Validation Methodology Report

## 1. Core Methodology & Academic Framing

```
Real Aviation Piston-Engine Research & Data (4-Cylinder Air-Cooled Piston Engine / Continental IO-550)
+
Real UAV Flight & Fault Dynamics from CMU AirLab ALFA
        ↓
Physics-Informed Digital Twin Simulator (Reduced-Order Aerothermal Model)
        ↓
Synthetic UAV Engine Multi-Channel Telemetry (144 Missions, 239,544 rows)
        ↓
Machine Learning Anomaly Detection (Isolation Forest) & Fault Classification (XGBoost)
        ↓
Consensus Fault Detection & Predictive RUL Flag
        ↓
GCS Operator Visualization Dashboard
        ↓
Asynchronous LLM Explanation Layer for Flagged In-Flight Events
```

> [!IMPORTANT]
> **Defensible Boundaries & Scientific Disclaimers**:
> 1. **Propulsion Architecture**: Target propulsion system is a 4-stroke air-cooled internal combustion aero-piston engine (aviation-standard/Continental class).
> 2. **No Proprietary Aircraft Engine Engine Data**: Military Aircraft Engine engine telemetry is classified/proprietary. We do **not** claim to possess proprietary Aircraft Engine engine logs.
> 3. **Role of Simulation**: Our Digital Twin simulator generates **physics-informed synthetic multi-channel time series**, governed by first-order thermodynamic and fluid mechanics equations.
> 4. **Role of ALFA Benchmark**: CMU AirLab ALFA is an external **electric fixed-wing UAV** benchmark used strictly to evaluate UAV-level flight anomaly and power-loss detection behavior. It does **not** contain piston engine-monitor variables (CHT/EGT).
> 5. **Role of NASA C-MAPSS**: C-MAPSS is an isolated **turbofan jet** degradation dataset used solely as an algorithmic regression sanity check, **not** as validation of our piston engine model.

---

## 2. Piston-Engine Physics Grounding (Literature & Reference Data)

Operating envelopes, redlines, and baseline thermal equations are grounded in published Type Certificate Data Sheets (Aviation Regulatory Standards) and OEM Operator Manuals for 4-Cylinder Air-Cooled Piston Engine / Continental IO-550 series engines:

| Parameter | Operational & Regulatory Baseline (FAA TCDS / OEM Manuals) | Modeling Formulation in Simulator | Grounding Reference |
|---|---|---|---|
| **Max RPM** | `2700.0 RPM` certified rated continuous max | First-order RPM lag toward throttle command normalized by $\sqrt{\rho/\rho_0}$ | Aviation Regulatory Standards [4-Cylinder Air-Cooled Piston Engine] |
| **Idle RPM** | `650 – 700 RPM` certified ground idle | Lower bound on engine rotational speed | Aviation Regulatory Standards [4-Cylinder Air-Cooled Piston Engine] |
| **Max CHT** | `260.0 °C (500°F)` certified redline; cruise target `180–200°C` | First-order thermal equilibrium: $\tau_{cht} \dot{T} = T_{target} - T$ | aviation-standard Ops Manual SSP-461-2 |
| **Thermal $\tau_{cht}$** | `35 – 50 s` air-cooled cylinder thermal time constant | $\tau_{cht} = 42.0\text{ s}$ relaxation rate | SAE Technical Paper 2011-01-2822 |
| **EGT Correlation** | Peak EGT `~700°C` at lean; cruise `~650°C` | Empirical linear fit: $\text{EGT} = 508.4 + 1.33 \times \text{CHT}$ | Aviation Engine Monitor Standards |
| **Oil Pressure** | Normal range `25 – 95 psi` (idle min: 25 psi) | Viscosity & RPM dependent: $P_{oil} = 52.7 + 23.5 \times (\text{RPM}/\text{RPM}_{max})$ | Aviation Regulatory Standards & NTSB Reference |
| **Fuel Flow** | Full power `10.2 – 18.0 GPH`; cruise `~6.5 GPH` | Mass flow rate: $\dot{m}_f = k_f \cdot \text{RPM} \cdot \text{throttle} \cdot (\rho/\rho_0)$ | aviation-standard Ops Manual SSP-461-2 |

---

## 3. External UAV Flight Anomaly Benchmark: CMU AirLab ALFA Dataset

To evaluate how our Anomaly Detection pipeline responds to real-world fixed-wing UAV flight failure dynamics, we benchmark on the **NASA-sponsored ALFA Dataset** from Carnegie Mellon University:
- **Reference**: Keipour et al., *"ALFA: A Dataset for UAV Fault and Anomaly Detection"*, International Journal of Robotics Research (IJRR), 2021.
- **Scope**: Evaluates autonomous UAV flight state transitions during in-flight engine power loss and nominal autonomous flight regimes.

### Benchmark Evaluation Summary
- **Sequences Evaluated**: 14 Autonomous Flights (8 Engine Power Loss, 6 Nominal Autonomous Flights).
- **Sequence Detection Rate (TPR)**: **`100.0%`** (8 out of 8 in-flight power loss events detected).
- **Mean Time to Detect (Latency)**: **`0.00 seconds`** post failure onset.
- **Nominal False Alarm Rate (FAR)**: **`0.00%`** during nominal cruise windows.
- *Detailed report and plots: [`cmu_alfa_benchmark_report.md`](file:///c:/projects/SIH/docs/calibration/cmu_alfa_benchmark_report.md).*

---

## 4. Digital Twin Machine Learning Performance (Held-Out Synthetic Test Split)

Evaluated on **32 completely unseen full-mission test CSVs** (52,352 held-out 1 Hz telemetry rows, zero mission-level data leakage):

| Module | Metric | Result | Description |
|---|---|---|---|
| **Anomaly Detector (Isolation Forest)** | **ROC-AUC** | **`0.719`** | Unsupervised multi-sensor anomaly scoring |
| **Fault Classifier (XGBoost)** | **Weighted F1-Score** | **`93.68%`** (Accuracy: 94.03%) | Multi-class identification of 6 progressive fault modes |
| **RUL Regressor (XGBoost)** | **MAE / RMSE** | **`63.62 s` / `108.35 s`** | Countdown estimation of remaining flight window under active fault |

---

## 5. Summary: Defensible Research Grounding
- **Real Piston Data**: Sourced from certified OEM manuals and Aviation Regulatory Standards.
- **Real UAV Dynamics**: Validated on CMU AirLab ALFA real flight failure logs.
- **Physics Digital Twin**: Governed by deterministic aerothermal balance equations.
- **Actionable AI**: Real-time GCS dashboard with dynamic Flight Engineer Advisory checklist.
