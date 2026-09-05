# AeroTwin — Physics-Informed Digital Twin for Aircraft Engine Health

[![Live Web Demo](https://img.shields.io/badge/Demo-Live%20on%20Render-brightgreen?style=flat-square&logo=render)](https://sih26054-male-uav-digital-twin.onrender.com)
[![Docker Ready](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square&logo=docker)](file:///Dockerfile)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow?style=flat-square&logo=python)](file:///requirements.txt)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)](file:///backend/main.py)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20TreeSHAP-orange?style=flat-square)](file:///ml/train_models.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](file:///LICENSE)

**A predictive, physics-calibrated, edge-deployable Digital Twin that monitors 14 channels of high-frequency aircraft engine telemetry in real time, diagnoses 6 distinct failure modes before onset, estimates Remaining Useful Life (RUL), and delivers explainable engineering advisories.**

🌐 **Live Flight Deck Operator Dashboard:** [https://sih26054-male-uav-digital-twin.onrender.com](https://sih26054-male-uav-digital-twin.onrender.com)  
📦 **GitHub Repository:** [https://github.com/Narayan1006/Aircraft-Engine-Flight-Deck](https://github.com/Narayan1006/Aircraft-Engine-Flight-Deck)

---

## 🎯 What This System Does

Traditional cockpit monitoring relies on static threshold idiot-lights that only illuminate after severe damage or flameout occurs. **AeroTwin replaces reactive alarms with continuous predictive health management:**

1. **Digital Twin Physics Engine:** Runs a reduced-order thermodynamic simulator of a 4-cylinder air-cooled aero engine at 1 Hz, modeling combustion heat release, convective cooling, friction wear, and atmospheric lapse ($15^\circ\text{C}$ to $-35^\circ\text{C}$, sea-level to 8,000m).
2. **Data Quality Guard:** Intercepts raw telemetry to filter statistical noise, detect sensor drift, and impute missing channels during simulated packet dropouts (tested up to 30% packet loss).
3. **Multi-Stage AI Diagnostics:**
   - **Anomaly Detection:** Unsupervised Isolation Forest detecting multivariate telemetry divergence (ROC-AUC `0.7188`).
   - **Fault Classification:** 150-tree XGBoost classifier identifying 6 distinct failure modes with **93.68% weighted F1** (94.03% test accuracy).
   - **RUL Forecasting:** XGBoost regressor predicting time-to-failure on active degradation trajectories (**MAE: 63.62 seconds**).
4. **Explainable AI (TreeSHAP):** Real-time Shapley attribution ranking the top-3 physical sensor drivers (e.g. rolling CHT mean, vibration variance) that caused the AI prediction.
5. **Adaptive Cockpit Advisories:** Automated flight engineer action plans, descent profiles, and emergency transponder **Squawk 7700** activation gated strictly to critical emergency states.

---

## 🏗️ System Architecture

```
[Aircraft Engine / Sensor Harness]
                │  14 Channels @ 1 Hz (RPM, CHT, EGT, Oil P/T, Fuel, Vib, Batt, Inj Timing...)
                ▼
  [Data Quality Guard Layer]
                │  Outlier filtering, sensor drift detection, rolling median imputation
                ▼
 [MAVLink v2 Framing (Msg ID 50001)]
                │  56-byte binary payload, CRC-16 MCRF4XX checksum
                ▼
[RFD900x 900 MHz FHSS RF Link]
                │  50 ms Gaussian latency, Bernoulli packet loss simulation
                ▼
 [Raspberry Pi 4 Edge Compute]
                │  Lightweight Isolation Forest inference (5.75 ms host / ~23.0 ms Pi 4)
                ▼
[Ground Station FastAPI Server]
 ┌──────────────┼──────────────┐
 ▼              ▼              ▼
[XGBoost Clf]  [RUL Regressor] [TreeSHAP XAI]
(93.68% F1)    (63.62s MAE)    (Top-3 features)
 └──────────────┬──────────────┘
                ▼
[AeroTwin Flight Deck UI (Browser)]
```

---

## 📊 Key Verified Metrics

All figures below are directly measured from this codebase:

| Metric Category | Measure | Verified Result | Verification Source |
|---|---|:---:|---|
| **Fault Classification** | Weighted F1-Score | **93.68%** | `ml/models/phase5_metrics.json` |
| **Fault Classification** | Test Set Accuracy | **94.03%** | `docs/DEPLOYMENT.md` |
| **RUL Prediction** | Mean Absolute Error (MAE) | **63.62 s** (~1.06 min) | `ml/models/phase5_metrics.json` |
| **RUL Prediction** | Root Mean Square Error (RMSE) | **108.35 s** (~1.80 min) | `ml/models/phase5_metrics.json` |
| **Anomaly Detection** | ROC-AUC | **0.7188** | `ml/models/anomaly_metrics.json` |
| **Edge Anomaly Latency** | Isolation Forest on Host CPU | **5.75 ms** | `backend/pipeline_test.py` |
| **Edge Compute (Pi 4)** | Estimated Cortex-A72 Inference | **~23.0 ms** | `backend/pipeline_test.py` |
| **Telemetry Budget Pass** | Frames under 100 ms threshold | **100.0%** | `backend/pipeline_test.py` |
| **Hardware BOM Cost** | Complete Sensor Harness + Compute | **$833 – $1,057** | `docs/DEPLOYMENT.md` |
| **Platform Cost Ratio** | Add-on vs $12k–$100k Airframe | **< 1% to 8%** | `docs/DEPLOYMENT.md` |

---

## 🔍 Supported Failure Modes

AeroTwin models and accurately classifies **6 in-flight fault modes** plus nominal baseline:

* **`cooling`**: Cylinder cooling fin blockage or cowl flap failure $\rightarrow$ thermal runaway on CHT & oil temp.
* **`injector`**: Fuel injector nozzle clogging $\rightarrow$ lean burn temperature runaway and EGT delta spikes.
* **`lubrication`**: Oil pump failure / line puncture $\rightarrow$ rapid oil pressure drop with bearing heat rise.
* **`misfire`**: Magneto / spark plug ignition degradation $\rightarrow$ rapid rotational vibration & unburnt fuel EGT drop.
* **`sensor_drift`**: Thermocouple calibration drift $\rightarrow$ isolated channel offset detected via consensus checks.
* **`vibration`**: Crankshaft bearing wear or propeller balance fault $\rightarrow$ high-g harmonic acceleration.
* **`none`**: Fully nominal certified operating envelope.

---

## 🚀 Quick Start

### Option 1 — Run with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Narayan1006/Aircraft-Engine-Flight-Deck.git
cd Aircraft-Engine-Flight-Deck

# Build and run container
docker build -t aerotwin .
docker run -p 8000:8000 aerotwin
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### Option 2 — Local Python Environment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the deployment pipeline test (MAVLink + RF + Edge compute)
python backend/pipeline_test.py

# 3. Launch the FastAPI server and Flight Deck UI
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 📂 Project Structure

```
Aircraft-Engine-Flight-Deck/
├── backend/
│   ├── main.py                  # FastAPI server, endpoints, and static file mount
│   ├── advisory_engine.py       # Adaptive mission planner & flight checklist generator
│   ├── explainability.py        # TreeSHAP local & global attribution explainer
│   ├── data_quality_guard.py    # Outlier filter, drift detector & imputation guard
│   ├── edge_compute.py          # Raspberry Pi 4 edge compute simulator
│   ├── mavlink_interface.py     # MAVLink v2 binary framing (Msg ID 50001, CRC-16)
│   ├── rf_link_simulator.py     # RFD900x 900MHz telemetry radio channel simulator
│   └── pipeline_test.py         # End-to-end 6-stage deployment benchmark
├── dashboard/
│   ├── index.html               # Aerospace Flight Deck cockpit interface
│   ├── css/style.css            # Dark mode aerospace design tokens & styles
│   └── js/dashboard.js          # Real-time telemetry, charts & XAI orchestrator
├── data/
│   └── raw/                     # 144 generated mission CSVs (239,544 rows)
├── ml/
│   ├── models/                  # Serialized XGBoost, Isolation Forest, and scaler weights
│   ├── calibrate_simulator.py   # Benchmark physics calibration against ALFA dataset
│   ├── train_models.py          # XGBoost classifier and RUL regressor training
│   └── train_anomaly_detector.py # Isolation Forest unsupervised training
├── docs/
│   ├── DEVPOST_SUBMISSION.md    # Ready-to-paste Devpost hackathon submission
│   ├── DEMO_SCRIPT.md           # Timed 2:45 video presentation & narration script
│   ├── PROJECT_SUMMARY.md       # Executive technical brief & metrics catalog
│   ├── DEPLOYMENT.md            # Hardware BOM ($833–$1,057), RF specs & roadmap
│   └── calibration/report.md    # Thermodynamic validation against benchmark datasets
├── Dockerfile                   # Production container definition
├── render.yaml                  # Render PaaS deployment configuration
├── requirements.txt             # Locked dependencies
└── README.md                    # Project documentation
```

---

## 📡 Hardware Bill of Materials (BOM)

| # | Component | Function | Sourced Market Price |
|---|---|---|:---:|
| 1 | **K-type Thermocouples + MAX31855 (x4)** | Cylinder Head Temp (CHT, 0–300°C) | $24 – $40 |
| 2 | **K-type Thermocouple + MAX31855 (x1)** | Exhaust Gas Temp (EGT, 0–1200°C) | $8 – $12 |
| 3 | **Pressure Transducer (0–100 psi)** | Oil Pressure Transducer (0.5–4.5V) | $20 – $30 |
| 4 | **MEMS Accelerometer (ADXL345)** | 3-Axis Engine Vibration ($\pm 16g$) | $3 – $5 |
| 5 | **Hall Effect Flow Sensor** | Fuel Flow Rate (1–30 L/min) | $12 – $18 |
| 6 | **Voltage Divider + ADS1115 ADC** | Alternator & Battery Voltage | $5 – $8 |
| 7 | **Raspberry Pi 4 Model B (8 GB RAM)** | Onboard Edge Compute Node | $80 – $95 |
| 8 | **Pi 4 Aluminum Passive Armor Case** | Vibration & Thermal Protection | $10 – $15 |
| 9 | **32 GB High-Endurance microSD** | OS & Model Weights Storage | $8 – $12 |
| 10 | **RFD900x Telemetry Radio Pair** | 900 MHz Air-to-Ground Data Link (40 km) | $350 – $400 |
| 11 | **Pixhawk 6C Autopilot** | MAVLink Telemetry Coordinator | $280 – $320 |
| 12 | **Aviation Shielded Wiring Harness** | Noise Shielding & Power Regulation | $33 – $52 |
| | **Total Hardware Add-on Cost** | **Sensor Harness + Compute + RF Link** | **$833 – $1,057** |

---

## 📜 Submission & Documentation Links

* 📄 **Devpost Submission:** [`docs/DEVPOST_SUBMISSION.md`](docs/DEVPOST_SUBMISSION.md)
* 🎙️ **Demo Video Script (2:45 Timed):** [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)
* 📊 **Executive Project Summary:** [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md)
* 🛠️ **Deployment & Hardware Architecture:** [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)

---

## 🛡️ Honesty Statement & Simulation Scope

* **Simulation Validated:** Thermodynamic physical modeling, ML training & validation on 239,544 rows of synthetic mission data, MAVLink v2 binary packing with CRC-16 checksums, RF latency modeling, and local edge CPU inference timing.
* **Requires Real Flight Hardware:** Long-term thermal throttling under engine cowling heat, physical sensor EMI coupling, and formal DO-178C/DO-254 aviation certification.
