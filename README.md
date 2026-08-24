# MALE UAV Digital Twin — SIH26054

**AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs**

> **PS ID:** SIH26054 | **Org:** DRDO / IDEX | **Theme:** Robotics and Drones | **Deadline:** 20 Sept 2026

---

## What this system does

Replaces threshold-based ("alert only after failure") engine monitoring with a predictive Digital Twin that:

- Mirrors a MALE UAV's aero-piston engine in real time using a physics-informed simulator
- Monitors 8 health parameters: RPM, CHT, EGT, oil pressure/temp, fuel flow, vibration, battery/alternator, injection timing
- Detects and predicts 6 fault types: misfire, injector degradation, cooling failure, lubrication loss, sensor drift, vibration anomaly
- Estimates Remaining Useful Life (RUL) and degradation trend
- Simulates 4 mission scenarios: endurance, high altitude, hot weather, rapid throttle
- Displays everything on a GCS-style operator dashboard

---

## Architecture

```
Physics-Informed Engine Simulator
        ↓
ML Core (live, low-latency)
  → Anomaly Score (Isolation Forest)
  → Fault Type + Confidence (XGBoost)
  → RUL Estimate (XGBoost Regressor)
        ↓                          ↓
  FastAPI Layer              LLM Explain Layer (async, advisory only)
        ↓                          ↓
              Dashboard (GCS-style operator UI)
```

The simulator is designed to be **swappable** for real CAN/ECU telemetry without changing anything downstream.

---

## Project structure

```
male-uav-digital-twin/
├── simulator/          # physics-informed engine simulator
├── data/
│   ├── raw/            # generated mission CSVs (train / val / test)
│   └── processed/      # windowed, feature-engineered versions
├── ml/                 # training scripts and saved models
├── backend/            # FastAPI app
├── dashboard/          # operator UI
├── docs/               # data dictionary, PS breakdown, architecture
├── generate_dataset.py # Phase 2 — run this to generate training data
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate full training dataset (~150-200 mission CSVs)
python generate_dataset.py

# 3. Validate data (Phase 3) — see ml/validate_data.py
# 4. Train models (Phase 4-5) — see ml/train_anomaly_detector.py

# Run the API (Phase 6)
uvicorn backend.main:app --reload
```

---

## Important disclaimer

This system uses a **reduced-order, physics-informed simulator** as the data source — not proprietary real MALE UAV engine data (which is classified). The PS explicitly permits simulated datasets for demonstration. All physical constants are calibrated to Rotax-912-class engines as defensible ballpark anchors. The system is described as a **prototype**, not a certified RUL predictor.

---

## Build roadmap

See [`build_roadmap.md`](docs/build_roadmap.md) for the full phase-by-phase plan with acceptance criteria.

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup — folders, git, requirements | ✅ Done |
| 2 | Dataset generation — 150-200 mission CSVs | 🔄 In progress |
| 3 | Data validation with plots | ⬜ |
| 4 | Baseline anomaly detector (Isolation Forest) | ⬜ |
| 5 | Fault classifier + RUL regressor (XGBoost) | ⬜ |
| 6 | FastAPI layer | ⬜ |
| 7 | Operator dashboard | ⬜ |
| 8 | LLM explain layer | ⬜ |
| 9 | Polish + stretch goals | ⬜ |
