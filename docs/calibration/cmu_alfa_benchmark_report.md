# External UAV Benchmark Analysis: CMU AirLab ALFA Dataset

## 1. Executive Summary & Forensic Determination

> [!IMPORTANT]
> **Forensic Audit & Scientific Status**:
> **Verdict: PARTIALLY VALID — External UAV Flight & Fault Dynamics Reference Only.**
> 
> Direct quantitative validation of our 14-channel aero-piston model against the ALFA dataset is **not scientifically valid** due to domain and feature differences:
> 1. **Propulsion Architecture**: ALFA was recorded on the *Carbon-Z T-28*, an **electric** fixed-wing autonomous UAV testbed. It does **not** contain piston engine thermodynamic telemetry (no CHT, EGT, oil pressure, or oil temperature).
> 2. **Telemetry Scope**: ALFA contains flight dynamics (Pixhawk/MAVROS IMU, GPS, attitude, control surface actuations, battery voltage/current, and ground-truth failure status).
> 3. **Methodological Role**: ALFA is used as an **external qualitative and architectural reference** for UAV in-flight failure progression and contingency reaction timelines, **not** as a direct quantitative telemetry benchmark for our piston engine ML models.
> 
> *All fabricated quantitative claims (e.g., "100% TPR / 0.00s latency / 0.00% FAR") derived from synthetic mapping have been formally deleted.*

---

## 2. CMU AirLab ALFA Dataset Specification

- **Authors**: Azarakhsh Keipour, Mohammadreza Mousaei, Sebastian Scherer (AirLab, Carnegie Mellon University)
- **Publication**: *"ALFA: A Dataset for UAV Fault and Anomaly Detection"*, International Journal of Robotics Research (IJRR), 2021.
- **Grant Sponsorship**: NASA Grant `NNX17CL06C` (*Contingency Detection and Reaction for Autonomous Unmanned Aircraft*).
- **Repository URL**: [http://theairlab.org/alfa-dataset](http://theairlab.org/alfa-dataset) (DOI: `10.1184/R1/12707963`)
- **Tools**: Integrated with CMU AirLab's official toolkit (`tools/alfa_tools/alfa-dataset-tools-master`).

### Actual Telemetry Channels in ALFA Processed Sequences
| ALFA Topic / Signal | Description | Present in ALFA | Usable for Piston CHT/EGT? |
|---|---|---|---|
| `mavros-imu-data` | 3-axis angular velocity and linear acceleration | ✅ Yes | ❌ No (Flight dynamics only) |
| `mavros-nav_info-velocity` | True airspeed and ground speed vectors | ✅ Yes | ❌ No |
| `mavros-battery` | Electric pack voltage, current draw, remaining % | ✅ Yes | ❌ No (Electric battery, not avgas) |
| `mavros-rc-in` / `rc-out` | Actuator control PWM signals (aileron, rudder, throttle) | ✅ Yes | ❌ No (Motor PWM only) |
| `failure_status-engine` | Binary flag for commanded power-loss injection | ✅ Yes | ❌ Ground truth label only |
| *CHT, EGT, Oil P, Oil T, Fuel Flow* | *Piston Engine Monitor Channels* | ❌ **NOT PRESENT** | ❌ **N/A (Electric UAV)** |

---

## 3. Clean Separation of Validation Layers

```
LAYER 1: Piston-Engine Physics Grounding
  ├── Lycoming IO-360 / Continental IO-550 certified specifications (FAA TCDS 1E10)
  └── NTSB Docket ERA21LA099 (1 Hz Garmin G1000 flight log baseline)
        ↓
LAYER 2: Physics-Informed Digital Twin Simulator (EngineSimulator)
  ├── 144 Mission CSVs (239,544 rows) across 4 mission profiles
  └── 6 Progressive Fault Modes (Injector, Cooling, Lubrication, Misfire, Drift, Vibration)
        ↓
LAYER 3: Machine Learning Evaluation (Held-Out Synthetic Test Split)
  ├── Isolation Forest Anomaly Detection (ROC-AUC: 0.719)
  ├── XGBoost Fault Classification (Weighted F1: 93.68%, Accuracy: 94.03%)
  └── XGBoost RUL Regression (MAE: 63.62 s, RMSE: 108.35 s)
        ↓
LAYER 4: External Literature References
  ├── CMU AirLab ALFA: Reference for UAV-level failure onset & reaction timelines
  └── NASA C-MAPSS FD001: Supplementary algorithmic sanity check for RUL regression on turbofans
```

---

## 4. Verified Machine Learning Metrics (Held-Out Synthetic Test Split)

Evaluated strictly on **32 completely unseen full-mission test CSVs** (52,352 held-out 1 Hz telemetry rows, zero mission-level data leakage):

### A. Fault Classification Performance (XGBoost Test Set)
| Class | Precision | Recall | F1-Score | Test Support (Frames) |
|---|---|---|---|---|
| **Cooling** | 0.9631 | 0.8360 | **0.8950** | 3,182 |
| **Injector** | 0.9984 | 0.9928 | **0.9956** | 3,871 |
| **Lubrication** | 0.9812 | 0.9622 | **0.9716** | 3,044 |
| **Misfire** | 0.9969 | 0.8782 | **0.9338** | 4,336 |
| **None (Healthy)** | 0.9121 | 0.9999 | **0.9540** | 28,742 |
| **Sensor Drift** | 0.9550 | 0.6205 | **0.7522** | 4,719 |
| **Vibration** | 0.9899 | 0.9681 | **0.9789** | 4,458 |
| **Weighted Average** | **0.9431** | **0.9403** | **0.9368** | **52,352** |

### B. Remaining Useful Life (RUL) Regressor Performance
- **Mean Absolute Error (MAE)**: **`63.62 seconds`**
- **Root Mean Squared Error (RMSE)**: **`108.35 seconds`**

---

## 5. Summary Conclusion for Defense / SIH Jury
1. We present an **end-to-end physics-informed Digital Twin prototype** for aero-piston UAV engines.
2. Physics constants are anchored in **certified Lycoming/Continental specifications**.
3. ML models are validated rigorously on a **held-out synthetic multi-mission dataset**.
4. External datasets (CMU ALFA, NASA C-MAPSS) are cited strictly for their legitimate scientific scope:
   - ALFA as a UAV in-flight failure transition dynamics reference.
   - C-MAPSS as an isolated algorithmic RUL regression sanity check.
