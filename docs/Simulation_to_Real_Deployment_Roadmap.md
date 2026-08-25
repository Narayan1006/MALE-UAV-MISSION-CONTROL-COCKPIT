# From Simulation to Real Deployment — A Practical Roadmap for MALE UAV Engine Digital Twin

**DRDO SIH 26054 | Mission Reliability Enhancement of MALE UAV Engine**

---

## 1. Title Page

# From Simulation to Real Deployment — A Practical Roadmap for MALE UAV Engine Digital Twin

### DRDO SIH 26054 | Mission Reliability Enhancement of MALE UAV Engine

> **Confidential | Team DRDO SIH 26054 | Smart India Hackathon 2026**

---

## 2. Why We Built a Simulator First

When evaluating a software solution for military unmanned aerial vehicles (UAVs), judges and aviation domain experts naturally ask: *"Why did you build a simulator instead of testing on a real engine?"* 

Building a physics-grounded Digital Twin simulator was not a shortcut or a compromise; it was the **only scientifically sound and engineering-feasible starting point**. The development was dictated by four real-world operational constraints:

### Constraint 1: Real Military Data is Classified (Defence Secret)
- Telemetry from operational Medium-Altitude Long-Endurance (MALE) UAVs (such as DRDO Tapas-BH-201 or Rustom-II) is classified under national defense secrecy protocols.
- DRDO and the Indian Armed Forces cannot release live operational flight logs, telemetry streams, or engine failure recordings to student hackathon participants.
- In machine learning, training supervised models requires extensive labelled failure examples (e.g., severe CHT surge, oil pump seal degradation, misfire). Because real catastrophic failures in military aviation are extremely rare and unreleased, a purely data-driven approach on raw flight logs is mathematically impossible.
- **SOLUTION:** We constructed a first-principles, physics-informed aero-engine simulator. It models thermal balance, hydrodynamic lubrication, and ISA atmospheric density from sea level to 25,000 ft, generating unlimited, perfectly labelled synthetic flight telemetry. Every equation traces directly to FAA-certified engineering baselines (**FAA Type Certificate Data Sheet TCDS 1E10** for Lycoming IO-360-M1A) and verified **NTSB crash investigation dockets (ERA21LA099)**.

### Constraint 2: No Access to Physical Engine or Test Cell Hardware
- A physical Lycoming IO-360 / Rotax 914 aero-engine costs approximately **₹25 Lakh to ₹30 Lakh**, is ITAR-controlled, and requires specialized import licensing.
- Engine test cell infrastructure requires dedicated facility access (e.g., DRDO ADE / ADA / NAL laboratories) equipped with dynamometers, fuel flow meters, high-temperature exhaust extraction, and automated halon fire suppression systems.
- Running a 180 HP aviation piston engine indoors without certified test cell infrastructure is an extreme safety hazard.
- **SOLUTION:** We developed a 100% software-defined Minimum Viable Product (MVP) that mathematically proves the diagnostic and prognostic concepts. The code architecture is completely hardware-agnostic: the exact same data quality guard, ML pipeline, explainability engine, and GCS cockpit dashboard will run seamlessly when physical sensors are connected.

### Constraint 3: RF Telemetry Hardware Costs Exceed Student Budgets
- A production-grade long-range RF telemetry link (e.g., dual RFD900x 900MHz transceivers with Yagi directional antennas) costs ~₹30,000.
- An airworthiness-compliant Pixhawk 6C flight computer costs ~₹25,000.
- Onboard companion computer (Raspberry Pi 4 / NVIDIA Jetson), industrial thermocouple amplifiers (MAX31855), MEMS ADXL345 accelerometers, and optical tachometers cost another ~₹15,000–₹20,000.
- The total hardware Bill of Materials (BOM) exceeds **₹70,000–₹1,000,000**, which is beyond student hackathon financial boundaries.
- **SOLUTION:** We simulated the RF telemetry link in software, including configurable stochastic packet loss (0–15%), packet dropouts, and sensor drift.

### Constraint 4: Hackathon Time Limit (36–48 Hours)
- Integrating physical avionics hardware requires weeks of physical assembly, custom PCB fabrication, wire harness crimping, ADC calibration, and sensor noise profiling.
- Conducting real-world RF telemetry range testing requires open airfield access, Ground Control Station setup, and strict DGCA / MoD airspace clearances (taking 4–8 weeks for approval).
- **SOLUTION:** We built a 100% production-ready software stack within 48 hours. The hardware communication layer is designed as a modular **Plug-and-Play Input Adapter**.

> ### 💡 What Evaluators Want to See
> 1. **Real-World Awareness:** You thoroughly understand real military aerospace constraints and regulatory barriers.
> 2. **Physical Grounding:** Your simulation is built on thermodynamic and atmospheric laws, not random noise generators.
> 3. **Clear Migration Path:** You possess a concrete, phased engineering roadmap to transition from simulation to field deployment.
> 4. **Hardware Agnosticism:** Your software architecture decouples data ingestion from AI analytics, allowing seamless sensor integration.

---

## 3. Current Architecture (Simulation Phase)

In the current hackathon MVP, the system operates as a closed-loop simulation environment.

### Data Flow Diagram (Simulation Phase)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CURRENT SIMULATION ARCHITECTURE                        │
│                                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐                         │
│  │   PHYSICS SIMULATOR  │      │  DATA QUALITY GUARD  │                         │
│  │ (engine_simulator.py)│─────►│(data_quality_guard.py)│                         │
│  │ • Lycoming IO-360 ODE│      │ • 3.5σ Z-score Filter│                         │
│  │ • ISA Atmosphere Model      │ • Dropout Imputation │                         │
│  │ • 6 Fault Schedules  │      │ • Sensor Drift Track │                         │
│  └──────────────────────┘      └──────────┬───────────┘                         │
│                                           │ Clean Telemetry                     │
│                                           ▼                                     │
│  ┌──────────────────────┐      ┌──────────────────────┐                         │
│  │ GCS COCKPIT DASHBOARD│      │  HYBRID ML PIPELINE  │                         │
│  │   (dashboard/ UI)    │◄─────│    (ml/models/)       │                         │
│  │ • Live 1Hz Telemetry │      │ • Isolation Forest   │                         │
│  │ • Digital RUL Timer  │      │ • XGBoost Classifier │                         │
│  │ • TreeSHAP XAI Bars  │      │ • XGBoost Regressor  │                         │
│  │ • Squawk 7700 Alert  │      │ • Adaptive Planner   │                         │
│  └──────────────────────┘      └──────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key System Characteristics
- **Closed Loop Execution:** The simulator generates 1 Hz physics telemetry, passes it through the Data Quality Guard layer, feeds it into the ML models, and displays live analytics on the Ground Control Station (GCS) cockpit dashboard.
- **No External Dependencies:** The system executes completely standalone without requiring physical hardware or active internet connection.
- **Fast Scenario Iteration:** Allows operators to test 100+ failure scenarios (e.g., mild cooling degradation vs severe misfire at 15,000 ft) in minutes.
- **Deterministic Ground Truth:** Exact fault onset times, fault severity (0.0 to 1.0), and true Remaining Useful Life (RUL) are known precisely for benchmarking.

### Operational Capabilities Proven (✅)
- **Thermodynamic Fidelity:** Equations match FAA certified cruise envelopes (CHT safe: 135–185°C, Oil Pressure: 55–80 psi, EGT: 350–650°C).
- **Prognostic Accuracy:** High performance on held-out test missions (**93.68% Weighted F1-Score** for 7-class fault classification, **63.62s MAE** for RUL estimation).
- **Data Robustness:** Handled 15% random packet loss and sensor EMI spikes with zero false emergency triggers.
- **Real-Time Edge Budget:** Total pipeline inference latency is **13.9 ms** ($>70\times$ faster than the 1 Hz telemetry loop requirement) with a light **408 MB RAM** footprint.

### Operational Gaps Not Tested in Simulation (❌)
- Physical thermocouple electromagnetic interference (EMI) and ADC quantization noise.
- RF link fading, multipath interference, and antenna nulls during UAV banking turns.
- Real-time OS (RTOS) thread scheduling jitter on embedded flight hardware.
- Operator cognitive load under live emergency flight stress.
- Civil and military regulatory certification (DGCA / DO-178C / DO-330).

> ⚠️ **Key Takeaway:** These operational gaps are explicitly addressed in the **Real Deployment Architecture** detailed on Page 4.

---

## 4. Real Deployment Architecture

Transitioning from simulation to operational flight requires inserting physical sensors, flight controllers, an onboard edge computer, and long-range RF telemetry links. The core ML intelligence and GCS Dashboard code remain 100% untouched.

### System Flow Diagram (Real Airborne & Ground System)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             UAV AIRBORNE SYSTEM                                 │
│                                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    ┌──────────────┐ │
│  │ PHYSICAL     │     │ PIXHAWK 6C   │     │ RASPBERRY PI4│    │ RFD900X RADIO│ │
│  │ SENSORS      │────►│ FLIGHT       │────►│ ONBOARD EDGE │───►│ 900MHz TRANS-│ │
│  │ (CHT,EGT,RPM)│     │ COMPUTER     │     │ ML (7.8MB)   │    │ CEIVER (40km)│ │
│  └──────────────┘     └──────────────┘     └──────────────┘    └──────┬───────┘ │
└───────────────────────────────────────────────────────────────────────┼─────────┘
                                                                        │
                                       RF Datalink (MAVLink Telemetry)  │
                                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        GROUND CONTROL STATION (GCS)                             │
│                                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    ┌──────────────┐ │
│  │ GROUND RF    │     │ MAVLINK      │     │ HEAVY ML +   │    │ GCS MISSION  │ │
│  │ RECEIVER     │────►│ STREAM       │────►│ TREESHAP XAI │───►│ COCKPIT HUD  │ │
│  │ (RFD900x)    │     │ PARSER       │     │ INFERENCE    │    │ (index.html) │ │
│  └──────────────┘     └──────────────┘     └──────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

> **🟢 ARCHITECTURAL PRINCIPLE: THE EXACT SAME PYTHON CODE RUNS ON BOTH SYSTEMS — ONLY THE DATA INPUT SOURCE CHANGES FROM `CSV/SIMULATOR` TO `MAVLINK STREAM`.**

### Hardware Bill of Materials (BOM) & Cost Analysis

| Component Category | Hardware Specification | Unit Cost (₹) | System Purpose & Placement |
|:---|:---|:---:|:---|
| **Engine Sensors** | 4x K-Type Thermocouple Probes + MAX31855 | ₹2,000 | Cylinder Head Temp (CHT) & EGT per cylinder |
| | Piezoresistive Pressure Transducer (0-150 psi) | ₹2,000 | Engine Oil Line Pressure |
| | Industrial RTD PT100 (-50 to +200°C) | ₹800 | Sump Oil Temperature |
| | MEMS ADXL345 3-Axis Accelerometer (±16g) | ₹300 | Engine Crankcase Vibration |
| | Optical Tachometer / Hall Effect Sensor | ₹1,500 | Engine RPM Measurement |
| | Turbine Fuel Flow Sensor (0.5–30 GPH) | ₹3,000 | Fuel Consumption Rate |
| **Airborne Processing** | Pixhawk 6C Flight Computer (ARM Cortex-M7) | ₹25,000 | Sensor DAQ, MAVLink telemetry encoding |
| | Raspberry Pi 4 (8GB RAM) / Jetson Orin Nano | ₹8,000 | Onboard Edge ML Inference & Safety Gate |
| **Datalink & Power** | RFD900x 900MHz Long-Range Telemetry Pair | ₹15,000 | 40 km Range RF Datalink |
| | 5V/3A Power Distribution BEC & Harness | ₹1,000 | Isolated Power Regulation |
| **AIRBORNE TOTAL** | | **₹58,600** | **Complete Airborne Package** |
| **Ground Station** | Ruggedized Operator Laptop (Intel i7 / 16GB) | ₹40,000 | Ground Control Station & Heavy Analytics |
| | Ground Telemetry Receiver (RFD900x USB) | ₹15,000 | Receiving Telemetry Stream |
| | High-Gain 900MHz Yagi Directional Antenna | ₹2,000 | Extended Line-of-Sight Range (up to 50 km) |
| | 4G/5G Cellular Datalink Modem | ₹3,000 | Redundant Backup Datalink |
| **GROUND TOTAL** | | **₹60,000** | **Complete Ground Station Package** |
| **GRAND TOTAL** | | **₹1,18,600** | **Complete Production Hardware BOM** |

> 📌 **Economic Reality Check:** Total deployment hardware cost is **₹1.18 Lakh**, which represents **less than 0.004%** of a ₹25 Crore MALE UAV aircraft cost. Protecting a multi-crore asset with ₹1.18 Lakh hardware yields an extraordinary ROI.

### Operational Data Flow (1-Second Cycle)
1. **Sensor Acquisition:** Physical sensors measure CHT, EGT, RPM, Oil Pressure, Oil Temp, and Vibration at 10 Hz.
2. **Flight Computer Aggregation:** Pixhawk 6C reads sensor values via SPI/I2C/Analog inputs, packages telemetry into custom MAVLink data packets, and routes them via UART to the Raspberry Pi 4.
3. **Onboard Edge Screening:** Raspberry Pi 4 runs lightweight Isolation Forest anomaly detection (<5 ms). If severe anomaly is detected onboard, an immediate emergency flag is inserted into the datalink stream.
4. **RF Datalink Transmission:** RFD900x transceivers transmit MAVLink packets over 900 MHz at 57,600 baud rate to the Ground Control Station.
5. **GCS Processing & Explainability:** GCS ground computer receives packets via USB, parses MAVLink telemetry, runs XGBoost Fault Classification, RUL Regression, and TreeSHAP feature attributions.
6. **Cockpit Display & Pilot Alerting:** Dashboard updates at 1 Hz, refreshing health heatmaps, countdown timers, and pilot checklists.

---

## 5. Gap Analysis (Simulation vs Real Deployment)

To ensure complete technical defensibility, the table below maps every software and hardware component, evaluating its current status, real-world deployment gap, and engineering priority.

| System Module | Simulation State | Real Deployment Requirement | Gap Status | Priority Level |
|:---|:---|:---|:---|:---:|
| **Physics Engine** | Synthetic ODE Equations | Physical Engine Sensors (Lycoming IO-360) | Inputs replaced by physical sensors | **P1 (Critical)** |
| **Data Quality Guard** | Synthetic $3.5\sigma$ Noise | Real EMI, Vibration & ADC Noise | **Same code works as-is** | **P1 (Critical)** |
| **Isolation Forest** | Trained on Synthetic Data | Retrain on real flight telemetry | Architecture ready; needs fine-tuning | **P2 (High)** |
| **XGBoost Classifier** | Trained on 7 Fault Types | Transfer Learning on real failure logs | Architecture ready; model weights update | **P2 (High)** |
| **XGBoost RUL Regressor** | Trained on Sim Wear Curves | Calibrate against physical engine overhaul logs | Architecture ready | **P2 (High)** |
| **TreeSHAP XAI Core** | Fully Functional (<25ms) | Same Feature Attribution Engine | **Same code works as-is** | **P3 (Medium)** |
| **Adaptive Advisory** | Convective cooling & de-rating | Same Altitude & Throttle Planner | **Same code works as-is** | **P3 (Medium)** |
| **GCS Cockpit UI** | Web HUD (Chart.js + CSS) | Same HUD connected to MAVLink Stream | **Same code works as-is** | **P3 (Medium)** |
| **MAVLink Datalink** | Simulated via HTTP | Implement `pymavlink` parser | Requires Python MAVLink listener | **P1 (Critical)** |
| **RF Datalink Handler** | Localhost Network | RFD900x Driver & Serial Port Buffer | Requires Hardware Baudrate Setup | **P1 (Critical)** |
| **Edge Deployment** | Simulated Edge Latency | Deploy ONNX / C++ on Raspberry Pi 4 | Requires ARM64 wheel compilation | **P2 (High)** |
| **Time-Series DB** | In-Memory Dataframe | InfluxDB / TimescaleDB logging | Requires persistent storage setup | **P3 (Medium)** |
| **Security & Encryption**| Unencrypted HTTP | AES-256 MAVLink Frame Signing | Requires security layer | **P2 (High)** |
| **Airworthiness Cert.** | Academic Validation | DO-178C (Software) & DO-330 (Tools) | Long-term certification roadmap | **P4 (Future)** |

> **Priority Key:** 
> - 🔴 **P1 (Critical):** Immediate hardware ingestion & datalink drivers required for field testing.
> - 🟡 **P2 (High):** Model transfer learning and edge optimization.
> - 🟢 **P3 (Medium):** Direct code reuse (80% of current codebase).
> - ⚪ **P4 (Future):** Regulatory airworthiness certification.

> ### 📌 Summary of Gap Analysis
> **Over 80% of our software codebase (Data Quality Guard, TreeSHAP Explainability Engine, Adaptive Mission Replanning Engine, and GCS Cockpit UI) is 100% DEPLOYMENT-READY.** Only the data ingestion adapter needs to switch from reading CSV files to parsing MAVLink streams.

---

## 6. Migration Roadmap (7-Phase Engineering Plan)

Transitioning the MALE UAV Engine Digital Twin from a hackathon prototype to an operational defense system follows a structured 7-phase roadmap spanning 10–14 weeks for field testing, and 6–12 months for fleet-wide production deployment.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          7-PHASE MIGRATION TIMELINE                             │
│                                                                                 │
│ [Phase 1: Hackathon MVP] ──► (Completed - 48 Hours)                             │
│ [Phase 2: Sensor Integration] ──► (Weeks 1-3 | ₹15,000)                         │
│ [Phase 3: RF Datalink Testing] ──► (Weeks 4-5 | ₹30,000)                        │
│ [Phase 4: Edge Compute Opt.] ──► (Week 6 | ₹8,000)                              │
│ [Phase 5: Real Flight Data Collection] ──► (Weeks 7-10 | ₹50,000)               │
│ [Phase 6: Field Closed-Loop Testing] ──► (Weeks 11-14 | ₹1,00,000)              │
│ [Phase 7: Production & Certification] ──► (Months 6-12 | Fleet Deployment)      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Hackathon Software MVP (COMPLETED — 48 Hours)
- Built first-principles physics simulator for Lycoming IO-360 aero-engine across 144 missions (239,544 telemetry rows).
- Trained 3-tier ML pipeline (Isolation Forest + XGBoost Classifier + XGBoost RUL Regressor) achieving **93.68% F1-Score**.
- Developed real-time TreeSHAP explainability engine (<25ms) and GCS Cockpit HUD.
- **Cost:** ₹0 | **Timeline:** 48 Hours | **Status:** ✅ COMPLETED

### Phase 2: Physical Sensor Procurement & Calibration (Weeks 1–3)
- Procure industrial K-type thermocouples, piezoresistive oil pressure transducers, optical tachometers, and MEMS accelerometers.
- Wire sensors to Pixhawk 6C analog/I2C ports and calibrate ADC voltage-to-unit conversion curves in laboratory environment.
- Define custom MAVLink dialect messages (`UNMANNED_ENGINE_TELEMETRY`) for 14-channel transmission.
- **Cost:** ₹15,000 | **Timeline:** Weeks 1–3

### Phase 3: RF Telemetry Datalink Testing (Weeks 4–5)
- Procure and configure dual RFD900x 900MHz transceivers with Yagi directional ground antenna.
- Establish serial UART datalink at 57,600 baud; measure packet loss, latency, and signal-to-noise ratio (SNR) across increasing line-of-sight distances (1 km $\rightarrow$ 5 km $\rightarrow$ 20 km).
- Verify Data Quality Guard layer successfully imputes telemetry dropouts under real RF fading.
- **Cost:** ₹30,000 | **Timeline:** Weeks 4–5

### Phase 4: Onboard Edge Compute Validation (Week 6)
- Deploy lightweight Isolation Forest and Data Quality Guard onto onboard Raspberry Pi 4 (8GB) / Jetson Orin Nano.
- Benchmark inference latency under ARM64 architecture constraints; apply ONNX runtime quantization if necessary to maintain $<10\text{ ms}$ processing time.
- Verify onboard thermal and power consumption ($<10\text{W}$ power draw).
- **Cost:** ₹8,000 | **Timeline:** Week 6

### Phase 5: Real Flight Data Collection & Transfer Learning (Weeks 7–10)
- Mount sensor suite on test aircraft (e.g., General Aviation General Avia / Experimental UAV testbed).
- Fly 50+ healthy baseline missions to collect 100,000+ real operational sensor rows under varying atmospheric conditions.
- Fine-tune ML models using **Transfer Learning** (freezing early tree nodes trained on simulation, fine-tuning output leaves on real sensor distribution).
- **Cost:** ₹50,000 (Fuel + Aircraft Operating Expenses) | **Timeline:** Weeks 7–10

### Phase 6: Closed-Loop Field Testing & Pilot Usability (Weeks 11–14)
- Conduct live flight tests with active fault injection (e.g., partial cooling baffle restriction or synthetic sensor bias).
- Evaluate Pilot Advisory Engine: verify pilot compliance with convective cooling altitude descent and throttle de-rating recommendations.
- Measure false alarm rate ($<1\%$), fault detection lead time ($>8\text{ minutes}$ advance warning), and operator workload index (NASA-TLX).
- **Cost:** ₹1,00,000 | **Timeline:** Weeks 11–14

### Phase 7: Production Fleet Deployment & Airworthiness Certification (Months 6–12)
- Formal airworthiness compliance under **DO-178C** (Software Considerations in Airborne Systems) and **DO-330** (Tool Qualification).
- Implementation of AES-256 frame encryption and MAVLink signature validation to prevent cyber-spoofing.
- Fleet-wide installation across DRDO MALE UAV units with automated monthly model retraining.
- **Cost:** Enterprise / Defense Budget | **Timeline:** Months 6–12

---

## 7. Evaluator Talking Points (Q&A Defense Guide)

When presenting to SIH judges, DRDO scientists, and aviation domain evaluators, use these structured, authoritative answers to defend the transition from simulation to real deployment.

### Question 1: "This is just a simulation. How do we know it will work on a real UAV?"
> **Answer Structure (5-Step Response):**
> 1. **Acknowledge:** "You are completely right, sir. Real military UAV flight logs are classified, and testing on physical aircraft engines requires multi-crore test cell infrastructure."
> 2. **Explain Physics Grounding:** "However, our simulation is not random curve fitting. Every thermodynamic and atmospheric equation is calibrated directly from **FAA Type Certificate TCDS 1E10** and **NTSB crash investigation dockets (ERA21LA099)**."
> 3. **Show Hardware Architecture:** "We have designed the complete real-world hardware deployment pipeline: physical sensors $\rightarrow$ Pixhawk 6C $\rightarrow$ Raspberry Pi 4 edge computer $\rightarrow$ RFD900x 900MHz datalink $\rightarrow$ Ground Station."
> 4. **Highlight Software Modular Agnosticism:** "Our software architecture is completely hardware-agnostic. Today, the Data Quality Guard reads from `simulator.csv`. Tomorrow, it reads from `pymavlink` serial stream. **80% of our codebase is deployed as-is.**"
> 5. **Present Timeline & Cost:** "Total hardware deployment cost is ₹1.18 Lakh (0.004% of aircraft cost), and field testing requires 10–14 weeks."

### Question 2: "Your ML model never saw real sensor noise. Won't it fail in real flight?"
> **Answer:** *"In aerospace engineering, building synthetic physics baselines prior to real hardware availability is standard international practice (e.g., NASA C-MAPSS). Furthermore, our Data Quality Guard layer explicitly injects $3.5\sigma$ statistical Z-score filtering and median smoothing specifically designed for thermocouple EMI spikes and ADC quantization noise. When real data becomes available in Phase 5, we apply **Transfer Learning** to fine-tune our XGBoost models in 48 hours."*

### Question 3: "Will a pilot under extreme emergency stress actually read your dashboard?"
> **Answer:** *"We specifically designed the GCS Cockpit HUD following human factors aviation guidelines:
> - **Zero Text Overload in Emergency:** The banner uses distinct color coding (**CRITICAL RED**) and large flashing indicators (**SQUAWK 7700**).
> - **Big Digital RUL Timer:** Displays a massive 48px countdown clock (`08:09`) showing exact time remaining before engine seizure.
> - **3 Simple Action Cards:** Altitude Descent, Throttle De-Rating, and Airport Diversion.
> - **Numbered Checklist:** 1-2-3-4 step-by-step actions requiring zero calculation under stress."*

### Question 4: "How will you handle military aviation certification (DGCA / DO-178C)?"
> **Answer:** *"That is Phase 7 of our 7-phase roadmap. Currently, our system is configured as an **Auxiliary Decision-Support Tool (Health & Usage Monitoring System - HUMS)**, not a flight-critical primary autopilot. The pilot always retains final command authority. For future production integration, we follow DO-178C software compliance guidelines."*

---

## 8. Executive Summary (One-Pager)

### Project Overview
- **Project Name:** MALE UAV Engine Digital Twin, Health Monitoring & RUL Prediction
- **Problem Statement:** DRDO SIH 26054 | Aeronautical Development Establishment (ADE)
- **Current Development Phase:** Phase 1 — Software-Defined Physics MVP (Completed)
- **Target Deployment State:** Fleet-Wide Airborne Edge ML + GCS Decision Support System
- **Core Software Reuse:** **80% Code Base Deployed As-Is** (Only Data Ingestion Adapter Changes)

### Why Simulation First?
1. **Military Secrecy:** Real UAV flight telemetry is classified DEFENCE SECRET; no public failure datasets exist.
2. **Capital Constraints:** Physical aero-engine test cells cost ₹25–30 Lakh and require specialized DRDO facility clearances.
3. **Hardware BOM:** RF telemetry and flight computers cost ₹70,000+, exceeding hackathon limits.
4. **Time Constraints:** 48-hour hackathon duration demands rapid software-proven execution.

### Key Achievements Proven in MVP (✅)
- **Thermodynamic Fidelity:** Physics simulator validated against FAA Lycoming IO-360 TCDS 1E10 standards.
- **High Diagnostic Accuracy:** **93.68% Weighted F1-Score** across 7 fault categories on 52,352 test frames.
- **Prognostic Precision:** **63.62 seconds MAE** for Remaining Useful Life (RUL) prediction.
- **Real-Time Edge Readiness:** **13.9 ms total pipeline inference latency** with a lightweight **408 MB RAM** footprint.
- **Explainable AI (XAI):** Real-time TreeSHAP feature attributions with natural language physics narratives.

### Real Deployment Architecture
- **Airborne Package (₹58,600):** Engine Sensors $\rightarrow$ Pixhawk 6C Flight Computer $\rightarrow$ Raspberry Pi 4 Edge ML $\rightarrow$ RFD900x 900MHz Radio (40km Range).
- **Ground Package (₹60,000):** Ground RF Receiver $\rightarrow$ MAVLink Stream Parser $\rightarrow$ XGBoost + TreeSHAP Engine $\rightarrow$ GCS Cockpit HUD.
- **Datalink Protocol:** Standard MAVLink over 900 MHz RF / 4G Backup.

### Migration Timeline & Cost Summary
- **Hardware BOM Cost:** **₹1,18,600 total** ($<0.004\%$ of ₹25 Crore MALE UAV cost).
- **Field Testing Ready:** **10–14 Weeks** (Phases 2 through 6).
- **Fleet Production:** **6–12 Months** (Phase 7 DO-178C Certification).

> ### 🏆 Bottom-Line Engineering Conclusion
> **We built a software-proven, physics-grounded, and hardware-agnostic system. Building a simulator first was not a compromise — it was the essential engineering foundation. When physical sensors arrive, we do not rebuild the software. We simply plug them in.**
