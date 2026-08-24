"""
FastAPI Telemetry, Live Interactive Digital Twin & Mission Replay Server
========================================================================
Endpoints:
  - GET  /health              : API status check & model load confirmation
  - GET  /missions            : List all available test mission scenarios
  - POST /telemetry           : Batch live telemetry prediction
  - POST /mission/replay      : Full mission CSV playback with AI analytics
  - POST /simulator/live/reset: Reset real-time interactive physics session
  - POST /simulator/live/step : Interactive live physics step with Digital Twin residuals & AI diagnosis
"""

import json
import pickle
import sys
import time
import asyncio
import math
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT))

from simulator.engine_simulator import (
    EngineSimulator, EngineConstants, FaultSchedule,
    atmosphere, REF_RHO, MissionPhase, MISSION_LIBRARY, mission_command
)
from backend.advisory_engine import AdaptiveMissionPlanner
from backend.explainability import XGBExplainer
from backend.data_quality_guard import DataQualityGuard
from backend.performance_monitor import PerformanceMonitor
from starlette.middleware.base import BaseHTTPMiddleware

MODEL_DIR = ROOT / "ml" / "models"
RAW_DATA_DIR = ROOT / "data" / "raw"

# Initialize FastAPI App
app = FastAPI(
    title="MALE UAV Digital Twin Engine Monitoring API",
    description="Real-Time Digital Twin Physics Simulator, Anomaly Detection & RUL Advisory",
    version="2.0.0"
)

perf_monitor = PerformanceMonitor(model_dir=str(MODEL_DIR))

class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path.startswith("/static") or path == "/favicon.ico":
            return await call_next(request)

        t0 = time.perf_counter()
        error = False
        try:
            response = await call_next(request)
            if response.status_code >= 500:
                error = True
            duration_ms = (time.perf_counter() - t0) * 1000.0
            perf_monitor.record(path, duration_ms, error=error)
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            return response
        except Exception:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            perf_monitor.record(path, duration_ms, error=True)
            raise

app.add_middleware(PerformanceMiddleware)

# Enable CORS for Operator Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model Context ---
models: Dict[str, Any] = {}

SENSOR_COLS = [
    "rpm", "true_cht", "sensor_cht", "egt",
    "oil_pressure", "oil_temp", "fuel_flow", "vibration",
    "battery_voltage", "injection_timing", "health_index",
    "altitude", "ambient_temp", "throttle",
]
WINDOWS = [30, 60]

# --- Global Live Simulator Session State ---
class LiveSession:
    def __init__(self):
        self.reset()

    def reset(self):
        self.physical_sim = EngineSimulator()
        self.nominal_twin = EngineSimulator()
        self.time_s = 0.0
        self.history: List[Dict[str, float]] = []

live_session = LiveSession()

@app.on_event("startup")
def load_models():
    try:
        with open(MODEL_DIR / "isolation_forest.pkl", "rb") as f:
            models["iforest"] = pickle.load(f)
        with open(MODEL_DIR / "scaler_anomaly.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
        with open(MODEL_DIR / "fault_classifier.pkl", "rb") as f:
            models["clf"] = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            models["le"] = pickle.load(f)
        with open(MODEL_DIR / "rul_regressor.pkl", "rb") as f:
            models["reg"] = pickle.load(f)
        with open(MODEL_DIR / "model_feature_cols.json", "r") as f:
            models["feature_cols"] = json.load(f)
        print("[OK] All Digital Twin ML models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading ML models: {e}")

# --- Pydantic Data Models ---
class TelemetryFrame(BaseModel):
    timestamp_s: float
    rpm: float
    true_cht: float
    sensor_cht: float
    egt: float
    oil_pressure: float
    oil_temp: float
    fuel_flow: float
    vibration: float
    battery_voltage: float
    injection_timing: float
    health_index: float
    altitude: float
    ambient_temp: float
    throttle: float

class LiveSimStepRequest(BaseModel):
    throttle: float = Field(0.70, ge=0.0, le=1.0, description="Throttle command 0.0 to 1.0")
    altitude_m: float = Field(1500.0, ge=0.0, le=8000.0, description="Target altitude in meters")
    ambient_offset_c: float = Field(0.0, ge=-30.0, le=30.0, description="Ambient temp delta C")
    injected_fault: str = Field("none", description="none | injector | cooling | lubrication | misfire | sensor_drift | vibration")
    fault_severity: float = Field(0.0, ge=0.0, le=1.0, description="Fault severity 0.0 (healthy) to 1.0 (failure)")
    dt: float = Field(1.0, ge=0.1, le=5.0, description="Time step delta")
    simulate_packet_loss: float = Field(0.0, ge=0.0, le=1.0, description="Simulate packet loss fraction (0.0 to 1.0)")

class ValidatedTelemetryRequest(BaseModel):
    telemetry: Dict[str, float]
    simulate_packet_loss: float = Field(0.0, ge=0.0, le=1.0, description="Simulate packet loss fraction (0.0 to 1.0)")
    simulate_outlier: bool = Field(False, description="Simulate transient outlier spike on CHT and EGT")
    simulate_sensor_drift: bool = Field(False, description="Simulate sensor drift offset")

class MissionReplayRequest(BaseModel):
    split: str = "test"
    filename: str

class ComponentHealthInput(BaseModel):
    cylinder_health: float = Field(0.95, ge=0.0, le=1.0, description="Cylinder & combustion health (0.0 - 1.0)")
    lubrication_health: float = Field(0.92, ge=0.0, le=1.0, description="Lubrication & oil system health (0.0 - 1.0)")
    cooling_health: float = Field(0.88, ge=0.0, le=1.0, description="Cooling jacket health (0.0 - 1.0)")
    vibration_health: float = Field(0.97, ge=0.0, le=1.0, description="Mechanical vibration & bearing health (0.0 - 1.0)")

class MissionReliabilityRequest(BaseModel):
    mission_profile: str = Field("endurance", description="endurance | high_altitude | hot_weather | rapid_throttle")
    planned_duration_minutes: float = Field(120.0, gt=0.0, description="Planned mission duration in minutes")
    ambient_temp_c: float = Field(35.0, description="Sea-level ambient temperature in °C")
    current_health: ComponentHealthInput

class WorstCaseMetrics(BaseModel):
    peak_cht_c: float
    peak_oil_temp_c: float
    min_health_index: float

class MissionReliabilityResponse(BaseModel):
    status: str = Field(..., description="go | caution | no_go")
    mission_success_probability_percent: float
    predicted_min_rul_seconds: int
    bottleneck_component: str
    worst_case_metrics: WorstCaseMetrics
    recommendations: List[str]

class PilotAdvisoryRequest(BaseModel):
    fault_type: str = Field("none", description="none | injector | cooling | lubrication | misfire | sensor_drift | vibration")
    severity: float = Field(0.0, ge=0.0, le=1.0)
    rul_seconds: Optional[float] = Field(None, description="Estimated Remaining Useful Life in seconds")
    current_altitude_m: float = Field(1500.0, ge=0.0, le=8500.0)
    current_throttle: float = Field(0.70, ge=0.0, le=1.0)
    mission_phase: str = Field("cruise", description="takeoff | climb | cruise | loiter | descent | landing")
    nearest_airbase_distance_km: float = Field(50.0, ge=0.0)
    current_groundspeed_kmh: float = Field(120.0, ge=10.0)
    current_health: Optional[Dict[str, float]] = Field(None)

class ExplainFaultRequest(BaseModel):
    features: Optional[Dict[str, float]] = None
    mission_id: Optional[str] = None

class ExplainRulRequest(BaseModel):
    current_features: Optional[Dict[str, float]] = None
    previous_features: Optional[Dict[str, float]] = None
    current_rul: Optional[float] = None
    previous_rul: Optional[float] = None
    timestep_context: Optional[Dict[str, float]] = None

class BenchmarkRequest(BaseModel):
    target_endpoint: str = Field("/telemetry", description="Target API route to benchmark")
    requests_count: int = Field(100, ge=1, le=1000, description="Total requests to execute")
    concurrency: int = Field(10, ge=1, le=50, description="Parallel concurrent workers")
    payload: Optional[Dict[str, Any]] = Field(None, description="Optional custom payload")

advisory_planner = AdaptiveMissionPlanner()
explainer = XGBExplainer(model_dir=str(MODEL_DIR))
quality_guard = DataQualityGuard()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df_in = df.copy()
    if "true_cht" not in df_in.columns and "cht" in df_in.columns:
        df_in["true_cht"] = df_in["cht"]
    if "sensor_cht" not in df_in.columns and "cht" in df_in.columns:
        df_in["sensor_cht"] = df_in["cht"]
    if "cht" not in df_in.columns and "true_cht" in df_in.columns:
        df_in["cht"] = df_in["true_cht"]

    defaults = {
        "rpm": 2400.0, "true_cht": 150.0, "sensor_cht": 150.0, "egt": 580.0,
        "oil_pressure": 55.0, "oil_temp": 85.0, "fuel_flow": 10.0, "vibration": 0.35,
        "battery_voltage": 13.8, "injection_timing": 20.0, "health_index": 0.98,
        "altitude": 1500.0, "ambient_temp": 15.0, "throttle": 0.70
    }
    for col in SENSOR_COLS:
        if col not in df_in.columns:
            df_in[col] = defaults.get(col, 0.0)

    features = df_in[SENSOR_COLS].copy()
    for col in SENSOR_COLS:
        series = df_in[col]
        for w in WINDOWS:
            features[f"{col}_rmean{w}"] = series.rolling(w, min_periods=1).mean()
            features[f"{col}_rstd{w}"]  = series.rolling(w, min_periods=1).std().fillna(0)
    return features.ffill().bfill().fillna(0)

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": len(models) == 6}

@app.get("/missions")
def list_missions():
    test_dir = RAW_DATA_DIR / "test"
    if not test_dir.exists():
        return {"missions": []}
    
    files = sorted([f.name for f in test_dir.glob("*.csv")])
    result = []
    for f in files:
        parts = f.replace(".csv", "").split("_")
        profile = parts[0] if parts[0] in ["endurance", "rapid"] else f"{parts[0]}_{parts[1]}"
        fault = "healthy" if "healthy" in f else [p for p in ["injector", "cooling", "lubrication", "misfire", "sensor_drift", "vibration"] if p in f]
        fault_name = fault[0] if isinstance(fault, list) and fault else ("healthy" if "healthy" in f else "faulted")
        result.append({
            "filename": f,
            "profile": profile.replace("_", " ").title(),
            "fault_type": fault_name.replace("_", " ").title(),
            "label": f"{profile.replace('_', ' ').title()} — {fault_name.replace('_', ' ').title()} ({f})"
        })
    return {"missions": result}

@app.post("/simulator/live/reset")
def reset_live_simulation():
    live_session.reset()
    quality_guard.reset()
    return {"status": "reset", "time_s": 0.0}

@app.post("/simulator/live/step")
def step_live_simulation(req: LiveSimStepRequest):
    sess = live_session
    dt = req.dt
    sess.time_s += dt
    t = sess.time_s

    # Build active fault schedule for physical engine
    fault_schedule = {
        "injector":     FaultSchedule("injector", onset_s=0 if req.injected_fault == "injector" else 1e9, failure_s=1e-6 if req.injected_fault == "injector" else 1e9),
        "lubrication":  FaultSchedule("lubrication", onset_s=0 if req.injected_fault == "lubrication" else 1e9, failure_s=1e-6 if req.injected_fault == "lubrication" else 1e9),
        "cooling":      FaultSchedule("cooling", onset_s=0 if req.injected_fault == "cooling" else 1e9, failure_s=1e-6 if req.injected_fault == "cooling" else 1e9),
        "misfire":      FaultSchedule("misfire", onset_s=0 if req.injected_fault == "misfire" else 1e9, failure_s=1e-6 if req.injected_fault == "misfire" else 1e9),
        "sensor_drift": FaultSchedule("sensor_drift", onset_s=0 if req.injected_fault == "sensor_drift" else 1e9, failure_s=1e-6 if req.injected_fault == "sensor_drift" else 1e9),
        "vibration":    FaultSchedule("vibration", onset_s=0 if req.injected_fault == "vibration" else 1e9, failure_s=1e-6 if req.injected_fault == "vibration" else 1e9),
    }

    # Override severity method dynamically with user input
    if req.injected_fault in fault_schedule:
        fault_schedule[req.injected_fault].severity = lambda t_val: req.fault_severity

    # 1. Step Physical Engine
    sess.physical_sim.step(dt, req.throttle, req.altitude_m, req.ambient_offset_c, fault_schedule, t)
    p_state = sess.physical_sim.state

    # 2. Step Nominal Digital Twin (Pure Ideal Baseline with 0 fault)
    nominal_faults = {k: FaultSchedule("none") for k in fault_schedule}
    sess.nominal_twin.step(dt, req.throttle, req.altitude_m, req.ambient_offset_c, nominal_faults, t)
    n_state = sess.nominal_twin.state

    # Ambient calculations
    _, T_amb_k, _ = atmosphere(p_state.altitude, req.ambient_offset_c)
    ambient_c = T_amb_k - 273.15

    # Sensor measurement with sensor drift fault effect
    drift_val = 15.0 * req.fault_severity if req.injected_fault == "sensor_drift" else 0.0
    sensor_cht = p_state.cht + drift_val + np.random.normal(0, 0.4)

    # Raw telemetry frame
    raw_frame = {
        "timestamp_s": t,
        "rpm": p_state.rpm,
        "true_cht": p_state.cht,
        "sensor_cht": sensor_cht,
        "egt": p_state.egt,
        "oil_pressure": p_state.oil_pressure,
        "oil_temp": p_state.oil_temp,
        "fuel_flow": p_state.fuel_flow,
        "vibration": p_state.vibration,
        "battery_voltage": p_state.battery_voltage,
        "injection_timing": p_state.injection_timing,
        "health_index": p_state.health_index,
        "altitude": p_state.altitude,
        "ambient_temp": ambient_c,
        "throttle": req.throttle,
    }

    # Optional Packet Loss Simulation
    simulated_loss = False
    processed_raw = raw_frame
    if req.simulate_packet_loss > 0.0:
        processed_raw = quality_guard.simulate_packet_loss(raw_frame, req.simulate_packet_loss)
        simulated_loss = True

    # Pass through DataQualityGuard
    t_start = time.perf_counter()
    model_preds = {
        "cht": n_state.cht,
        "sensor_cht": n_state.cht,
        "egt": n_state.egt,
        "oil_pressure": n_state.oil_pressure,
        "oil_temp": n_state.oil_temp,
        "rpm": n_state.rpm,
        "fuel_flow": n_state.fuel_flow
    }
    t0_dq = time.perf_counter()
    dq_res = quality_guard.process_telemetry_frame(
        processed_raw,
        dt=dt,
        model_predicted_values=model_preds,
        simulated_packet_loss=simulated_loss
    )
    lat_dq = (time.perf_counter() - t0_dq) * 1000.0
    cleaned_frame = dq_res["cleaned_data"]

    # Guarantee non-NaN for internal simulator channels
    if "true_cht" not in cleaned_frame or (isinstance(cleaned_frame.get("true_cht"), float) and math.isnan(cleaned_frame["true_cht"])):
        cleaned_frame["true_cht"] = cleaned_frame.get("cht", cleaned_frame.get("sensor_cht", p_state.cht))
    if "sensor_cht" not in cleaned_frame or (isinstance(cleaned_frame.get("sensor_cht"), float) and math.isnan(cleaned_frame["sensor_cht"])):
        cleaned_frame["sensor_cht"] = cleaned_frame.get("cht", sensor_cht)

    # Record cleaned frame in rolling history
    sess.history.append(cleaned_frame)
    if len(sess.history) > 120:
        sess.history.pop(0)

    # Compute rolling window features for AI inference
    df_hist = pd.DataFrame(sess.history)
    feat = compute_features(df_hist)
    X_latest = feat[models["feature_cols"]].iloc[[-1]]

    # 3. AI Inference
    # Anomaly Score
    t0_anom = time.perf_counter()
    X_scaled = models["scaler"].transform(X_latest)
    raw_if = models["iforest"].decision_function(X_scaled)[0]
    anom_score = float(1.0 / (1.0 + np.exp(raw_if * 12.0)))
    is_anomaly = bool(raw_if < -0.02)
    lat_anom = (time.perf_counter() - t0_anom) * 1000.0

    # Fault Classifier
    t0_clf = time.perf_counter()
    probs = models["clf"].predict_proba(X_latest)[0]
    pred_idx = int(np.argmax(probs))
    pred_fault = str(models["le"].inverse_transform([pred_idx])[0])
    pred_conf = float(probs[pred_idx])
    lat_clf = (time.perf_counter() - t0_clf) * 1000.0

    # Class probabilities dictionary for XAI radar/bar
    class_probs = {str(c): round(float(p), 4) for c, p in zip(models["le"].classes_, probs)}

    # RUL Prediction
    t0_reg = time.perf_counter()
    pred_rul = float(models["reg"].predict(X_latest)[0]) if pred_fault != "none" else None
    if pred_rul is not None:
        pred_rul = max(0.0, round(pred_rul, 1))
    lat_reg = (time.perf_counter() - t0_reg) * 1000.0

    # 4. Physical vs Digital Twin Residuals
    residuals = {
        "cht_delta_c": round(p_state.cht - n_state.cht, 2),
        "egt_delta_c": round(p_state.egt - n_state.egt, 2),
        "oil_pressure_delta_psi": round(p_state.oil_pressure - n_state.oil_pressure, 2),
        "oil_temp_delta_c": round(p_state.oil_temp - n_state.oil_temp, 2),
        "vibration_delta": round(p_state.vibration - n_state.vibration, 3),
        "fuel_flow_delta": round(p_state.fuel_flow - n_state.fuel_flow, 3)
    }

    # 5. Component Subsystem Health Heatmap (0.0 to 1.0)
    comp_health = getattr(p_state, "component_health", None) or {
        "cylinder_health": 1.0,
        "lubrication_health": 1.0,
        "cooling_health": 1.0,
        "vibration_health": 1.0
    }

    subsystem_health = {
        "cylinders": comp_health.get("cylinder_health", 1.0),
        "fuel_system": round(max(0.0, 1.0 - (0.8 if req.injected_fault == "injector" else 0.0) * req.fault_severity), 2),
        "oil_lubrication": comp_health.get("lubrication_health", 1.0),
        "cooling_jacket": comp_health.get("cooling_health", 1.0),
        "avionics_sensors": round(max(0.0, 1.0 - (0.7 if req.injected_fault == "sensor_drift" else 0.0) * req.fault_severity), 2),
    }

    # 6. Actionable Pilot Advisory Generation
    t0_adv = time.perf_counter()
    if pred_fault != "none" and p_state.health_index < 0.92:
        advisory_level = "CRITICAL"
        action_plan = [
            f"1. Reduce throttle to {(req.throttle * 0.75):.2f} to lower thermal load.",
            "2. Pitch down slightly to maintain airspeed for cylinder head ram-air cooling.",
            f"3. INITIATE DIVERT: Estimated safe flight window is {pred_rul or 0:.0f} seconds.",
            "4. Alert Ground Control Station (GCS) and squawk 7700 emergency."
        ]
    elif anom_score > 0.65 and p_state.health_index < 0.88:
        advisory_level = "WARNING"
        action_plan = [
            "1. Multichannel telemetry variance exceeding nominal band.",
            "2. Monitor oil pressure and CHT trend.",
            "3. Prepare diversion plan if health index continues downward trend."
        ]
    else:
        advisory_level = "NOMINAL"
        action_plan = [
            "All engine systems operating within FAA/Lycoming certified limits.",
            "Cruise parameters nominal. No pilot intervention required."
        ]
    lat_adv = (time.perf_counter() - t0_adv) * 1000.0

    # 7. SHAP Explanation
    t0_shap = time.perf_counter()
    shap_explanation = explainer.explain_fault_prediction(X_latest) if (pred_fault != "none" and pred_conf > 0.60) else None
    lat_shap = (time.perf_counter() - t0_shap) * 1000.0

    total_pipeline_ms = (time.perf_counter() - t_start) * 1000.0
    stage_breakdown = {
        "data_quality_ms": round(lat_dq, 2),
        "anomaly_detection_ms": round(lat_anom, 2),
        "fault_classification_ms": round(lat_clf, 2),
        "rul_regression_ms": round(lat_reg, 2),
        "advisory_generation_ms": round(lat_adv, 2),
        "shap_explanation_ms": round(lat_shap, 2)
    }
    perf_monitor.record_model_latencies(stage_breakdown)

    return {
        "timestamp_s": round(t, 1),
        "physical_telemetry": {
            "rpm": round(cleaned_frame.get("rpm", p_state.rpm), 1),
            "true_cht": round(cleaned_frame.get("true_cht", p_state.cht), 1),
            "sensor_cht": round(cleaned_frame.get("sensor_cht", sensor_cht), 1),
            "egt": round(cleaned_frame.get("egt", p_state.egt), 1),
            "oil_pressure": round(cleaned_frame.get("oil_pressure", p_state.oil_pressure), 1),
            "oil_temp": round(cleaned_frame.get("oil_temp", p_state.oil_temp), 1),
            "fuel_flow": round(cleaned_frame.get("fuel_flow", p_state.fuel_flow), 2),
            "vibration": round(cleaned_frame.get("vibration", p_state.vibration), 3),
            "battery_voltage": round(cleaned_frame.get("battery_voltage", p_state.battery_voltage), 2),
            "injection_timing": round(cleaned_frame.get("injection_timing", p_state.injection_timing), 1),
            "health_index": round(cleaned_frame.get("health_index", p_state.health_index), 3),
            "component_health": comp_health,
            "altitude": round(cleaned_frame.get("altitude", p_state.altitude), 1),
            "ambient_temp": round(ambient_c, 1),
            "throttle": req.throttle,
        },
        "digital_twin_nominal": {
            "nominal_rpm": round(n_state.rpm, 1),
            "nominal_cht": round(n_state.cht, 1),
            "nominal_egt": round(n_state.egt, 1),
            "nominal_oil_pressure": round(n_state.oil_pressure, 1),
            "nominal_oil_temp": round(n_state.oil_temp, 1),
            "nominal_fuel_flow": round(n_state.fuel_flow, 2),
        },
        "residuals": residuals,
        "data_quality": dq_res["data_quality_summary"],
        "quality_flags": dq_res["quality_flags"],
        "ai_diagnostics": {
            "anomaly_score": round(anom_score, 4),
            "is_anomaly": is_anomaly,
            "predicted_fault": pred_fault,
            "confidence": round(pred_conf, 4),
            "class_probabilities": class_probs,
            "estimated_rul_seconds": pred_rul,
            "subsystem_health": subsystem_health,
            "explanation": shap_explanation
        },
        "advisory": {
            "level": advisory_level,
            "action_plan": action_plan
        },
        "performance": {
            "inference_latency_ms": round(total_pipeline_ms, 2),
            "pipeline_stage_breakdown": stage_breakdown,
            "total_pipeline_ms": round(total_pipeline_ms, 2)
        }
    }

@app.post("/mission/replay")
def replay_mission(req: MissionReplayRequest):
    filepath = RAW_DATA_DIR / req.split / req.filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Mission file {req.filename} not found in {req.split}")

    df = pd.read_csv(filepath)
    feat = compute_features(df)
    X = feat[models["feature_cols"]]

    # Predictions
    X_scaled = models["scaler"].transform(X)
    raw_if = models["iforest"].decision_function(X_scaled)
    anom_score = (1.0 / (1.0 + np.exp(raw_if * 12.0))).tolist()
    is_anom = (raw_if < -0.02).tolist()

    probs = models["clf"].predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    fault_labels = models["le"].inverse_transform(pred_idx)
    confidences = np.max(probs, axis=1).tolist()

    rul_preds = models["reg"].predict(X).tolist()

    records = []
    for i in range(len(df)):
        f_type = fault_labels[i]
        rul_val = round(float(rul_preds[i]), 1) if f_type != "none" else None

        row_dict = {
            "timestamp_s": float(df.loc[i, "timestamp_s"]),
            "rpm": float(df.loc[i, "rpm"]),
            "true_cht": float(df.loc[i, "true_cht"]),
            "sensor_cht": float(df.loc[i, "sensor_cht"]),
            "egt": float(df.loc[i, "egt"]),
            "oil_pressure": float(df.loc[i, "oil_pressure"]),
            "oil_temp": float(df.loc[i, "oil_temp"]),
            "fuel_flow": float(df.loc[i, "fuel_flow"]),
            "vibration": float(df.loc[i, "vibration"]),
            "battery_voltage": float(df.loc[i, "battery_voltage"]),
            "injection_timing": float(df.loc[i, "injection_timing"]),
            "health_index": float(df.loc[i, "health_index"]),
            "altitude": float(df.loc[i, "altitude"]),
            "ambient_temp": float(df.loc[i, "ambient_temp"]),
            "throttle": float(df.loc[i, "throttle"]),
            "true_fault_type": str(df.loc[i, "fault_type"]),
            "true_fault_severity": float(df.loc[i, "fault_severity"]),
            "pred_anomaly_score": round(float(anom_score[i]), 4),
            "pred_is_anomaly": bool(is_anom[i] and f_type != "none"),
            "pred_fault_type": str(f_type),
            "pred_confidence": round(float(confidences[i]), 4),
            "pred_rul_seconds": rul_val,
        }
        records.append(row_dict)

    return {
        "filename": req.filename,
        "total_frames": len(records),
        "telemetry": records
    }

@app.post("/mission/reliability-check", response_model=MissionReliabilityResponse)
def preflight_reliability_check(req: MissionReliabilityRequest):
    """
    Pre-Flight Go/No-Go Advisor & Digital Twin Reliability Assessment:
    - Simulates planned mission profile under specified ambient conditions and current health state
    - Executes 50 Monte Carlo runs with stochastic noise and operational wear
    - Returns Go/Caution/No-Go verdict, success probability, worst-case metrics, and actionable recommendations
    """
    total_s = req.planned_duration_minutes * 60.0
    ambient_offset_c = req.ambient_temp_c - 15.0  # ISA sea-level deviation

    # Scale mission phase durations to planned duration
    base_phases = MISSION_LIBRARY.get(req.mission_profile, MISSION_LIBRARY["endurance"])
    base_total = sum(p.duration_s for p in base_phases)
    scale = total_s / max(base_total, 1.0)
    scaled_phases = [
        MissionPhase(duration_s=p.duration_s * scale, throttle=p.throttle, altitude_m=p.altitude_m)
        for p in base_phases
    ]

    # Map initial health state to fault severities (0.0 = healthy, 1.0 = failure)
    inj_sev_init = float(np.clip(1.0 - req.current_health.cylinder_health, 0.0, 1.0))
    lub_sev_init = float(np.clip(1.0 - req.current_health.lubrication_health, 0.0, 1.0))
    cool_sev_init = float(np.clip(1.0 - req.current_health.cooling_health, 0.0, 1.0))
    vib_sev_init = float(np.clip(1.0 - req.current_health.vibration_health, 0.0, 1.0))

    # Identify Bottleneck Component
    comp_map = {
        "cylinders": req.current_health.cylinder_health,
        "oil_lubrication": req.current_health.lubrication_health,
        "cooling_jacket": req.current_health.cooling_health,
        "vibration_assembly": req.current_health.vibration_health
    }
    bottleneck_component = min(comp_map, key=comp_map.get)

    # Monte Carlo simulation (50 runs)
    n_runs = 50
    dt = 2.0  # 2.0s timestep for smooth numerical integration
    success_count = 0
    peak_chts = []
    peak_oil_temps = []
    min_health_indices = []
    first_failure_times = []

    # Wear accumulation factor scales with flight duration (~1.5% per flight hour)
    wear_factor = 0.015 * (req.planned_duration_minutes / 60.0)

    for _ in range(n_runs):
        sim = EngineSimulator()
        sim.state.cht = max(25.0, req.ambient_temp_c + 15.0)
        sim.state.oil_temp = max(25.0, req.ambient_temp_c + 10.0)

        # Stochastic noise on degradation slopes
        noise_inj = np.random.normal(1.0, 0.05)
        noise_cool = np.random.normal(1.0, 0.05)
        noise_lub = np.random.normal(1.0, 0.05)
        noise_vib = np.random.normal(1.0, 0.05)

        faults = {
            "injector": FaultSchedule("injector"),
            "lubrication": FaultSchedule("lubrication"),
            "cooling": FaultSchedule("cooling"),
            "misfire": FaultSchedule("misfire"),
            "sensor_drift": FaultSchedule("sensor_drift"),
            "vibration": FaultSchedule("vibration")
        }
        faults["injector"].severity = lambda t, n=noise_inj: float(np.clip(inj_sev_init + (wear_factor * (t / total_s) * n), 0.0, 1.0))
        faults["lubrication"].severity = lambda t, n=noise_lub: float(np.clip(lub_sev_init + (wear_factor * (t / total_s) * n), 0.0, 1.0))
        faults["cooling"].severity = lambda t, n=noise_cool: float(np.clip(cool_sev_init + (wear_factor * (t / total_s) * n), 0.0, 1.0))
        faults["vibration"].severity = lambda t, n=noise_vib: float(np.clip(vib_sev_init + (wear_factor * (t / total_s) * n), 0.0, 1.0))
        faults["misfire"].severity = lambda t: float(np.clip(inj_sev_init * 0.4, 0.0, 1.0))

        run_min_h = 1.0
        run_peak_cht = 0.0
        run_peak_oil_t = 0.0
        t_fail = total_s

        t = 0.0
        while t <= total_s:
            thr, alt = mission_command(t, scaled_phases)
            out = sim.step(dt, thr, alt, ambient_offset_c, faults, t)
            h = out["health_index"]
            if h < run_min_h:
                run_min_h = h
            if out["true_cht"] > run_peak_cht:
                run_peak_cht = out["true_cht"]
            if out["oil_temp"] > run_peak_oil_t:
                run_peak_oil_t = out["oil_temp"]
            if h < 0.30 and t_fail == total_s:
                t_fail = t
            t += dt

        if run_min_h >= 0.30:
            success_count += 1

        peak_chts.append(run_peak_cht)
        peak_oil_temps.append(run_peak_oil_t)
        min_health_indices.append(run_min_h)
        first_failure_times.append(t_fail)

    success_pct = round((success_count / n_runs) * 100.0, 1)
    worst_min_h = round(float(np.min(min_health_indices)), 3)
    worst_peak_cht = round(float(np.max(peak_chts)), 1)
    worst_peak_oil = round(float(np.max(peak_oil_temps)), 1)
    predicted_min_rul = int(np.min(first_failure_times))

    # Status Determination & Dynamic Recommendations
    recommendations = []
    if success_pct >= 90.0 and worst_min_h >= 0.70:
        status = "go"
        recommendations.append("Engine health and thermal reserves are nominal for the planned mission envelope.")
        recommendations.append("Mission cleared for departure under standard operating procedures.")
    elif success_pct >= 65.0 or worst_min_h >= 0.45:
        status = "caution"
        if req.ambient_temp_c > 32.0:
            recommendations.append(f"High ambient OAT ({req.ambient_temp_c:.1f}°C) reduces cylinder head cooling margins during initial climb.")
            recommendations.append("Perform step-climb in 1,000m increments to allow cylinder head heat dissipation.")
        if req.planned_duration_minutes > 90.0:
            recommendations.append(f"Consider reducing mission duration from {req.planned_duration_minutes:.0f} min to 90 min to avoid thermal fatigue accumulation.")
        if req.mission_profile == "rapid_throttle":
            recommendations.append("Avoid rapid full-throttle transients; limit rate of power application.")
        if comp_map[bottleneck_component] < 0.85:
            recommendations.append(f"Bottleneck component '{bottleneck_component}' shows reduced pre-flight health ({comp_map[bottleneck_component]*100:.0f}%). Monitor closely on telemetry.")
    else:
        status = "no_go"
        recommendations.append(f"CRITICAL: High risk of in-flight engine health degradation ({100.0 - success_pct:.1f}% failure probability).")
        recommendations.append(f"Pre-flight maintenance required for '{bottleneck_component}' (initial health: {comp_map[bottleneck_component]*100:.0f}%).")
        if req.ambient_temp_c > 35.0:
            recommendations.append("Delay mission until ambient temperature drops below 30°C.")
        recommendations.append("Do not authorize flight until technician inspects and signs off engine subassemblies.")

    return MissionReliabilityResponse(
        status=status,
        mission_success_probability_percent=success_pct,
        predicted_min_rul_seconds=predicted_min_rul,
        bottleneck_component=bottleneck_component,
        worst_case_metrics=WorstCaseMetrics(
            peak_cht_c=worst_peak_cht,
            peak_oil_temp_c=worst_peak_oil,
            min_health_index=worst_min_h
        ),
        recommendations=recommendations
    )

@app.post("/advisory/generate")
def generate_advisory(req: PilotAdvisoryRequest):
    """
    In-Flight Pilot Emergency & Adaptive Mission Replanning Advisory:
    - Calculates safe throttle de-rating envelope
    - Determines optimal altitude replanning based on thermodynamic fault physics
    - Evaluates emergency diversion urgency to nearest airbase
    - Generates actionable pilot checklists and emergency squawk codes
    """
    return advisory_planner.generate_pilot_advisory(
        fault_type=req.fault_type,
        severity=req.severity,
        rul_seconds=req.rul_seconds,
        current_altitude_m=req.current_altitude_m,
        current_throttle=req.current_throttle,
        mission_phase=req.mission_phase,
        nearest_airbase_distance_km=req.nearest_airbase_distance_km,
        current_health=req.current_health
    )

@app.post("/explain/fault")
def explain_fault_endpoint(req: ExplainFaultRequest):
    """
    Explainable AI (TreeSHAP) Fault Prediction Breakdown:
    - Analyzes provided telemetry features or falls back to live session state
    - Returns top-3 contributing physical features with directional SHAP impacts,
      nominal envelopes, confidence breakdown, and physics narrative.
    """
    if req.features:
        return explainer.explain_fault_prediction(req.features)

    # Fallback to most recent live simulator telemetry
    if live_session.history:
        df_hist = pd.DataFrame(live_session.history)
        feat = compute_features(df_hist)
        X_latest = feat[models.get("feature_cols", feat.columns)].iloc[[-1]]
        return explainer.explain_fault_prediction(X_latest)

    return explainer.explain_fault_prediction(None)

@app.post("/explain/rul")
def explain_rul_endpoint(req: ExplainRulRequest):
    """
    Explainable AI (TreeSHAP) RUL Degradation Attribution:
    - Identifies primary physical driver of remaining engine life
    - Calculates impact of rapid throttle transients, thermal shocks, or lubrication loss
    - Returns physics-informed degradation narrative.
    """
    curr_feat = req.current_features
    if curr_feat is None and live_session.history:
        df_hist = pd.DataFrame(live_session.history)
        feat = compute_features(df_hist)
        curr_feat = feat[models.get("feature_cols", feat.columns)].iloc[[-1]]

    return explainer.explain_rul_drop(
        current_features=curr_feat,
        previous_features=req.previous_features,
        current_rul=req.current_rul,
        previous_rul=req.previous_rul,
        timestep_context=req.timestep_context
    )

@app.get("/explain/importance")
def get_global_importance():
    """
    Returns global feature importance ranked by TreeSHAP attribution.
    """
    return explainer.get_feature_importance_summary()

@app.post("/telemetry")
def process_telemetry(frame: Dict[str, Any]):
    """
    Telemetry ingestion endpoint:
    - Filters raw sensor readings through DataQualityGuard
    - Runs multi-class fault classification and RUL prediction on cleaned data
    - Returns AI diagnostics and sensor data quality report
    """
    dq_res = quality_guard.process_telemetry_frame(frame)
    cleaned = dq_res["cleaned_data"]

    df = pd.DataFrame([cleaned])
    feat = compute_features(df)
    X = feat[models["feature_cols"]]

    X_scaled = models["scaler"].transform(X)
    raw_if = float(models["iforest"].decision_function(X_scaled)[0])
    anom_score = float(1.0 / (1.0 + np.exp(raw_if * 12.0)))

    probs = models["clf"].predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_fault = str(models["le"].inverse_transform([pred_idx])[0])
    pred_conf = float(probs[pred_idx])
    is_anomaly = bool(raw_if < -0.02 and pred_fault != "none")

    pred_rul = None
    if pred_fault != "none":
        pred_rul = round(float(models["reg"].predict(X)[0]), 1)

    explanation = None
    if pred_fault != "none" and pred_conf > 0.60:
        explanation = explainer.explain_fault_prediction(X)

    return {
        "cleaned_data": cleaned,
        "quality_flags": dq_res["quality_flags"],
        "data_quality_summary": dq_res["data_quality_summary"],
        "ai_predictions": {
            "anomaly_score": round(anom_score, 4),
            "is_anomaly": is_anomaly,
            "predicted_fault": pred_fault,
            "confidence": round(pred_conf, 4),
            "estimated_rul_seconds": pred_rul
        },
        "explanation": explanation
    }

@app.post("/telemetry/validated")
def process_validated_telemetry(req: ValidatedTelemetryRequest):
    """
    Robust telemetry processing endpoint with synthetic degradation injections:
    - Simulates packet loss, outlier spikes, or sensor drift
    - DataQualityGuard recovers dropped channels and corrects spikes
    - AI models run inference on clean imputed streams
    - Returns complete raw, cleaned, quality, AI, and SHAP payload
    """
    raw_frame = dict(req.telemetry)

    # 1. Outlier Injection
    if req.simulate_outlier:
        raw_frame["cht"] = 350.0
        raw_frame["egt"] = 950.0

    # 2. Sensor Drift Injection
    if req.simulate_sensor_drift:
        raw_frame["cht"] = raw_frame.get("cht", 150.0) + 45.0

    # 3. Packet Loss Simulation
    processed_frame = raw_frame
    simulated_loss = False
    if req.simulate_packet_loss > 0.0:
        processed_frame = quality_guard.simulate_packet_loss(raw_frame, req.simulate_packet_loss)
        simulated_loss = True

    # 4. Data Quality Guard Processing
    dq_res = quality_guard.process_telemetry_frame(
        processed_frame,
        simulated_packet_loss=simulated_loss
    )
    cleaned = dq_res["cleaned_data"]

    # 5. Feature Computation & Model Inference
    df = pd.DataFrame([cleaned])
    feat = compute_features(df)
    X = feat[models["feature_cols"]]

    X_scaled = models["scaler"].transform(X)
    raw_if = float(models["iforest"].decision_function(X_scaled)[0])
    anom_score = float(1.0 / (1.0 + np.exp(raw_if * 12.0)))

    probs = models["clf"].predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_fault = str(models["le"].inverse_transform([pred_idx])[0])
    pred_conf = float(probs[pred_idx])
    is_anomaly = bool(raw_if < -0.02 and pred_fault != "none")

    pred_rul = None
    if pred_fault != "none":
        pred_rul = round(float(models["reg"].predict(X)[0]), 1)

    explanation = None
    if pred_fault != "none" and pred_conf > 0.60:
        explanation = explainer.explain_fault_prediction(X)

    return {
        "raw_data": raw_frame,
        "cleaned_data": cleaned,
        "quality_flags": dq_res["quality_flags"],
        "data_quality_summary": dq_res["data_quality_summary"],
        "ai_predictions": {
            "anomaly_score": round(anom_score, 4),
            "is_anomaly": is_anomaly,
            "predicted_fault": pred_fault,
            "confidence": round(pred_conf, 4),
            "estimated_rul_seconds": pred_rul
        },
        "explanation": explanation
    }

@app.get("/metrics/performance")
def get_performance_metrics():
    """
    Returns aggregated system resource metrics, ASGI latency percentiles, and ML model profile.
    """
    return perf_monitor.get_dashboard_metrics()

@app.get("/metrics/endpoint/{endpoint_name:path}")
def get_endpoint_metrics(endpoint_name: str):
    """
    Returns granular latency and throughput percentiles for a specific API endpoint.
    """
    clean_name = f"/{endpoint_name.lstrip('/')}"
    return perf_monitor.get_endpoint_stats(clean_name)

@app.post("/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    """
    Asynchronous Stress-Testing & High-Throughput Benchmarking:
    - Fires concurrent requests using asyncio workers
    - Calculates latency percentiles (p50, p95, p99), throughput (RPS), and system stability
    """
    default_payload = {
        "cht": 145.0, "egt": 620.0, "rpm": 2400.0, "oil_pressure": 65.0,
        "oil_temp": 90.0, "fuel_flow": 10.5, "vibration": 0.25,
        "battery_voltage": 13.8, "injection_timing": 20.0, "health_index": 0.95,
        "altitude": 3000.0, "ambient_temp": 15.0, "throttle": 0.55, "sensor_cht": 145.0
    }
    payload = req.payload if req.payload is not None else default_payload
    ep = req.target_endpoint if req.target_endpoint.startswith("/") else f"/{req.target_endpoint}"

    latencies = []
    success_count = 0
    fail_count = 0
    semaphore = asyncio.Semaphore(req.concurrency)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30.0) as client:
        async def call_worker():
            nonlocal success_count, fail_count
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    r = await client.post(ep, json=payload)
                    duration = (time.perf_counter() - t0) * 1000.0
                    if r.status_code == 200:
                        success_count += 1
                        latencies.append(duration)
                    else:
                        fail_count += 1
                except Exception:
                    fail_count += 1

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(call_worker()) for _ in range(req.requests_count)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    arr = np.array(latencies) if latencies else np.array([0.0])
    throughput = round(len(latencies) / max(0.001, total_time), 1)

    p95 = float(np.percentile(arr, 95))
    if p95 < 50.0 and fail_count == 0:
        grade = "excellent"
        rec = "System handles real-time concurrency with high headroom for edge deployment."
    elif p95 < 100.0 and fail_count == 0:
        grade = "good"
        rec = "Good real-time performance within MALE UAV GCS telemetry budgets."
    elif p95 < 500.0:
        grade = "fair"
        rec = "Acceptable latency; consider batching or quantization for low-power edge."
    else:
        grade = "degraded"
        rec = "High latency detected; inspect model inference bottlenecks."

    return {
        "benchmark_id": f"bench_{int(time.time())}",
        "target_endpoint": ep,
        "requests_sent": req.requests_count,
        "successful_responses": success_count,
        "failed_responses": fail_count,
        "total_duration_seconds": round(total_time, 3),
        "throughput_rps": throughput,
        "latency_stats_ms": {
            "min": round(float(np.min(arr)), 2),
            "avg": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(p95, 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(np.max(arr)), 2)
        },
        "performance_grade": grade,
        "recommendation": rec
    }

# ---------------------------------------------------------------------------
# Mount GCS Mission Control Cockpit Static Files
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

DASHBOARD_DIR = ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

