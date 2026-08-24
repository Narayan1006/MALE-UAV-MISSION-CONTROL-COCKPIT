"""
Reduced-Order Physics-Informed Engine Simulator
=================================================
MALE UAV aero-piston engine — Digital Twin data generation layer.

PHYSICS CONSTANTS — REAL DATA CALIBRATION & SOURCE CITATION
-------------------------------------------------------------
All default EngineConstants values are calibrated directly from an authentic
1 Hz Garmin G1000 flight datalog recovered during an official National
Transportation Safety Board (NTSB) investigation, harmonized with FAA TCDS 1E10.

Source Citation:
  - NTSB Docket: ERA21LA099 (Project ID: 102515)
  - Docket URL: https://data.ntsb.gov/Docket?ProjectID=102515
  - Aircraft: Diamond DA40-180 powered by Lycoming IO-360-M1A (Air-cooled, 4-cylinder, direct-drive)
  - Telemetry File: log_210103_103720_KBVY-Rel.csv (2,190 recorded seconds)
  - FAA Type Certificate Data Sheet: TCDS 1E10

Why these equations exist:
- Every output depends on physically meaningful inputs (throttle, altitude, ambient temp)
  through a chain of first-order thermal and aerodynamic balance equations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. ATMOSPHERE MODEL — simple standard atmosphere (why altitude matters)
# ---------------------------------------------------------------------------
def atmosphere(altitude_m: float, ambient_offset_c: float = 0.0):
    """Returns (pressure_pa, temperature_k, air_density_kg_m3) at altitude.
    Uses the standard lapse rate: temperature and pressure both fall with altitude,
    which is WHY engine performance changes at high altitude (less oxygen)."""
    T0, P0, L, R, g = 288.15, 101325.0, 0.0065, 287.05, 9.80665
    T = (T0 + ambient_offset_c) - L * altitude_m
    T = max(T, 216.65)  # stratosphere floor, MALE UAVs rarely exceed this
    P = P0 * (T / (T0 + ambient_offset_c)) ** (g / (R * L))
    rho = P / (R * T)
    return P, T, rho


REF_RHO = atmosphere(0.0)[2]  # sea-level reference density, used to normalize


# ---------------------------------------------------------------------------
# 2. MISSION PROFILE — defines throttle(t), altitude(t), ambient(t)
# ---------------------------------------------------------------------------
@dataclass
class MissionPhase:
    duration_s: float
    throttle: float      # 0-1
    altitude_m: float     # target altitude for this phase


MISSION_LIBRARY = {
    "endurance": [
        MissionPhase(60, 0.9, 500),      # takeoff
        MissionPhase(180, 0.75, 3000),   # climb
        MissionPhase(300, 0.55, 3000),   # cruise
        MissionPhase(1800, 0.45, 3000),  # long loiter (endurance)
        MissionPhase(180, 0.6, 1500),    # return
        MissionPhase(90, 0.3, 0),        # landing
    ],
    "high_altitude": [
        MissionPhase(60, 0.95, 500),
        MissionPhase(300, 0.85, 7000),   # climb to high altitude
        MissionPhase(400, 0.6, 7000),    # cruise
        MissionPhase(600, 0.5, 7000),    # loiter
        MissionPhase(200, 0.35, 1000),   # descent
        MissionPhase(90, 0.3, 0),
    ],
    "hot_weather": [
        MissionPhase(60, 0.9, 500),
        MissionPhase(180, 0.75, 2500),
        MissionPhase(1200, 0.55, 2500),  # extended cruise/loiter in heat
        MissionPhase(90, 0.3, 0),
    ],
    "rapid_throttle": [
        MissionPhase(60, 0.9, 500),
        MissionPhase(120, 0.3, 1500),
        MissionPhase(120, 0.7, 1500),
        MissionPhase(120, 0.9, 1500),
        MissionPhase(120, 0.4, 1500),
        MissionPhase(120, 0.8, 1500),
        MissionPhase(90, 0.3, 0),
    ],
}


def mission_command(t: float, phases: list[MissionPhase]):
    """Given elapsed time, return (throttle_target, altitude_target) for that instant."""
    elapsed = 0.0
    for p in phases:
        if t < elapsed + p.duration_s:
            return p.throttle, p.altitude_m
        elapsed += p.duration_s
    last = phases[-1]
    return last.throttle, last.altitude_m


# ---------------------------------------------------------------------------
# 3. FAULT INJECTION — progressive, labelled degradation for ML ground truth
# ---------------------------------------------------------------------------
@dataclass
class FaultSchedule:
    fault_type: str = "none"          # injector | lubrication | cooling | misfire | sensor_drift | vibration | none
    onset_s: float = 1e9              # when degradation starts
    failure_s: float = 1e9            # when it would hit failure threshold (100% severity)

    def severity(self, t: float) -> float:
        """0.0 = healthy, 1.0 = failure threshold. Linear ramp for simplicity —
        swap for exponential/piecewise if you want to match CMAPSS-style curves."""
        if t < self.onset_s or self.fault_type == "none":
            return 0.0
        frac = (t - self.onset_s) / max(self.failure_s - self.onset_s, 1e-6)
        return float(np.clip(frac, 0.0, 1.0))

    def rul_seconds(self, t: float) -> float:
        if self.fault_type == "none":
            return 1e9  # effectively "no known failure"
        return max(self.failure_s - t, 0.0)


# ---------------------------------------------------------------------------
# 4. ENGINE STATE + PHYSICS STEP
# ---------------------------------------------------------------------------
@dataclass
class EngineConstants:
    # --- RPM [NTSB G1000 Fit: Idle 809 RPM, Max 2641 RPM | TCDS 1E10 rated max: 2700 RPM] ---
    idle_rpm: float = 809.1          # RPM — taxi/idle from G1000 log
    max_rpm: float = 2641.3          # RPM — full throttle takeoff/climb from G1000 log
    tau_rpm: float = 3.2             # s   — first-order RPM response time constant
    k_fuel: float = 0.012            # fuel-flow proportionality constant

    # --- CHT [NTSB G1000 Fit: Max Rise 186.8°C | Redline 260°C] ---
    cht_rise_max: float = 186.8      # degC — peak CHT rise above ambient
    tau_cht: float = 42.0            # s    — thermal time constant from climb transition

    # --- EGT [NTSB G1000 Empirical Linear Fit: EGT = 508.35 + 1.3317 * CHT] ---
    egt_base: float = 508.35         # degC — EGT intercept
    egt_gain: float = 1.3317         # degC/degC — EGT/CHT linear coefficient

    # --- Oil Temp [NTSB G1000 Fit: Max Rise 78.7°C] ---
    oil_rise_max: float = 78.7       # degC — max oil temp rise above ambient
    tau_oil: float = 80.0            # s    — oil thermal time constant

    # --- Oil Pressure [NTSB G1000 Fit: Base 52.7 psi, Slope 23.5 psi | Max ~76.2 psi] ---
    pressure_base: float = 52.7      # psi  — oil pressure at idle
    pressure_slope: float = 23.5     # psi  — pressure range to max RPM
    pressure_temp_penalty: float = 0.12  # psi/degC — viscosity-driven pressure drop with temp


@dataclass
class EngineState:
    rpm: float = 1700.0
    cht: float = 25.0
    egt: float = 25.0
    oil_temp: float = 25.0
    oil_pressure: float = 0.0
    fuel_flow: float = 0.0
    vibration: float = 0.2
    battery_voltage: float = 12.6
    injection_timing: float = 20.0   # degrees BTDC, nominal
    health_index: float = 1.0
    altitude: float = 0.0


class EngineSimulator:
    def __init__(self, constants: EngineConstants = None):
        self.c = constants or EngineConstants()
        self.state = EngineState()

    def step(self, dt: float, throttle_cmd: float, altitude_target: float,
             ambient_offset_c: float, faults: dict[str, FaultSchedule], t: float):
        c, s = self.c, self.state

        # --- altitude rate-limited climb/descent (UAV can't teleport altitude) ---
        max_climb_rate = 8.0  # m/s
        s.altitude += np.clip(altitude_target - s.altitude, -max_climb_rate * dt, max_climb_rate * dt)
        _, T_amb_k, rho = atmosphere(s.altitude, ambient_offset_c)
        ambient_c = T_amb_k - 273.15
        rho_ratio = rho / REF_RHO   # <1 at altitude -> less available oxygen

        # --- fault severities this instant ---
        inj_sev = faults["injector"].severity(t)
        lub_sev = faults["lubrication"].severity(t)
        cool_sev = faults["cooling"].severity(t)
        vib_sev = faults["vibration"].severity(t)
        misfire_sev = faults["misfire"].severity(t)
        drift_sev = faults["sensor_drift"].severity(t)

        injector_efficiency = 1.0 - 0.5 * inj_sev      # 100% -> 50% at full failure
        lubrication_health = 1.0 - 0.8 * lub_sev
        cooling_efficiency = 1.0 - 0.6 * cool_sev

        # --- misfire: intermittent, not constant (real misfire is sporadic) ---
        misfire_active = misfire_sev > 0 and (np.random.random() < 0.15 * misfire_sev)
        misfire_kick = -400.0 if misfire_active else 0.0

        # --- RPM: first-order lag toward target + fault disturbance ---
        target_rpm = c.idle_rpm + throttle_cmd * (c.max_rpm - c.idle_rpm) * rho_ratio ** 0.5
        s.rpm += (target_rpm - s.rpm) / c.tau_rpm * dt
        s.rpm += misfire_kick + np.random.normal(0, 15 * inj_sev)
        s.rpm = max(s.rpm, 0)

        load = (s.rpm / c.max_rpm) * throttle_cmd  # engine load 0-1

        # --- fuel flow: throttle + rpm + air density + injector health ---
        s.fuel_flow = c.k_fuel * s.rpm * throttle_cmd * rho_ratio * injector_efficiency
        if misfire_active:
            s.fuel_flow *= 0.6  # unburned fuel doesn't register as useful flow

        # --- thermal balance: CHT lags toward a load/fault-dependent equilibrium ---
        # More throttle -> more fuel burned -> more heat -> higher equilibrium CHT.
        # Weak cooling or a fouled injector both push the equilibrium higher.
        load_effect = 0.25 + 0.75 * load          # baseline heat even near idle + load scaling
        cht_target = ambient_c + c.cht_rise_max * load_effect * \
            (1 + 0.4 * (1 - injector_efficiency)) / cooling_efficiency
        s.cht += (cht_target - s.cht) / c.tau_cht * dt

        # --- EGT: correlates with CHT + combustion completeness ---
        s.egt = c.egt_base + c.egt_gain * s.cht + 60 * (1 - injector_efficiency)
        if misfire_active:
            s.egt += np.random.normal(0, 60)

        # --- lubrication: oil temp lags toward a friction/health-dependent equilibrium ---
        oil_target = ambient_c + c.oil_rise_max * load_effect * (2 - lubrication_health)
        s.oil_temp += (oil_target - s.oil_temp) / c.tau_oil * dt
        base_pressure = c.pressure_base + c.pressure_slope * (s.rpm / c.max_rpm)
        s.oil_pressure = max(
            base_pressure * lubrication_health - c.pressure_temp_penalty * max(s.oil_temp - 100, 0), 0)

        # --- vibration: baseline + lubrication wear + bearing degradation + misfire ---
        s.vibration = 0.2 + 1.2 * (1 - lubrication_health) + 1.5 * vib_sev
        if misfire_active:
            s.vibration += 0.8

        # --- battery/alternator: simple charging curve with rpm ---
        s.battery_voltage = 12.0 + 1.6 * min(s.rpm / c.max_rpm, 1.0) + np.random.normal(0, 0.05)

        # --- injection timing drift with injector wear ---
        s.injection_timing = 20.0 - 6.0 * inj_sev + np.random.normal(0, 0.2)

        # --- composite health index: realistic healthy bands & weighted deviations ---
        # Healthy bands:
        # CHT: ambient + 30°C to ambient + 120°C (Redline 260°C)
        # EGT: 350°C to 650°C (Redline 950°C)
        # Oil Pressure: 30 to 80 psi (Redline High 115 psi)
        # Vibration: 0.1 to 0.8 (Redline 2.5)

        cht_band_lower = ambient_c + 30.0
        cht_band_upper = ambient_c + 120.0
        cht_redline = 260.0
        dev_cht_high = max(0.0, (s.cht - cht_band_upper) / max(cht_redline - cht_band_upper, 1e-6))
        dev_cht_low = max(0.0, (cht_band_lower - s.cht) / max(cht_band_lower, 1e-6))
        dev_cht = max(dev_cht_high, dev_cht_low)

        egt_band_lower = 350.0
        egt_band_upper = 650.0
        egt_redline = 950.0
        dev_egt_high = max(0.0, (s.egt - egt_band_upper) / max(egt_redline - egt_band_upper, 1e-6))
        dev_egt_low = max(0.0, (egt_band_lower - s.egt) / max(egt_band_lower, 1e-6))
        dev_egt = max(dev_egt_high, dev_egt_low)

        oil_p_band_lower = 30.0
        oil_p_band_upper = 80.0
        oil_p_redline_high = 115.0
        dev_oil_p_high = max(0.0, (s.oil_pressure - oil_p_band_upper) / max(oil_p_redline_high - oil_p_band_upper, 1e-6))
        dev_oil_p_low = max(0.0, (oil_p_band_lower - s.oil_pressure) / max(oil_p_band_lower, 1e-6))
        dev_oil_p = max(dev_oil_p_high, dev_oil_p_low)

        vib_band_lower = 0.1
        vib_band_upper = 0.8
        vib_redline = 2.5
        dev_vib_high = max(0.0, (s.vibration - vib_band_upper) / max(vib_redline - vib_band_upper, 1e-6))
        dev_vib_low = max(0.0, (vib_band_lower - s.vibration) / max(vib_band_lower, 1e-6))
        dev_vib = max(dev_vib_high, dev_vib_low)

        # Weights: CHT 0.30, EGT 0.20, oil_pressure 0.25, vibration 0.25
        weighted_dev = 0.30 * dev_cht + 0.20 * dev_egt + 0.25 * dev_oil_p + 0.25 * dev_vib
        s.health_index = float(np.clip(1.0 - weighted_dev, 0.0, 1.0))

        # Individual component health scores (0.0 = failed, 1.0 = pristine)
        cylinder_health = float(np.clip(1.0 - 0.6 * dev_cht - 0.4 * dev_egt, 0.0, 1.0))
        lubrication_health = float(np.clip(1.0 - dev_oil_p, 0.0, 1.0))
        cooling_health = float(np.clip(1.0 - dev_cht, 0.0, 1.0))
        vibration_health = float(np.clip(1.0 - dev_vib, 0.0, 1.0))

        s.component_health = {
            "cylinder_health": round(cylinder_health, 3),
            "lubrication_health": round(lubrication_health, 3),
            "cooling_health": round(cooling_health, 3),
            "vibration_health": round(vibration_health, 3),
        }

        # --- ground truth vs sensor-reported (sensor drift fault) ---
        drift_bias_cht = 10.0 * drift_sev  # sensor slowly lies about CHT while engine is fine
        sensor_cht = s.cht + drift_bias_cht + np.random.normal(0, 0.5)

        return {
            "true_cht": s.cht, "sensor_cht": sensor_cht,
            "egt": s.egt + np.random.normal(0, 3),
            "rpm": s.rpm + np.random.normal(0, 5),
            "oil_pressure": s.oil_pressure + np.random.normal(0, 0.5),
            "oil_temp": s.oil_temp + np.random.normal(0, 0.5),
            "fuel_flow": s.fuel_flow,
            "vibration": s.vibration + np.random.normal(0, 0.05),
            "battery_voltage": s.battery_voltage,
            "injection_timing": s.injection_timing,
            "health_index": s.health_index,
            "component_health": s.component_health,
            "altitude": s.altitude,
            "ambient_temp": ambient_c,
            "throttle": throttle_cmd,
        }


# ---------------------------------------------------------------------------
# 5. MISSION RUNNER — ties everything together, produces labelled CSV
# ---------------------------------------------------------------------------
def run_mission(mission_name: str, fault_type: str = "none", onset_s: float = 1e9,
                 failure_s: float = 1e9, hot_weather_offset: float = 0.0,
                 dt: float = 1.0, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    phases = MISSION_LIBRARY[mission_name]
    total_duration = sum(p.duration_s for p in phases)

    faults = {k: FaultSchedule() for k in
              ["injector", "lubrication", "cooling", "misfire", "sensor_drift", "vibration"]}
    if fault_type != "none":
        faults[fault_type] = FaultSchedule(fault_type, onset_s, failure_s)

    sim = EngineSimulator()
    rows = []
    t = 0.0
    while t <= total_duration:
        throttle, alt_target = mission_command(t, phases)
        out = sim.step(dt, throttle, alt_target, hot_weather_offset, faults, t)
        active_fault = fault_type if faults[fault_type if fault_type != "none" else "injector"].severity(t) > 0 else "none"
        severity = faults[fault_type].severity(t) if fault_type != "none" else 0.0
        rul = faults[fault_type].rul_seconds(t) if fault_type != "none" else None
        rows.append({
            "timestamp_s": t,
            **out,
            "fault_type": active_fault,
            "fault_severity": round(severity, 3),
            "rul_seconds": rul,
        })
        t += dt

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    os.makedirs("missions", exist_ok=True)

    # Healthy baseline missions
    for name in MISSION_LIBRARY:
        df = run_mission(name)
        df.to_csv(f"missions/{name}_healthy.csv", index=False)
        print(f"generated missions/{name}_healthy.csv  ({len(df)} rows)")

    # Faulted mission examples — degradation ramps in during the loiter phase
    df = run_mission("endurance", fault_type="injector", onset_s=600, failure_s=1800)
    df.to_csv("missions/endurance_injector_fault.csv", index=False)

    df = run_mission("endurance", fault_type="lubrication", onset_s=500, failure_s=2000)
    df.to_csv("missions/endurance_lubrication_fault.csv", index=False)

    df = run_mission("endurance", fault_type="cooling", onset_s=400, failure_s=1600)
    df.to_csv("missions/endurance_cooling_fault.csv", index=False)

    df = run_mission("endurance", fault_type="sensor_drift", onset_s=300, failure_s=2200)
    df.to_csv("missions/endurance_sensor_drift.csv", index=False)

    df = run_mission("hot_weather", hot_weather_offset=15.0)
    df.to_csv("missions/hot_weather_extreme.csv", index=False)

    print("\nAll mission CSVs generated in ./missions/")
    print("Each row has both sensor readings (noisy) and ground-truth fault labels for ML training.")
