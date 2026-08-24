"""
Adaptive Mission Replanning & Pilot Emergency Advisory Engine
=============================================================
Module: backend.advisory_engine
Author: SIH Digital Twin Team
Description:
  Rule-based physics and aeronautical expert system for in-flight contingency
  reaction, dynamic flight envelope de-rating, optimal altitude replanning,
  diversion urgency assessment, and pilot emergency checklists.
"""

from typing import Dict, List, Optional, Any


class AdaptiveMissionPlanner:
    def __init__(self):
        """Rule-based physics and aeronautical contingency advisor."""
        pass

    def suggest_optimal_altitude(
        self,
        current_altitude_m: float,
        fault_type: str,
        severity: float,
        current_health: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates optimal altitude adjustment to maximize engine thermal margins and RUL.
        """
        f = (fault_type or "none").lower()
        sev = float(severity or 0.0)

        if f == "cooling" and sev > 0.3:
            target_alt = max(500, int(current_altitude_m - 1500))
            reasoning = "Descend to denser air for improved convective cooling and higher heat dissipation"
            expected_gain = int(180 + 320 * sev)
        elif f == "injector" and sev > 0.3:
            target_alt = max(500, int(current_altitude_m - 1000))
            reasoning = "Descend for richer air-fuel mixture and increased manifold oxygen pressure"
            expected_gain = int(150 + 250 * sev)
        elif f == "lubrication":
            target_alt = int(current_altitude_m)
            reasoning = "Maintain level flight to reduce oil slosh, g-load spikes, and bearing hydrodynamic stress"
            expected_gain = 0
        elif f == "misfire" and sev > 0.2:
            target_alt = max(1000, int(current_altitude_m - 500))
            reasoning = "Descend to reduce engine aerodynamic load and stabilize cylinder combustion"
            expected_gain = int(120 + 200 * sev)
        elif f == "vibration":
            target_alt = int(current_altitude_m)
            reasoning = "Maintain altitude; vibration is mechanical / shaft imbalance, not atmospheric"
            expected_gain = 0
        else:
            target_alt = int(current_altitude_m)
            reasoning = "Maintain assigned mission altitude. Engine operating within safe margins."
            expected_gain = 0

        return {
            "target_altitude_m": target_alt,
            "reasoning": reasoning,
            "expected_rul_gain_seconds": expected_gain
        }

    def suggest_throttle_envelope(
        self,
        current_throttle: float,
        fault_type: str,
        severity: float
    ) -> Dict[str, Any]:
        """
        Calculates safe throttle de-rating envelope to prevent thermal runaway or catastrophic mechanical seizure.
        """
        f = (fault_type or "none").lower()
        sev = float(severity or 0.0)
        min_throttle = 0.20  # Avoid flameout / idle stall

        if sev < 0.3:
            max_throttle = 0.85
            recommended = min(current_throttle, 0.75)
            reasoning = "Nominal throttle envelope active; maintain standard cruise power"
        elif 0.3 <= sev <= 0.6:
            max_throttle = 0.60
            recommended = 0.50
            reasoning = "De-rate engine power to 50-60% to mitigate thermal and mechanical degradation"
        else:
            max_throttle = 0.45
            recommended = 0.40
            reasoning = "Critical power de-rate: limit power to minimum sustainable flight to prolong RUL"

        # Fault-specific envelope adjustments
        if f == "lubrication" and sev > 0.2:
            max_throttle = max(0.25, round(max_throttle - 0.10, 2))
            recommended = max(0.20, min(recommended, max_throttle))
            reasoning += " (Reduced further to minimize hydrodynamic friction and bearing seizure risk)"
        elif f == "misfire" and sev > 0.2:
            max_throttle = max(0.25, round(max_throttle - 0.05, 2))
            recommended = max(0.20, min(recommended, max_throttle))
            reasoning += " (Apply smooth, gradual throttle inputs to avoid triggering acute cylinder misfire)"

        return {
            "min_throttle": round(min_throttle, 2),
            "max_throttle": round(max_throttle, 2),
            "recommended_throttle": round(recommended, 2),
            "reasoning": reasoning
        }

    def calculate_diversion_urgency(
        self,
        rul_seconds: float,
        current_altitude_m: float,
        nearest_airbase_distance_km: float = 50.0,
        current_groundspeed_kmh: float = 120.0
    ) -> Dict[str, Any]:
        """
        Computes emergency diversion feasibility, airbase reachability, and safety time buffer.
        """
        speed = max(10.0, float(current_groundspeed_kmh))
        time_to_airbase_s = int((nearest_airbase_distance_km / speed) * 3600.0)
        buffer_s = int(rul_seconds - time_to_airbase_s)

        if buffer_s < 300:  # < 5 minutes buffer
            urgency_level = "critical"
            diversion_recommended = True
        elif buffer_s < 900:  # < 15 minutes buffer
            urgency_level = "warning"
            diversion_recommended = True
        elif buffer_s < 1800:  # < 30 minutes buffer
            urgency_level = "monitor"
            diversion_recommended = False
        else:
            urgency_level = "nominal"
            diversion_recommended = False

        descent_note = "Standard descent profile feasible"
        if current_altitude_m > 4000 and urgency_level in ["critical", "warning"]:
            descent_note = "Begin descent immediately to reduce time-to-land"

        return {
            "urgency_level": urgency_level,
            "time_to_airbase_seconds": time_to_airbase_s,
            "buffer_seconds": buffer_s,
            "diversion_recommended": diversion_recommended,
            "descent_note": descent_note
        }

    def generate_pilot_advisory(
        self,
        fault_type: str,
        severity: float,
        rul_seconds: float,
        current_altitude_m: float,
        current_throttle: float,
        mission_phase: str = "cruise",
        nearest_airbase_distance_km: float = 50.0,
        current_health: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Master advisor: Synthesizes altitude, throttle, diversion, and pilot checklists.
        """
        f = (fault_type or "none").lower()
        sev = float(severity or 0.0)
        rul = float(rul_seconds if rul_seconds is not None else 1e9)

        # 1. Emergency Level Mapping
        if sev == 0.0 or f in ["none", "sensor_drift"]:
            emergency_level = "none"
        elif sev < 0.3:
            emergency_level = "advisory"
        elif 0.3 <= sev <= 0.5:
            emergency_level = "caution"
        elif 0.5 < sev <= 0.7:
            emergency_level = "warning"
        else:
            emergency_level = "critical"

        # 2. Sub-assessments
        alt_rec = self.suggest_optimal_altitude(current_altitude_m, f, sev, current_health)
        thr_rec = self.suggest_throttle_envelope(current_throttle, f, sev)
        div_rec = self.calculate_diversion_urgency(rul, current_altitude_m, nearest_airbase_distance_km)

        # 3. Immediate Actions
        actions_map = {
            "cooling": [
                f"Reduce throttle to {thr_rec['recommended_throttle']:.2f} to lower combustion heat",
                f"Initiate descent to {alt_rec['target_altitude_m']}m for denser cooling airflow",
                "Monitor Cylinder Head Temperature (CHT) and Oil Temp every 30 seconds"
            ],
            "lubrication": [
                f"De-rate engine to {thr_rec['recommended_throttle']:.2f} power to reduce bearing load",
                "Maintain steady level flight without high-g turns",
                "Prepare for precautionary landing; verify nearest diversion field"
            ],
            "injector": [
                f"Descend to {alt_rec['target_altitude_m']}m for richer ambient oxygen mixture",
                "Avoid rapid throttle transients to prevent cylinder detonation",
                "Monitor Exhaust Gas Temperature (EGT) for lean thermal runaway"
            ],
            "misfire": [
                f"Smoothly set power to {thr_rec['recommended_throttle']:.2f} to stabilize firing cycle",
                "Cross-check magneto/ignition channel telemetry",
                "Prepare contingency route if RPM fluctuation exceeds ±100 RPM"
            ],
            "vibration": [
                "Reduce throttle to limit propeller and crankshaft stress",
                "Avoid abrupt pitch and rudder control inputs",
                "Inspect airframe sensors for mechanical looseness or harmonic resonance"
            ],
            "sensor_drift": [
                "Cross-check CHT reading against EGT and Oil Temperature sensors",
                "Ignore false single-channel alarm if secondary thermal channels are nominal",
                "Log sensor calibration offset for post-flight maintenance"
            ],
            "none": [
                "Maintain standard flight plan and scan instruments",
                "All engine subsystems operating within FAA/Lycoming certified envelopes"
            ]
        }
        immediate_actions = actions_map.get(f, actions_map["none"])

        # 4. Mission Continue Probability
        if emergency_level == "none":
            continue_prob = 98.0
        elif emergency_level == "advisory":
            continue_prob = 88.0 if rul > 1800 else 75.0
        elif emergency_level == "caution":
            continue_prob = 65.0 if rul > 1200 else 50.0
        elif emergency_level == "warning":
            continue_prob = 35.0 if rul > 600 else 20.0
        else:
            continue_prob = 8.0

        # 5. Pilot Actionable Checklist
        checklist_map = {
            "cooling": [
                f"1. Set Throttle: {thr_rec['recommended_throttle']:.2f}",
                f"2. Altitude: Descend to {alt_rec['target_altitude_m']}m",
                "3. Airspeed: Increase slightly to maximize ram-air cooling",
                f"4. Diversion: {'Initiate divert to nearest airbase' if div_rec['diversion_recommended'] else 'Monitor CHT trend'}",
                "5. Contact ATC / GCS with status update"
            ],
            "lubrication": [
                f"1. Set Throttle: {thr_rec['recommended_throttle']:.2f}",
                "2. Maintain level flight attitude",
                "3. Scan oil pressure gauge continuously (Min limit: 25 psi)",
                f"4. Diversion: {'Immediate divert required' if div_rec['diversion_recommended'] else 'Plan precautionary return'}",
                "5. Contact ATC / GCS with status update"
            ],
            "injector": [
                f"1. Adjust Throttle: {thr_rec['recommended_throttle']:.2f}",
                f"2. Descend to {alt_rec['target_altitude_m']}m",
                "3. Verify EGT balance across cylinders",
                f"4. Diversion: {'Divert to nearest runway' if div_rec['diversion_recommended'] else 'Continue with caution'}",
                "5. Contact ATC / GCS with status update"
            ],
            "misfire": [
                f"1. Set smooth throttle: {thr_rec['recommended_throttle']:.2f}",
                f"2. Descend to {alt_rec['target_altitude_m']}m",
                "3. Monitor RPM stability and fuel flow",
                f"4. Diversion: {'Initiate emergency return' if div_rec['diversion_recommended'] else 'Hold over safe terrain'}",
                "5. Contact ATC / GCS with status update"
            ],
            "vibration": [
                f"1. Reduce power to {thr_rec['recommended_throttle']:.2f}",
                "2. Limit control surface deflection rates",
                "3. Check vibration spectrum telemetry",
                f"4. Diversion: {'Divert immediately' if div_rec['diversion_recommended'] else 'Maintain altitude and monitor'}",
                "5. Contact ATC / GCS with status update"
            ],
            "sensor_drift": [
                "1. Confirm engine physical sound and vibration are nominal",
                "2. Correlate sensor CHT with EGT and Oil Temp",
                "3. Continue mission on backup telemetry channels",
                "4. Contact ATC / GCS with status update"
            ],
            "none": [
                "1. Cruising parameters verified nominal",
                "2. Maintain assigned waypoint track",
                "3. Contact ATC / GCS with status update"
            ]
        }
        pilot_checklist = checklist_map.get(f, checklist_map["none"])

        # 6. Squawk Code
        squawk_code = "7700" if (emergency_level == "critical" and div_rec["diversion_recommended"]) else None

        return {
            "fault_detected": f,
            "severity": round(sev, 3),
            "emergency_level": emergency_level,
            "immediate_actions": immediate_actions,
            "altitude_recommendation": alt_rec,
            "throttle_recommendation": thr_rec,
            "diversion_assessment": div_rec,
            "mission_continue_probability": continue_prob,
            "pilot_checklist": pilot_checklist,
            "squawk_code": squawk_code
        }
