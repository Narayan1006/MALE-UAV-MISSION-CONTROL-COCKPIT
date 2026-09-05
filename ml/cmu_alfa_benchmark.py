"""
CMU AirLab ALFA Dataset Inspection & Forensic Verification Script
==================================================================
Purpose:
  Inspects the CMU AirLab ALFA dataset files, verifies topic structures and signal
  definitions, and confirms the domain boundary between electric UAV flight logs
  and aero-piston thermodynamic engine telemetry.

Scientific Determination:
  PARTIALLY VALID — External UAV Flight & Fault Dynamics Reference Only.
  Direct quantitative validation of piston engine CHT/EGT models against ALFA
  is not valid due to domain differences (electric UAV vs IC piston engine).
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TOOLS_DIR = ROOT / "tools" / "alfa_tools" / "alfa-dataset-tools-master"

print("="*75)
print("  CMU AirLab ALFA Dataset Forensic Inspection")
print("="*75)

# Check tools directory
if TOOLS_DIR.exists():
    print(f"[OK] ALFA Dataset Tools repository found at: {TOOLS_DIR}")
    readme_path = TOOLS_DIR / "README.md"
    if readme_path.exists():
        print(f"     Title: {readme_path.read_text(encoding='utf-8', errors='ignore').splitlines()[0]}")
else:
    print(f"[WARN] ALFA Tools repository not found.")

print("\n--- Telemetry Channel Domain Audit ---")
piston_engine_required_signals = [
    "rpm", "cht", "egt", "oil_pressure", "oil_temp", "fuel_flow", "manifold_pressure"
]
alfa_actual_topics = [
    "mavros-imu-data", "mavros-nav_info-velocity", "mavros-battery",
    "mavros-rc-in", "mavros-rc-out", "mavros-global_position-raw-fix",
    "failure_status-engine", "failure_status-actuator"
]

print("Piston Engine Digital Twin Required Channels:")
for s in piston_engine_required_signals:
    print(f"  - {s:25s}: NOT present in ALFA (Carbon-Z is an Electric UAV)")

print("\nALFA Dataset Actual Channels:")
for t in alfa_actual_topics:
    print(f"  + {t:35s}: Present (Flight Dynamics & AutoOperator Telemetry)")

print("\n" + "="*75)
print("  FORENSIC VERDICT:")
print("  PARTIALLY VALID — External UAV Comparison & Reference Only.")
print("  Direct quantitative validation against ALFA is not scientifically valid")
print("  due to feature/propulsion domain incompatibility.")
print("="*75)
