import requests
import json
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def simulate_frontend_shap_decision(ai_diagnostics):
    """Mirror exact frontend renderShapBars logic in JS."""
    pred_fault = ai_diagnostics.get("predicted_fault", "none")
    conf = ai_diagnostics.get("confidence", 0.0)
    explanation = ai_diagnostics.get("explanation")
    
    is_fault_met = (pred_fault != "none") and (conf > 0.60)
    has_features = bool(explanation and explanation.get("top_3_features") and len(explanation["top_3_features"]) > 0)
    
    if not is_fault_met or not has_features:
        return {
            "rendered": "EMPTY_STATE",
            "message": "No active fault — explainability panel activates when a fault is detected above 60% confidence",
            "fault": pred_fault,
            "conf": conf
        }
    else:
        return {
            "rendered": "ATTRIBUTION_BARS",
            "top_3": [f['feature_name'] for f in explanation["top_3_features"]],
            "fault": pred_fault,
            "conf": conf
        }

print("=================================================================")
print("  AEROTWIN DASHBOARD ISSUE 1 & ISSUE 2 VERIFICATION TEST")
print("=================================================================\n")

# STEP 1: Reset simulator, no fault
print("Step 1: Resetting simulator (no fault)...")
r_reset = requests.post(f"{BASE_URL}/simulator/live/reset").json()
print("  Status:", r_reset)

# STEP 2 & 3: Ramp throttle from idle to 70% over 30 seconds
print("\nStep 2 & 3: Ramping throttle from idle (15%) to 70% over 30 seconds...")
ramp_shap_rendered_count = 0
observed_anomaly_scores = []

for sec in range(1, 31):
    throttle = round(min(0.70, 0.15 + (0.55 * (sec / 30.0))), 3)
    payload = {
        'throttle': throttle,
        'altitude_m': 1000.0,
        'ambient_offset_c': 0.0,
        'injected_fault': 'none',
        'fault_severity': 0.0,
        'simulate_packet_loss': 0.0,
        'dt': 1.0
    }
    resp = requests.post(f"{BASE_URL}/simulator/live/step", json=payload).json()
    ai = resp["ai_diagnostics"]
    p = resp["physical_telemetry"]
    
    frontend_state = simulate_frontend_shap_decision(ai)
    if frontend_state["rendered"] != "EMPTY_STATE":
        ramp_shap_rendered_count += 1
        
    observed_anomaly_scores.append(ai["anomaly_score"])
    
    if sec in [5, 10, 15, 20, 25, 30]:
        print(f"  t={sec:2d}s | Thr={throttle:.2f} | RPM={p['rpm']:4.0f} | Health={p['health_index']:.2f} | AnomScore={ai['anomaly_score']:.2f} | Fault={ai['predicted_fault']} ({ai['confidence']*100:.0f}%) | SHAP={frontend_state['rendered']}")

print(f"\n  Result: Across 30s ramp, SHAP fault attribution appeared {ramp_shap_rendered_count} times.")
assert ramp_shap_rendered_count == 0, "SHAP panel must NOT show attribution during nominal ramp!"
print("  [PASS] SHAP panel stayed in clean empty state throughout nominal throttle ramp.")

# STEP 4 & 5: Inject fault at severity 0.6+
print("\nStep 4 & 5: Injecting injector failure at severity 0.75...")
fault_payload = {
    'throttle': 0.70,
    'altitude_m': 1000.0,
    'ambient_offset_c': 0.0,
    'injected_fault': 'injector',
    'fault_severity': 0.75,
    'simulate_packet_loss': 0.0,
    'dt': 1.0
}
# Step a few seconds so fault manifests
fault_resp = None
for s in range(5):
    fault_resp = requests.post(f"{BASE_URL}/simulator/live/step", json=fault_payload).json()

ai_fault = fault_resp["ai_diagnostics"]
frontend_state_fault = simulate_frontend_shap_decision(ai_fault)

print(f"  Detected Fault: {ai_fault['predicted_fault']} (Conf: {ai_fault['confidence']*100:.1f}%)")
print(f"  SHAP Panel State: {frontend_state_fault['rendered']}")
if frontend_state_fault["rendered"] == "ATTRIBUTION_BARS":
    print(f"  Top 3 Attribution Features: {frontend_state_fault['top_3']}")
    narrative = ai_fault['explanation'].get('physics_explanation') or ai_fault['explanation'].get('physics_narrative')
    print(f"  Physics Narrative: {narrative}")

assert frontend_state_fault["rendered"] == "ATTRIBUTION_BARS", "SHAP panel must render attribution when fault detected > 60% conf!"
print("  [PASS] SHAP panel properly rendered top-3 features for detected fault.")

# STEP 6: Reset again, confirm clears back to empty
print("\nStep 6: Resetting simulator again...")
requests.post(f"{BASE_URL}/simulator/live/reset")
post_reset_payload = {
    'throttle': 0.30,
    'altitude_m': 1000.0,
    'ambient_offset_c': 0.0,
    'injected_fault': 'none',
    'fault_severity': 0.0,
    'simulate_packet_loss': 0.0,
    'dt': 1.0
}
post_reset_resp = requests.post(f"{BASE_URL}/simulator/live/step", json=post_reset_payload).json()
ai_post_reset = post_reset_resp["ai_diagnostics"]
frontend_state_post_reset = simulate_frontend_shap_decision(ai_post_reset)
print(f"  Post-Reset Fault: {ai_post_reset['predicted_fault']} (Conf: {ai_post_reset['confidence']*100:.1f}%)")
print(f"  Post-Reset SHAP Panel: {frontend_state_post_reset['rendered']}")
assert frontend_state_post_reset["rendered"] == "EMPTY_STATE", "SHAP panel must return to empty state after reset!"
print("  [PASS] SHAP panel cleared back to empty state after reset.")

print("\n=================================================================")
print("  ALL 6 TEST STEPS PASSED SUCCESSFULLY!")
print("=================================================================")
