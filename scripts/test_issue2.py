import requests
import json
import numpy as np

res_reset = requests.post('http://127.0.0.1:8000/simulator/live/reset')
print("Reset status:", res_reset.json())

# Ramp throttle from idle (0.15) to 70% (0.70) over 30 seconds
print("\n--- Ramping throttle from idle to 70% over 30s ---")
for sec in range(1, 36):
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
    r = requests.post('http://127.0.0.1:8000/simulator/live/step', json=payload).json()
    ai = r['ai_diagnostics']
    p = r['physical_telemetry']
    adv = r['advisory']
    print(f"t={sec:2d}s | thr={throttle:.2f} | RPM={p['rpm']:4.0f} | CHT={p['true_cht']:5.1f}C | anom={ai['anomaly_score']:.4f} | is_anom={ai['is_anomaly']} | fault={ai['predicted_fault']} ({ai['confidence']*100:.0f}%) | shap={ai['explanation'] is not None}")
