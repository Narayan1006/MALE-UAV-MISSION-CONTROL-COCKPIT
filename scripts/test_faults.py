import requests

for fault in ['cooling', 'misfire', 'lubrication', 'injector']:
    requests.post('http://127.0.0.1:8000/simulator/live/reset')
    # Prime engine at cruise for 15s
    for _ in range(15):
        requests.post('http://127.0.0.1:8000/simulator/live/step', json={
            'throttle': 0.70, 'altitude_m': 1000.0, 'ambient_offset_c': 0.0,
            'injected_fault': 'none', 'fault_severity': 0.0, 'simulate_packet_loss': 0.0, 'dt': 1.0
        })
    # Inject fault
    detected = False
    for s in range(1, 30):
        r = requests.post('http://127.0.0.1:8000/simulator/live/step', json={
            'throttle': 0.70, 'altitude_m': 1000.0, 'ambient_offset_c': 0.0,
            'injected_fault': fault, 'fault_severity': 0.75, 'simulate_packet_loss': 0.0, 'dt': 1.0
        }).json()
        ai = r['ai_diagnostics']
        if ai['predicted_fault'] != 'none' and ai['confidence'] > 0.60:
            print(f"Fault {fault}: DETECTED at s={s} as '{ai['predicted_fault']}' (conf={ai['confidence']*100:.1f}%), SHAP={ai['explanation'] is not None}")
            if ai['explanation']:
                top3 = [f['feature_name'] for f in ai['explanation']['top_3_features']]
                print(f"  Top 3 SHAP features: {top3}")
            detected = True
            break
    if not detected:
        print(f"Fault {fault}: not detected within 30s")
