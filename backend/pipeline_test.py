"""
AeroTwin End-to-End Deployment Pipeline Test [SIMULATED]
=========================================================
Tests the full round-trip:
    Sensor data (dict)
    -> MAVLink pack   (mavlink_interface.py)
    -> RF transmit    (rf_link_simulator.py)
    -> MAVLink parse  (mavlink_interface.py)
    -> Edge inference (edge_compute.py)

Prints latency breakdown for each stage. All hardware is SIMULATED.
"""

import time
import sys
from pathlib import Path

# Make sure backend/ is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.mavlink_interface  import MAVLinkInterface
from backend.rf_link_simulator  import RFLinkSimulator
from backend.edge_compute       import EdgeComputeNode


def run_pipeline_test(n_frames: int = 20):
    print("=" * 65)
    print("  AeroTwin Deployment Pipeline Test  [ALL SIMULATED]")
    print("=" * 65)

    iface = MAVLinkInterface(system_id=1, component_id=200)
    rf    = RFLinkSimulator(baud_rate=57600, latency_ms=50.0, latency_std_ms=8.0, packet_loss_rate=0.03)
    edge  = EdgeComputeNode(target_latency_ms=100.0)

    # Varied telemetry frames (mix of nominal and degraded)
    test_frames = []
    for i in range(n_frames):
        sev = (i / n_frames) * 0.7
        test_frames.append({
            "rpm":              5200.0 - sev * 600,
            "true_cht":         165.0  + sev * 95.0,
            "sensor_cht":       166.0  + sev * 95.0,
            "egt":              610.0  + sev * 190.0,
            "oil_pressure":     68.0   - sev * 38.0,
            "oil_temp":         92.0   + sev * 30.0,
            "fuel_flow":        5.6,
            "vibration":        0.12   + sev * 0.40,
            "battery_voltage":  13.8,
            "injection_timing": 28.5,
            "health_index":     max(0.1, 1.0 - sev),
            "altitude":         2800.0,
            "ambient_temp":     288.15,
            "throttle":         0.68,
        })

    pack_times   = []
    rf_times     = []
    parse_times  = []
    infer_times  = []
    total_times  = []
    dropped      = 0
    results      = []

    for telem in test_frames:
        t_total_start = time.perf_counter()

        # Stage 1: MAVLink pack
        t0 = time.perf_counter()
        packet = iface.pack_telemetry(telem)
        pack_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 2: RF transmit
        t0 = time.perf_counter()
        received = rf.transmit(packet)
        rf_ms = (time.perf_counter() - t0) * 1000.0

        if received is None:
            dropped += 1
            continue   # packet dropped -- skip this frame

        # Stage 3: MAVLink parse
        t0 = time.perf_counter()
        parsed_telem = iface.parse_telemetry(received)
        parse_ms = (time.perf_counter() - t0) * 1000.0

        if parsed_telem is None:
            print("  [WARN] CRC failure on received packet -- skipping")
            continue

        # Stage 4: Edge inference
        infer_result = edge.run_lightweight_inference(parsed_telem)
        infer_ms = infer_result["edge_processing_time_ms"]

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        pack_times.append(pack_ms)
        rf_times.append(rf_ms)
        parse_times.append(parse_ms)
        infer_times.append(infer_ms)
        total_times.append(total_ms)
        results.append(infer_result)

    # Summary
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    def mx(lst):  return max(lst) if lst else 0.0

    print(f"\n  Frames tested:         {n_frames}")
    print(f"  Frames processed:      {len(total_times)}")
    print(f"  Dropped (RF loss sim): {dropped}")
    print()
    print(f"  Stage latencies (mean / max) over {len(total_times)} frames:")
    print(f"    MAVLink pack:   {avg(pack_times):.3f} ms  /  {mx(pack_times):.3f} ms")
    print(f"    RF transmit:    {avg(rf_times):.3f} ms  /  {mx(rf_times):.3f} ms   [simulated 50ms mean]")
    print(f"    MAVLink parse:  {avg(parse_times):.3f} ms  /  {mx(parse_times):.3f} ms")
    print(f"    Edge inference: {avg(infer_times):.3f} ms  /  {mx(infer_times):.3f} ms")
    print(f"                    {'─'*42}")
    print(f"    Total pipeline: {avg(total_times):.3f} ms  /  {mx(total_times):.3f} ms")
    print()

    link = rf.get_link_status()
    print(f"  RF Link Status:")
    print(f"    Packets sent:       {link['packets_sent']}")
    print(f"    Packets received:   {link['packets_received']}")
    print(f"    Packet loss:        {link['packet_loss_percent']}%")
    print(f"    Avg latency:        {link['avg_latency_ms']} ms")
    print(f"    Link quality:       {link['link_quality']}%")
    print()

    # Sample predictions
    print(f"  Sample predictions (last 5 frames):")
    for r in results[-5:]:
        print(f"    anomaly={r['anomaly_score']:.3f}  fault={r['preliminary_fault']:12s}  "
              f"conf={r['confidence']:.2f}  edge_ms={r['edge_processing_time_ms']:.2f}")

    print()
    edge_bench = edge.benchmark(100)
    print(f"  Edge Compute 100-run Benchmark:")
    print(f"    Mean:   {edge_bench['mean_ms']} ms (host)  /  ~{edge_bench['pi4_estimate_mean_ms']} ms (Pi 4 est.)")
    print(f"    P95:    {edge_bench['p95_ms']} ms")
    print(f"    P99:    {edge_bench['p99_ms']} ms")
    print(f"    Budget: {edge_bench['budget_pass_rate']} frames under {edge_bench['target_ms']} ms")
    print()
    print(f"  NOTE: {results[0]['simulation_note']}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline_test(n_frames=20)
