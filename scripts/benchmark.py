#!/usr/bin/env python3
"""
AeroTwin Aircraft Engine Digital Twin API High-Throughput Benchmarking Tool
===========================================================
Usage:
  python scripts/benchmark.py
"""

import asyncio
import httpx
import time
import numpy as np
from typing import Dict, List, Any, Tuple


BASE_URL = "http://127.0.0.1:8000"

SAMPLE_TELEMETRY = {
    "cht": 145.0, "egt": 620.0, "rpm": 2400.0, "oil_pressure": 65.0,
    "oil_temp": 90.0, "fuel_flow": 10.5, "vibration": 0.25,
    "battery_voltage": 13.8, "injection_timing": 20.0, "health_index": 0.95,
    "altitude": 3000.0, "ambient_temp": 15.0, "throttle": 0.55, "sensor_cht": 145.0
}


async def benchmark_endpoint_sequential(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Dict[str, Any],
    count: int = 50
) -> List[float]:
    """Sends `count` sequential HTTP POST requests and records individual latencies."""
    latencies = []
    for _ in range(count):
        t0 = time.perf_counter()
        try:
            r = await client.post(endpoint, json=payload, timeout=10.0)
            if r.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass
    return latencies


async def benchmark_endpoint_concurrent(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Dict[str, Any],
    count: int = 50,
    concurrency: int = 10
) -> Tuple[List[float], float]:
    """Sends `count` HTTP POST requests with `concurrency` parallel tasks."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker():
        async with semaphore:
            t0 = time.perf_counter()
            try:
                r = await client.post(endpoint, json=payload, timeout=10.0)
                if r.status_code == 200:
                    latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                pass

    t_start = time.perf_counter()
    tasks = [asyncio.create_task(worker()) for _ in range(count)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start
    return latencies, total_time


async def main():
    print("\n" + "=" * 78)
    print(" [*] AeroTwin Aircraft Engine Digital Twin API HIGH-THROUGHPUT BENCHMARK")
    print("=" * 78)

    endpoints = [
        ("/telemetry", SAMPLE_TELEMETRY),
        ("/telemetry/validated", {"telemetry": SAMPLE_TELEMETRY, "simulate_packet_loss": 0.0}),
        ("/simulator/live/step", {"throttle": 0.55, "altitude_m": 3000.0, "ambient_offset_c": 0.0, "injected_fault": "none", "fault_severity": 0.0, "dt": 1.0}),
        ("/advisory/generate", {
            "fault_type": "cooling", "severity": 0.65, "rul_seconds": 840,
            "current_altitude_m": 3500, "current_throttle": 0.75, "mission_phase": "cruise"
        }),
        ("/explain/fault", {"features": SAMPLE_TELEMETRY})
    ]

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Check server health
        try:
            h = await client.get("/health")
            if h.status_code != 200:
                print(f"[ERROR] API is unreachable at {BASE_URL}")
                return
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return

        print(f"\nTarget Server: {BASE_URL} (FastAPI + TreeSHAP + DataQualityGuard)")
        print(f"{'Endpoint':<25} | {'Mode':<11} | {'Reqs':<5} | {'Avg (ms)':<9} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9} | {'RPS':<7}")
        print("-" * 92)

        for ep, payload in endpoints:
            # 1. Sequential Run
            seq_lats = await benchmark_endpoint_sequential(client, ep, payload, count=30)
            if seq_lats:
                arr = np.array(seq_lats)
                rps = round(len(arr) / (sum(arr) / 1000.0), 1)
                print(f"{ep:<25} | {'Sequential':<11} | {len(arr):<5} | {np.mean(arr):<9.2f} | {np.percentile(arr, 50):<9.2f} | {np.percentile(arr, 95):<9.2f} | {np.percentile(arr, 99):<9.2f} | {rps:<7}")

            # 2. Concurrent Run (concurrency=8)
            conc_lats, total_wall_time = await benchmark_endpoint_concurrent(client, ep, payload, count=30, concurrency=8)
            if conc_lats:
                arr = np.array(conc_lats)
                conc_rps = round(len(arr) / max(0.001, total_wall_time), 1)
                print(f"{ep:<25} | {'Concurrent':<11} | {len(arr):<5} | {np.mean(arr):<9.2f} | {np.percentile(arr, 50):<9.2f} | {np.percentile(arr, 95):<9.2f} | {np.percentile(arr, 99):<9.2f} | {conc_rps:<7}")
            print("-" * 92)

        # Full System Health & Grade
        r_met = await client.get("/metrics/performance")
        if r_met.status_code == 200:
            met = r_met.json()
            print("\n[+] SYSTEM RESOURCE PROFILE:")
            print(f"  * CPU Utilization : {met['system']['cpu_percent']}%")
            print(f"  * Memory Footprint: {met['system']['memory_usage_mb']} MB ({met['system']['memory_percent']}%)")
            print(f"  * Total Uptime    : {met['system']['uptime_human']}")
            print(f"  * Overall Grade   : {met['performance_grade'].upper()}")

    print("=" * 78 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
