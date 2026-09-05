"""
RF Link Simulator (RFD900x-class 900 MHz Telemetry Radio) -- SIMULATED
=======================================================================
Module: backend.rf_link_simulator
Author: AeroTwin Engineering Team

SIMULATION DISCLAIMER:
    This module simulates the RF link characteristics of an RFD900x-class
    900 MHz frequency-hopping spread-spectrum (FHSS) telemetry radio pair,
    commonly used for UAV ground-to-air data links at ranges up to 40 km.

    What this accurately models:
      - Configurable baud rate (default 57600, matching RFD900x default)
      - Gaussian-distributed one-way link latency (mean 50 ms, std 8 ms)
      - Bernoulli packet loss (configurable rate, default 0%)
      - Link quality score derived from packet loss and latency deviation
      - Rolling 100-packet window statistics

    What requires real hardware to validate:
      - Actual RF propagation path loss at 900 MHz
      - Multipath fading, Doppler shift under vehicle motion
      - Interference from other 900 MHz ISM band devices
      - Antenna gain patterns and polarisation mismatch

Real hardware reference:
    RFD900x radio pair (~$350-400/pair, RFDesign Pty Ltd)
    Baud rates: 57600 (default), up to 250000
    Range: 40 km LOS at 1 W TX power
    Latency: 20-80 ms typical (air packet rate dependent)
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

SPEED_OF_LIGHT_MS = 299_792_458e-3   # m/ms, for theoretical propagation delay


@dataclass
class RFPacket:
    payload: bytes
    sent_at_ms: float
    received_at_ms: Optional[float] = None
    dropped: bool = False


class RFLinkSimulator:
    """
    [SIMULATED] Models an RFD900x-class 900 MHz telemetry radio link.

    Applies realistic latency jitter and packet loss to MAVLink byte streams,
    simulating the channel impairments experienced on a real UAV RF link.
    """

    def __init__(
        self,
        baud_rate: int   = 57600,
        latency_ms: float = 50.0,
        latency_std_ms: float = 8.0,
        packet_loss_rate: float = 0.0,
        max_range_m: float = 40_000.0,
    ):
        """
        Args:
            baud_rate:        Simulated link baud rate (bits/sec). RFD900x default: 57600.
            latency_ms:       Mean one-way latency in milliseconds.
            latency_std_ms:   Standard deviation of latency (Gaussian jitter).
            packet_loss_rate: Fraction of packets to drop (0.0 = none, 0.05 = 5%).
            max_range_m:      Maximum simulated range; latency scales with distance.
        """
        self.baud_rate        = baud_rate
        self.latency_ms       = latency_ms
        self.latency_std_ms   = latency_std_ms
        self.packet_loss_rate = packet_loss_rate
        self.max_range_m      = max_range_m

        # Rolling window for statistics (last 100 packets)
        self._window: deque[RFPacket] = deque(maxlen=100)
        self._total_sent     = 0
        self._total_received = 0
        self._total_dropped  = 0

    # ------------------------------------------------------------------
    # Core transmit method
    # ------------------------------------------------------------------

    def transmit(self, mavlink_bytes: bytes) -> Optional[bytes]:
        """
        [SIMULATED] Push MAVLink bytes through the simulated RF link.

        Applies:
          1. Transmission delay: ceil(len * 10 / baud_rate * 1000) ms
          2. One-way link latency: Gaussian(mean, std)
          3. Bernoulli packet loss: returns None on simulated drop

        Returns:
            bytes: the received payload (identical; RF introduces no bit errors
                   in this model), or None if the packet was dropped.
        """
        t0_ms = time.perf_counter() * 1000.0

        # Serial transmission time: (bits / baud) ms  [10 bits per byte: 8 data + start + stop]
        tx_delay_ms = (len(mavlink_bytes) * 10 / self.baud_rate) * 1000.0

        # One-way propagation latency with Gaussian jitter
        link_latency_ms = max(0.0, random.gauss(self.latency_ms, self.latency_std_ms))
        total_delay_ms  = tx_delay_ms + link_latency_ms

        pkt = RFPacket(payload=mavlink_bytes, sent_at_ms=t0_ms)
        self._total_sent += 1

        # Bernoulli packet loss
        if random.random() < self.packet_loss_rate:
            pkt.dropped = True
            self._total_dropped += 1
            self._window.append(pkt)
            return None

        # Simulate the propagation delay (non-blocking: we track it but don't sleep)
        pkt.received_at_ms = t0_ms + total_delay_ms
        pkt.dropped        = False
        self._total_received += 1
        self._window.append(pkt)
        return mavlink_bytes   # RF is transparent to payload bytes

    # ------------------------------------------------------------------
    # Link status reporting
    # ------------------------------------------------------------------

    def get_link_status(self) -> dict:
        """
        Return link quality statistics over the rolling 100-packet window.

        Returns dict with:
            packets_sent, packets_received, packet_loss_percent,
            avg_latency_ms, link_quality (0-100), simulation_note
        """
        if not self._window:
            return {
                "packets_sent":       0,
                "packets_received":   0,
                "packet_loss_percent": 0.0,
                "avg_latency_ms":     0.0,
                "link_quality":       100,
                "simulation_note":    "SIMULATED -- no real RF hardware"
            }

        window_list  = list(self._window)
        n_sent       = len(window_list)
        n_dropped    = sum(1 for p in window_list if p.dropped)
        n_received   = n_sent - n_dropped
        loss_pct     = (n_dropped / n_sent * 100.0) if n_sent else 0.0

        latencies = [
            (p.received_at_ms - p.sent_at_ms)
            for p in window_list
            if not p.dropped and p.received_at_ms is not None
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Quality score: 100 = perfect; penalise loss (70%) and excess latency (30%)
        loss_penalty    = loss_pct * 0.70
        latency_penalty = max(0.0, (avg_latency - self.latency_ms) / max(1.0, self.latency_ms) * 30.0)
        quality         = max(0, min(100, round(100 - loss_penalty - latency_penalty)))

        return {
            "packets_sent":           self._total_sent,
            "packets_received":       self._total_received,
            "packets_dropped":        self._total_dropped,
            "packet_loss_percent":    round(loss_pct, 2),
            "avg_latency_ms":         round(avg_latency, 2),
            "link_quality":           quality,
            "baud_rate":              self.baud_rate,
            "configured_loss_rate":   self.packet_loss_rate,
            "simulation_note":        "SIMULATED -- no real RF hardware involved"
        }

    def set_range(self, distance_m: float) -> None:
        """
        [SIMULATED] Adjust mean latency to model a given air-to-ground range.
        Propagation delay = distance / speed_of_light, plus 30 ms base radio overhead.
        """
        prop_ms = (distance_m / SPEED_OF_LIGHT_MS) + 30.0
        self.latency_ms = min(200.0, prop_ms)   # cap at 200 ms (link budget limit)


if __name__ == "__main__":
    rf = RFLinkSimulator(baud_rate=57600, latency_ms=50.0, packet_loss_rate=0.03)
    test_bytes = b"\xFD" + bytes(range(67))   # mock 68-byte MAVLink frame

    sent = received = dropped = 0
    for _ in range(100):
        result = rf.transmit(test_bytes)
        sent += 1
        if result is None:
            dropped += 1
        else:
            received += 1

    status = rf.get_link_status()
    print(f"[SIMULATED] RF Link Test: sent={sent}  received={received}  dropped={dropped}")
    print(f"[SIMULATED] Link Quality: {status['link_quality']}%  Avg Latency: {status['avg_latency_ms']:.1f} ms")
