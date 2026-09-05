"""
Sensor Robustness & Real-Time Data Quality Guard Layer
======================================================
Module: backend.data_quality_guard
Author: AeroTwin Engineering Team
Description:
  High-frequency (1 Hz) telemetry hygiene guard. Performs statistical outlier
  filtering, packet loss dropout detection, adaptive last-known/model-based imputation,
  multi-cylinder consensus check, and sensor health tracking.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


RAW_SENSOR_CHANNELS = [
    "cht", "egt", "rpm", "oil_pressure", "oil_temp",
    "fuel_flow", "vibration", "battery_voltage",
    "injection_timing", "health_index", "altitude",
    "ambient_temp", "throttle", "sensor_cht"
]

NOMINAL_DEFAULTS = {
    "cht": 150.0,
    "sensor_cht": 150.0,
    "egt": 580.0,
    "rpm": 2400.0,
    "oil_pressure": 55.0,
    "oil_temp": 85.0,
    "fuel_flow": 10.0,
    "vibration": 0.35,
    "battery_voltage": 13.8,
    "injection_timing": 20.0,
    "health_index": 0.98,
    "altitude": 1500.0,
    "ambient_temp": 15.0,
    "throttle": 0.70
}


@dataclass
class SensorChannelState:
    """Tracks rolling statistics and quality state for one sensor channel."""
    name: str
    window_size: int = 30
    values: deque = field(default_factory=lambda: deque(maxlen=30))
    last_good_value: Optional[float] = None
    confidence: float = 1.0  # Decays on dropout / noise
    missing_streak: int = 0
    outlier_suspicion_score: float = 0.0  # Increments on statistical outlier
    is_degraded: bool = False

    def get_rolling_stats(self) -> Tuple[float, float]:
        """Return (mean, std) of current window. If window < 2, return (last_val, 0.0)."""
        valid_vals = [v for v in self.values if v is not None and not math.isnan(v)]
        if len(valid_vals) < 2:
            base = valid_vals[0] if valid_vals else (self.last_good_value or NOMINAL_DEFAULTS.get(self.name, 0.0))
            return float(base), 0.0
        return float(np.mean(valid_vals)), float(np.std(valid_vals))

    def get_rolling_median(self) -> float:
        """Return median of rolling window or default."""
        valid_vals = [v for v in self.values if v is not None and not math.isnan(v)]
        if not valid_vals:
            return float(self.last_good_value or NOMINAL_DEFAULTS.get(self.name, 0.0))
        return float(np.median(valid_vals))

    def update(
        self,
        raw_value: Optional[float],
        dt: float = 1.0,
        model_predicted_value: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Process a new raw sensor reading through the quality pipeline:
        1. Check if NaN / missing / None
        2. Statistical Z-score outlier filtering
        3. Imputation (last-known or model-predicted)
        4. State update & quality flag return
        """
        is_missing = (
            raw_value is None or
            (isinstance(raw_value, float) and (math.isnan(raw_value) or math.isinf(raw_value)))
        )

        # ----------------------------------------------------
        # 1. Missing Value / Dropout Handling
        # ----------------------------------------------------
        if is_missing:
            self.missing_streak += 1
            self.confidence = max(0.10, self.confidence * 0.95)
            self.is_degraded = True

            if self.missing_streak <= 5:
                # Short-term dropout: use last good value
                if self.last_good_value is not None:
                    cleaned_val = self.last_good_value
                elif self.values:
                    cleaned_val = self.get_rolling_median()
                else:
                    cleaned_val = NOMINAL_DEFAULTS.get(self.name, 0.0)
                flag_status = "imputed_last_known"
            else:
                # Extended dropout: use Digital Twin model or median
                if model_predicted_value is not None and not math.isnan(model_predicted_value):
                    cleaned_val = model_predicted_value
                else:
                    cleaned_val = self.get_rolling_median()
                flag_status = "imputed_model"

            self.values.append(cleaned_val)
            quality_info = {
                "status": flag_status,
                "confidence": round(self.confidence, 3),
                "imputed": True,
                "missing_streak": self.missing_streak,
                "original_value": None,
                "corrected_value": round(float(cleaned_val), 2)
            }
            return cleaned_val, quality_info

        # ----------------------------------------------------
        # 2. Present Value Processing & Outlier / Drift Check
        # ----------------------------------------------------
        val = float(raw_value)
        self.missing_streak = 0
        self.confidence = min(1.0, self.confidence + 0.10)

        mu, sigma = self.get_rolling_stats()
        imputed = False
        original_val = val
        cleaned_val = val

        # Need at least 5 samples in rolling window for reliable sigma
        if len(self.values) >= 5:
            effective_sigma = max(sigma, 0.03 * max(abs(mu), 1.0))
            z_score = abs(val - mu) / effective_sigma
            if z_score > 3.5:
                self.outlier_suspicion_score += 1.0
                if self.outlier_suspicion_score <= 2.0:
                    # Isolated noise spike / EMI spike: Correct to rolling median
                    cleaned_val = self.get_rolling_median()
                    flag_status = "outlier_corrected"
                    imputed = True
                    self.confidence = max(0.50, self.confidence * 0.85)
                else:
                    # Persistent outlier: Suspect sensor drift / calibration offset
                    flag_status = "sensor_drift_suspected"
                    cleaned_val = val
                    self.confidence = 0.50
                    self.is_degraded = True
            else:
                self.outlier_suspicion_score = max(0.0, self.outlier_suspicion_score - 0.5)
                flag_status = "valid"
        else:
            self.outlier_suspicion_score = max(0.0, self.outlier_suspicion_score - 0.2)
            flag_status = "valid"

        # Update last good value if reading is valid or corrected
        if flag_status in ["valid", "outlier_corrected"]:
            self.last_good_value = cleaned_val

        self.values.append(cleaned_val)
        self.is_degraded = (self.confidence < 0.90 or flag_status != "valid")

        quality_info = {
            "status": flag_status,
            "confidence": round(self.confidence, 3),
            "imputed": imputed,
            "missing_streak": 0,
            "original_value": round(original_val, 2),
            "corrected_value": round(float(cleaned_val), 2)
        }
        return cleaned_val, quality_info


class DataQualityGuard:
    """
    Complete Telemetry Data Hygiene Guard with multi-sensor consensus and
    dropout/packet loss recovery.
    """
    def __init__(self, channels: Optional[List[str]] = None, window_size: int = 30):
        self.channel_names = channels or RAW_SENSOR_CHANNELS
        self.window_size = window_size
        self.channels: Dict[str, SensorChannelState] = {
            ch: SensorChannelState(name=ch, window_size=window_size)
            for ch in self.channel_names
        }

    def reset(self):
        """Resets all channel tracking states."""
        self.channels = {
            ch: SensorChannelState(name=ch, window_size=self.window_size)
            for ch in self.channel_names
        }

    def simulate_packet_loss(self, frame: Dict[str, float], loss_rate: float = 0.05) -> Dict[str, Any]:
        """
        Simulates telemetry packet loss by setting random channels to NaN.
        loss_rate: 0.0 to 1.0 (fraction of channels dropped)
        """
        if loss_rate <= 0.0:
            return dict(frame)

        corrupted = dict(frame)
        for ch in corrupted:
            if random.random() < loss_rate:
                corrupted[ch] = float("nan")
        return corrupted

    def process_telemetry_frame(
        self,
        frame: Dict[str, Any],
        dt: float = 1.0,
        model_predicted_values: Optional[Dict[str, float]] = None,
        simulated_packet_loss: bool = False
    ) -> Dict[str, Any]:
        """
        Processes an entire telemetry frame (all 14 raw sensors + optional synthetic cylinder channels).
        """
        cleaned_data: Dict[str, float] = {}
        quality_flags: Dict[str, Any] = {}

        missing_count = 0
        outlier_count = 0
        imputed_count = 0

        # Process each registered sensor channel
        for ch_name in self.channel_names:
            if ch_name not in self.channels:
                self.channels[ch_name] = SensorChannelState(name=ch_name, window_size=self.window_size)

            ch_state = self.channels[ch_name]
            raw_val = frame.get(ch_name)
            if raw_val is None and ch_name == "sensor_cht" and "cht" in frame:
                raw_val = frame.get("cht")
            elif raw_val is None and ch_name == "cht" and "sensor_cht" in frame:
                raw_val = frame.get("sensor_cht")

            model_pred = model_predicted_values.get(ch_name) if model_predicted_values else None
            cleaned_val, q_info = ch_state.update(raw_val, dt=dt, model_predicted_value=model_pred)

            cleaned_data[ch_name] = cleaned_val
            quality_flags[ch_name] = q_info

            if q_info["status"] in ["imputed_last_known", "imputed_model"]:
                missing_count += 1
                imputed_count += 1
            elif q_info["status"] == "outlier_corrected":
                outlier_count += 1
                imputed_count += 1

        # Preserve any extra metadata fields present in frame (e.g. timestamp, fault_type)
        for k, v in frame.items():
            if k not in cleaned_data:
                cleaned_data[k] = v

        # ----------------------------------------------------
        # Multi-Cylinder Consensus Check (if cylinders present)
        # ----------------------------------------------------
        cyl_keys = ["cht_cyl_1", "cht_cyl_2", "cht_cyl_3", "cht_cyl_4"]
        present_cyls = [k for k in cyl_keys if k in frame and frame[k] is not None and not math.isnan(frame[k])]
        if len(present_cyls) >= 3:
            vals = [float(frame[k]) for k in present_cyls]
            med = float(np.median(vals))
            mad = float(np.median([abs(v - med) for v in vals])) or 1.0
            for k in present_cyls:
                v = float(frame[k])
                if abs(v - med) > 2.5 * mad and abs(v - med) > 15.0:
                    quality_flags[k] = {
                        "status": "cylinder_sensor_drift",
                        "confidence": 0.60,
                        "imputed": True,
                        "original_value": v,
                        "corrected_value": med
                    }
                    cleaned_data[k] = med
                    imputed_count += 1

        # ----------------------------------------------------
        # Compute Overall Telemetry Quality Summary
        # ----------------------------------------------------
        confidences = [self.channels[ch].confidence for ch in self.channel_names if ch in self.channels]
        overall_health = round(float(np.mean(confidences)), 3) if confidences else 1.0

        degraded_channels = [
            ch for ch, q in quality_flags.items()
            if q["confidence"] < 0.90 or q["status"] != "valid"
        ]

        sensor_drift_suspected = any(
            self.channels[ch].outlier_suspicion_score > 3.0 or q.get("status") == "sensor_drift_suspected"
            for ch, q in quality_flags.items()
            if ch in self.channels
        )

        return {
            "cleaned_data": cleaned_data,
            "quality_flags": quality_flags,
            "data_quality_summary": {
                "overall_health": overall_health,
                "degraded_channels": degraded_channels,
                "missing_count": missing_count,
                "outlier_count": outlier_count,
                "imputed_count": imputed_count,
                "sensor_drift_suspected": sensor_drift_suspected,
                "packet_loss_simulated": simulated_packet_loss
            }
        }

    def get_channel_health_report(self) -> Dict[str, dict]:
        """Returns diagnostic health summary of all monitored sensor channels."""
        report = {}
        for name, ch in self.channels.items():
            mu, sigma = ch.get_rolling_stats()
            report[name] = {
                "confidence": round(ch.confidence, 3),
                "missing_streak": ch.missing_streak,
                "outlier_suspicion_score": round(ch.outlier_suspicion_score, 2),
                "is_degraded": ch.is_degraded,
                "rolling_mean": round(mu, 2),
                "rolling_std": round(sigma, 2),
                "last_good_value": round(ch.last_good_value, 2) if ch.last_good_value is not None else None
            }
        return report
