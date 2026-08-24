# Phase 3 — Data Validation Report

**Status: ✅ ALL CHECKS PASSED**

---

## Dataset Summary

| Split | Files | Total rows |
|-------|-------|------------|
| train | 80 | 134,840 |
| val | 32 | 52,352 |
| test | 32 | 52,352 |
| **total** | **144** | **239,544** |

---

## Check 1 — NaN Audit

**Result:** PASS — zero NaNs across all CSVs

---

## Check 2 — Physical Range Bounds

**Result:** PASS — all channels within plausible physical bounds

---

## Check 3 — Fault Severity Ramp

**Result:** PASS — healthy files have severity=0, faulted files ramp to ≥0.5

---

## Check 4 — Class Balance (row counts)

                 all  train    val   test
fault_type                               
cooling        15046   8364   3500   3182
injector       15046   7595   3580   3871
lubrication    15046   8011   3991   3044
misfire        15046   6304   4406   4336
none          149268  91579  28947  28742
sensor_drift   15046   6074   4253   4719
vibration      15046   6913   3675   4458

> **Note:** Class imbalance (healthy >> individual fault types) is expected and normal.
> Will be addressed in Phase 5 via XGBoost `scale_pos_weight` or class weighting.

---

## Plots Generated

All plots saved to `docs/validation/plots/`.

- `timeseries_injector.png` — healthy vs injector fault, 4-channel time series
- `timeseries_lubrication.png` — healthy vs lubrication fault, 4-channel time series
- `timeseries_cooling.png` — healthy vs cooling fault, 4-channel time series
- `timeseries_misfire.png` — healthy vs misfire fault, 4-channel time series
- `timeseries_sensor_drift.png` — healthy vs sensor_drift fault, 4-channel time series
- `timeseries_vibration.png` — healthy vs vibration fault, 4-channel time series
- `severity_vs_health.png` — fault_severity vs health_index scatter per fault type
- `class_balance.png` — row-level class distribution across train/val/test

---

## Phase 3 Verdict

- [x] No NaNs
- [x] Physical ranges plausible
- [x] Severity ramp correct in faulted files
- [x] All fault types in all splits

**Proceed to Phase 4 (anomaly detector).**