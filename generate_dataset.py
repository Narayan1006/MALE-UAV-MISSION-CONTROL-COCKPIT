"""
Phase 2 — Full Dataset Generation Script
==========================================
Generates 150-200 labelled mission CSVs by calling run_mission() with varying
seeds, fault types, and onset/failure timings, then splits them into
train / val / test at the whole-mission level (no data leakage).

Usage (from project root):
    python generate_dataset.py

Output:
    data/raw/train/  — ~70% of missions
    data/raw/val/    — ~15% of missions
    data/raw/test/   — ~15% of missions

Each CSV filename encodes its metadata, e.g.:
    endurance_healthy_s001.csv
    high_altitude_injector_onset600_s042.csv
"""

import sys
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------------------
# Path setup — works whether run from project root or scripts/ subfolder
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from simulator.engine_simulator import run_mission, MISSION_LIBRARY

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_SEED = 2026          # master seed for reproducibility of the split
RAW_DIR = ROOT / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
VAL_DIR   = RAW_DIR / "val"
TEST_DIR  = RAW_DIR / "test"

MISSIONS = list(MISSION_LIBRARY.keys())   # endurance, high_altitude, hot_weather, rapid_throttle
FAULTS   = ["injector", "lubrication", "cooling", "misfire", "sensor_drift", "vibration"]

# Mission durations (seconds) — used to place onset/failure relative to each mission's length.
# Must stay in sync with MISSION_LIBRARY in engine_simulator.py.
MISSION_DURATIONS = {
    "endurance":      60 + 180 + 300 + 1800 + 180 + 90,   # 2610 s
    "high_altitude":  60 + 300 + 400 + 600  + 200 + 90,   # 1650 s
    "hot_weather":    60 + 180 + 1200 + 90,                # 1530 s
    "rapid_throttle": 60 + 120*5 + 90,                     # 750 s
}

# For missions with hot weather context, add ambient temperature offset
HOT_WEATHER_OFFSET = {"hot_weather": 15.0}   # +15 C above standard atmosphere

# Train / val / test fractions
SPLIT = (0.70, 0.15, 0.15)

# --------------------------------------------------------------------------
# Build the generation manifest
# --------------------------------------------------------------------------
def build_manifest():
    """
    Returns a list of dicts, each describing one mission run to generate.
    Designed to hit ~150-200 total CSVs with good variety.

    Counts:
        Healthy : 4 missions × 12 seeds           = 48
        Faulted : 4 missions × 6 faults × 4 seeds = 96
        ──────────────────────────────────────────────
        Total                                      144
    (Rounded up to 156 by using 5 seeds for high-duration missions.)
    """
    manifest = []

    # ── HEALTHY RUNS ──────────────────────────────────────────────────────
    for mission in MISSIONS:
        # More seeds for longer missions (more temporal diversity in healthy data)
        n_seeds = 14 if mission in ("endurance", "high_altitude") else 10
        for seed in range(1, n_seeds + 1):
            manifest.append({
                "mission":        mission,
                "fault_type":     "none",
                "onset_s":        1e9,
                "failure_s":      1e9,
                "hot_offset":     HOT_WEATHER_OFFSET.get(mission, 0.0),
                "seed":           seed,
                "label":          "healthy",
            })

    # ── FAULTED RUNS ──────────────────────────────────────────────────────
    # Onset at different points in the mission so the model sees early, mid,
    # and late faults, and doesn't just learn "fault starts at timestamp X".
    #
    # onset_frac: fraction of total mission duration at which degradation begins
    # window_frac: duration of the ramp as a fraction of remaining mission time
    ONSET_CONFIGS = [
        (0.25, 0.55),   # early onset, long window
        (0.35, 0.50),   # mid onset, medium window
        (0.50, 0.40),   # mid-late onset
        (0.60, 0.35),   # late onset, short window (rapid degradation scenario)
    ]
    FAULT_SEEDS = [100, 200, 300, 400]   # one seed per onset config

    for mission in MISSIONS:
        duration = MISSION_DURATIONS[mission]
        for fault in FAULTS:
            for (onset_frac, window_frac), seed in zip(ONSET_CONFIGS, FAULT_SEEDS):
                onset_s   = duration * onset_frac
                failure_s = onset_s + duration * window_frac
                # Clamp so failure never exceeds mission end
                failure_s = min(failure_s, duration * 0.98)
                # Ensure onset < failure (minimum 60 s window)
                if failure_s - onset_s < 60:
                    failure_s = onset_s + 60

                manifest.append({
                    "mission":    mission,
                    "fault_type": fault,
                    "onset_s":    round(onset_s, 1),
                    "failure_s":  round(failure_s, 1),
                    "hot_offset": HOT_WEATHER_OFFSET.get(mission, 0.0),
                    "seed":       seed,
                    "label":      fault,
                })

    return manifest


# --------------------------------------------------------------------------
# Filename builder
# --------------------------------------------------------------------------
def make_filename(entry: dict) -> str:
    if entry["fault_type"] == "none":
        return f"{entry['mission']}_healthy_s{entry['seed']:03d}.csv"
    else:
        onset = int(entry["onset_s"])
        return f"{entry['mission']}_{entry['fault_type']}_onset{onset}_s{entry['seed']:03d}.csv"


# --------------------------------------------------------------------------
# Split manifest into train / val / test (at whole-mission level)
# --------------------------------------------------------------------------
def split_manifest(manifest: list, seed: int):
    """
    Stratified split: for each (mission, fault_type) group, split runs
    proportionally so every fault type appears in train, val, AND test.

    With n=4 runs per group:
        test  = 1  (guaranteed minimum)
        val   = 1  (guaranteed minimum)
        train = 2  (remainder)
    """
    rng = random.Random(seed)

    from collections import defaultdict
    groups = defaultdict(list)
    for entry in manifest:
        groups[(entry["mission"], entry["fault_type"])].append(entry)

    train, val, test = [], [], []
    for key, entries in groups.items():
        rng.shuffle(entries)
        n = len(entries)
        # Guarantee at least 1 in test and val; give remainder to train
        n_test  = max(1, round(n * SPLIT[2]))
        n_val   = max(1, round(n * SPLIT[1]))
        n_train = max(1, n - n_val - n_test)
        test.extend(entries[:n_test])
        val.extend(entries[n_test:n_test + n_val])
        train.extend(entries[n_test + n_val:])

    return train, val, test


# --------------------------------------------------------------------------
# Generate one CSV
# --------------------------------------------------------------------------
def generate_one(entry: dict, out_dir: Path) -> Path:
    df = run_mission(
        mission_name=entry["mission"],
        fault_type=entry["fault_type"],
        onset_s=entry["onset_s"],
        failure_s=entry["failure_s"],
        hot_weather_offset=entry["hot_offset"],
        dt=1.0,
        seed=entry["seed"],
    )
    # Inject metadata columns for traceability
    df.insert(0, "mission_type",  entry["mission"])
    df.insert(1, "mission_seed",  entry["seed"])
    df.insert(2, "fault_label",   entry["label"])

    out_path = out_dir / make_filename(entry)
    df.to_csv(out_path, index=False)
    return out_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    # Ensure output directories exist and are clean
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Building generation manifest...")
    manifest = build_manifest()
    print(f"  Total missions planned: {len(manifest)}")
    print(f"  Healthy: {sum(1 for e in manifest if e['fault_type'] == 'none')}")
    print(f"  Faulted: {sum(1 for e in manifest if e['fault_type'] != 'none')}")
    print()

    train_m, val_m, test_m = split_manifest(manifest, seed=RANDOM_SEED)
    print(f"Split -> train: {len(train_m)} | val: {len(val_m)} | test: {len(test_m)}")
    print()

    # Fault representation check
    from collections import Counter
    for split_name, split_data, out_dir in [
        ("TRAIN", train_m, TRAIN_DIR),
        ("VAL",   val_m,   VAL_DIR),
        ("TEST",  test_m,  TEST_DIR),
    ]:
        fault_counts = Counter(e["fault_type"] for e in split_data)
        print(f"[{split_name}] fault type counts: {dict(fault_counts)}")

    print()
    print("Generating CSVs...")

    total = len(manifest)
    generated = 0
    errors = []

    for split_name, split_data, out_dir in [
        ("train", train_m, TRAIN_DIR),
        ("val",   val_m,   VAL_DIR),
        ("test",  test_m,  TEST_DIR),
    ]:
        for entry in tqdm(split_data, desc=f"  {split_name:5s}", unit="mission"):
            try:
                path = generate_one(entry, out_dir)
                generated += 1
            except Exception as exc:
                errors.append((entry, exc))
                print(f"\n  ERROR: {make_filename(entry)} — {exc}")

    print()
    print("=" * 60)
    print(f"  Generated : {generated} / {total} CSVs")
    print(f"  Errors    : {len(errors)}")
    print(f"  Train dir : {TRAIN_DIR}  ({len(list(TRAIN_DIR.glob('*.csv')))} files)")
    print(f"  Val dir   : {VAL_DIR}   ({len(list(VAL_DIR.glob('*.csv')))} files)")
    print(f"  Test dir  : {TEST_DIR}  ({len(list(TEST_DIR.glob('*.csv')))} files)")
    print("=" * 60)

    if errors:
        print("\nFailed entries:")
        for entry, exc in errors:
            print(f"  {make_filename(entry)}: {exc}")
        sys.exit(1)
    else:
        print("\nPhase 2 complete. Run ml/validate_data.py next (Phase 3).")


if __name__ == "__main__":
    main()
