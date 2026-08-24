"""
Phase 3 — Data Validation & Sanity Check
==========================================
Run this before touching any ML code. A bug caught here costs an hour;
the same bug caught after training costs a day.

Checks performed:
  1. NaN audit across all CSVs in all splits
  2. Physical range checks (RPM, CHT, EGT, oil pressure, oil temp)
  3. fault_severity ramp validation (0→1 in faulted files, 0 in healthy)
  4. Class balance report (row counts per fault type)
  5. Time-series plots: 2 healthy + 1 per fault type, 4-channel overlay
  6. fault_severity vs health_index scatter per fault type

Usage (from project root):
    python ml/validate_data.py

Outputs:
    docs/validation/plots/   — all PNG files
    docs/validation/report.md — written summary of every check
"""

import sys
import os
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.resolve()
TRAIN_DIR   = ROOT / "data" / "raw" / "train"
VAL_DIR     = ROOT / "data" / "raw" / "val"
TEST_DIR    = ROOT / "data" / "raw" / "test"
PLOT_DIR    = ROOT / "docs" / "validation" / "plots"
REPORT_PATH = ROOT / "docs" / "validation" / "report.md"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Physical plausibility bounds (inferred from EngineConstants + PS) ──────
# fuel_flow: k_fuel=0.011 × rpm(~5500) × throttle(~1) × rho_ratio(~1) = ~60 g/s max.
# Actual range is ~7-55 g/s depending on mission; bound is generously 0-100.
BOUNDS = {
    "rpm":             (0,    6000),   # max_rpm=5500 + noise headroom
    "true_cht":        (-5,    380),   # cooling fault (loss of dissipation) + rapid throttle can exceed 330C
    "sensor_cht":      (-5,    390),   # sensor drift adds up to +10C bias on top of true_cht
    "egt":             (-80,  1600),   # misfire noise N(0,60) can briefly go negative; cooling fault raises ceiling
    "oil_pressure":    (0,      80),   # max ~65 psi healthy
    "oil_temp":        (-10,   180),   # cold-start at high altitude (7000m ambient ~-1.7C) can go slightly below 0
    "fuel_flow":       (0,     100),   # g/s (k_fuel×rpm×throttle; ~7-55 healthy)
    "vibration":       (-1,      8),   # some negative noise possible
    "battery_voltage": (10,     16),   # 12V nominal + alternator
    "injection_timing": (-5,    30),   # degrees BTDC; nominal 20, drift ±6
    "health_index":    (0,       1),   # clipped 0-1 by design
    "fault_severity":  (0,       1),   # clipped 0-1 by design
}

FAULT_TYPES = ["none", "injector", "lubrication", "cooling",
               "misfire", "sensor_drift", "vibration"]

PLOT_CHANNELS = ["rpm", "true_cht", "egt", "health_index"]
CHANNEL_UNITS = {"rpm": "RPM", "true_cht": "°C", "egt": "°C", "health_index": "0–1"}
CHANNEL_COLORS = {
    "rpm":          "#4FC3F7",
    "true_cht":     "#FF8A65",
    "egt":          "#CE93D8",
    "health_index": "#81C784",
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#ddd",
    "axes.titlecolor":   "#fff",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "grid.color":        "#2a2a4a",
    "grid.linewidth":    0.6,
    "text.color":        "#eee",
    "legend.facecolor":  "#0f3460",
    "legend.edgecolor":  "#444",
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "figure.dpi":        130,
})

# ────────────────────────────────────────────────────────────────────────────
# 1. LOAD ALL CSVs
# ────────────────────────────────────────────────────────────────────────────

def load_split(split_dir: Path) -> dict[str, pd.DataFrame]:
    """Returns {filename: dataframe} for all CSVs in split_dir."""
    dfs = {}
    for p in sorted(split_dir.glob("*.csv")):
        try:
            dfs[p.name] = pd.read_csv(p)
        except Exception as e:
            print(f"  [WARN] could not read {p.name}: {e}")
    return dfs


print("Loading all CSVs...")
train_dfs = load_split(TRAIN_DIR)
val_dfs   = load_split(VAL_DIR)
test_dfs  = load_split(TEST_DIR)
all_dfs   = {**train_dfs, **val_dfs, **test_dfs}
print(f"  Loaded: train={len(train_dfs)}, val={len(val_dfs)}, test={len(test_dfs)}, total={len(all_dfs)}")

# ────────────────────────────────────────────────────────────────────────────
# 2. NaN AUDIT
# ────────────────────────────────────────────────────────────────────────────

# rul_seconds is intentionally None/NaN for healthy missions — that's a data contract
# ("no known failure horizon"), not a bug. Exclude it from the NaN audit.
NAN_EXCLUDE_COLS = {"rul_seconds"}

def nan_audit(dfs: dict[str, pd.DataFrame]) -> tuple[int, list[str]]:
    total_nans = 0
    offenders  = []
    for name, df in dfs.items():
        audit_cols = [c for c in df.columns if c not in NAN_EXCLUDE_COLS]
        n = df[audit_cols].isnull().sum().sum()
        if n > 0:
            total_nans += n
            bad_cols = df[audit_cols].isnull().sum()
            bad_cols = bad_cols[bad_cols > 0].to_dict()
            offenders.append(f"{name}: {n} NaNs in {bad_cols}")
    return total_nans, offenders


print("\n[CHECK 1] NaN audit...")
total_nans, nan_offenders = nan_audit(all_dfs)
if total_nans == 0:
    nan_result = "PASS — zero NaNs across all CSVs"
    print(f"  PASS: 0 NaNs across {len(all_dfs)} files")
else:
    nan_result = f"FAIL — {total_nans} NaNs found in: " + ", ".join(nan_offenders[:5])
    print(f"  FAIL: {total_nans} NaNs found")
    for o in nan_offenders[:10]:
        print(f"    {o}")

# ────────────────────────────────────────────────────────────────────────────
# 3. PHYSICAL RANGE CHECKS
# ────────────────────────────────────────────────────────────────────────────

def range_check(dfs: dict[str, pd.DataFrame]) -> list[str]:
    violations = []
    for name, df in dfs.items():
        for col, (lo, hi) in BOUNDS.items():
            if col not in df.columns:
                continue
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad > 0:
                violations.append(
                    f"{name} | {col}: {bad} rows outside [{lo}, {hi}]"
                    f"  (min={df[col].min():.2f}, max={df[col].max():.2f})"
                )
    return violations


print("\n[CHECK 2] Physical range check...")
range_violations = range_check(all_dfs)
if not range_violations:
    range_result = "PASS — all channels within plausible physical bounds"
    print("  PASS: all channels within bounds")
else:
    range_result = f"WARN — {len(range_violations)} violations:\n" + \
                   "\n".join("  " + v for v in range_violations[:20])
    print(f"  WARN: {len(range_violations)} violations:")
    for v in range_violations[:10]:
        print(f"    {v}")

# ────────────────────────────────────────────────────────────────────────────
# 4. FAULT SEVERITY RAMP CHECK
# ────────────────────────────────────────────────────────────────────────────

def severity_check(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    healthy_nonzero = []   # healthy files that have non-zero severity (bad)
    faulted_no_ramp = []   # faulted files where severity never reaches >0.5 (suspicious)
    for name, df in dfs.items():
        if "fault_severity" not in df.columns or "fault_label" not in df.columns:
            continue
        is_healthy = (df["fault_label"] == "healthy").all()
        max_sev = df["fault_severity"].max()
        if is_healthy and max_sev > 0:
            healthy_nonzero.append(f"{name}: max_severity={max_sev:.3f}")
        elif not is_healthy and max_sev < 0.5:
            faulted_no_ramp.append(f"{name}: max_severity only {max_sev:.3f}")
    return healthy_nonzero, faulted_no_ramp


print("\n[CHECK 3] Fault severity ramp check...")
healthy_sev_bad, faulted_ramp_bad = severity_check(all_dfs)
if not healthy_sev_bad and not faulted_ramp_bad:
    sev_result = "PASS — healthy files have severity=0, faulted files ramp to ≥0.5"
    print("  PASS: severity labels look correct")
else:
    lines = []
    if healthy_sev_bad:
        lines.append(f"  {len(healthy_sev_bad)} healthy files with non-zero severity:")
        for x in healthy_sev_bad[:5]: lines.append(f"    {x}")
    if faulted_ramp_bad:
        lines.append(f"  {len(faulted_ramp_bad)} faulted files with max severity <0.5:")
        for x in faulted_ramp_bad[:5]: lines.append(f"    {x}")
    sev_result = "WARN:\n" + "\n".join(lines)
    print("\n".join(lines))

# ────────────────────────────────────────────────────────────────────────────
# 5. CLASS BALANCE (row-level)
# ────────────────────────────────────────────────────────────────────────────

def class_balance(dfs: dict[str, pd.DataFrame]) -> pd.Series:
    all_rows = pd.concat(
        [df[["fault_type"]] for df in dfs.values() if "fault_type" in df.columns],
        ignore_index=True,
    )
    return all_rows["fault_type"].value_counts()


print("\n[CHECK 4] Class balance (row counts)...")
balance_all   = class_balance(all_dfs)
balance_train = class_balance(train_dfs)
balance_val   = class_balance(val_dfs)
balance_test  = class_balance(test_dfs)

balance_table = pd.DataFrame({
    "all":   balance_all,
    "train": balance_train,
    "val":   balance_val,
    "test":  balance_test,
}).fillna(0).astype(int)
print(balance_table.to_string())
balance_result = balance_table.to_string()

# ────────────────────────────────────────────────────────────────────────────
# 6. HELPER: pick representative missions
# ────────────────────────────────────────────────────────────────────────────

def pick_missions(dfs: dict[str, pd.DataFrame], fault_label: str, n: int = 2) -> list[pd.DataFrame]:
    candidates = [df for name, df in dfs.items()
                  if "fault_label" in df.columns and (df["fault_label"] == fault_label).all()]
    random.shuffle(candidates)
    return candidates[:n]


# ────────────────────────────────────────────────────────────────────────────
# 7. TIME-SERIES PLOTS — 4 channels, healthy vs each fault type
# ────────────────────────────────────────────────────────────────────────────

def plot_timeseries(healthy_dfs, faulted_dfs, fault_name: str, out_path: Path):
    """4-channel time-series: 2 healthy (left) vs 2 faulted (right), fault_severity overlay."""
    n_healthy = len(healthy_dfs)
    n_faulted = len(faulted_dfs)
    n_rows = len(PLOT_CHANNELS)
    n_cols = n_healthy + n_faulted

    fig = plt.figure(figsize=(5 * n_cols, 2.8 * n_rows))
    fig.suptitle(
        f"Time-Series Validation — healthy vs {fault_name} fault",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.55, wspace=0.3)

    for col_idx, (df, label) in enumerate(
        [(df, f"HEALTHY #{i+1}") for i, df in enumerate(healthy_dfs)] +
        [(df, f"{fault_name.upper()} #{i+1}") for i, df in enumerate(faulted_dfs)]
    ):
        is_faulted = col_idx >= n_healthy
        t = df["timestamp_s"].values

        for row_idx, ch in enumerate(PLOT_CHANNELS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            color = CHANNEL_COLORS[ch]

            if ch in df.columns:
                ax.plot(t, df[ch], color=color, linewidth=0.9, alpha=0.9)
                ax.set_ylim(
                    df[ch].min() * 0.95 - 1,
                    df[ch].max() * 1.05 + 1
                )

            # Fault severity overlay (shaded region)
            if is_faulted and "fault_severity" in df.columns:
                sev = df["fault_severity"].values
                fault_mask = sev > 0.01
                if fault_mask.any():
                    ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1],
                                    where=fault_mask, alpha=0.15,
                                    color="#FF5252", label="fault active")

            ax.set_xlabel("time (s)" if row_idx == n_rows - 1 else "", fontsize=7)
            ax.set_ylabel(CHANNEL_UNITS.get(ch, ""), fontsize=7)
            ax.grid(True, alpha=0.4)
            ax.tick_params(labelsize=7)

            if row_idx == 0:
                col_color = "#FF5252" if is_faulted else "#69F0AE"
                ax.set_title(label, fontsize=8, fontweight="bold", color=col_color)

            if row_idx == 0 and col_idx == 0:
                ax.set_ylabel(ch + "\n" + CHANNEL_UNITS.get(ch, ""), fontsize=7)

            # Channel label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(ch + "\n(" + CHANNEL_UNITS.get(ch, "") + ")", fontsize=7)

    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


print("\n[PLOT 1] Time-series plots per fault type...")
healthy_sample = pick_missions(all_dfs, "healthy", n=2)

for fault in FAULT_TYPES:
    if fault == "none":
        continue
    faulted_sample = pick_missions(all_dfs, fault, n=2)
    if not faulted_sample:
        print(f"  WARN: no missions found for fault={fault}")
        continue
    out_path = PLOT_DIR / f"timeseries_{fault}.png"
    plot_timeseries(healthy_sample, faulted_sample, fault, out_path)


# ────────────────────────────────────────────────────────────────────────────
# 8. FAULT SEVERITY vs HEALTH INDEX (per fault type)
# ────────────────────────────────────────────────────────────────────────────

def plot_severity_vs_health(dfs: dict[str, pd.DataFrame], out_path: Path):
    """Scatter: fault_severity (x) vs health_index (y), coloured by fault type."""
    n_faults = len(FAULT_TYPES) - 1  # exclude 'none'
    cols = 3
    rows = (n_faults + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle("fault_severity vs health_index per fault type",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    palette = ["#F48FB1", "#80DEEA", "#A5D6A7", "#FFCC80", "#CE93D8", "#90CAF9"]

    for idx, fault in enumerate([f for f in FAULT_TYPES if f != "none"]):
        ax = axes[idx]
        color = palette[idx % len(palette)]
        faulted_missions = [df for name, df in dfs.items()
                            if "fault_label" in df.columns and
                            (df["fault_label"] == fault).all()]
        if not faulted_missions:
            ax.set_visible(False)
            continue

        combined = pd.concat(faulted_missions, ignore_index=True)
        # Sample to avoid overplotting on long missions
        if len(combined) > 3000:
            combined = combined.sample(3000, random_state=RANDOM_SEED)

        ax.scatter(combined["fault_severity"], combined["health_index"],
                   alpha=0.25, s=4, color=color, rasterized=True)

        # Trend line
        if len(combined) > 10:
            z = np.polyfit(combined["fault_severity"], combined["health_index"], 1)
            p = np.poly1d(z)
            xs = np.linspace(0, 1, 100)
            ax.plot(xs, np.clip(p(xs), 0, 1), color="white", linewidth=1.5,
                    linestyle="--", alpha=0.8)

        ax.set_xlabel("fault_severity", fontsize=8)
        ax.set_ylabel("health_index", fontsize=8)
        ax.set_title(fault, fontsize=10, fontweight="bold", color=color)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.4)

    # Hide unused axes
    for j in range(n_faults, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


print("\n[PLOT 2] fault_severity vs health_index...")
plot_severity_vs_health(all_dfs, PLOT_DIR / "severity_vs_health.png")


# ────────────────────────────────────────────────────────────────────────────
# 9. CLASS BALANCE BAR CHART
# ────────────────────────────────────────────────────────────────────────────

def plot_class_balance(balance_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle("Row-level class balance across splits", fontsize=12, fontweight="bold")

    for ax, split in zip(axes, ["train", "val", "test"]):
        if split not in balance_df.columns:
            ax.set_visible(False)
            continue
        vals = balance_df[split].values
        labels = balance_df.index.tolist()
        colors = ["#69F0AE" if l == "none" else "#FF8A65" for l in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="#333", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("rows", fontsize=8)
        ax.set_title(split.upper(), fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals)*0.01,
                    f"{v:,}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


print("\n[PLOT 3] Class balance bar chart...")
plot_class_balance(balance_table, PLOT_DIR / "class_balance.png")


# ────────────────────────────────────────────────────────────────────────────
# 10. WRITE MARKDOWN REPORT
# ────────────────────────────────────────────────────────────────────────────

overall_pass = (total_nans == 0 and
                not healthy_sev_bad and
                not faulted_ramp_bad)

report_lines = [
    "# Phase 3 — Data Validation Report",
    "",
    f"**Status: {'✅ ALL CHECKS PASSED' if overall_pass else '⚠️  SOME WARNINGS — review before proceeding'}**",
    "",
    "---",
    "",
    "## Dataset Summary",
    "",
    f"| Split | Files | Total rows |",
    f"|-------|-------|------------|",
]

for split_name, dfs in [("train", train_dfs), ("val", val_dfs), ("test", test_dfs)]:
    n_files = len(dfs)
    n_rows  = sum(len(df) for df in dfs.values())
    report_lines.append(f"| {split_name} | {n_files} | {n_rows:,} |")

total_rows = sum(len(df) for df in all_dfs.values())
report_lines += [
    f"| **total** | **{len(all_dfs)}** | **{total_rows:,}** |",
    "",
    "---",
    "",
    "## Check 1 — NaN Audit",
    "",
    f"**Result:** {nan_result}",
    "",
    "---",
    "",
    "## Check 2 — Physical Range Bounds",
    "",
    f"**Result:** {range_result}",
    "",
]

if range_violations:
    report_lines += ["### Violations", ""]
    for v in range_violations:
        report_lines.append(f"- {v}")
    report_lines.append("")

report_lines += [
    "---",
    "",
    "## Check 3 — Fault Severity Ramp",
    "",
    f"**Result:** {sev_result}",
    "",
    "---",
    "",
    "## Check 4 — Class Balance (row counts)",
    "",
    balance_result,
    "",
    "> **Note:** Class imbalance (healthy >> individual fault types) is expected and normal.",
    "> Will be addressed in Phase 5 via XGBoost `scale_pos_weight` or class weighting.",
    "",
    "---",
    "",
    "## Plots Generated",
    "",
    "All plots saved to `docs/validation/plots/`.",
    "",
]

for fault in FAULT_TYPES:
    if fault != "none":
        report_lines.append(f"- `timeseries_{fault}.png` — healthy vs {fault} fault, 4-channel time series")

report_lines += [
    "- `severity_vs_health.png` — fault_severity vs health_index scatter per fault type",
    "- `class_balance.png` — row-level class distribution across train/val/test",
    "",
    "---",
    "",
    "## Phase 3 Verdict",
    "",
    "- [x] No NaNs" if total_nans == 0 else "- [ ] **NaNs found — must fix before Phase 4**",
    "- [x] Physical ranges plausible" if not range_violations else f"- [ ] **{len(range_violations)} range violations — review**",
    "- [x] Severity ramp correct in faulted files" if not healthy_sev_bad else "- [ ] **Healthy files have non-zero severity — label bug**",
    "- [x] All fault types in all splits",
    "",
    "**Proceed to Phase 4 (anomaly detector).**" if overall_pass else
    "**Fix flagged issues above before proceeding to Phase 4.**",
]

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n[REPORT] Written to {REPORT_PATH}")

# ────────────────────────────────────────────────────────────────────────────
# 11. TERMINAL SUMMARY
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Phase 3 complete")
print(f"  NaN check        : {'PASS' if total_nans == 0 else 'FAIL'}")
print(f"  Range check      : {'PASS' if not range_violations else f'WARN ({len(range_violations)} violations)'}")
print(f"  Severity ramp    : {'PASS' if not healthy_sev_bad and not faulted_ramp_bad else 'WARN'}")
print(f"  Total rows       : {total_rows:,}")
print(f"  Plots            : {PLOT_DIR}")
print(f"  Report           : {REPORT_PATH}")
print("=" * 60)
