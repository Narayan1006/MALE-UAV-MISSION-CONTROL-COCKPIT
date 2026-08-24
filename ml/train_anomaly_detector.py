"""
Phase 4 — Fast CPU/Multi-core Optimized Anomaly Detector
=========================================================
Strategy:
  1. Use subsampled / vectorized feature computation for high speed.
  2. Train IsolationForest with n_jobs=-1 and max_samples=2048 for fast execution.
  3. Save model artefacts.
"""

import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT      = Path(__file__).parent.parent.resolve()
TRAIN_DIR = ROOT / "data" / "raw" / "train"
VAL_DIR   = ROOT / "data" / "raw" / "val"
MODEL_DIR = ROOT / "ml" / "models"
PLOT_DIR  = ROOT / "docs" / "validation" / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLS = [
    "rpm", "true_cht", "sensor_cht", "egt",
    "oil_pressure", "oil_temp", "fuel_flow", "vibration",
    "battery_voltage", "injection_timing", "health_index",
    "altitude", "ambient_temp", "throttle",
]

WINDOWS = [30, 60]
RANDOM_SEED = 2026

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[SENSOR_COLS].copy()
    for col in SENSOR_COLS:
        series = df[col]
        for w in WINDOWS:
            features[f"{col}_rmean{w}"] = series.rolling(w, min_periods=1).mean()
            features[f"{col}_rstd{w}"]  = series.rolling(w, min_periods=1).std().fillna(0)
    features = features.ffill().bfill().fillna(0)
    return features

def load_and_featurize(split_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(split_dir.glob("*.csv")):
        df = pd.read_csv(p)
        feat = compute_rolling_features(df)
        feat["fault_type"]     = df["fault_type"].values
        feat["fault_severity"] = df["fault_severity"].values
        feat["fault_label"]    = df["fault_label"].values
        feat["mission_file"]   = p.name
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)

print("Loading training data...")
train_df = load_and_featurize(TRAIN_DIR)
print(f"Train rows: {len(train_df):,}")

print("Loading validation data...")
val_df = load_and_featurize(VAL_DIR)
print(f"Val rows: {len(val_df):,}")

META_COLS = {"fault_type", "fault_severity", "fault_label", "mission_file"}
FEATURE_COLS = [c for c in train_df.columns if c not in META_COLS]

healthy_mask = train_df["fault_label"] == "healthy"
X_healthy    = train_df.loc[healthy_mask, FEATURE_COLS].values
print(f"Healthy training samples: {len(X_healthy):,}")

scaler = StandardScaler()
X_healthy_scaled = scaler.fit_transform(X_healthy)

# Optimized Isolation Forest with max_samples capped for high speed
iforest = IsolationForest(
    n_estimators=100,
    max_samples=2048,
    contamination=0.01,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
print("Training Isolation Forest...")
iforest.fit(X_healthy_scaled)
print("Model training complete.")

X_val_scaled = scaler.transform(val_df[FEATURE_COLS].values)
raw_scores = iforest.decision_function(X_val_scaled)
anomaly_scores = -raw_scores
anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-9)

val_df["anomaly_score_norm"] = anomaly_scores_norm
y_true = (val_df["fault_type"] != "none").astype(int).values
y_pred = (iforest.predict(X_val_scaled) == -1).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
roc_auc = roc_auc_score(y_true, anomaly_scores_norm)
avg_prec = average_precision_score(y_true, anomaly_scores_norm)
corr = np.corrcoef(anomaly_scores_norm, val_df["fault_severity"].values)[0, 1]

print(f"\nResults:\nPrecision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f} | Corr: {corr:.3f}")

with open(MODEL_DIR / "isolation_forest.pkl", "wb") as f:
    pickle.dump(iforest, f)
with open(MODEL_DIR / "scaler_anomaly.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MODEL_DIR / "anomaly_feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

metrics = {
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "corr": round(float(corr), 4),
}
with open(MODEL_DIR / "anomaly_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model artefacts to ml/models/")
