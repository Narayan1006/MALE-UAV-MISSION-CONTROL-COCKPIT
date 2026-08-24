"""
Phase 5 — Fault Classifier & RUL Regressor
=============================================
1. XGBoost Multi-class Classifier (predicts fault_type: none, injector, lubrication, cooling, misfire, sensor_drift, vibration)
2. XGBoost Regressor (predicts remaining useful life in seconds, trained on active fault data)
3. Evaluated strictly on the held-out test split.
"""

import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, root_mean_squared_error, f1_score

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent.resolve()
TRAIN_DIR = ROOT / "data" / "raw" / "train"
VAL_DIR   = ROOT / "data" / "raw" / "val"
TEST_DIR  = ROOT / "data" / "raw" / "test"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLS = [
    "rpm", "true_cht", "sensor_cht", "egt",
    "oil_pressure", "oil_temp", "fuel_flow", "vibration",
    "battery_voltage", "injection_timing", "health_index",
    "altitude", "ambient_temp", "throttle",
]
WINDOWS = [30, 60]

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[SENSOR_COLS].copy()
    for col in SENSOR_COLS:
        series = df[col]
        for w in WINDOWS:
            features[f"{col}_rmean{w}"] = series.rolling(w, min_periods=1).mean()
            features[f"{col}_rstd{w}"]  = series.rolling(w, min_periods=1).std().fillna(0)
    features = features.ffill().bfill().fillna(0)
    return features

def load_split(split_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(split_dir.glob("*.csv")):
        df = pd.read_csv(p)
        feat = compute_rolling_features(df)
        feat["fault_type"]   = df["fault_type"].values
        feat["rul_seconds"]  = df["rul_seconds"].values
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)

print("Loading dataset splits...")
train_df = load_split(TRAIN_DIR)
val_df   = load_split(VAL_DIR)
test_df  = load_split(TEST_DIR)

META_COLS = {"fault_type", "rul_seconds"}
FEATURE_COLS = [c for c in train_df.columns if c not in META_COLS]

print(f"Train rows: {len(train_df):,}, Val rows: {len(val_df):,}, Test rows: {len(test_df):,}")

# --- 1. Fault Classifier ---
print("\n--- Training Fault Classifier (XGBoost) ---")
le = LabelEncoder()
y_train_cls = le.fit_transform(train_df["fault_type"])
y_val_cls   = le.transform(val_df["fault_type"])
y_test_cls  = le.transform(test_df["fault_type"])

clf = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2026,
    tree_method="hist",
    n_jobs=-1,
)

clf.fit(train_df[FEATURE_COLS], y_train_cls, eval_set=[(val_df[FEATURE_COLS], y_val_cls)], verbose=False)

y_test_pred_cls = clf.predict(test_df[FEATURE_COLS])
test_f1 = f1_score(y_test_cls, y_test_pred_cls, average="weighted")
print(f"Test Set Weighted F1-Score: {test_f1:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test_cls, y_test_pred_cls, target_names=le.classes_, digits=4))

# --- 2. RUL Regressor ---
print("\n--- Training RUL Regressor (XGBoost) ---")
train_rul_mask = train_df["fault_type"] != "none"
val_rul_mask   = val_df["fault_type"] != "none"
test_rul_mask  = test_df["fault_type"] != "none"

X_train_rul = train_df.loc[train_rul_mask, FEATURE_COLS]
y_train_rul = train_df.loc[train_rul_mask, "rul_seconds"].astype(float)

X_val_rul   = val_df.loc[val_rul_mask, FEATURE_COLS]
y_val_rul   = val_df.loc[val_rul_mask, "rul_seconds"].astype(float)

X_test_rul  = test_df.loc[test_rul_mask, FEATURE_COLS]
y_test_rul  = test_df.loc[test_rul_mask, "rul_seconds"].astype(float)

reg = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2026,
    tree_method="hist",
    n_jobs=-1,
)

reg.fit(X_train_rul, y_train_rul, eval_set=[(X_val_rul, y_val_rul)], verbose=False)

y_test_pred_rul = reg.predict(X_test_rul)
mae = mean_absolute_error(y_test_rul, y_test_pred_rul)
rmse = root_mean_squared_error(y_test_rul, y_test_pred_rul)

print(f"Test Set RUL Performance:")
print(f"  MAE : {mae:.2f} seconds")
print(f"  RMSE: {rmse:.2f} seconds")

# --- Save Models ---
print("\nSaving Phase 5 artefacts...")
with open(MODEL_DIR / "fault_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open(MODEL_DIR / "rul_regressor.pkl", "wb") as f:
    pickle.dump(reg, f)
with open(MODEL_DIR / "model_feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

metrics = {
    "classifier_weighted_f1": round(float(test_f1), 4),
    "rul_mae_seconds": round(float(mae), 2),
    "rul_rmse_seconds": round(float(rmse), 2),
}
with open(MODEL_DIR / "phase5_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Phase 5 Models & Metrics successfully saved to ml/models/")
