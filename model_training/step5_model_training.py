"""
Solar Flare Forecasting — Model Training & Save
================================================
Trains a 4-class model (No-flare / C / M / X) and saves it as a .pkl file.

Inputs
------
  solar_flare_features.csv          (Step 1 output)
  data_extended_v3.csv              (raw data, for label creation)
  class_imbalance_config.py         (Step 3 output)

Outputs
-------
  solar_flare_model_multiclass.pkl  — trained model
"""

import re
import pickle
import importlib.util
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES  = {0: "No-flare", 1: "C-class", 2: "M-class", 3: "X-class"}
CLASS_LABELS = ["No-flare", "C-class", "M-class", "X-class"]
LEAKY_FEATURES = ["goes_flux", "goes_ordinal", "log_goes_flux"]

RAW_DATA_FILE = "data_extended_v3.csv"
FEAT_FILE     = "solar_flare_features.csv"
MODEL_FILE    = "solar_flare_model_multiclass.pkl"
CONFIG_FILE   = "class_imbalance_config.py"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def goes_to_class4(goes_str: str) -> int:
    """Map GOES class string → 4-class integer label."""
    g = str(goes_str).strip()
    if g == "FQ":
        return 0
    m = re.match(r"^([ABCMX])", g)
    if not m:
        return 0
    return {"A": 0, "B": 0, "C": 1, "M": 2, "X": 3}[m.group(1)]


def load_config():
    """Load feature columns from config, excluding leaky features."""
    spec = importlib.util.spec_from_file_location("cfg", CONFIG_FILE)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return [c for c in cfg.FEATURE_COLS if c not in LEAKY_FEATURES]


# ─────────────────────────────────────────────
# TRAIN & SAVE
# ─────────────────────────────────────────────
def train_model(feat_cols):
    print("=" * 62)
    print("  Training multiclass model (4 classes)")
    print("=" * 62)

    # ── Build 4-class shifted target ──────────
    # At hour t, target = class that will occur at t+1
    raw = pd.read_csv(RAW_DATA_FILE, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw["class4"]      = raw["max_goes_class"].apply(goes_to_class4)
    raw["class4_next"] = raw["class4"].shift(-1)
    raw = raw.dropna(subset=["class4_next"])
    raw["class4_next"] = raw["class4_next"].astype(int)

    # ── Merge with engineered features ────────
    feat   = pd.read_csv(FEAT_FILE, parse_dates=["timestamp"])
    merged = feat.merge(raw[["timestamp", "class4_next"]],
                        on="timestamp", how="inner")

    # ── Temporal split ─────────────────────────
    train = merged[merged["timestamp"] <= "2022-12-31 23:00:00"]
    val   = merged[(merged["timestamp"] >= "2023-01-01") &
                   (merged["timestamp"] <= "2023-12-31 23:00:00")]
    test  = merged[merged["timestamp"] >= "2024-01-01"]

    X_train = train[feat_cols].values;  y_train = train["class4_next"].values
    X_val   = val[feat_cols].values;    y_val   = val["class4_next"].values
    X_test  = test[feat_cols].values;   y_test  = test["class4_next"].values

    print(f"\n  Train : {len(train):,} rows  "
          f"({train['timestamp'].min().date()} → "
          f"{train['timestamp'].max().date()})")
    print(f"  Val   : {len(val):,} rows  "
          f"({val['timestamp'].min().date()} → "
          f"{val['timestamp'].max().date()})")
    print(f"  Test  : {len(test):,} rows  "
          f"({test['timestamp'].min().date()} → "
          f"{test['timestamp'].max().date()})")

    print(f"\n  Class distribution (train):")
    for k, name in CLASS_NAMES.items():
        n = (y_train == k).sum()
        print(f"    {name:<12}: {n:>6,}  ({n/len(y_train)*100:.1f}%)")

    # ── Train ──────────────────────────────────
    print("\n  Training HistGradientBoostingClassifier ...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter            = 1000,
        max_depth           = 6,
        learning_rate       = 0.05,
        min_samples_leaf    = 30,
        class_weight        = "balanced",
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 30,
        random_state        = 42,
        verbose             = 0,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Done: {model.n_iter_} trees in {elapsed:.1f}s")

    # ── Evaluate ───────────────────────────────
    print("\n  Validation results (2023):")
    val_preds = model.predict(X_val)
    print(classification_report(y_val, val_preds,
          target_names=CLASS_LABELS, digits=3, zero_division=0))

    print("  Test results (2024):")
    test_2024 = test[test["timestamp"].dt.year == 2024]
    if len(test_2024) > 0:
        preds_24 = model.predict(test_2024[feat_cols].values)
        print(classification_report(test_2024["class4_next"].values,
              preds_24, target_names=CLASS_LABELS, digits=3, zero_division=0))

    # ── Save ───────────────────────────────────
    payload = {
        "model"       : model,
        "feature_cols": feat_cols,
        "class_names" : CLASS_NAMES,
        "classes"     : model.classes_.tolist(),
        "n_trees"     : model.n_iter_,
        "train_time"  : elapsed,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Model saved → {MODEL_FILE}")

    return payload


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    feat_cols = load_config()
    train_model(feat_cols)