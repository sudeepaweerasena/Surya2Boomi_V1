"""
Solar Flare Forecasting — Step 2: Temporal Train / Val / Test Split
====================================================================
Input : solar_flare_features.csv        (output of Step 1)
Output: split_train.csv
        split_val.csv
        split_test.csv
        split_summary.txt

Split strategy — Option C (chronological, no leakage):
  Train : 2017-02-08 → 2022-12-31   (51,672 rows,  6.3% flare rate)
  Val   : 2023-01-01 → 2023-12-31   ( 8,760 rows, 44.3% flare rate)
  Test  : 2024-01-01 → 2026-03-21   (19,464 rows, 34.2% flare rate)

Run:
    python step2_temporal_split.py
"""

import textwrap, sys, os
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE = config.get_data_path("solar_flare_features.csv")

# Strict chronological boundaries (inclusive)
TRAIN_END  = "2022-12-31 23:00:00"
VAL_START  = "2023-01-01 00:00:00"
VAL_END    = "2023-12-31 23:00:00"
TEST_START = "2024-01-01 00:00:00"

# Columns that are labels (never go into X)
LABEL_COLS = ["label_max", "label_cum"]

# Columns to exclude from features
NON_FEATURE_COLS = ["timestamp"] + LABEL_COLS


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print("Loading feature dataset ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"  Rows : {len(df):,}")
print(f"  Cols : {len(df.columns)}  "
      f"(features={len(df.columns) - len(NON_FEATURE_COLS)}  labels=2)")


# ─────────────────────────────────────────────
# SPLIT — strict time boundaries, no shuffling
# ─────────────────────────────────────────────
print("\nApplying temporal split ...")

train_df = df[df["timestamp"] <= TRAIN_END].copy().reset_index(drop=True)
val_df   = df[(df["timestamp"] >= VAL_START) &
              (df["timestamp"] <= VAL_END)].copy().reset_index(drop=True)
test_df  = df[df["timestamp"] >= TEST_START].copy().reset_index(drop=True)


# ─────────────────────────────────────────────
# EXTRACT X / y for each split
# (saved separately for convenience in Step 3+)
# ─────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

X_train = train_df[feature_cols]
y_train = train_df["label_max"]

X_val   = val_df[feature_cols]
y_val   = val_df["label_max"]

X_test  = test_df[feature_cols]
y_test  = test_df["label_max"]


# ─────────────────────────────────────────────
# VALIDATION — zero overlap, no leakage,
# full row count, no NaNs
# ─────────────────────────────────────────────
print("\nRunning integrity checks ...")

total_rows  = len(train_df) + len(val_df) + len(test_df)
assert total_rows == len(df), \
    f"Row count mismatch: {total_rows} != {len(df)}"

train_ts = set(train_df["timestamp"])
val_ts   = set(val_df["timestamp"])
test_ts  = set(test_df["timestamp"])
assert len(train_ts & val_ts)  == 0, "Train/Val timestamp overlap!"
assert len(val_ts   & test_ts) == 0, "Val/Test timestamp overlap!"
assert len(train_ts & test_ts) == 0, "Train/Test timestamp overlap!"

assert train_df["timestamp"].max() < val_df["timestamp"].min(), \
    "Temporal leakage: train bleeds into val!"
assert val_df["timestamp"].max() < test_df["timestamp"].min(), \
    "Temporal leakage: val bleeds into test!"

assert X_train.isnull().sum().sum() == 0, "NaNs in X_train!"
assert X_val.isnull().sum().sum()   == 0, "NaNs in X_val!"
assert X_test.isnull().sum().sum()  == 0, "NaNs in X_test!"

print("  All checks passed.")


# ─────────────────────────────────────────────
# CLASS IMBALANCE STATS
# (needed for Step 3 model config)
# ─────────────────────────────────────────────
neg_train = (y_train == 0).sum()
pos_train = (y_train == 1).sum()
scale_pos_weight = round(neg_train / pos_train, 2)   # for XGBoost


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────
def split_stats(name, part_df, y):
    n  = len(part_df)
    f  = y.sum()
    nf = n - f
    return (f"  {name:<8}  "
            f"{part_df['timestamp'].min().date()} → "
            f"{part_df['timestamp'].max().date()}  "
            f"rows={n:>6,}  "
            f"flares={f:>5,} ({f/n*100:5.1f}%)  "
            f"no-flare={nf:>6,} ({nf/n*100:5.1f}%)")

summary = textwrap.dedent(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          Solar Flare Forecasting — Split Summary (Option C)          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Strategy : Strict chronological  — ZERO data leakage               ║
║  Target   : label_max  (binary: 1 = flare in next 24 h window)       ║
║  Features : {len(feature_cols)} columns                                           ║
╠══════════════════════════════════════════════════════════════════════╣
{split_stats('Train', train_df, y_train)}
{split_stats('Val  ', val_df,   y_val)}
{split_stats('Test ', test_df,  y_test)}
╠══════════════════════════════════════════════════════════════════════╣
║  Total rows        : {len(df):,}                                        ║
║  Train / Val / Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}                   ║
║  Train split share : {len(train_df)/len(df)*100:.1f}%                                       ║
║  Val   split share : {len(val_df)/len(df)*100:.1f}%                                        ║
║  Test  split share : {len(test_df)/len(df)*100:.1f}%                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Class imbalance (train)                                             ║
║    Negative (no-flare) : {neg_train:,}                                    ║
║    Positive (flare)    : {pos_train:,}                                     ║
║    Ratio neg/pos       : {scale_pos_weight}  → use as scale_pos_weight in XGBoost  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Solar cycle context                                                 ║
║    Train covers : solar minimum (2017-19) + ramp-up through 2022    ║
║    Val  covers  : active sun, mid-cycle peak  (2023)                 ║
║    Test covers  : solar maximum + post-peak decline (2024-2026)      ║
║    → Model trained on rising activity, evaluated on peak conditions  ║
╚══════════════════════════════════════════════════════════════════════╝
""").strip()

print()
print(summary)


# ─────────────────────────────────────────────
# SAVE SPLITS
# ─────────────────────────────────────────────
print("\nSaving split files ...")

train_df.to_csv(config.get_data_path("split_train.csv"), index=False)
val_df.to_csv(config.get_data_path("split_val.csv"),     index=False)
test_df.to_csv(config.get_data_path("split_test.csv"),   index=False)

with open(config.get_report_path("split_summary.txt"), "w") as f:
    f.write(summary + "\n")
    f.write(f"\nscale_pos_weight = {scale_pos_weight}\n")
    f.write(f"feature_cols ({len(feature_cols)}):\n")
    for c in feature_cols:
        f.write(f"  {c}\n")

print("  split_train.csv")
print("  split_val.csv")
print("  split_test.csv")
print("  split_summary.txt")
print("\nDone. Pass scale_pos_weight =", scale_pos_weight,
      "to XGBoost in Step 3.")
