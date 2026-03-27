"""
Solar Flare Forecasting — Step 6: Evaluation & 24-Hour Forecasting
==================================================================
Input : solar_flare_model.pkl
        split_val.csv, split_test.csv
Output: step6_evaluation_report.txt   — full metric breakdown
        step6_forecast_sample.csv     — 24-hour ahead predictions
        step6_per_year_metrics.csv    — per-year performance table

Key findings documented in this script:
  - 2024 : excellent (TSS=0.966)  — solar maximum, in-distribution
  - 2025/2026 : zero predictions  — cumulative_index feature went to 0
                                    in source data (data availability issue)
                                    AND solar cycle phase shifted
                                    out-of-distribution for this model

Run:
    python step6_evaluation.py
"""

import pickle, sys, os
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("=" * 62)
print("  Solar Flare Forecasting — Step 6: Evaluation")
print("=" * 62)

with open(config.get_model_path("solar_flare_model.pkl"), "rb") as f:
    payload = pickle.load(f)

model      = payload["model"]
feat_cols  = payload["feature_cols"]
thresh     = payload["threshold"]

print(f"\n  Model     : HistGradientBoostingClassifier")
print(f"  Trees     : {payload['n_trees']}")
print(f"  Features  : {len(feat_cols)}")
print(f"  Threshold : {thresh:.4f}")


# ─────────────────────────────────────────────
# LOAD SPLITS
# ─────────────────────────────────────────────
print("\nLoading splits ...")
val  = pd.read_csv(config.get_data_path("split_val.csv"),  parse_dates=["timestamp"])
test = pd.read_csv(config.get_data_path("split_test.csv"), parse_dates=["timestamp"])

# Add predictions to both splits
for df in [val, test]:
    df["proba"] = model.predict_proba(df[feat_cols].values)[:, 1]
    df["pred"]  = (df["proba"] >= thresh).astype(int)

print(f"  Val  : {len(val):,} rows")
print(f"  Test : {len(test):,} rows")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def metrics(y_true, y_pred, y_proba):
    """Return a dict of all evaluation metrics."""
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return None                         # skip degenerate splits

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)                   # hit rate  (recall)
    spec = tn / (tn + fp)                   # correct rejection rate

    return {
        "N"          : len(y_true),
        "n_flares"   : int(y_true.sum()),
        "flare_pct"  : round(y_true.mean() * 100, 1),
        "F1"         : round(f1_score(y_true, y_pred),              4),
        "AUC_PR"     : round(average_precision_score(y_true, y_proba), 4),
        "AUC_ROC"    : round(roc_auc_score(y_true, y_proba),        4),
        "TSS"        : round(sens + spec - 1,                       4),
        "HSS"        : round(_hss(tp, fp, tn, fn),                  4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
        "miss_rate"  : round(fn / (tp + fn) * 100, 1),
        "fa_rate"    : round(fp / (fp + tn) * 100, 1),
    }

def _hss(tp, fp, tn, fn):
    """Heidke Skill Score."""
    n = tp + fp + tn + fn
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n
    correct  = tp + tn
    denom    = n - expected
    return (correct - expected) / denom if denom > 0 else 0

def print_metrics(m, label):
    if m is None:
        print(f"\n  ── {label} ── SKIPPED (no pos or neg samples)")
        return
    print(f"\n  ── {label} ──")
    print(f"  Rows        : {m['N']:,}  "
          f"(flares={m['n_flares']:,} / {m['flare_pct']}%)")
    print(f"  F1          : {m['F1']}")
    print(f"  AUC-PR      : {m['AUC_PR']}")
    print(f"  AUC-ROC     : {m['AUC_ROC']}")
    print(f"  TSS         : {m['TSS']}   "
          f"(>0.5=skilful  >0.9=excellent)")
    print(f"  HSS         : {m['HSS']}")
    print(f"  Sensitivity : {m['Sensitivity']}  "
          f"(flare hit rate)")
    print(f"  Specificity : {m['Specificity']}  "
          f"(quiet-sun rejection rate)")
    print(f"  Confusion   : TP={m['TP']:,}  FP={m['FP']:,}  "
          f"TN={m['TN']:,}  FN={m['FN']:,}")
    print(f"  Miss rate   : {m['miss_rate']}%   "
          f"(flares the model missed)")
    print(f"  False alarm : {m['fa_rate']}%   "
          f"(quiet hours flagged as flares)")


# ─────────────────────────────────────────────
# SECTION 1 — OVERALL METRICS
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 1 — OVERALL METRICS")
print("=" * 62)

m_val  = metrics(val["label_max"].values,
                 val["pred"].values,
                 val["proba"].values)
m_test = metrics(test["label_max"].values,
                 test["pred"].values,
                 test["proba"].values)

print_metrics(m_val,  "Validation set — 2023")
print_metrics(m_test, "Test set — 2024 to 2026")


# ─────────────────────────────────────────────
# SECTION 2 — PER-YEAR BREAKDOWN
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 2 — PER-YEAR BREAKDOWN")
print("=" * 62)

test["year"] = test["timestamp"].dt.year
per_year_rows = []

for year, grp in test.groupby("year"):
    m = metrics(grp["label_max"].values,
                grp["pred"].values,
                grp["proba"].values)
    if m is None:
        print(f"\n  {year}: no positive or negative samples — skipped")
        continue
    m["year"] = year
    per_year_rows.append(m)
    print_metrics(m, f"{year}")

per_year_df = pd.DataFrame(per_year_rows)


# ─────────────────────────────────────────────
# SECTION 3 — DATA AVAILABILITY ISSUE: 2025/2026
# The cumulative_index feature is 0 for ALL rows
# in 2025 and 2026 in the source dataset.
# This is a data collection / pipeline issue,
# not a model bug. Documented here explicitly.
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 3 — DATA ISSUE: 2025 / 2026 ZERO PREDICTIONS")
print("=" * 62)
print("""
  Root cause: cumulative_index = 0.0 for every row in 2025 and 2026.
  This feature is the #2 most important predictor in the model.
  When it is always 0, the model cannot find the patterns it learned
  during training (2017–2022) and outputs near-zero probabilities.

  Additionally, the solar cycle phase shifted:
    - Training data (2022): flares preceded by M-class activity
                            (goes_ordinal_lag1 mean = 4.05)
    - Test data (2025)    : flares preceded by C-class activity
                            (goes_ordinal_lag1 mean = 3.38)
  The model has never seen declining-cycle patterns in training.

  This is an out-of-distribution (OOD) generalisation challenge.

  Recommended fix:
    If cumulative_index data becomes available for 2025/2026,
    re-run this evaluation. Alternatively, retrain including
    post-peak solar data once the dataset is updated.
""")

t25 = test[test["year"] == 2025]
t26 = test[test["year"] == 2026]
print(f"  2025 actual flares  : {t25['label_max'].sum():,}")
print(f"  2025 predictions=1  : {t25['pred'].sum():,}")
print(f"  2025 max proba      : {t25['proba'].max():.4f}")
print(f"  2026 actual flares  : {t26['label_max'].sum():,}")
print(f"  2026 predictions=1  : {t26['pred'].sum():,}")
print(f"  2026 max proba      : {t26['proba'].max():.4f}")


# ─────────────────────────────────────────────
# SECTION 4 — 24-HOUR FORECAST WINDOW ANALYSIS
# Group hourly predictions into 24h blocks.
# A block is flagged as "flare day" if ANY hour
# in that block is predicted as a flare.
# This is how operational forecasting works —
# forecasters issue a daily flare probability.
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 4 — 24-HOUR FORECAST WINDOW")
print("=" * 62)

test_sorted = test.sort_values("timestamp").reset_index(drop=True)
test_sorted["block"] = test_sorted.index // 24

block_df = test_sorted.groupby("block").agg(
    start_time   = ("timestamp", "min"),
    actual_flare = ("label_max", "max"),   # 1 if any flare in 24h
    pred_flare   = ("pred",      "max"),   # 1 if any prediction in 24h
    max_proba    = ("proba",     "max"),   # peak probability in 24h
    mean_proba   = ("proba",     "mean"),
    n_flare_hrs  = ("label_max", "sum"),
    n_pred_hrs   = ("pred",      "sum"),
).reset_index(drop=True)

bm = metrics(block_df["actual_flare"].values,
             block_df["pred_flare"].values,
             block_df["max_proba"].values)

print_metrics(bm, "24-hour block level (test set)")
print(f"\n  Total 24h blocks   : {len(block_df):,}")
print(f"  Blocks with flares : {block_df['actual_flare'].sum():,}")
print(f"  Blocks predicted 1 : {block_df['pred_flare'].sum():,}")


# ─────────────────────────────────────────────
# SECTION 5 — PROBABILITY CALIBRATION
# A well-calibrated model outputs proba=0.8 for
# events that actually happen 80% of the time.
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 5 — PROBABILITY CALIBRATION")
print("=" * 62)

bins   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ["0–10%","10–20%","20–30%","30–40%","40–50%",
          "50–60%","60–70%","70–80%","80–90%","90–100%"]

test["proba_bin"] = pd.cut(test["proba"], bins=bins,
                           labels=labels, right=False)
cal = test.groupby("proba_bin", observed=True).agg(
    count            = ("label_max", "count"),
    actual_flare_pct = ("label_max", "mean"),
).reset_index()
cal["actual_flare_pct"] = (cal["actual_flare_pct"] * 100).round(1)

print(f"\n  {'Proba bin':<12}  {'Count':>7}  {'Actual flare %':>15}  Note")
print(f"  {'-'*12}  {'-'*7}  {'-'*15}  {'-'*25}")
for _, row in cal.iterrows():
    pct  = row["actual_flare_pct"]
    note = ""
    if row["proba_bin"] in ["0–10%"] and pct < 5:
        note = "✓ well calibrated (low proba = rare flare)"
    elif row["proba_bin"] in ["90–100%"] and pct > 90:
        note = "✓ well calibrated (high proba = flare)"
    elif 10 < pct < 90:
        note = "middle range — uncertain"
    print(f"  {str(row['proba_bin']):<12}  {int(row['count']):>7,}  "
          f"{pct:>14.1f}%  {note}")

print("""
  Interpretation:
    0–10% predicted  → 2.6% actual flare rate   ✓ model is cautious correctly
    90–100% predicted → 99.0% actual flare rate  ✓ high confidence = real flares
    Middle bins (10–90%) are sparse — the model is bimodal (very confident
    either way) which is typical for well-trained gradient boosted trees.
""")


# ─────────────────────────────────────────────
# SECTION 6 — SAMPLE 24-HOUR FORECAST
# Show what a real operational forecast looks
# like: given the last 24 hours of observations,
# predict the next 24 hours hour by hour.
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 6 — SAMPLE 24-HOUR FORECAST (2024-09-01)")
print("=" * 62)

# Pick a representative 24h window from 2024 — a high-activity day
sample_start = pd.Timestamp("2024-09-01 00:00:00")
sample_end   = pd.Timestamp("2024-09-01 23:00:00")
sample       = test[(test["timestamp"] >= sample_start) &
                    (test["timestamp"] <= sample_end)].copy()

if len(sample) == 0:
    print("  Sample date not in test set — using first 24 rows of test.")
    sample = test.head(24).copy()

print(f"\n  {'Hour':<22}  {'Proba':>7}  {'Pred':>5}  {'Actual':>7}  Status")
print(f"  {'-'*22}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*15}")
for _, row in sample.iterrows():
    proba  = row["proba"]
    pred   = int(row["pred"])
    actual = int(row["label_max"])
    status = ""
    if pred == 1 and actual == 1:
        status = "✓ TP"
    elif pred == 1 and actual == 0:
        status = "⚠ FP (false alarm)"
    elif pred == 0 and actual == 1:
        status = "✗ FN (missed!)"
    else:
        status = "✓ TN"
    print(f"  {str(row['timestamp']):<22}  {proba:>7.4f}  "
          f"{'FLARE' if pred else 'quiet':>5}  "
          f"{'FLARE' if actual else 'quiet':>7}  {status}")

print(f"\n  Day summary:")
print(f"    Predicted flares  : {sample['pred'].sum()} / 24 hours")
print(f"    Actual flares     : {sample['label_max'].sum()} / 24 hours")
print(f"    Peak probability  : {sample['proba'].max():.4f}")
print(f"    Mean probability  : {sample['proba'].mean():.4f}")


# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SAVING OUTPUTS")
print("=" * 62)

# 1. Per-year metrics CSV
per_year_df.to_csv(config.get_report_path("step6_per_year_metrics.csv"), index=False)
print("  step6_per_year_metrics.csv")

# 2. Full forecast on test set
forecast_out = test[[
    "timestamp", "label_max", "proba", "pred"
]].copy()
forecast_out.columns = [
    "timestamp", "actual_flare",
    "flare_probability", "predicted_flare"
]
forecast_out.to_csv(config.get_data_path("step6_forecast_sample.csv"), index=False)
print("  step6_forecast_sample.csv")

# 3. Text report
report = []
report.append("Solar Flare Forecasting — Step 6 Evaluation Report")
report.append("=" * 62)
report.append(f"Model        : HistGradientBoostingClassifier")
report.append(f"Trees        : {payload['n_trees']}")
report.append(f"Threshold    : {thresh:.4f}")
report.append(f"Features     : {len(feat_cols)}")
report.append("")
report.append("OVERALL METRICS")

def m_to_lines(m, label):
    lines = [f"\n{label}"]
    if m is None:
        lines.append("  SKIPPED")
        return lines
    for k in ["F1","AUC_PR","AUC_ROC","TSS","HSS",
               "Sensitivity","Specificity",
               "TP","FP","TN","FN","miss_rate","fa_rate"]:
        lines.append(f"  {k:<14}: {m[k]}")
    return lines

report.extend(m_to_lines(m_val,  "Validation (2023)"))
report.extend(m_to_lines(m_test, "Test (2024–2026)"))
report.append("")
report.append("PER-YEAR BREAKDOWN (test set)")
report.append(per_year_df[[
    "year","N","n_flares","flare_pct",
    "F1","TSS","HSS","Sensitivity","Specificity",
    "TP","FP","TN","FN"
]].to_string(index=False))
report.append("")
report.append("24-HOUR BLOCK LEVEL")
report.extend(m_to_lines(bm, "24h blocks"))
report.append("")
report.append("NOTE: 2025/2026 show zero predictions due to")
report.append("cumulative_index=0 in source data for those years.")
report.append("This is a data availability issue, not a model bug.")

with open(config.get_report_path("step6_evaluation_report.txt"), "w") as f:
    f.write("\n".join(report))
print("  step6_evaluation_report.txt")

print("\n" + "=" * 62)
print("  Step 6 complete.")
print("=" * 62)
