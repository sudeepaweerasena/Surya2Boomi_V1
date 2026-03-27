"""
SURYA2BOOMI — Model Testing Report Generator
=============================================
Generates evaluation metrics and ROC curve plots for both models:

  Model 1: solar_flare_model.pkl         (Binary: Flare / No-flare)
  Model 2: solar_flare_model_multiclass.pkl (4-class: No-flare/C/M/X)

Outputs:
  model1_roc_curve.png          ROC curve for binary model
  model2_roc_curve.png          ROC curve for multiclass model (OvR)
  model_testing_report.txt      Full metrics report for both models

Run:
    python step_testing_report.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

CLASS_NAMES = {0: "No-flare", 1: "C-class", 2: "M-class", 3: "X-class"}

import os
import sys

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# LOAD BINARY MODEL
# ─────────────────────────────────────────────
print("Loading Model 1 (binary) ...")
with open(config.get_model_path("solar_flare_model.pkl"), "rb") as f:
    p1 = pickle.load(f)
m1   = p1["model"]
fc1  = p1["feature_cols"]
thr  = p1["threshold"]
n_trees_1 = p1.get("n_trees", m1.n_iter_)

val  = pd.read_csv(config.get_data_path("split_val.csv"),  parse_dates=["timestamp"])
test = pd.read_csv(config.get_data_path("split_test.csv"), parse_dates=["timestamp"])

X_v1 = val[fc1].values;  y_v1 = val["label_max"].values
X_t1 = test[fc1].values; y_t1 = test["label_max"].values

prob_v1 = m1.predict_proba(X_v1)[:, 1]
prob_t1 = m1.predict_proba(X_t1)[:, 1]
pred_v1 = (prob_v1 >= thr).astype(int)
pred_t1 = (prob_t1 >= thr).astype(int)

# ─────────────────────────────────────────────
# LOAD MULTICLASS MODEL
# ─────────────────────────────────────────────
print("Loading Model 2 (multiclass) ...")
with open(config.get_model_path("solar_flare_model_multiclass.pkl"), "rb") as f:
    p2 = pickle.load(f)
m2   = p2["model"]
fc2  = p2["feature_cols"]
n_trees_2 = p2.get("n_trees", m2.n_iter_)

# Build multiclass labels from raw data
raw = pd.read_csv(config.get_data_path("data_extended_v3.csv"), parse_dates=["timestamp"])
raw = raw.sort_values("timestamp").reset_index(drop=True)

def goes_to_class4(g):
    g = str(g).strip()
    if g == "FQ": return 0
    m = re.match(r"^([ABCMX])", g)
    return {"A": 0, "B": 0, "C": 1, "M": 2, "X": 3}.get(m.group(1), 0) if m else 0

raw["class4"] = raw["max_goes_class"].apply(goes_to_class4)
raw["class4_next"] = raw["class4"].shift(-1)
raw = raw.dropna(subset=["class4_next"])
raw["class4_next"] = raw["class4_next"].astype(int)

feat   = pd.read_csv(config.get_data_path("solar_flare_features.csv"), parse_dates=["timestamp"])
merged = feat.merge(raw[["timestamp", "class4_next"]], on="timestamp", how="inner")

val2  = merged[merged["timestamp"].dt.year == 2023]
test2 = merged[merged["timestamp"].dt.year == 2024]

X_v2 = val2[fc2].values;  y_v2 = val2["class4_next"].values
X_t2 = test2[fc2].values; y_t2 = test2["class4_next"].values

prob_v2 = m2.predict_proba(X_v2)
prob_t2 = m2.predict_proba(X_t2)
pred_v2 = m2.predict(X_v2)
pred_t2 = m2.predict(X_t2)


# ─────────────────────────────────────────────
# ROC CURVE — MODEL 1 (BINARY)
# ─────────────────────────────────────────────
print("Plotting ROC curve — Model 1 ...")

fpr_v, tpr_v, thresh_v = roc_curve(y_v1, prob_v1)
fpr_t, tpr_t, thresh_t = roc_curve(y_t1, prob_t1)
auc_v = auc(fpr_v, tpr_v)
auc_t = auc(fpr_t, tpr_t)

# Operating point
op_idx = np.argmin(np.abs(thresh_v - thr))
fpr_op = fpr_v[op_idx]
tpr_op = tpr_v[op_idx]

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#0a0e27")
ax.set_facecolor("#0f1630")

ax.grid(True, color="#1e2a50", linestyle="--", linewidth=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor("#2a3060")

ax.plot([0, 1], [0, 1], color="#4a5568", linewidth=1.2,
        linestyle="--", label="Random Classifier  (AUC = 0.50)")

ax.fill_between(fpr_v, tpr_v, alpha=0.12, color="#00d9ff")
ax.plot(fpr_v, tpr_v, color="#00d9ff", linewidth=2.5,
        label=f"Validation 2023  (AUC = {auc_v:.4f})")

ax.fill_between(fpr_t, tpr_t, alpha=0.10, color="#ffd700")
ax.plot(fpr_t, tpr_t, color="#ffd700", linewidth=2.5,
        label=f"Test 2024–2026  (AUC = {auc_t:.4f})")

ax.scatter([fpr_op], [tpr_op], color="#ff4500", s=100,
           zorder=5, edgecolors="#fff", linewidths=1.2,
           label=f"Operating point  (threshold = {thr:.4f})")

ax.annotate(
    f"  threshold = {thr:.4f}\n  FPR = {fpr_op:.4f}\n  TPR = {tpr_op:.4f}",
    xy=(fpr_op, tpr_op),
    xytext=(fpr_op + 0.10, tpr_op - 0.14),
    color="#ff4500", fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color="#ff4500", lw=1.2)
)

ax.set_xlabel("False Positive Rate  (1 − Specificity)",
              color="#8b9dc3", fontsize=11, labelpad=10)
ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)",
              color="#8b9dc3", fontsize=11, labelpad=10)
ax.set_title(
    "ROC Curve — Model 1: Binary Classifier (Flare / No-flare)\n"
    f"HistGradientBoostingClassifier · {n_trees_1} trees · 122 features",
    color="#e2e8f0", fontsize=12, fontweight="bold", pad=14
)
ax.tick_params(colors="#8b9dc3", labelsize=9)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)

legend = ax.legend(loc="lower right", fontsize=9.5, framealpha=0.25,
                   facecolor="#0a0e27", edgecolor="#2a3060", labelcolor="#e2e8f0")

ax.text(0.02, 0.97, "Perfect classifier = top-left corner",
        transform=ax.transAxes, color="#4ade80",
        fontsize=8, fontstyle="italic", va="top", alpha=0.7)

plt.tight_layout()
plt.savefig(config.get_report_path("model1_roc_curve.png"), dpi=180,
            bbox_inches="tight", facecolor="#0a0e27")
plt.close()
print(f"  Saved → {config.get_report_path('model1_roc_curve.png')}")


# ─────────────────────────────────────────────
# ROC CURVE — MODEL 2 (MULTICLASS, OvR)
# ─────────────────────────────────────────────
print("Plotting ROC curve — Model 2 ...")

COLORS = ["#4ade80", "#ffd700", "#ff8c00", "#ff4500"]
LABELS = ["No-flare (0)", "C-class (1)", "M-class (2)", "X-class (3)"]

yb_v = label_binarize(y_v2, classes=[0, 1, 2, 3])
yb_t = label_binarize(y_t2, classes=[0, 1, 2, 3])

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor("#0a0e27")

for ax_idx, (ax, yb, prob, split_name) in enumerate(
    [(axes[0], yb_v, prob_v2, "Validation 2023"),
     (axes[1], yb_t, prob_t2, "Test 2024")]
):
    ax.set_facecolor("#0f1630")
    ax.grid(True, color="#1e2a50", linestyle="--", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3060")

    ax.plot([0, 1], [0, 1], color="#4a5568", linewidth=1.2,
            linestyle="--", label="Random (AUC=0.50)")

    auc_scores = []
    for i, (cls_label, color) in enumerate(zip(LABELS, COLORS)):
        fpr_i, tpr_i, _ = roc_curve(yb[:, i], prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        auc_scores.append(auc_i)
        ax.fill_between(fpr_i, tpr_i, alpha=0.08, color=color)
        ax.plot(fpr_i, tpr_i, color=color, linewidth=2.2,
                label=f"{cls_label}  AUC={auc_i:.4f}")

    macro_auc = np.mean(auc_scores)
    ax.set_xlabel("False Positive Rate", color="#8b9dc3",
                  fontsize=10, labelpad=8)
    ax.set_ylabel("True Positive Rate", color="#8b9dc3",
                  fontsize=10, labelpad=8)
    ax.set_title(
        f"ROC Curve — {split_name}\n"
        f"Macro AUC = {macro_auc:.4f}",
        color="#e2e8f0", fontsize=11, fontweight="bold", pad=12
    )
    ax.tick_params(colors="#8b9dc3", labelsize=8.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    legend = ax.legend(loc="lower right", fontsize=9, framealpha=0.25,
                       facecolor="#0a0e27", edgecolor="#2a3060",
                       labelcolor="#e2e8f0")

fig.suptitle(
    "ROC Curves — Model 2: Multiclass Classifier (No-flare / C / M / X)\n"
    f"HistGradientBoostingClassifier · {n_trees_2} trees · One-vs-Rest (OvR)",
    color="#e2e8f0", fontsize=12, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(config.get_report_path("model2_roc_curve.png"), dpi=180,
            bbox_inches="tight", facecolor="#0a0e27")
plt.close()
print(f"  Saved → {config.get_report_path('model2_roc_curve.png')}")


# ─────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────
def div(c="=", n=66): return c * n

lines = []

lines += [
    div("═"),
    "  SURYA2BOOMI — MODEL TESTING REPORT",
    "  AI/ML Model Evaluation · Confusion Matrix · Metrics · AUC-ROC",
    div("═"),
    "",
    "  System has TWO models:",
    "  Model 1 : solar_flare_model.pkl          (Binary classifier)",
    "  Model 2 : solar_flare_model_multiclass.pkl (4-class classifier)",
    "",
]

# ── MODEL 1 ──
lines += [
    div(),
    "  MODEL 1 — BINARY CLASSIFIER",
    "  Task    : Predict Flare (1) vs No-flare (0)",
    "  Used for: Evaluation and reporting (step6, step_evaluation_report)",
    div(),
    f"  Algorithm     : HistGradientBoostingClassifier",
    f"  Trees built   : {n_trees_1}  (early stopping, n_iter_no_change=30)",
    f"  Features      : {len(fc1)}",
    f"  Threshold     : {thr:.4f}  (optimised on Val F1)",
    f"  Training data : 2017–2022  (51,672 rows)",
    "",
]

for split_name, y, pred, prob in [
    ("VALIDATION SET (2023 — 8,760 rows)", y_v1, pred_v1, prob_v1),
    ("TEST SET (2024–2026 — 19,464 rows)", y_t1, pred_t1, prob_t1),
]:
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y, pred)
    f1   = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    aucs = roc_auc_score(y, prob)

    lines += [
        f"\n  ── {split_name} ──",
        "",
        "  CONFUSION MATRIX",
        f"  {'':28}  Predicted: No-flare   Predicted: Flare",
        f"  {'Actual: No-flare':<28}  {tn:>18,}   {fp:>15,}",
        f"  {'Actual: Flare':<28}  {fn:>18,}   {tp:>15,}",
        "",
        f"  {'TP (correct flare)':<30} {tp:>10,}",
        f"  {'TN (correct no-flare)':<30} {tn:>10,}",
        f"  {'FP (false alarm)':<30} {fp:>10,}",
        f"  {'FN (missed flare)':<30} {fn:>10,}",
        "",
        "  METRICS",
        f"  {'Accuracy':<30} {acc:.4f}   ({acc*100:.2f}%)",
        f"  {'F1 Score':<30} {f1:.4f}",
        f"  {'Precision':<30} {prec:.4f}",
        f"  {'Recall (Sensitivity)':<30} {rec:.4f}",
        f"  {'AUC-ROC':<30} {aucs:.4f}",
        "",
    ]

lines += ["  ROC Curve saved → model1_roc_curve.png", ""]

# ── MODEL 2 ──
lines += [
    div(),
    "  MODEL 2 — MULTICLASS CLASSIFIER",
    "  Task    : Predict No-flare / C-class / M-class / X-class",
    "  Used for: All live forecasts (step7, step10)",
    div(),
    f"  Algorithm     : HistGradientBoostingClassifier",
    f"  Trees built   : {n_trees_2}  (early stopping, n_iter_no_change=15)",
    f"  Features      : {len(fc2)}",
    f"  Classes       : 0=No-flare  1=C-class  2=M-class  3=X-class",
    f"  Training data : split_train_smote.csv (SMOTE-balanced)",
    "",
]

for split_name, y, pred, prob in [
    ("VALIDATION SET (2023)", y_v2, pred_v2, prob_v2),
    ("TEST SET (2024 only — clean data)", y_t2, pred_t2, prob_t2),
]:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    acc  = accuracy_score(y, pred)
    f1   = f1_score(y, pred, average="macro", zero_division=0)
    prec = precision_score(y, pred, average="macro", zero_division=0)
    rec  = recall_score(y, pred, average="macro", zero_division=0)
    yb   = label_binarize(y, classes=[0, 1, 2, 3])
    aucs = roc_auc_score(yb, prob, multi_class="ovr", average="macro")

    lines += [
        f"\n  ── {split_name} ──",
        "",
        "  CONFUSION MATRIX (rows=Actual, cols=Predicted)",
        f"  {'':12}  No-flare   C-class   M-class   X-class",
        f"  {'No-flare':<12}  {cm[0,0]:>8,}  {cm[0,1]:>8,}  {cm[0,2]:>8,}  {cm[0,3]:>8,}",
        f"  {'C-class':<12}  {cm[1,0]:>8,}  {cm[1,1]:>8,}  {cm[1,2]:>8,}  {cm[1,3]:>8,}",
        f"  {'M-class':<12}  {cm[2,0]:>8,}  {cm[2,1]:>8,}  {cm[2,2]:>8,}  {cm[2,3]:>8,}",
        f"  {'X-class':<12}  {cm[3,0]:>8,}  {cm[3,1]:>8,}  {cm[3,2]:>8,}  {cm[3,3]:>8,}",
        "",
        "  METRICS (macro average across all 4 classes)",
        f"  {'Accuracy':<30} {acc:.4f}   ({acc*100:.2f}%)",
        f"  {'F1 Score (macro)':<30} {f1:.4f}",
        f"  {'Precision (macro)':<30} {prec:.4f}",
        f"  {'Recall (macro)':<30} {rec:.4f}",
        f"  {'AUC-ROC (macro OvR)':<30} {aucs:.4f}",
        "",
        "  PER-CLASS F1",
    ]
    f1_per = f1_score(y, pred, average=None, zero_division=0, labels=[0,1,2,3])
    for i, name in CLASS_NAMES.items():
        lines.append(f"    {name:<12} F1 = {f1_per[i]:.4f}")
    lines.append("")

lines += ["  ROC Curve saved → model2_roc_curve.png", ""]

lines += [
    div("═"),
    "  SUMMARY",
    div("═"),
    "",
    f"  {'Model':<42}  {'Val AUC':>9}  {'Test AUC':>9}  {'Val F1':>7}  {'Test F1':>8}",
    f"  {'-'*42}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*8}",
    f"  {'Model 1 — Binary (solar_flare_model.pkl)':<42}  "
    f"{roc_auc_score(y_v1,prob_v1):>9.4f}  "
    f"{roc_auc_score(y_t1,prob_t1):>9.4f}  "
    f"{f1_score(y_v1,pred_v1,zero_division=0):>7.4f}  "
    f"{f1_score(y_t1,pred_t1,zero_division=0):>8.4f}",
]
yb_v2b = label_binarize(y_v2, classes=[0,1,2,3])
yb_t2b = label_binarize(y_t2, classes=[0,1,2,3])
lines += [
    f"  {'Model 2 — Multiclass (multiclass.pkl)':<42}  "
    f"{roc_auc_score(yb_v2b,prob_v2,multi_class='ovr',average='macro'):>9.4f}  "
    f"{roc_auc_score(yb_t2b,prob_t2,multi_class='ovr',average='macro'):>9.4f}  "
    f"{f1_score(y_v2,pred_v2,average='macro',zero_division=0):>7.4f}  "
    f"{f1_score(y_t2,pred_t2,average='macro',zero_division=0):>8.4f}",
    "",
    "  OUTPUT FILES",
    f"  {config.get_report_path('model1_roc_curve.png')} — Binary model ROC curve (Val + Test)",
    f"  {config.get_report_path('model2_roc_curve.png')} — Multiclass model ROC curves (OvR, Val + Test)",
    f"  {config.get_report_path('model_testing_report.txt')} — This report",
    "",
    div("═"),
    "  SURYA2BOOMI · Sudeepa Weerasena · IIT Sri Lanka",
    div("═"),
    "",
]

report = "\n".join(lines)
with open(config.get_report_path("model_testing_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nReport saved → {config.get_report_path('model_testing_report.txt')}")
print()
print("SUMMARY:")
print(f"  Model 1 Binary    — Val AUC={roc_auc_score(y_v1,prob_v1):.4f}  Test AUC={roc_auc_score(y_t1,prob_t1):.4f}")
yb_v2b = label_binarize(y_v2, classes=[0,1,2,3])
yb_t2b = label_binarize(y_t2, classes=[0,1,2,3])
print(f"  Model 2 Multiclass — Val AUC={roc_auc_score(yb_v2b,prob_v2,multi_class='ovr',average='macro'):.4f}  Test AUC={roc_auc_score(yb_t2b,prob_t2,multi_class='ovr',average='macro'):.4f}")
print()
print("Files generated:")
print(f"  {config.get_report_path('model1_roc_curve.png')}")
print(f"  {config.get_report_path('model2_roc_curve.png')}")
print(f"  {config.get_report_path('model_testing_report.txt')}")
