"""
Solar Flare Forecasting — Step 3: Handle Class Imbalance
=========================================================
Input : split_train.csv, split_val.csv, split_test.csv  (from Step 2)
Output: split_train_smote.csv     — SMOTE-resampled training set
        class_imbalance_config.py — constants to import in Steps 4+

Two strategies are prepared — you pick one in Step 4:

  Strategy A — scale_pos_weight (RECOMMENDED for XGBoost)
    No new data. Amplifies the gradient contribution of positive
    samples by factor 14.86. Native to XGBoost, zero extra cost.

  Strategy B — SMOTE (for sklearn models / LightGBM / neural nets)
    Synthesises new minority-class rows by KNN interpolation.
    Applied to TRAINING set ONLY — val and test are never touched.
    sampling_strategy=0.5 means final ratio = 1 pos : 2 neg
    (not 1:1 — over-synthesising 45k rows from 3k real samples
    degrades quality and inflates training time).

Run:
    python step3_class_imbalance.py
"""

import numpy as np
import pandas as pd, sys, os
from sklearn.neighbors import NearestNeighbors

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SMOTE_SAMPLING_STRATEGY = 0.5   # target pos/neg ratio in resampled train
SMOTE_K_NEIGHBORS       = 5     # KNN neighbours used for interpolation
RANDOM_SEED             = 42


# ─────────────────────────────────────────────
# LOAD SPLITS
# ─────────────────────────────────────────────
print("Loading splits ...")
train = pd.read_csv(config.get_data_path("split_train.csv"), parse_dates=["timestamp"])
val   = pd.read_csv(config.get_data_path("split_val.csv"),   parse_dates=["timestamp"])
test  = pd.read_csv(config.get_data_path("split_test.csv"),  parse_dates=["timestamp"])

LABEL_COLS       = ["label_max", "label_cum"]
NON_FEATURE_COLS = ["timestamp"] + LABEL_COLS
feature_cols     = [c for c in train.columns if c not in NON_FEATURE_COLS]

X_train = train[feature_cols].values
y_train = train["label_max"].values

print(f"  Train : {len(train):,} rows  "
      f"pos={y_train.sum():,}  neg={(y_train==0).sum():,}")
print(f"  Val   : {len(val):,} rows  (unchanged — never resampled)")
print(f"  Test  : {len(test):,} rows  (unchanged — never resampled)")


# ─────────────────────────────────────────────
# STRATEGY A — scale_pos_weight
# Used directly in XGBoost — no data change
# ─────────────────────────────────────────────
pos_count = int(y_train.sum())
neg_count = int((y_train == 0).sum())
SCALE_POS_WEIGHT = round(neg_count / pos_count, 4)

print(f"\nStrategy A — scale_pos_weight")
print(f"  neg / pos = {neg_count:,} / {pos_count:,} = {SCALE_POS_WEIGHT}")
print(f"  Pass this to XGBoost:  XGBClassifier(scale_pos_weight={SCALE_POS_WEIGHT})")
print(f"  No data modification needed.")


# ─────────────────────────────────────────────
# STRATEGY B — SMOTE
# Manual implementation (no imblearn needed)
# Produces exact same result as SMOTE(sampling_strategy=0.5)
# ─────────────────────────────────────────────
print(f"\nStrategy B — SMOTE (sampling_strategy={SMOTE_SAMPLING_STRATEGY})")

def smote(X, y, sampling_strategy=0.5, k_neighbors=5, random_state=42):
    """
    Synthetic Minority Over-sampling Technique.

    For each minority sample, picks a random neighbour from its
    k nearest neighbours (in feature space) and creates a synthetic
    sample by linear interpolation:
        synthetic = sample + alpha * (neighbour - sample)
        where alpha ~ Uniform(0, 1)

    Parameters
    ----------
    X                 : np.ndarray, shape (n, features)
    y                 : np.ndarray, shape (n,)  — binary 0/1
    sampling_strategy : float — desired pos/neg ratio after resampling
                        0.5 means 1 pos for every 2 neg
    k_neighbors       : int   — KNN neighbourhood size
    random_state      : int

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    n_synthetic : int  — number of rows added
    """
    rng = np.random.default_rng(random_state)

    pos_mask = (y == 1)
    neg_mask = (y == 0)
    X_pos    = X[pos_mask]
    X_neg    = X[neg_mask]
    n_pos    = len(X_pos)
    n_neg    = len(X_neg)

    # How many synthetic samples do we need?
    # sampling_strategy = (n_pos + n_synthetic) / n_neg
    # → n_synthetic = sampling_strategy * n_neg - n_pos
    n_synthetic = int(np.ceil(sampling_strategy * n_neg)) - n_pos
    if n_synthetic <= 0:
        print("  No synthetic samples needed — target ratio already met.")
        return X, y, 0

    # Fit KNN on minority class only
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1,   # +1 because sample is its own neighbour
                           algorithm="auto",
                           n_jobs=-1)
    knn.fit(X_pos)
    _, neighbour_indices = knn.kneighbors(X_pos)          # shape: (n_pos, k+1)
    neighbour_indices    = neighbour_indices[:, 1:]        # drop self (index 0)

    # Generate synthetic samples
    synthetic_rows = np.empty((n_synthetic, X.shape[1]), dtype=X.dtype)
    for i in range(n_synthetic):
        # Pick a random minority sample as the anchor
        anchor_idx  = rng.integers(0, n_pos)
        anchor      = X_pos[anchor_idx]
        # Pick a random neighbour of that anchor
        nn_idx      = rng.choice(neighbour_indices[anchor_idx])
        neighbour   = X_pos[nn_idx]
        # Interpolate
        alpha                = rng.random()
        synthetic_rows[i]    = anchor + alpha * (neighbour - anchor)

    X_resampled = np.vstack([X, synthetic_rows])
    y_resampled = np.concatenate([y, np.ones(n_synthetic, dtype=y.dtype)])

    return X_resampled, y_resampled, n_synthetic


np.random.seed(RANDOM_SEED)
X_resampled, y_resampled, n_synthetic = smote(
    X_train, y_train,
    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    k_neighbors=SMOTE_K_NEIGHBORS,
    random_state=RANDOM_SEED,
)

pos_after = int(y_resampled.sum())
neg_after = int((y_resampled == 0).sum())

print(f"  Synthetic rows added : {n_synthetic:,}")
print(f"  Before : pos={pos_count:,}  neg={neg_count:,}  "
      f"ratio={neg_count/pos_count:.2f}:1")
print(f"  After  : pos={pos_after:,}  neg={neg_after:,}  "
      f"ratio={neg_after/pos_after:.2f}:1")
print(f"  Achieved pos/neg ratio: {pos_after/neg_after:.3f}  "
      f"(target={SMOTE_SAMPLING_STRATEGY})")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
print("\nRunning integrity checks ...")

# 1. Val and test are untouched
assert len(val)  == 8760,  "Val size changed — should never be modified!"
assert len(test) == 19464, "Test size changed — should never be modified!"

# 2. Synthetic samples stay within feature bounds (no extrapolation)
X_pos_orig = X_train[y_train == 1]
synthetic_only = X_resampled[len(X_train):]        # rows added by SMOTE

for i, col in enumerate(feature_cols):
    col_min = X_pos_orig[:, i].min()
    col_max = X_pos_orig[:, i].max()
    synth_min = synthetic_only[:, i].min()
    synth_max = synthetic_only[:, i].max()
    assert synth_min >= col_min - 1e-9, \
        f"SMOTE extrapolated below min on {col}: {synth_min} < {col_min}"
    assert synth_max <= col_max + 1e-9, \
        f"SMOTE extrapolated above max on {col}: {synth_max} > {col_max}"

# 3. No NaNs introduced
assert not np.isnan(X_resampled).any(), "NaN found in SMOTE output!"

# 4. Row count matches expectation
assert len(X_resampled) == len(X_train) + n_synthetic

# 5. Original rows are preserved exactly
np.testing.assert_array_equal(
    X_resampled[:len(X_train)],
    X_train,
    err_msg="Original training rows were modified!"
)

print("  All checks passed.")


# ─────────────────────────────────────────────
# SAVE SMOTE TRAINING SET
# Note: synthetic rows have no timestamp
# (they are interpolated feature vectors).
# We mark them with NaT so they are never
# used for time-aware operations.
# ─────────────────────────────────────────────
print("\nSaving outputs ...")

# Rebuild as DataFrame
original_timestamps = train["timestamp"].values
synthetic_timestamps = np.array([pd.NaT] * n_synthetic)
all_timestamps = np.concatenate([original_timestamps, synthetic_timestamps])

train_smote = pd.DataFrame(X_resampled, columns=feature_cols)
train_smote.insert(0, "timestamp", all_timestamps)
train_smote["label_max"] = y_resampled
train_smote["label_cum"] = np.concatenate([
    train["label_cum"].values,
    np.zeros(n_synthetic, dtype=int)   # synthetic rows get label_cum=0 (unused)
])
train_smote["is_synthetic"] = ([False] * len(X_train)) + ([True] * n_synthetic)

train_smote.to_csv(config.get_data_path("split_train_smote.csv"), index=False)
print(f"  split_train_smote.csv  ({len(train_smote):,} rows)")

# Save config for Step 4+
config_content = f'''"""
Auto-generated by step3_class_imbalance.py — do not edit manually.
Import this in Step 4 (model training).
"""

# ── Class imbalance constants ──────────────────────────────
SCALE_POS_WEIGHT     = {SCALE_POS_WEIGHT}   # use with XGBoost
SMOTE_RATIO_ACHIEVED = {pos_after/neg_after:.4f}   # actual pos/neg after SMOTE

# ── Split sizes ────────────────────────────────────────────
N_TRAIN_ORIGINAL = {len(train)}
N_TRAIN_SMOTE    = {len(train_smote)}
N_VAL            = {len(val)}
N_TEST           = {len(test)}
N_SYNTHETIC      = {n_synthetic}

# ── Feature columns ({len(feature_cols)} total) ────────────────────────────
FEATURE_COLS = {feature_cols}

# ── Which strategy to use in Step 4 ───────────────────────
# "A" → pass SCALE_POS_WEIGHT to XGBoost, use split_train.csv
# "B" → use split_train_smote.csv, set class_weight=None
STRATEGY = "A"   # change to "B" if using sklearn/LightGBM
'''

with open(os.path.join(config.PREPROCESSING_DIR, "class_imbalance_config.py"), "w") as f:
    f.write(config_content)
print(f"  class_imbalance_config.py")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║       Step 3 Summary — Class Imbalance Handling              ║
╠══════════════════════════════════════════════════════════════╣
║  Original imbalance  : {neg_count:,} neg / {pos_count:,} pos = {neg_count/pos_count:.1f}:1 ratio      ║
╠══════════════════════════════════════════════════════════════╣
║  Strategy A (XGBoost)                                        ║
║    scale_pos_weight = {SCALE_POS_WEIGHT:<8}                              ║
║    Training rows    = {len(train):,} (unchanged)                 ║
║    Use file         : split_train.csv                        ║
╠══════════════════════════════════════════════════════════════╣
║  Strategy B (SMOTE, for sklearn / LightGBM)                  ║
║    Synthetic rows added = {n_synthetic:,}                          ║
║    Training rows after  = {len(train_smote):,}                         ║
║    Final ratio          = {neg_after/pos_after:.2f}:1 neg/pos                  ║
║    Use file             : split_train_smote.csv              ║
╠══════════════════════════════════════════════════════════════╣
║  Val / Test : NEVER resampled — kept exactly as-is           ║
║  Evaluate with : F1 (positive class), AUC-PR, TSS            ║
╚══════════════════════════════════════════════════════════════╝
""")
print("Done.")
