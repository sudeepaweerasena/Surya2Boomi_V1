"""
Solar Flare Forecasting — Step 1: Feature Engineering
======================================================
Input : data_extended_v3.csv
Output: solar_flare_features.csv  (125 features + labels + timestamp)

Run:
    python step1_feature_engineering.py
"""

import re, sys, os
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE  = config.get_data_path("data_extended_v3.csv")
OUTPUT_FILE = config.get_data_path("solar_flare_features.csv")

LAG_STEPS    = [1, 3, 6, 12, 24]          # hours to look back (lag features)
ROLL_WINDOWS = [6, 12, 24]                 # rolling window sizes in hours

# Solar Cycle 25 minimum (start reference for cycle-phase feature)
CYCLE25_START = pd.Timestamp("2019-12-01")
CYCLE_MONTHS  = 132                        # 11-year cycle in months


# ─────────────────────────────────────────────
# HELPER: GOES CLASS → NUMERIC
# ─────────────────────────────────────────────
_GOES_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
_GOES_ORD  = {"FQ": 0,   "A": 1,    "B": 2,    "C": 3,    "M": 4,   "X": 5}
_GOES_RE   = re.compile(r"^([ABCMX])(\d+\.?\d*)")

def goes_to_flux(goes_str: str) -> float:
    """Convert GOES class string (e.g. 'M5.4') to W/m² flux value."""
    g = str(goes_str).strip()
    if g == "FQ":
        return 0.0
    m = _GOES_RE.match(g)
    if m:
        return _GOES_BASE[m.group(1)] * float(m.group(2))
    return 0.0                             # fallback — never triggered in this dataset

def goes_to_ordinal(goes_str: str) -> int:
    """Convert GOES class string to ordinal: FQ=0, A=1, B=2, C=3, M=4, X=5."""
    g = str(goes_str).strip()
    if g == "FQ":
        return 0
    for letter in ("X", "M", "C", "B", "A"):
        if g.startswith(letter):
            return _GOES_ORD[letter]
    return 0


# ─────────────────────────────────────────────
# LOAD & SORT
# ─────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 1A — IMPUTE MISSING VALUES
# (must happen BEFORE lag features so rolling
#  windows are not broken by mid-series NaNs)
# ─────────────────────────────────────────────
print("\nStep 1a: Imputing missing values ...")

df["magnetic_field"]  = df["magnetic_field"].ffill().bfill()
df["xray_flux_short"] = df["xray_flux_short"].ffill().bfill()

print(f"  Remaining NaNs: {df.isnull().sum().sum()}")   # should be 0


# ─────────────────────────────────────────────
# STEP 1B — GOES CLASS ENCODING
# ─────────────────────────────────────────────
print("Step 1b: Encoding GOES class ...")

goes_flux    = df["max_goes_class"].apply(goes_to_flux).rename("goes_flux")
goes_ordinal = df["max_goes_class"].apply(goes_to_ordinal).rename("goes_ordinal")


# ─────────────────────────────────────────────
# STEP 1C — LOG TRANSFORMS
# (xray_flux_short and goes_flux span 5+ orders
#  of magnitude; log scale helps tree models
#  and is required for any linear/distance-
#  based model)
# ─────────────────────────────────────────────
print("Step 1c: Log-transforming skewed columns ...")

log_xray_flux  = np.log10(df["xray_flux_short"].clip(lower=1e-10)).rename("log_xray_flux")
log_goes_flux  = np.log10(goes_flux.clip(lower=1e-10)).rename("log_goes_flux")
log_cumulative = np.log1p(df["cumulative_index"]).rename("log_cumulative")


# ─────────────────────────────────────────────
# STEP 1D — LAG FEATURES
# shift(n) moves the column DOWN by n rows,
# so row t sees the value from t-n (no leakage)
# ─────────────────────────────────────────────
print("Step 1d: Building lag features ...")

lag_source_cols = {
    "magnetic_field"  : df["magnetic_field"],
    "sunspot_number"  : df["sunspot_number"],
    "xray_flux_short" : df["xray_flux_short"],
    "cumulative_index": df["cumulative_index"],
    "goes_flux"       : goes_flux,
    "goes_ordinal"    : goes_ordinal,
    "log_xray_flux"   : log_xray_flux,
    "log_goes_flux"   : log_goes_flux,
}

lag_frames = []
for col_name, series in lag_source_cols.items():
    for lag in LAG_STEPS:
        lag_frames.append(series.shift(lag).rename(f"{col_name}_lag{lag}"))


# ─────────────────────────────────────────────
# STEP 1E — ROLLING WINDOW FEATURES
# shift(1) before rolling ensures the window
# covers [t-w, t-1] — never includes t itself
# ─────────────────────────────────────────────
print("Step 1e: Building rolling window features ...")

roll_source_cols = {
    "magnetic_field"  : df["magnetic_field"],
    "sunspot_number"  : df["sunspot_number"],
    "xray_flux_short" : df["xray_flux_short"],
    "cumulative_index": df["cumulative_index"],
    "goes_flux"       : goes_flux,
    "log_xray_flux"   : log_xray_flux,
}

roll_frames = []
for col_name, series in roll_source_cols.items():
    base = series.shift(1)               # exclude current hour
    for w in ROLL_WINDOWS:
        roll_frames.append(base.rolling(w).mean().rename(f"{col_name}_roll{w}_mean"))
        roll_frames.append(base.rolling(w).max() .rename(f"{col_name}_roll{w}_max"))
        roll_frames.append(base.rolling(w).std() .rename(f"{col_name}_roll{w}_std"))


# ─────────────────────────────────────────────
# STEP 1F — DELTA / RATE-OF-CHANGE FEATURES
# Captures how fast each signal is rising or
# falling — important precursor to flares
# ─────────────────────────────────────────────
print("Step 1f: Building delta (rate-of-change) features ...")

delta_source_cols = {
    "magnetic_field"  : df["magnetic_field"],
    "sunspot_number"  : df["sunspot_number"],
    "xray_flux_short" : df["xray_flux_short"],
    "cumulative_index": df["cumulative_index"],
    "log_xray_flux"   : log_xray_flux,
}

delta_frames = []
for col_name, series in delta_source_cols.items():
    for d in [1, 6, 24]:
        delta_frames.append(series.diff(d).rename(f"{col_name}_delta{d}"))


# ─────────────────────────────────────────────
# STEP 1G — TIME / SOLAR CYCLE FEATURES
# Cyclical encoding prevents discontinuities at
# midnight (hour 23 → 0) and year boundaries
# ─────────────────────────────────────────────
print("Step 1g: Building time and solar cycle features ...")

hour = df["timestamp"].dt.hour
doy  = df["timestamp"].dt.dayofyear
solar_cycle_month = ((df["timestamp"] - CYCLE25_START).dt.days / 30.44).clip(lower=0)

time_features = pd.DataFrame({
    "hour_sin"         : np.sin(2 * np.pi * hour / 24),
    "hour_cos"         : np.cos(2 * np.pi * hour / 24),
    "doy_sin"          : np.sin(2 * np.pi * doy  / 365.25),
    "doy_cos"          : np.cos(2 * np.pi * doy  / 365.25),
    "solar_cycle_month": solar_cycle_month,
    "solar_cycle_sin"  : np.sin(2 * np.pi * solar_cycle_month / CYCLE_MONTHS),
    "solar_cycle_cos"  : np.cos(2 * np.pi * solar_cycle_month / CYCLE_MONTHS),
})


# ─────────────────────────────────────────────
# ASSEMBLE FINAL DATAFRAME
# Using pd.concat once avoids the DataFrame
# fragmentation warning from repeated inserts
# ─────────────────────────────────────────────
print("\nAssembling final feature dataframe ...")

base_cols = df[["timestamp",
                "magnetic_field", "sunspot_number",
                "cumulative_index", "xray_flux_short",
                "label_max", "label_cum"]].copy()

df_final = pd.concat(
    [base_cols,
     goes_flux, goes_ordinal,
     log_xray_flux, log_goes_flux, log_cumulative,
     *lag_frames,
     *roll_frames,
     *delta_frames,
     time_features],
    axis=1
)


# ─────────────────────────────────────────────
# STEP 1H — DROP LAG WARMUP ROWS
# First 24 rows have NaN in lag-24 features
# ─────────────────────────────────────────────
rows_before = len(df_final)
df_final = df_final.dropna().reset_index(drop=True)
rows_dropped = rows_before - len(df_final)

print(f"  Dropped {rows_dropped} warmup rows (lag-24 window)")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
feature_cols = [c for c in df_final.columns
                if c not in ("timestamp", "label_max", "label_cum")]

print("\n── Validation ─────────────────────────────")
print(f"  Final rows     : {len(df_final):,}")
print(f"  Feature columns: {len(feature_cols)}")
print(f"  Remaining NaNs : {df_final[feature_cols].isnull().sum().sum()}")
print(f"  label_max=1    : {df_final['label_max'].sum():,}  "
      f"({df_final['label_max'].mean()*100:.1f}%)")
print(f"  label_cum=1    : {df_final['label_cum'].sum():,}  "
      f"({df_final['label_cum'].mean()*100:.1f}%)")
print(f"  Date range     : {df_final['timestamp'].min().date()} "
      f"→ {df_final['timestamp'].max().date()}")
print("────────────────────────────────────────────")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE}")

print("\nFeature groups summary:")
print(f"  Raw base features    :  9  (magnetic, sunspot, xray, cumulative, goes_flux,")
print(f"                              goes_ordinal, log_xray, log_goes, log_cumulative)")
print(f"  Lag features         : {len(lag_frames):>2}  ({len(lag_source_cols)} cols × {len(LAG_STEPS)} lags)")
print(f"  Rolling features     : {len(roll_frames):>2}  ({len(roll_source_cols)} cols × {len(ROLL_WINDOWS)} windows × 3 stats)")
print(f"  Delta features       : {len(delta_frames):>2}  ({len(delta_source_cols)} cols × 3 deltas)")
print(f"  Time/cycle features  :  7  (hour sin/cos, doy sin/cos, cycle month/sin/cos)")
print(f"  ─────────────────────────")
print(f"  Total                : {len(feature_cols):>2}")
