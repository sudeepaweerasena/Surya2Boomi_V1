"""
live_data.py — Real-time NOAA data fetcher & feature engineer
=============================================================
Fetches live solar data from NOAA SWPC and engineers the exact
122-feature snapshot that the trained model expects.

Correct endpoints (verified from directory listing):
  xrays-7-day.json         → xray_flux_short (0.05-0.4nm), xray_flux_long
  magnetometers-7-day.json → Hp magnetic field component
  sunspot_report.json      → observed daily sunspot number

Feature engineering exactly mirrors step1_feature_engineering.py:
  lags (1,3,6,12,24h), rolling (6/12/24h mean/max/std),
  deltas (1,6,24h), log transforms, time encodings.

Falls back to last row of solar_flare_features.csv if any fetch fails.

Usage:
    from live_data import get_live_snapshot
    x0, now, source = get_live_snapshot(feat_cols)
"""

import json, math, warnings, sys, os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CORRECT NOAA URLS  (verified 2026-03-25)
# ─────────────────────────────────────────────
BASE             = "https://services.swpc.noaa.gov"
URL_XRAY         = f"{BASE}/json/goes/primary/xrays-7-day.json"
URL_MAG          = f"{BASE}/json/goes/primary/magnetometers-7-day.json"
URL_SUNSPOT      = f"{BASE}/json/sunspot_report.json"

TIMEOUT          = 15
FALLBACK_CSV     = config.get_data_path("solar_flare_features.csv")
LEAKY_FEATURES   = ["goes_flux", "goes_ordinal", "log_goes_flux"]
CYCLE25_START    = pd.Timestamp("2019-12-01")


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────
def _fetch(url):
    req = Request(url, headers={"User-Agent": "SuryaBoomi-Solar-Forecast/1.0"})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


# ─────────────────────────────────────────────
# 1. X-RAY FLUX  (xrays-7-day.json)
# Keys: time_tag, satellite, flux, energy
# energy values: "0.05-0.4nm" = short,  "0.1-0.8nm" = long
# ─────────────────────────────────────────────
def fetch_xray(verbose=True):
    data = _fetch(URL_XRAY)
    rows = []
    for d in data:
        flux   = d.get("flux")
        energy = d.get("energy", "")
        ts_str = d.get("time_tag", "")
        if flux is None or float(flux) <= 0:
            continue
        try:
            rows.append({
                "timestamp": pd.Timestamp(ts_str),
                "energy"   : str(energy),
                "flux"     : float(flux),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("xrays-7-day.json returned no usable rows")

    short = (df[df["energy"].str.contains("0.05", na=False)]
             [["timestamp","flux"]].copy().rename(columns={"flux":"xray_flux_short"}))
    long_ = (df[df["energy"].str.contains("0.1-0.8", na=False)]
             [["timestamp","flux"]].copy().rename(columns={"flux":"xray_flux_long"}))

    merged = (pd.merge(short, long_, on="timestamp", how="outer")
              .sort_values("timestamp")
              .set_index("timestamp")
              .resample("1h").mean()
              .interpolate("linear")
              .reset_index()
              .dropna(subset=["xray_flux_short"]))

    if verbose:
        print(f"  [live] X-ray   : {len(merged)} hourly rows  "
              f"({merged['timestamp'].min().date()} → {merged['timestamp'].max().date()})")
    return merged


# ─────────────────────────────────────────────
# 2. MAGNETOMETER  (magnetometers-7-day.json)
# Keys: time_tag, satellite, Hp, He, Hn, total_field
# Hp = field-parallel component (nT) — use as magnetic_field
# ─────────────────────────────────────────────
def fetch_magnetometer(verbose=True):
    data = _fetch(URL_MAG)
    rows = []
    for d in data:
        hp     = d.get("Hp")
        ts_str = d.get("time_tag", "")
        if hp is None:
            continue
        try:
            val = float(hp)
            if not math.isfinite(val):
                continue
            rows.append({"timestamp": pd.Timestamp(ts_str), "magnetic_field": val})
        except Exception:
            continue

    df = (pd.DataFrame(rows)
          .sort_values("timestamp")
          .set_index("timestamp")
          .resample("1h").mean()
          .interpolate("linear")
          .reset_index()
          .dropna(subset=["magnetic_field"]))

    if verbose:
        print(f"  [live] Mag     : {len(df)} hourly rows  "
              f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


# ─────────────────────────────────────────────
# 3. SUNSPOT NUMBER  (sunspot_report.json)
# Keys: time_tag, ssn  (or similar — check first row)
# Returns the most recent daily SSN as an integer.
# ─────────────────────────────────────────────
def fetch_sunspot(verbose=True):
    try:
        data = _fetch(URL_SUNSPOT)
        if not data:
            raise RuntimeError("Empty response")

        # Find the SSN key — field name varies slightly
        sample = data[0] if isinstance(data, list) else data
        if verbose and isinstance(data, list):
            print(f"  [live] SSN keys: {list(sample.keys())}")

        # Try known key names
        ssn_key = None
        for k in ["ssn", "SSN", "sunspot_number", "SN", "sn"]:
            if k in sample:
                ssn_key = k
                break

        if ssn_key is None:
            raise RuntimeError(f"SSN key not found in: {list(sample.keys())}")

        # Take most recent (last element)
        last = data[-1] if isinstance(data, list) else data
        ssn  = int(float(last[ssn_key]))

        if verbose:
            print(f"  [live] Sunspot : SSN = {ssn}  "
                  f"(date: {last.get('time_tag','?')})")
        return ssn

    except Exception as e:
        if verbose:
            print(f"  [live] Sunspot fetch failed ({e}) → using 0")
        return 0


# ─────────────────────────────────────────────
# TIME FEATURES
# ─────────────────────────────────────────────
def _time_features(ts):
    h   = ts.hour
    doy = ts.day_of_year
    cm  = max(0, (ts.year - CYCLE25_START.year)*12
                 + (ts.month - CYCLE25_START.month))
    return {
        "hour_sin"          : math.sin(2*math.pi*h/24),
        "hour_cos"          : math.cos(2*math.pi*h/24),
        "doy_sin"           : math.sin(2*math.pi*doy/365.25),
        "doy_cos"           : math.cos(2*math.pi*doy/365.25),
        "solar_cycle_month" : cm,
        "solar_cycle_sin"   : math.sin(2*math.pi*cm/132),
        "solar_cycle_cos"   : math.cos(2*math.pi*cm/132),
    }


# ─────────────────────────────────────────────
# BUILD FEATURE WINDOW
# ─────────────────────────────────────────────
def build_feature_window(xray_df, mag_df, ssn, verbose=True):
    df = xray_df.copy().sort_values("timestamp").reset_index(drop=True)

    # Merge magnetic field
    if not mag_df.empty:
        df = df.merge(mag_df[["timestamp","magnetic_field"]], on="timestamp", how="left")
    else:
        df["magnetic_field"] = np.nan

    df["magnetic_field"] = (df["magnetic_field"]
                            .fillna(method="ffill")
                            .fillna(method="bfill")
                            .fillna(10.0))

    df["sunspot_number"]  = float(ssn)
    df["goes_flux"]       = df["xray_flux_short"].clip(lower=1e-9)
    df["cumulative_index"] = (df["xray_flux_short"]
                               .rolling(24, min_periods=1).sum().fillna(0))

    df["log_xray_flux"]   = np.log10(df["xray_flux_short"].clip(lower=1e-9))
    df["log_cumulative"]  = np.log10(df["cumulative_index"].clip(lower=1e-9))

    def flux_to_ord(f):
        if f >= 1e-4: return 5.0
        if f >= 1e-5: return 4.0
        if f >= 1e-6: return 3.0
        if f >= 1e-7: return 2.0
        return 1.0

    df["goes_ordinal"]    = df["goes_flux"].apply(flux_to_ord)
    df["log_goes_flux"]   = np.log10(df["goes_flux"].clip(lower=1e-9))

    # Rolling stats
    base_cols = ["magnetic_field","sunspot_number","xray_flux_short",
                 "cumulative_index","goes_flux","log_xray_flux"]
    for col in base_cols:
        for w, wn in [(6,"roll6"),(12,"roll12"),(24,"roll24")]:
            r = df[col].rolling(w, min_periods=1)
            df[f"{col}_{wn}_mean"] = r.mean()
            df[f"{col}_{wn}_max"]  = r.max()
            df[f"{col}_{wn}_std"]  = r.std().fillna(0)

    # Lags
    lag_cols = ["magnetic_field","sunspot_number","xray_flux_short",
                "cumulative_index","goes_flux","goes_ordinal",
                "log_goes_flux","log_xray_flux"]
    for col in lag_cols:
        for lag in [1, 3, 6, 12, 24]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).fillna(method="bfill")

    # Deltas
    delta_cols = ["magnetic_field","sunspot_number","xray_flux_short",
                  "cumulative_index","goes_flux","log_xray_flux"]
    for col in delta_cols:
        for d in [1, 6, 24]:
            df[f"{col}_delta{d}"] = df[col].diff(d).fillna(0)

    # Time features
    tf_list = df["timestamp"].apply(_time_features)
    for key in ["hour_sin","hour_cos","doy_sin","doy_cos",
                "solar_cycle_month","solar_cycle_sin","solar_cycle_cos"]:
        df[key] = [row[key] for row in tf_list]

    df = df.dropna(subset=["xray_flux_short"]).reset_index(drop=True)

    if verbose:
        print(f"  [live] Window  : {len(df)} rows, "
              f"latest = {df['timestamp'].iloc[-1]}")
    return df


# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────
def _fallback_snapshot(feat_cols, verbose=True):
    if verbose:
        print(f"  [fallback] Loading last row from {FALLBACK_CSV} ...")
    df  = pd.read_csv(FALLBACK_CSV, parse_dates=["timestamp"]).sort_values("timestamp")
    row = df.iloc[-1]
    now = row["timestamp"]
    x0  = np.array([float(row[c]) if c in row.index else 0.0 for c in feat_cols])
    return x0, now, "fallback"


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def get_live_snapshot(feat_cols, verbose=True):
    """
    Fetch live NOAA data and return feature vector for current hour.

    Returns
    -------
    x0     : np.ndarray  (122,)
    now    : pd.Timestamp  UTC floor-hour of snapshot
    source : "live" | "fallback"
    """
    if verbose:
        print(f"\nFetching live NOAA data "
              f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC) ...")

    try:
        xray_df = fetch_xray(verbose)
        mag_df  = fetch_magnetometer(verbose)
        ssn     = fetch_sunspot(verbose)
        window  = build_feature_window(xray_df, mag_df, ssn, verbose)

        if len(window) < 25:
            raise RuntimeError(
                f"Window too short ({len(window)} rows, need ≥25 for lag24)")

        last = window.iloc[-1]
        now  = pd.Timestamp(last["timestamp"]).floor("h")
        x0   = np.array([
            float(last[c]) if c in last.index else 0.0
            for c in feat_cols
        ])

        if verbose:
            print(f"\n  [live] Snapshot ready:")
            print(f"    timestamp      = {now} UTC")
            print(f"    xray_flux_short= {last['xray_flux_short']:.3e} W/m²")
            print(f"    magnetic_field = {last['magnetic_field']:.1f} nT")
            print(f"    sunspot_number = {int(ssn)}")
            print(f"    goes_ordinal   = {last['goes_ordinal']:.0f}  "
                  f"(1=A/B, 3=C, 4=M, 5=X)")
            print(f"    non-zero feats = {(x0!=0).sum()} / {len(x0)}")

        return x0, now, "live"

    except Exception as e:
        if verbose:
            print(f"\n  [WARNING] Live fetch failed: {e}")
            print(f"  [WARNING] Falling back to last historical row.\n")
        return _fallback_snapshot(feat_cols, verbose)


# ─────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import pickle
    with open(config.get_model_path("solar_flare_model_multiclass.pkl"),"rb") as f:
        payload = pickle.load(f)
    feat_cols = [c for c in payload["feature_cols"] if c not in LEAKY_FEATURES]
    x0, now, source = get_live_snapshot(feat_cols, verbose=True)
    print(f"\n{'='*50}")
    print(f"  source = {source}")
    print(f"  now    = {now}")
    print(f"  shape  = {x0.shape}")
    print(f"{'='*50}")
