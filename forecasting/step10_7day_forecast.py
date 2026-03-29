"""
Solar Flare Forecasting — Step 10: 7-Day Daily Forecast
========================================================
Fetches LIVE solar data from NOAA, then runs 168-hour
autoregressive rollout and aggregates to daily d+1..d+7.

Same model as step7 — only the horizon changes.
Live data via live_data.py, falls back gracefully.

Key functions (used by pipeline.py):
  rollout()       — autoregressive 168h forecast
  daily_rollup()  — aggregate to d+1..d+7
  run()           — full step10 pipeline, returns DataFrame

Usage
-----
  python step10_7day_forecast.py             # live data
  python step10_7day_forecast.py --fallback  # force historical
"""

import pickle, argparse, os, sys
# Add root to sys.path for standalone runs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
from forecasting.live_data import get_live_snapshot, _fallback_snapshot

FORECAST_HOURS    = 168
MODEL_FILE        = config.get_model_path("solar_flare_model_multiclass.pkl")
LEAKY_FEATURES    = ["goes_flux", "goes_ordinal", "log_goes_flux"]
CLASS_NAMES       = {0:"No-flare", 1:"C-class", 2:"M-class", 3:"X-class"}
CLASS4_TO_ORDINAL = {0:0.0, 1:3.0, 2:4.0, 3:5.0}
CLASS4_TO_FLUX    = {0:0.0, 1:1e-6, 2:1e-5, 3:1e-4}
CONFIDENCE        = {1:"High",2:"High",3:"High",4:"Medium",5:"Medium",6:"Low",7:"Low"}


def rollout(model, feat_cols, x0, now):
    def get_idx(name):
        try: return feat_cols.index(name)
        except ValueError: return -1

    # Initialize a 25-hour history buffer by interpolating the known lags from x0
    # This serves as the rolling window to simulate all 50+ derived features dynamically.
    hist_flux = np.zeros(25)
    f24, f12, f6, f3, f1 = [x0[get_idx(f"xray_flux_short_lag{L}")] if get_idx(f"xray_flux_short_lag{L}") != -1 else 1e-8 for L in [24,12,6,3,1]]
    hist_flux[0] = f24
    for i in range(1, 13): hist_flux[i] = f24 + (f12-f24)*(i/12.0)
    for i in range(13, 19): hist_flux[i] = f12 + (f6-f12)*((i-12)/6.0)
    for i in range(19, 22): hist_flux[i] = f6 + (f3-f6)*((i-18)/3.0)
    for i in range(22, 24): hist_flux[i] = f3 + (f1-f3)*((i-21)/2.0)
    hist_flux[24] = f1
    hist_flux = list(hist_flux)

    def flux_to_ord(f):
        if f >= 1e-4: return 5.0
        if f >= 1e-5: return 4.0
        if f >= 1e-6: return 3.0
        if f >= 1e-7: return 2.0
        return 1.0

    x_cur, rows = x0.copy(), []
    import math

    for h in range(1, FORECAST_HOURS+1):
        ts    = now + pd.Timedelta(hours=h)
        proba = model.predict_proba(x_cur.reshape(1,-1))[0]
        pred  = int(np.argmax(proba))
        
        rows.append({"timestamp":ts, "date":ts.date(), "pred_class":pred,
                     "p_noflare":float(proba[0]),"p_c":float(proba[1]),
                     "p_m":float(proba[2]),"p_x":float(proba[3])})

        # Expected instantaneous flux
        raw_exp = proba[0]*1e-8 + proba[1]*1e-6 + proba[2]*1e-5 + proba[3]*1e-4
        
        # Decay aggressively to simulate the short-lived transient nature of solar flares
        # This breaks the autoregressive equilibrium loop where a continuous high average is assumed
        baseline = 1e-8
        exp_flux = baseline + (raw_exp - baseline) * 0.15 + (hist_flux[-1] - baseline) * 0.15
        hist_flux.append(exp_flux)
        
        hf_arr = np.array(hist_flux)
        
        # Dynamically update the 50+ derived rolling and delta features that the model relies on
        def update_lag_roll_delta(base_name, arr):
            if get_idx(f"{base_name}_lag1") != -1: x_cur[get_idx(f"{base_name}_lag1")] = arr[-2]
            if get_idx(f"{base_name}_lag3") != -1: x_cur[get_idx(f"{base_name}_lag3")] = arr[-4]
            if get_idx(f"{base_name}_lag6") != -1: x_cur[get_idx(f"{base_name}_lag6")] = arr[-7]
            if get_idx(f"{base_name}_lag12") != -1: x_cur[get_idx(f"{base_name}_lag12")] = arr[-13]
            if get_idx(f"{base_name}_lag24") != -1: x_cur[get_idx(f"{base_name}_lag24")] = arr[-25]
            
            for w in [6, 12, 24]:
                w_arr = arr[-w-1:-1]
                if len(w_arr) > 0:
                    if get_idx(f"{base_name}_roll{w}_mean") != -1: x_cur[get_idx(f"{base_name}_roll{w}_mean")] = np.mean(w_arr)
                    if get_idx(f"{base_name}_roll{w}_max") != -1: x_cur[get_idx(f"{base_name}_roll{w}_max")] = np.max(w_arr)
                    if get_idx(f"{base_name}_roll{w}_std") != -1: x_cur[get_idx(f"{base_name}_roll{w}_std")] = np.std(w_arr)
                
            if len(arr) >= 26:
                if get_idx(f"{base_name}_delta1") != -1: x_cur[get_idx(f"{base_name}_delta1")] = arr[-2] - arr[-3]
                if get_idx(f"{base_name}_delta6") != -1: x_cur[get_idx(f"{base_name}_delta6")] = arr[-2] - arr[-8]
                if get_idx(f"{base_name}_delta24") != -1: x_cur[get_idx(f"{base_name}_delta24")] = arr[-2] - arr[-26]

        update_lag_roll_delta("xray_flux_short", hf_arr)
        update_lag_roll_delta("goes_flux", np.clip(hf_arr, 1e-9, None))
        update_lag_roll_delta("log_xray_flux", np.log10(np.clip(hf_arr, 1e-9, None)))
        update_lag_roll_delta("log_goes_flux", np.log10(np.clip(hf_arr, 1e-9, None)))
        
        ord_arr = np.array([flux_to_ord(f) for f in hf_arr])
        update_lag_roll_delta("goes_ordinal", ord_arr)
        
        cum_arr = [np.sum(hf_arr[max(0, i-23):i+1]) for i in range(len(hf_arr))]
        update_lag_roll_delta("cumulative_index", cum_arr)

        idx_hsin, idx_hcos = get_idx("hour_sin"), get_idx("hour_cos")
        idx_dsin, idx_dcos = get_idx("doy_sin"), get_idx("doy_cos")
        if idx_hsin != -1: x_cur[idx_hsin] = math.sin(2*math.pi*ts.hour/24)
        if idx_hcos != -1: x_cur[idx_hcos] = math.cos(2*math.pi*ts.hour/24)
        if idx_dsin != -1: x_cur[idx_dsin] = math.sin(2*math.pi*ts.day_of_year/365.25)
        if idx_dcos != -1: x_cur[idx_dcos] = math.cos(2*math.pi*ts.day_of_year/365.25)
        
    return pd.DataFrame(rows)


def daily_rollup(hourly_df, now):
    today, daily = now.date(), []
    for date, grp in hourly_df.groupby("date"):
        d = (pd.Timestamp(date)-pd.Timestamp(today)).days
        if d<=0 or d>7: continue
        pc   = int(grp["pred_class"].max())
        pcol = ["p_noflare","p_c","p_m","p_x"][pc]
        daily.append({"date":date,"day_offset":f"d+{d}","peak_class":pc,
                      "peak_name":CLASS_NAMES[pc],
                      "peak_prob":round(float(grp[pcol].max())*100,2),
                      "max_p_c":round(float(grp["p_c"].max())*100,2),
                      "max_p_m":round(float(grp["p_m"].max())*100,2),
                      "max_p_x":round(float(grp["p_x"].max())*100,2),
                      "confidence":CONFIDENCE.get(d,"Low")})
    return pd.DataFrame(daily)


def print_forecast(now, source, daily_df):
    src = "LIVE NOAA DATA" if source=="live" else "FALLBACK (historical CSV)"
    b = "="*70
    print(f"\n{b}\n  Solar Flare 7-Day Daily Forecast")
    print(f"  Source : {src}")
    print(f"  Issued : {now.strftime('%Y-%m-%d %H:%M')} UTC\n{b}")
    print(f"\n  {'Offset':<6}  {'Date':<12}  {'Class':<12}  "
          f"{'P':>6}  {'P(M)':>6}  {'P(X)':>6}  Conf")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
    for _, r in daily_df.iterrows():
        flag = " ⚠X" if r["peak_class"]==3 else " ⚠M" if r["peak_class"]==2 else ""
        print(f"  {r['day_offset']:<6}  {str(r['date']):<12}  {r['peak_name']:<12}  "
              f"{r['peak_prob']:>5.1f}%  {r['max_p_m']:>5.1f}%  "
              f"{r['max_p_x']:>5.1f}%  {r['confidence']}{flag}")
    print(f"\n{b}\n")


# ── RUN (importable) ──────────────────────────────────────────────────
def run(model=None, feat_cols=None, x0=None, now=None, source=None, verbose=True):
    """
    Run the full step10 pipeline and return DataFrames.

    If model/feat_cols/x0/now are provided, uses them directly.
    Otherwise loads model and fetches live data.

    Returns: (daily_df, now, source)
    """
    if model is None or feat_cols is None or x0 is None or now is None:
        with open(MODEL_FILE, "rb") as f:
            payload = pickle.load(f)
        model     = payload["model"]
        feat_cols = [c for c in payload["feature_cols"] if c not in LEAKY_FEATURES]
        x0, now, source = get_live_snapshot(feat_cols, verbose=verbose)

    hourly_df = rollout(model, feat_cols, x0, now)
    daily_df  = daily_rollup(hourly_df, now)
    return daily_df, now, source


# ── MAIN (CLI only, no file output) ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", action="store_true")
    args = ap.parse_args()

    with open(MODEL_FILE,"rb") as f:
        payload = pickle.load(f)
    model     = payload["model"]
    feat_cols = [c for c in payload["feature_cols"] if c not in LEAKY_FEATURES]
    print(f"\nModel: {payload['n_trees']} trees | {len(feat_cols)} features")

    if args.fallback:
        x0, now, source = _fallback_snapshot(feat_cols, verbose=True)
    else:
        x0, now, source = get_live_snapshot(feat_cols, verbose=True)

    print(f"Snapshot: {now} UTC  [source={source}]")
    print(f"Running {FORECAST_HOURS}-hour rollout ...")
    daily_df, now, source = run(model, feat_cols, x0, now, source, verbose=True)
    print_forecast(now, source, daily_df)


if __name__ == "__main__":
    main()
