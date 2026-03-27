"""
Solar Flare Forecasting — Step 7: 72-Hour Hourly Forecast
=========================================================
Fetches LIVE solar data from NOAA, engineers features,
then runs the trained multiclass model autoregressively
for 72 hours (t+1 to t+72).

Model was trained on 2017–2022 historical data (step4 / --train).
At forecast time: live NOAA APIs replace the historical CSV snapshot.

Live data sources (via live_data.py)
-------------------------------------
  NOAA xray-fluxes-7-day.json  → xray_flux_short, goes_flux features
  NOAA magnetometers-6-hour    → magnetic_field features
  NOAA solar_regions.json      → sunspot_number proxy

Fallback: if any live fetch fails, last row of solar_flare_features.csv
is used and a warning is printed.

Key functions (used by pipeline.py):
  rollout()       — autoregressive 72h forecast
  daily_rollup()  — aggregate to d+1..d+3
  run()           — full step7 pipeline, returns DataFrames

Usage
-----
  python step7_72h_forecast.py                # live data, current UTC
  python step7_72h_forecast.py --fallback     # force historical fallback
  python step7_72h_forecast.py --train        # retrain then forecast
"""

import pickle, argparse, re, sys, os
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from forecasting.live_data import get_live_snapshot, _fallback_snapshot

FORECAST_HOURS    = 72
MODEL_FILE        = config.get_model_path("solar_flare_model_multiclass.pkl")
LEAKY_FEATURES    = ["goes_flux", "goes_ordinal", "log_goes_flux"]
CLASS_NAMES       = {0:"No-flare", 1:"C-class", 2:"M-class", 3:"X-class"}
CLASS4_TO_ORDINAL = {0:0.0, 1:3.0, 2:4.0, 3:5.0}
CLASS4_TO_FLUX    = {0:0.0, 1:1e-6, 2:1e-5, 3:1e-4}


# ── TRAIN ──────────────────────────────────────────────────────────────
def train_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    print("Loading training data ...")
    train = pd.read_csv(config.get_data_path("split_train_smote.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(config.get_data_path("split_val.csv"),          parse_dates=["timestamp"])

    def goes_class(g):
        g = str(g).strip()
        if g == "FQ": return 0
        m = re.match(r"^([ABCMX])", g)
        return {"A":0,"B":0,"C":1,"M":2,"X":3}.get(m.group(1), 0) if m else 0

    for df in [train, val]:
        df["class4"] = df["max_goes_class"].apply(goes_class)

    drop = ["timestamp","max_goes_class","label_max","label_cum","class4"] + LEAKY_FEATURES
    feat_cols = [c for c in train.columns if c not in drop]

    X_tr  = train[feat_cols].values
    y_tr  = train["class4"].shift(-1).fillna(0).astype(int).values

    print(f"Training on {len(X_tr):,} rows, {len(feat_cols)} features ...")
    model = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.05, max_depth=6,
        min_samples_leaf=3, class_weight="balanced",
        early_stopping=True, n_iter_no_change=15,
        random_state=42, verbose=1,
    )
    model.fit(X_tr, y_tr)
    n_trees = model.n_iter_
    print(f"Trained: {n_trees} trees")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model":model, "feature_cols":feat_cols, "n_trees":n_trees}, f)
    print(f"Saved → {MODEL_FILE}")


# ── ROLLOUT ────────────────────────────────────────────────────────────
def rollout(model, feat_cols, x0, now):
    idx_ord  = feat_cols.index("goes_ordinal_lag1")
    idx_flux = feat_cols.index("goes_flux_lag1")
    idx_log  = feat_cols.index("log_goes_flux_lag1")
    x_cur, rows = x0.copy(), []

    for h in range(1, FORECAST_HOURS + 1):
        ts    = now + pd.Timedelta(hours=h)
        proba = model.predict_proba(x_cur.reshape(1,-1))[0]
        pred  = int(np.argmax(proba))
        rows.append({
            "timestamp"  : ts,
            "hour_offset": f"t+{h}",
            "pred_class" : pred,
            "pred_name"  : CLASS_NAMES[pred],
            "probability": round(float(proba[pred])*100, 2),
            "p_noflare"  : round(float(proba[0])*100, 2),
            "p_c"        : round(float(proba[1])*100, 2),
            "p_m"        : round(float(proba[2])*100, 2),
            "p_x"        : round(float(proba[3])*100, 2),
        })
        x_cur[idx_ord]  = CLASS4_TO_ORDINAL[pred]
        x_cur[idx_flux] = CLASS4_TO_FLUX[pred]
        x_cur[idx_log]  = np.log10(max(CLASS4_TO_FLUX[pred], 1e-10))

    return pd.DataFrame(rows)


# ── DAILY ROLLUP ────────────────────────────────────────────────────────
def daily_rollup(hourly_df, now):
    today, daily = now.date(), []
    for date, grp in hourly_df.groupby(hourly_df["timestamp"].dt.date):
        d = (pd.Timestamp(date) - pd.Timestamp(today)).days
        if d <= 0 or d > 3: continue
        pc   = int(grp["pred_class"].max())
        pcol = ["p_noflare","p_c","p_m","p_x"][pc]
        daily.append({"date":date, "day_offset":f"d+{d}",
                      "peak_class":pc, "peak_name":CLASS_NAMES[pc],
                      "probability":round(float(grp[pcol].max()),2)})
    return pd.DataFrame(daily)


# ── PRINT ──────────────────────────────────────────────────────────────
def print_forecast(now, source, hourly_df, daily_df):
    src = "LIVE NOAA DATA" if source == "live" else "FALLBACK (historical CSV)"
    b = "=" * 68
    print(f"\n{b}\n  Solar Flare 72-Hour Forecast")
    print(f"  Source : {src}")
    print(f"  Issued : {now.strftime('%Y-%m-%d %H:%M')} UTC\n{b}")
    print(f"\n  HOURLY  (t+1..t+{FORECAST_HOURS})")
    print(f"  {'Offset':<8}  {'Timestamp':<22}  {'Class':<12}  Prob    P(M)   P(X)")
    print(f"  {'-'*8}  {'-'*22}  {'-'*12}  {'-'*6}  {'-'*5}  {'-'*5}")
    for _, r in hourly_df.iterrows():
        flag = " ⚠X" if r["pred_class"]==3 else " ⚠M" if r["pred_class"]==2 else ""
        print(f"  {r['hour_offset']:<8}  {str(r['timestamp']):<22}  "
              f"{r['pred_name']:<12}  {r['probability']:>5.1f}%  "
              f"{r['p_m']:>5.1f}%  {r['p_x']:>5.1f}%{flag}")
    print(f"\n  DAILY  (d+1..d+3)")
    for _, r in daily_df.iterrows():
        print(f"  {r['day_offset']}  {str(r['date'])}  {r['peak_name']}  {r['probability']:.1f}%")
    print(f"\n{b}\n")


# ── RUN (importable) ──────────────────────────────────────────────────
def run(model=None, feat_cols=None, x0=None, now=None, source=None, verbose=True):
    """
    Run the full step7 pipeline and return DataFrames.

    If model/feat_cols/x0/now are provided, uses them directly.
    Otherwise loads model and fetches live data.

    Returns: (hourly_df, daily_df, now, source)
    """
    if model is None or feat_cols is None or x0 is None or now is None:
        with open(MODEL_FILE, "rb") as f:
            payload = pickle.load(f)
        model     = payload["model"]
        feat_cols = [c for c in payload["feature_cols"] if c not in LEAKY_FEATURES]
        x0, now, source = get_live_snapshot(feat_cols, verbose=verbose)

    hourly_df = rollout(model, feat_cols, x0, now)
    daily_df  = daily_rollup(hourly_df, now)
    return hourly_df, daily_df, now, source


# ── MAIN (CLI only, no file output) ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",    action="store_true", help="Retrain model first")
    ap.add_argument("--fallback", action="store_true", help="Force historical fallback")
    args = ap.parse_args()

    if args.train:
        train_model()

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
    hourly_df, daily_df, now, source = run(model, feat_cols, x0, now, source)
    print_forecast(now, source, hourly_df, daily_df)


if __name__ == "__main__":
    main()
