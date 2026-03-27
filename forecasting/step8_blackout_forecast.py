"""
HF Blackout Forecasting — Step 8: 72-Hour Hourly Forecast
==========================================================
Converts hourly flare class probability distributions (from step7)
into HF radio blackout probabilities using the Law of Total
Probability with empirically derived conditional rates.

NO live data fetch here. NO model rollout here.
This is a pure post-processor of step7 output.

Empirical conditional rates (NOAA 2017-2025 frequency counts):
    Class       R1       R2       R3
    No-flare  0.02%   0.00%   0.00%
    C-class   0.88%   0.08%   0.01%
    M-class  13.65%   1.51%   0.25%
    X-class  17.44%   2.56%   7.19%

Formula per hour:
    P(Rx) = P(no-flare)×rate_nf + P(C)×rate_C + P(M)×rate_M + P(X)×rate_X

Key functions (used by pipeline.py):
  compute_blackout(df)  — flare probs → R1/R2/R3 probs
  daily_rollup(df)      — aggregate to daily
  run(flare_hourly_df)  — full step8 pipeline, returns DataFrames
"""


import pandas as pd, sys, os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# EMPIRICAL CONDITIONAL RATES
# Source: NOAA 2017-2025 blackout events joined
# to hourly GOES flare class data. Frequency
# counts — no model, no fitting.
# ─────────────────────────────────────────────
BLACKOUT_PROBS = {
    "p_noflare": {"R1": 0.0002, "R2": 0.0000, "R3": 0.0000},
    "p_c"      : {"R1": 0.0088, "R2": 0.0008, "R3": 0.0001},
    "p_m"      : {"R1": 0.1365, "R2": 0.0151, "R3": 0.0025},
    "p_x"      : {"R1": 0.1744, "R2": 0.0256, "R3": 0.0719},
}

R_SCALE = {0: "No blackout", 1: "R1 Minor", 2: "R2 Moderate", 3: "R3 Strong"}





# ─────────────────────────────────────────────
# APPLY BLACKOUT RATES
# Probabilities are percentages (0-100),
# so divide by 100 first.
# ─────────────────────────────────────────────
def compute_blackout(df):
    rows = []
    for _, r in df.iterrows():
        pnf = float(r["p_noflare"]) / 100.0
        pc  = float(r["p_c"])       / 100.0
        pm  = float(r["p_m"])       / 100.0
        px  = float(r["p_x"])       / 100.0

        p_r1 = (pnf * BLACKOUT_PROBS["p_noflare"]["R1"] +
                pc  * BLACKOUT_PROBS["p_c"]["R1"] +
                pm  * BLACKOUT_PROBS["p_m"]["R1"] +
                px  * BLACKOUT_PROBS["p_x"]["R1"])

        p_r2 = (pnf * BLACKOUT_PROBS["p_noflare"]["R2"] +
                pc  * BLACKOUT_PROBS["p_c"]["R2"] +
                pm  * BLACKOUT_PROBS["p_m"]["R2"] +
                px  * BLACKOUT_PROBS["p_x"]["R2"])

        p_r3 = (pnf * BLACKOUT_PROBS["p_noflare"]["R3"] +
                pc  * BLACKOUT_PROBS["p_c"]["R3"] +
                pm  * BLACKOUT_PROBS["p_m"]["R3"] +
                px  * BLACKOUT_PROBS["p_x"]["R3"])

        # Determine the most likely alert level
        if   p_r3 >= 0.01:  ll, llab = 3, "R3 Strong"
        elif p_r2 >= 0.005: ll, llab = 2, "R2 Moderate"
        elif p_r1 >= 0.01:  ll, llab = 1, "R1 Minor"
        else:               ll, llab = 0, "No blackout"

        rows.append({
            "timestamp"   : r["timestamp"],
            "hour_offset" : r["hour_offset"],
            "flare_class" : r["pred_name"],
            "p_noflare_pct": round(float(r["p_noflare"]), 2),
            "p_c_pct"     : round(float(r["p_c"]), 2),
            "p_m_pct"     : round(float(r["p_m"]), 2),
            "p_x_pct"     : round(float(r["p_x"]), 2),
            "p_R1"        : round(p_r1 * 100, 4),
            "p_R2"        : round(p_r2 * 100, 4),
            "p_R3"        : round(p_r3 * 100, 4),
            "likely_level": ll,
            "likely_label": llab,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# DAILY ROLLUP
# ─────────────────────────────────────────────
def daily_rollup(hourly_df):
    first_ts  = hourly_df["timestamp"].min()
    today     = first_ts.date()
    daily     = []

    for date, grp in hourly_df.groupby(hourly_df["timestamp"].dt.date):
        d = (pd.Timestamp(date) - pd.Timestamp(today)).days
        if d <= 0 or d > 3:
            continue

        mr1 = round(float(grp["p_R1"].max()), 4)
        mr2 = round(float(grp["p_R2"].max()), 4)
        mr3 = round(float(grp["p_R3"].max()), 4)
        pl  = 3 if mr3 >= 1.0 else 2 if mr2 >= 0.5 else 1 if mr1 >= 0.5 else 0
        dom = grp["flare_class"].mode()[0]

        daily.append({
            "date"          : date,
            "day_offset"    : f"d+{d}",
            "dominant_flare": dom,
            "peak_level"    : pl,
            "peak_label"    : R_SCALE[pl],
            "peak_prob"     : mr1,
            "max_p_R1"      : mr1,
            "max_p_R2"      : mr2,
            "max_p_R3"      : mr3,
        })

    return pd.DataFrame(daily)


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
def print_forecast(hourly_df, daily_df):
    issued = hourly_df["timestamp"].min() - pd.Timedelta(hours=1)
    b = "=" * 72
    print(f"\n{b}")
    print(f"  HF Radio Blackout 72-Hour Forecast")
    print(f"  Issued  : {issued.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{b}")

    print(f"\n  HOURLY  (t+1..t+72)")
    print(f"  {'Offset':<8}  {'Timestamp':<22}  {'Flare':<12}  "
          f"{'R1%':>7}  {'R2%':>7}  {'R3%':>7}  Status")
    print(f"  {'-'*8}  {'-'*22}  {'-'*12}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*14}")

    for _, r in hourly_df.iterrows():
        flag = (" ⚠R3" if r["likely_level"] == 3
                else " ⚠R2" if r["likely_level"] == 2 else "")
        print(f"  {r['hour_offset']:<8}  {str(r['timestamp']):<22}  "
              f"{r['flare_class']:<12}  {r['p_R1']:>6.4f}%  "
              f"{r['p_R2']:>6.4f}%  {r['p_R3']:>6.4f}%  "
              f"{r['likely_label']}{flag}")

    print(f"\n  DAILY  (d+1..d+3)")
    print(f"  {'Offset':<6}  {'Date':<12}  {'Flare':<12}  "
          f"{'R1%':>7}  {'R2%':>7}  {'R3%':>7}  Peak")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*12}")

    for _, r in daily_df.iterrows():
        print(f"  {r['day_offset']:<6}  {str(r['date']):<12}  "
              f"{r['dominant_flare']:<12}  {r['max_p_R1']:>6.4f}%  "
              f"{r['max_p_R2']:>6.4f}%  {r['max_p_R3']:>6.4f}%  "
              f"{r['peak_label']}")

    print(f"\n{b}\n")


# ─────────────────────────────────────────────
# RUN (importable)
# ─────────────────────────────────────────────
def run(flare_hourly_df):
    """
    Run full step8 pipeline on a flare hourly DataFrame.
    Returns: (hourly_df, daily_df)
    """
    hourly_df = compute_blackout(flare_hourly_df)
    daily_df  = daily_rollup(hourly_df)
    return hourly_df, daily_df


# ─────────────────────────────────────────────
# MAIN (CLI only, no file output)
# ─────────────────────────────────────────────
def main():
    from forecasting.step7_72h_forecast import run as run_step7
    fh, fd, now, source = run_step7()
    hourly_df, daily_df = run(fh)
    print_forecast(hourly_df, daily_df)


if __name__ == "__main__":
    main()
