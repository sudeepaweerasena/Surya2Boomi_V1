"""
HF Blackout Forecasting — Step 11: 7-Day Daily Forecast
========================================================
Converts daily flare class probabilities (from step10) into
HF radio blackout probabilities using the Law of Total Probability.

NO live data fetch here. NO model rollout here.
This is a pure post-processor of step10 output.

The "no-flare" probability per day is derived as:
    p_noflare = 1 - max(max_p_c + max_p_m + max_p_x) / 100

Key functions (used by pipeline.py):
  compute_blackout(df)  — daily flare probs → R1/R2/R3 probs
  run(flare_7day_df)    — full step11 pipeline, returns DataFrame
"""

import pandas as pd, sys, os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# EMPIRICAL CONDITIONAL RATES
# ─────────────────────────────────────────────
BLACKOUT_PROBS = {
    "p_noflare": {"R1": 0.0002, "R2": 0.0000, "R3": 0.0000},
    "p_c"      : {"R1": 0.0088, "R2": 0.0008, "R3": 0.0001},
    "p_m"      : {"R1": 0.1365, "R2": 0.0151, "R3": 0.0025},
    "p_x"      : {"R1": 0.1744, "R2": 0.0256, "R3": 0.0719},
}

R_SCALE    = {0: "None", 1: "R1 Minor", 2: "R2 Moderate", 3: "R3 Strong"}
CONFIDENCE = {1:"High", 2:"High", 3:"High", 4:"Medium", 5:"Medium", 6:"Low", 7:"Low"}


# ─────────────────────────────────────────────
# APPLY BLACKOUT RATES
# max_p_c/m/x are percentages (0–100).
# Uses peak_class probabilities per day as
# the best available daily class distribution.
# ─────────────────────────────────────────────
def compute_blackout(df):
    rows = []
    for _, r in df.iterrows():
        # Daily peak class probabilities as fractions
        pc  = float(r["max_p_c"]) / 100.0
        pm  = float(r["max_p_m"]) / 100.0
        px  = float(r["max_p_x"]) / 100.0
        # Derive no-flare: remaining probability
        pnf = max(0.0, 1.0 - pc - pm - px)

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

        pl   = 3 if p_r3 >= 0.01 else 2 if p_r2 >= 0.005 else 1 if p_r1 >= 0.01 else 0

        # Extract day number for confidence lookup
        day_offset_str = str(r["day_offset"])  # e.g. "d+3"
        try:
            d = int(day_offset_str.replace("d+","").strip())
        except ValueError:
            d = 0

        rows.append({
            "date"          : r["date"],
            "day_offset"    : r["day_offset"],
            "dominant_flare": r["peak_name"],
            "peak_level"    : pl,
            "peak_label"    : R_SCALE[pl],
            "peak_prob"     : round(p_r1 * 100, 4),
            "max_p_R1"      : round(p_r1 * 100, 4),
            "max_p_R2"      : round(p_r2 * 100, 4),
            "max_p_R3"      : round(p_r3 * 100, 4),
            "flare_p_c"     : round(float(r["max_p_c"]), 2),
            "flare_p_m"     : round(float(r["max_p_m"]), 2),
            "flare_p_x"     : round(float(r["max_p_x"]), 2),
            "confidence"    : r.get("confidence", CONFIDENCE.get(d, "Low")),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
def print_forecast(daily_df):
    b = "=" * 74
    print(f"\n{b}")
    print(f"  HF Radio Blackout 7-Day Daily Forecast")
    print(f"{b}")
    print(f"\n  Confidence: d+1..d+3 = High | d+4..d+5 = Medium | d+6..d+7 = Low\n")

    print(f"  {'Offset':<6}  {'Date':<12}  {'Flare':<12}  "
          f"{'R1%':>8}  {'R2%':>8}  {'R3%':>8}  {'Peak':<14}  Conf")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*8}")

    for _, r in daily_df.iterrows():
        flag = (" ⚠R3" if r["peak_level"] == 3
                else " ⚠R2" if r["peak_level"] == 2 else "")
        print(f"  {r['day_offset']:<6}  {str(r['date']):<12}  "
              f"{r['dominant_flare']:<12}  {r['max_p_R1']:>7.4f}%  "
              f"{r['max_p_R2']:>7.4f}%  {r['max_p_R3']:>7.4f}%  "
              f"{r['peak_label']:<14}  {r['confidence']}{flag}")

    print(f"\n  Flare input used:")
    print(f"  {'Offset':<6}  {'Date':<12}  {'Flare':<12}  "
          f"{'P(C)':>7}  {'P(M)':>7}  {'P(X)':>7}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}")
    for _, r in daily_df.iterrows():
        print(f"  {r['day_offset']:<6}  {str(r['date']):<12}  "
              f"{r['dominant_flare']:<12}  {r['flare_p_c']:>6.2f}%  "
              f"{r['flare_p_m']:>6.2f}%  {r['flare_p_x']:>6.2f}%")

    print(f"\n{b}\n")


# ─────────────────────────────────────────────
# RUN (importable)
# ─────────────────────────────────────────────
def run(flare_7day_df):
    """
    Run full step11 pipeline on a 7-day flare DataFrame.
    Returns: daily_df
    """
    return compute_blackout(flare_7day_df)


# ─────────────────────────────────────────────
# MAIN (CLI only, no file output)
# ─────────────────────────────────────────────
def main():
    from forecasting.step10_7day_forecast import run as run_step10
    f7, now, source = run_step10()
    daily_df = run(f7)
    print_forecast(daily_df)


if __name__ == "__main__":
    main()
