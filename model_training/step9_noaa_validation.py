"""
Solar Flare & HF Blackout Forecast — Step 9: NOAA Validation
=============================================================
Fetches NOAA's current 3-day forecast from the live API and
compares it side-by-side with our Step 7 and Step 8 forecasts.

NOAA API endpoints used (no API key required — public):
  3-day forecast text  : https://services.swpc.noaa.gov/text/3-day-forecast.txt
  Flare probabilities  : https://services.swpc.noaa.gov/json/solar_probabilities.json
  Recent flare events  : https://services.swpc.noaa.gov/json/edited_events.json
  Current NOAA scales  : https://services.swpc.noaa.gov/products/noaa-scales.json

What NOAA forecasts (3-day, daily resolution):
  - P(C-class flare)   for each of next 3 days
  - P(M-class flare)   for each of next 3 days
  - P(X-class flare)   for each of next 3 days
  - P(R1-R2 blackout)  for each of next 3 days
  - P(R3+ blackout)    for each of next 3 days

What we forecast (72-hour, hourly + daily resolution):
  - Predicted class + probability per hour (t+1 to t+72)
  - Peak class + probability per day (d+1 to d+3)
  - R1/R2/R3 blackout probability per hour and day

Comparison strategy:
  NOAA gives daily probabilities. We aggregate our hourly forecasts
  to daily level (max probability per day) and compare directly.

Run:
    python step9_noaa_validation.py

Outputs:
    validation_report.txt    — side-by-side comparison table
    validation_summary.csv   — machine-readable comparison
"""

import re, sys, os
import json
import pickle
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NOAA_3DAY_URL   = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"
NOAA_PROBS_URL  = "https://services.swpc.noaa.gov/json/solar_probabilities.json"
NOAA_EVENTS_URL = "https://services.swpc.noaa.gov/json/edited_events.json"
NOAA_SCALES_URL = "https://services.swpc.noaa.gov/products/noaa-scales.json"

MODEL_FILE = config.get_model_path("solar_flare_model_multiclass.pkl")
FEAT_FILE  = config.get_data_path("solar_flare_features.csv")

CLASS_NAMES        = {0: "No-flare", 1: "C-class", 2: "M-class", 3: "X-class"}
CLASS4_TO_ORDINAL  = {0: 0.0,  1: 3.0,  2: 4.0,  3: 5.0}
CLASS4_TO_FLUX     = {0: 0.0,  1: 1e-6, 2: 1e-5, 3: 1e-4}
LEAKY_FEATURES     = ["goes_flux", "goes_ordinal", "log_goes_flux"]
FORECAST_HOURS     = 72

# Empirical blackout probabilities per flare class (from Step 8)
BLACKOUT_PROBS = {
    0: {"R1": 0.0002, "R2": 0.0000, "R3": 0.0000},
    1: {"R1": 0.0088, "R2": 0.0008, "R3": 0.0001},
    2: {"R1": 0.1365, "R2": 0.0151, "R3": 0.0025},
    3: {"R1": 0.1744, "R2": 0.0256, "R3": 0.0719},
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fetch_url(url, as_json=False, timeout=15):
    """Fetch a URL. Returns text or parsed JSON. Returns None on failure."""
    try:
        req = urllib.request.Request(url,
            headers={"User-Agent": "SolarFlareResearch/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data) if as_json else data
    except urllib.error.URLError as e:
        print(f"  [WARN] Could not reach {url}: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Error fetching {url}: {e}")
        return None


def parse_3day_forecast(text):
    """
    Parse NOAA 3-day-forecast.txt into structured dict.
    Returns dict with:
      issued_utc, days[], flare_c[], flare_m[], flare_x[],
      blackout_r1r2[], blackout_r3plus[]
    """
    result = {
        "issued_utc"    : None,
        "days"          : [],
        "flare_c"       : [],
        "flare_m"       : [],
        "flare_x"       : [],
        "blackout_r1r2" : [],
        "blackout_r3"   : [],
        "raw_text"      : text,
    }
    if not text:
        return result

    # Parse issue time
    m = re.search(r":Issued:\s+(.+UTC)", text)
    if m:
        result["issued_utc"] = m.group(1).strip()

    # Parse flare probabilities section
    # Pattern: lines like "Jan 20  Jan 21  Jan 22"
    # then "C      55%     45%     35%"
    lines = text.splitlines()

    # Find the solar activity section
    in_solar = False
    day_header = []
    for i, line in enumerate(lines):
        # Detect day header row (e.g. "            Jan 25       Jan 26       Jan 27")
        day_match = re.findall(r'([A-Z][a-z]{2}\s+\d{1,2})', line)
        if len(day_match) >= 3 and not in_solar:
            day_header = day_match[:3]
            result["days"] = day_header
            in_solar = True
            continue

        if in_solar:
            # C-class line
            if re.match(r'\s*C\s+\d+%', line):
                probs = re.findall(r'(\d+)%', line)
                result["flare_c"] = [int(p) for p in probs[:3]]
            # M-class line
            elif re.match(r'\s*M\s+\d+%', line):
                probs = re.findall(r'(\d+)%', line)
                result["flare_m"] = [int(p) for p in probs[:3]]
            # X-class line
            elif re.match(r'\s*X\s+\d+%', line):
                probs = re.findall(r'(\d+)%', line)
                result["flare_x"] = [int(p) for p in probs[:3]]

    # Parse radio blackout section
    in_blackout = False
    for i, line in enumerate(lines):
        if "Radio Blackout Forecast" in line:
            in_blackout = True
            continue
        if in_blackout:
            if re.match(r'\s*R1-R2', line):
                probs = re.findall(r'(\d+)%', line)
                result["blackout_r1r2"] = [int(p) for p in probs[:3]]
            elif re.match(r'\s*R3', line):
                probs = re.findall(r'(\d+)%', line)
                result["blackout_r3"] = [int(p) for p in probs[:3]]
            elif line.strip().startswith("Rationale"):
                in_blackout = False

    return result


# ─────────────────────────────────────────────
# OUR FORECAST — rerun Step 7 + 8 inline
# ─────────────────────────────────────────────
def run_our_forecast(now):
    """Run Step 7 (flare) + Step 8 (blackout) inline and return daily summary."""
    print("  Loading multiclass model ...")
    with open(MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    model     = payload["model"]
    feat_cols = [c for c in payload["feature_cols"] if c not in LEAKY_FEATURES]

    print("  Loading feature snapshot ...")
    feat = pd.read_csv(FEAT_FILE, parse_dates=["timestamp"])
    row  = feat[feat["timestamp"] == now]
    if len(row) == 0:
        diffs   = (feat["timestamp"] - now).abs()
        nearest = feat.loc[diffs.idxmin(), "timestamp"]
        print(f"  [snapshot] Nearest timestamp: {nearest}")
        row = feat[feat["timestamp"] == nearest]
    x0 = row[feat_cols].values[0].copy()

    # Autoregressive 72h forecast
    idx_ord  = feat_cols.index("goes_ordinal_lag1")
    idx_flux = feat_cols.index("goes_flux_lag1")
    idx_log  = feat_cols.index("log_goes_flux_lag1")
    x_cur    = x0.copy()
    hourly   = []

    for h in range(1, FORECAST_HOURS + 1):
        ts    = now + pd.Timedelta(hours=h)
        proba = model.predict_proba(x_cur.reshape(1, -1))[0]
        pred  = int(np.argmax(proba))

        p_r1 = sum(proba[c] * BLACKOUT_PROBS[c]["R1"] for c in range(4))
        p_r2 = sum(proba[c] * BLACKOUT_PROBS[c]["R2"] for c in range(4))
        p_r3 = sum(proba[c] * BLACKOUT_PROBS[c]["R3"] for c in range(4))

        hourly.append({
            "timestamp"  : ts,
            "date"       : ts.date(),
            "pred_class" : pred,
            "pred_name"  : CLASS_NAMES[pred],
            "p_c"        : round(float(proba[1]) * 100, 1),
            "p_m"        : round(float(proba[2]) * 100, 1),
            "p_x"        : round(float(proba[3]) * 100, 1),
            "p_r1"       : round(p_r1 * 100, 1),
            "p_r2"       : round(p_r2 * 100, 1),
            "p_r3"       : round(p_r3 * 100, 1),
        })

        x_cur[idx_ord]  = CLASS4_TO_ORDINAL[pred]
        x_cur[idx_flux] = CLASS4_TO_FLUX[pred]
        x_cur[idx_log]  = np.log10(max(CLASS4_TO_FLUX[pred], 1e-10))

    df = pd.DataFrame(hourly)
    today = now.date()

    # Daily aggregation — max probability per day
    daily = []
    for date, grp in df.groupby("date"):
        day_offset = (pd.Timestamp(date) - pd.Timestamp(today)).days
        if day_offset <= 0:
            continue
        daily.append({
            "date"       : date,
            "day_offset" : day_offset,
            "peak_class" : CLASS_NAMES[int(grp["pred_class"].max())],
            "our_p_c"    : round(float(grp["p_c"].max()), 1),
            "our_p_m"    : round(float(grp["p_m"].max()), 1),
            "our_p_x"    : round(float(grp["p_x"].max()), 1),
            "our_p_r1"   : round(float(grp["p_r1"].max()), 1),
            "our_p_r2"   : round(float(grp["p_r2"].max()), 1),
            "our_p_r3"   : round(float(grp["p_r3"].max()), 1),
        })

    return daily[:3]   # d+1, d+2, d+3


# ─────────────────────────────────────────────
# COMPARISON SCORE
# ─────────────────────────────────────────────
def agreement_label(our_val, noaa_val, tolerance=15):
    """
    Simple agreement label.
    tolerance: percentage points within which we call it 'close'.
    """
    if noaa_val is None:
        return "N/A"
    diff = abs(our_val - noaa_val)
    if diff <= tolerance:
        return f"✓ close  (Δ{diff:.0f}pp)"
    elif our_val > noaa_val:
        return f"↑ higher (Δ{diff:.0f}pp)"
    else:
        return f"↓ lower  (Δ{diff:.0f}pp)"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    border = "=" * 70

    now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    print(f"\n{border}")
    print(f"  NOAA Validation  —  run at {now} UTC")
    print(f"{border}\n")

    # ── Step 1: Fetch NOAA data ───────────────
    print("Fetching NOAA live forecasts ...")
    raw_3day   = fetch_url(NOAA_3DAY_URL)
    raw_scales = fetch_url(NOAA_SCALES_URL, as_json=True)
    raw_events = fetch_url(NOAA_EVENTS_URL, as_json=True)

    noaa = parse_3day_forecast(raw_3day)

    if not noaa["days"]:
        print("  [WARN] Could not parse NOAA 3-day forecast. Check network.")
    else:
        print(f"  NOAA forecast issued: {noaa['issued_utc']}")
        print(f"  NOAA forecast days  : {' | '.join(noaa['days'])}")

    # ── Step 2: Run our forecast ──────────────
    print("\nRunning our forecast ...")
    our_daily = run_our_forecast(now)

    # ── Step 3: Current NOAA R-scale ─────────
    current_r = "N/A"
    if raw_scales:
        try:
            r_val = raw_scales.get("R", {}).get("Scale", "0")
            current_r = f"R{r_val}" if r_val and r_val != "0" else "None active"
        except Exception:
            pass

    # ── Step 4: Recent observed flare events ──
    recent_flares = []
    if raw_events:
        try:
            events = raw_events if isinstance(raw_events, list) else []
            for ev in events[-20:]:   # last 20 events
                if isinstance(ev, list) and len(ev) >= 8:
                    # Format: [time, observatory, frequency, class, ...]
                    cls = str(ev[3]) if len(ev) > 3 else ""
                    if re.match(r'^[BCMX]\d', cls):
                        recent_flares.append({
                            "time" : ev[0],
                            "class": cls,
                        })
        except Exception:
            pass

    # ── Step 5: Print comparison report ───────
    print(f"\n{border}")
    print(f"  CURRENT CONDITIONS")
    print(f"{border}")
    print(f"  Current NOAA R-scale : {current_r}")
    if recent_flares:
        print(f"  Recent observed flares (last available):")
        for f in recent_flares[-5:]:
            print(f"    {f['time']}  {f['class']}")
    else:
        print("  Recent flares        : none reported / unavailable")

    print(f"\n{border}")
    print(f"  SIDE-BY-SIDE COMPARISON  (daily, d+1 to d+3)")
    print(f"{border}")
    print(f"\n  {'Metric':<28}  {'NOAA':>8}  {'Ours':>8}  Agreement")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*24}")

    report_rows = []
    for i, our in enumerate(our_daily):
        d = i + 1
        date_str    = str(our["date"])
        noaa_day    = noaa["days"][i] if i < len(noaa["days"]) else "N/A"
        noaa_c      = noaa["flare_c"][i]      if i < len(noaa["flare_c"])      else None
        noaa_m      = noaa["flare_m"][i]      if i < len(noaa["flare_m"])      else None
        noaa_x      = noaa["flare_x"][i]      if i < len(noaa["flare_x"])      else None
        noaa_r1r2   = noaa["blackout_r1r2"][i] if i < len(noaa["blackout_r1r2"]) else None
        noaa_r3     = noaa["blackout_r3"][i]   if i < len(noaa["blackout_r3"])   else None

        print(f"\n  ── d+{d}  {date_str}  (NOAA: {noaa_day}) ──")

        rows = [
            ("P(C-class flare)",  noaa_c,    our["our_p_c"],  15),
            ("P(M-class flare)",  noaa_m,    our["our_p_m"],  15),
            ("P(X-class flare)",  noaa_x,    our["our_p_x"],  10),
            ("P(R1-R2 blackout)", noaa_r1r2, our["our_p_r1"], 20),
            ("P(R3+ blackout)",   noaa_r3,   our["our_p_r3"], 10),
        ]

        for metric, noaa_val, our_val, tol in rows:
            noaa_str = f"{noaa_val}%" if noaa_val is not None else "N/A"
            agr      = agreement_label(our_val, noaa_val, tol)
            print(f"  {metric:<28}  {noaa_str:>8}  {our_val:>7.1f}%  {agr}")
            report_rows.append({
                "day"       : f"d+{d}",
                "date"      : date_str,
                "metric"    : metric,
                "noaa_pct"  : noaa_val,
                "our_pct"   : our_val,
                "diff_pp"   : round(our_val - noaa_val, 1) if noaa_val is not None else None,
                "agreement" : agr,
            })

        print(f"  {'Our peak class':<28}  {'---':>8}  {our['peak_class']:>8}")

    # ── Step 6: Overall score ─────────────────
    print(f"\n{border}")
    print(f"  OVERALL AGREEMENT SUMMARY")
    print(f"{border}")

    valid_rows = [r for r in report_rows if r["noaa_pct"] is not None and r["diff_pp"] is not None]
    if valid_rows:
        diffs     = [abs(r["diff_pp"]) for r in valid_rows]
        mean_diff = sum(diffs) / len(diffs)
        close     = sum(1 for r in valid_rows if abs(r["diff_pp"]) <= 15)
        total     = len(valid_rows)
        print(f"\n  Metrics compared      : {total}")
        print(f"  Within 15pp of NOAA   : {close} / {total}  ({close/total*100:.0f}%)")
        print(f"  Mean absolute diff    : {mean_diff:.1f} percentage points")

        if mean_diff <= 10:
            verdict = "Excellent — our model closely tracks NOAA forecasts"
        elif mean_diff <= 20:
            verdict = "Good — our model is broadly consistent with NOAA"
        elif mean_diff <= 30:
            verdict = "Fair — some divergence, expected for different methods"
        else:
            verdict = "Divergent — review model or data currency"
        print(f"  Verdict               : {verdict}")
    else:
        print("  Could not compute agreement — NOAA data unavailable.")

    print(f"\n  Note: NOAA forecasts daily probabilities (3-day resolution).")
    print(f"  Our model forecasts hourly probabilities (72h resolution).")
    print(f"  Comparison uses our max daily probability vs NOAA daily probability.")
    print(f"  A 'close' result (within 15pp) is a strong validation signal")
    print(f"  given the different methodologies and time resolutions.\n")

    # ── Step 7: Save outputs ──────────────────
    out_df = pd.DataFrame(report_rows)
    out_df.to_csv(config.get_report_path("validation_summary.csv"), index=False)

    report_lines = [
        "NOAA Validation Report",
        f"Run at: {now} UTC",
        f"NOAA forecast issued: {noaa.get('issued_utc', 'N/A')}",
        f"Current NOAA R-scale: {current_r}",
        "",
        f"{'Day':<6}  {'Date':<12}  {'Metric':<28}  {'NOAA%':>6}  {'Ours%':>6}  {'Diff':>6}  Agreement",
    ]
    for r in report_rows:
        noaa_str = f"{r['noaa_pct']}%" if r["noaa_pct"] is not None else "N/A"
        diff_str = f"{r['diff_pp']:+.1f}" if r["diff_pp"] is not None else "N/A"
        report_lines.append(
            f"{r['day']:<6}  {r['date']:<12}  {r['metric']:<28}  "
            f"{noaa_str:>6}  {r['our_pct']:>5.1f}%  {diff_str:>6}  {r['agreement']}"
        )
    if valid_rows:
        report_lines.extend([
            "",
            f"Mean absolute difference: {mean_diff:.1f}pp",
            f"Within 15pp: {close}/{total}",
            f"Verdict: {verdict}",
        ])

    with open(config.get_report_path("validation_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("Saved:")
    print("  validation_report.txt")
    print("  validation_summary.csv")
    print(f"\n{border}\n")


if __name__ == "__main__":
    main()
