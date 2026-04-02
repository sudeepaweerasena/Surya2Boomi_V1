import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime

from pipeline import run_full_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CACHE CONFIGURATION
# FIX: session_state alone never expires on Streamlit Cloud — the server
#      process stays alive indefinitely. We store a timestamp alongside the
#      data and force a fresh pipeline run when TTL_SECONDS have elapsed.
# ─────────────────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600  # refresh live NOAA data every 1 hour

# Set page config
st.set_page_config(
    page_title="SURYA2BOOMI — HF Blackout Forecast",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS to hide Streamlit header, footer and menu, and remove padding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    /* Make iframe fill the space */
    iframe {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """Load and format data from in-memory pipeline (cached in session_state with TTL)."""
    try:
        now_ts = time.time()

        # FIX: expire the cache when TTL has passed, or if it was never set
        cache_expired = (
            'pipeline_data' not in st.session_state
            or 'pipeline_fetched_at' not in st.session_state
            or (now_ts - st.session_state.pipeline_fetched_at) > CACHE_TTL_SECONDS
        )

        if cache_expired:
            # Show loading screen while pipeline runs
            loading = st.empty()
            loading.markdown("""
            <div style="position:fixed;inset:0;z-index:99999;background:linear-gradient(135deg,#0a0e27 0%,#1a1f3a 50%,#0a0e27 100%);
            display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1.2rem;">
                <div style="font-size:3rem;">📡</div>
                <div style="font-family:'Orbitron',sans-serif;font-size:1.3rem;font-weight:700;color:#00d9ff;letter-spacing:3px;">
                SURYA2BOOMI</div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:0.85rem;color:#8b9dc3;letter-spacing:3px;text-transform:uppercase;">
                Initializing Forecast Pipeline</div>
                <div style="margin-top:1rem;">
                    <div style="width:220px;height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;">
                        <div style="width:100%;height:100%;background:linear-gradient(90deg,#00d9ff,#667eea,#00d9ff);
                        background-size:200% 100%;border-radius:2px;animation:loading 1.5s ease-in-out infinite;">
                        </div>
                    </div>
                </div>
                <div style="margin-top:1.2rem;text-align:left;">
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#4ade80;line-height:2;">
                    ▸ Fetching live NOAA satellite data ...<br>
                    ▸ Running 72-hour flare forecast ...<br>
                    ▸ Computing HF blackout probabilities ...<br>
                    ▸ Generating 7-day outlook ...<br>
                    </div>
                </div>
                <style>@keyframes loading{0%{background-position:200% 0}100%{background-position:-200% 0}}</style>
            </div>
            """, unsafe_allow_html=True)

            # Run fresh pipeline and stamp the fetch time
            st.session_state.pipeline_data = run_full_pipeline(verbose=True)
            st.session_state.pipeline_fetched_at = now_ts  # FIX: record when data was fetched
            loading.empty()

        data = st.session_state.pipeline_data
        fh_df = data["fh"]
        fd_df = data["fd"]
        bh_df = data["bh"]
        bd_df = data["bd"]
        f7_df = data["f7"]
        b7_df = data["b7"]

        # Parse hour integers from hour_offset (e.g. "t+1h" -> 1)
        fh_df = fh_df.copy()
        bh_df = bh_df.copy()
        fh_df['hour_int'] = fh_df['hour_offset'].str.replace('t+', '', regex=False).str.replace('h', '', regex=False).astype(int)
        bh_df['hour_int'] = bh_df['hour_offset'].str.replace('t+', '', regex=False).str.replace('h', '', regex=False).astype(int)

        # Class color mapping
        class_colors = {"No-flare": "#4ade80", "C-class": "#ffd700", "M-class": "#ff8c00", "X-class": "#ff4500"}

        # Class distribution counts
        class_counts = fh_df['pred_name'].value_counts()
        hours_noflare = int(class_counts.get('No-flare', 0))
        hours_c = int(class_counts.get('C-class', 0))
        hours_m = int(class_counts.get('M-class', 0))
        hours_x = int(class_counts.get('X-class', 0))

        # Prepare D object for JS
        D = {
            "fh_hours": fh_df['hour_int'].tolist(),
            "fh_pm": fh_df['p_m'].tolist(),
            "fh_px": fh_df['p_x'].tolist(),
            "fh_pc": fh_df['p_c'].tolist(),
            "fh_pred_name": fh_df['pred_name'].tolist(),
            "fh_prob": fh_df['probability'].tolist(),
            "fh_noflare": fh_df['p_noflare'].tolist(),
            "bh_r1": bh_df['p_R1'].tolist(),
            "bh_r2": bh_df['p_R2'].tolist(),
            "bh_r3": bh_df['p_R3'].tolist(),
            "bh_likely_label": bh_df['likely_label'].tolist(),
            "fh_ts": fh_df['timestamp'].apply(lambda x: str(x).split()[0][5:] + " " + str(x).split()[1][:5]).tolist(),
            "fd": fd_df.assign(date=fd_df['date'].astype(str)).to_dict(orient='records') if len(fd_df) > 0 else [],
            "bd": bd_df.assign(date=bd_df['date'].astype(str)).to_dict(orient='records') if len(bd_df) > 0 else [],
            "f7": f7_df.assign(date=f7_df['date'].astype(str)).to_dict(orient='records'),
            "b7": b7_df.assign(date=b7_df['date'].astype(str)).to_dict(orient='records'),
            "class_dist": [hours_noflare, hours_c, hours_m, hours_x],
            "class_colors": class_colors,
        }

        # KPIs
        kpis = {
            "current_class": fh_df.iloc[0]['pred_name'],
            "current_prob": f"{fh_df.iloc[0]['probability']:.2f}%",
            "current_color": class_colors.get(fh_df.iloc[0]['pred_name'], '#4ade80'),
            "p_c_t1": f"{fh_df.iloc[0]['p_c']:.2f}%",
            "p_m_t1": f"{fh_df.iloc[0]['p_m']:.2f}%",
            "p_x_t1": f"{fh_df.iloc[0]['p_x']:.2f}%",
            "r1_t1": f"{bh_df.iloc[0]['p_R1']:.3f}%",
            "r2_t1": f"{bh_df.iloc[0]['p_R2']:.3f}%",
            "r3_t1": f"{bh_df.iloc[0]['p_R3']:.3f}%",
            "likely_label_t1": bh_df.iloc[0]['likely_label'],
            "peak_m_72h": f"{fh_df['p_m'].max():.2f}%",
            "peak_x_72h": f"{fh_df['p_x'].max():.2f}%",
            "peak_r1_72h": f"{bh_df['p_R1'].max():.3f}%",
            "peak_r1_72h_full": f"{bh_df['p_R1'].max():.4f}%",
            "peak_r2_72h": f"{bh_df['p_R2'].max():.4f}%",
            "gauge_val": round(min(100.0, (fh_df.iloc[0]['p_c'] * 0.1 + fh_df.iloc[0]['p_m'] * 1.5 + fh_df.iloc[0]['p_x'] * 8.0)), 1),
            "alert_code": "QUIET" if bh_df.iloc[0]['p_R1'] < 5 else "ELEVATED",
            "alert_desc": "NO SIGNIFICANT ACTIVITY" if bh_df.iloc[0]['p_R1'] < 5 else "INCREASED BLACKOUT RISK",
            "alert_color": "#4ade80" if bh_df.iloc[0]['p_R1'] < 5 else "#ffd700",
            "hours_noflare": hours_noflare,
            "hours_c": hours_c,
            "hours_m": hours_m,
            "hours_x": hours_x,
        }

        return D, kpis
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Embedded HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SURYA2BOOMI — HF Blackout Forecast</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:linear-gradient(135deg,#0a0e27 0%,#1a1f3a 50%,#0a0e27 100%);min-height:100vh;font-family:'Rajdhani',sans-serif;color:rgba(255,255,255,0.85);}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:rgba(255,255,255,0.04)}::-webkit-scrollbar-thumb{background:rgba(0,217,255,0.35);border-radius:3px}

/* Layout */
.layout{display:flex;min-height:100vh;}
.sidebar{width:240px;min-width:240px;background:rgba(10,14,39,0.97);border-right:1px solid rgba(0,217,255,0.18);padding:1.2rem;display:flex;flex-direction:column;gap:0.5rem;}
.main{flex:1;padding:1.2rem 1.4rem;overflow-y:auto;}

/* Sidebar */
.sb-logo{font-family:'Orbitron',sans-serif;font-size:0.95rem;font-weight:700;color:#00d9ff;letter-spacing:2px;margin-bottom:0.15rem;}
.sb-sub{font-family:'Rajdhani',sans-serif;font-size:0.68rem;color:#8b9dc3;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;}
.sb-sec{font-family:'Orbitron',sans-serif;font-size:0.7rem;font-weight:700;color:#00d9ff;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid rgba(0,217,255,0.2);padding-bottom:0.3rem;margin:0.8rem 0 0.5rem;text-shadow:0 0 8px rgba(0,217,255,0.3);}
.sb-stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;background:rgba(255,255,255,0.03);border:1px solid rgba(0,217,255,0.15);border-radius:8px;padding:0.7rem;}
.sb-stat-lbl{font-family:'Orbitron',sans-serif;font-size:0.55rem;color:#8b9dc3;letter-spacing:1px;text-transform:uppercase;}
.sb-stat-val{font-family:'Share Tech Mono',monospace;font-size:1rem;}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#4ade80;box-shadow:0 0 5px #4ade80;margin-right:6px;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.r-row{display:flex;align-items:center;gap:0.5rem;margin:0.28rem 0;}
.r-box{width:24px;height:24px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-family:'Orbitron',sans-serif;font-size:0.58rem;font-weight:700;flex-shrink:0;}
.r-lbl{font-size:0.75rem;color:#8b9dc3;}
.sb-btn{background:linear-gradient(135deg,#00d9ff 0%,#667eea 100%);color:#0a0e27;border:none;border-radius:7px;font-family:'Orbitron',sans-serif;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:0.55rem;cursor:pointer;width:100%;transition:all 0.25s;margin-top:0.3rem;}
.sb-btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,217,255,0.45);}
.sb-btn-sm{font-size:0.62rem;padding:0.42rem;}
.pipe-row{display:flex;justify-content:space-between;align-items:center;padding:0.22rem 0;border-bottom:1px solid rgba(0,217,255,0.06);}
.pipe-lbl{font-family:'Orbitron',sans-serif;font-size:0.58rem;color:#8b9dc3;letter-spacing:1px;}
.pipe-ok{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#4ade80;}
.sb-footer{margin-top:auto;padding-top:1.2rem;border-top:1px solid rgba(0,217,255,0.1);font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#4a5568;text-align:center;line-height:1.6;}

/* Header */
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.9rem;}
.hdr-left{display:flex;align-items:center;gap:1rem;}
.hdr-icon{font-size:2.2rem;}
.hdr-title{font-family:'Orbitron',sans-serif;font-size:1.45rem;font-weight:700;color:#e2e8f0;letter-spacing:2.5px;}
.hdr-sub{font-family:'Rajdhani',sans-serif;font-size:0.82rem;color:#8b9dc3;letter-spacing:3.5px;text-transform:uppercase;}
.hdr-right{text-align:right;}
.hdr-time{font-family:'Share Tech Mono',monospace;font-size:1.35rem;color:#00d9ff;text-shadow:0 0 12px rgba(0,217,255,0.5);letter-spacing:2px;}
.hdr-date{font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8b9dc3;margin-top:0.15rem;}
.live-badge{display:inline-flex;align-items:center;gap:0.45rem;border:1px solid #4ade80;border-radius:4px;padding:0.15rem 0.5rem;background:rgba(74,222,128,0.08);margin-top:0.3rem;}
.live-dot{width:5px;height:5px;border-radius:50%;background:#4ade80;box-shadow:0 0 4px #4ade80;}
.live-txt{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#4ade80;letter-spacing:2px;}

/* Divider */
.divider{height:1px;background:rgba(0,217,255,0.13);margin:0.6rem 0 1.1rem;}

/* KPI strip */
.kpi-strip{display:grid;grid-template-columns:repeat(6,1fr);gap:0.55rem;margin-bottom:1rem;}
.kpi{background:rgba(255,255,255,0.04);border:1px solid rgba(0,217,255,0.15);border-radius:10px;padding:0.85rem;text-align:center;cursor:default;position:relative;}
.kpi:hover{border-color:rgba(0,217,255,0.35);background:rgba(0,217,255,0.04);}
.kpi-lbl{font-family:'Orbitron',sans-serif;font-size:0.56rem;color:#8b9dc3;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:0.32rem;}
.kpi-val{font-family:'Share Tech Mono',monospace;font-size:1.3rem;}
.kpi-sub{font-family:'Rajdhani',sans-serif;font-size:0.68rem;color:#8b9dc3;margin-top:0.18rem;}
.tooltip{display:none;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:rgba(10,14,39,0.97);border:1px solid rgba(0,217,255,0.3);border-radius:6px;padding:0.5rem 0.7rem;font-size:0.7rem;color:rgba(255,255,255,0.85);white-space:nowrap;z-index:100;pointer-events:none;max-width:220px;white-space:normal;text-align:center;}
.kpi:hover .tooltip{display:block;}

/* Alert banner */
.alert-banner{border:2px solid;border-radius:12px;padding:1.1rem 1.8rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;}
.alert-code{font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:700;letter-spacing:3px;}
.alert-lbl{font-family:'Orbitron',sans-serif;font-size:0.58rem;color:#8b9dc3;letter-spacing:2px;margin-bottom:0.25rem;}
.alert-desc{font-family:'Rajdhani',sans-serif;font-size:1rem;color:#e2e8f0;font-weight:500;}
.alert-mono{font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8b9dc3;margin-top:0.2rem;}

/* Tabs */
.tabs{display:flex;gap:0;border-bottom:1px solid rgba(0,217,255,0.15);margin-bottom:1.1rem;}
.tab{font-family:'Orbitron',sans-serif;font-size:0.68rem;letter-spacing:1px;text-transform:uppercase;padding:0.55rem 1rem;cursor:pointer;color:#8b9dc3;border-bottom:2px solid transparent;transition:all 0.2s;background:none;border-top:none;border-left:none;border-right:none;}
.tab:hover{color:#00d9ff;}
.tab.active{color:#00d9ff;border-bottom:2px solid #00d9ff;}

/* Tab content */
.tab-content{display:none;}
.tab-content.active{display:block;}

/* Section header */
.sec-hdr{font-family:'Orbitron',sans-serif;font-size:0.75rem;font-weight:700;color:#00d9ff;letter-spacing:2px;text-transform:uppercase;border-bottom:2px solid rgba(0,217,255,0.22);padding-bottom:0.38rem;margin:1.1rem 0 0.8rem;text-shadow:0 0 8px rgba(0,217,255,0.3);}
.sec-hdr span{font-family:'Rajdhani',sans-serif;font-size:0.68rem;color:#8b9dc3;margin-left:0.6rem;letter-spacing:0;text-transform:none;}

/* Two-column layout */
.two-col{display:grid;grid-template-columns:2.1fr 1fr;gap:1.2rem;}

/* Glass card */
.glass{background:rgba(255,255,255,0.04);border:1px solid rgba(0,217,255,0.16);border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.7rem;}

/* Chart container */
.chart-wrap{position:relative;width:100%;margin-bottom:1rem;}

/* Day card */
.day-card{background:rgba(255,255,255,0.04);border:1px solid rgba(0,217,255,0.14);border-radius:9px;padding:0.85rem 1rem;margin-bottom:0.6rem;}
.day-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;}
.day-offset{font-family:'Orbitron',sans-serif;font-size:0.72rem;color:#00d9ff;}
.day-date{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#8b9dc3;}
.day-class{font-family:'Rajdhani',sans-serif;font-size:0.9rem;font-weight:600;}
.day-prob{font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#00d9ff;}
.r-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.3rem;text-align:center;margin-top:0.45rem;padding-top:0.4rem;border-top:1px solid rgba(0,217,255,0.08);}
.r-cell-lbl{font-size:0.58rem;color:#8b9dc3;}
.r-cell-val{font-family:'Share Tech Mono',monospace;font-size:0.78rem;}
.bar-wrap{background:rgba(255,255,255,0.07);height:5px;border-radius:2px;overflow:hidden;margin-top:0.25rem;}
.bar-fill{height:100%;border-radius:2px;}

/* Hour row */
.hr-row{background:rgba(255,255,255,0.025);border:1px solid rgba(0,217,255,0.07);border-radius:6px;padding:0.45rem 0.7rem;margin:0.22rem 0;display:flex;align-items:center;gap:0.65rem;}
.hr-off{min-width:32px;font-family:'Orbitron',sans-serif;font-size:0.62rem;color:#00d9ff;}
.hr-bar-wrap{flex:1;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;}
.hr-bar{height:100%;border-radius:2px;}
.hr-class{min-width:68px;font-family:'Rajdhani',sans-serif;font-size:0.78rem;font-weight:500;text-align:right;}
.hr-r1{min-width:58px;font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#ffd700;text-align:right;}

/* Rate table */
.rate-table{width:100%;border-collapse:collapse;font-size:0.82rem;}
.rate-table th{font-family:'Orbitron',sans-serif;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;color:#8b9dc3;padding:0.5rem 0.8rem;border-bottom:1px solid rgba(0,217,255,0.18);}
.rate-table td{padding:0.45rem 0.8rem;border-bottom:1px solid rgba(0,217,255,0.07);font-family:'Share Tech Mono',monospace;font-size:0.78rem;}

/* Badge */
.badge{display:inline-block;padding:0.18rem 0.6rem;border-radius:20px;font-family:'Orbitron',sans-serif;font-size:0.62rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;}

/* Formula box */
.formula{background:rgba(0,217,255,0.04);border:1px solid rgba(0,217,255,0.2);border-radius:8px;padding:0.9rem 1.1rem;font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#e2e8f0;line-height:2;}

/* Data table */
.data-tbl{width:100%;border-collapse:collapse;font-size:0.75rem;}
.data-tbl th{background:rgba(10,14,39,0.95);font-family:'Orbitron',sans-serif;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#8b9dc3;padding:0.45rem 0.6rem;border-bottom:1px solid rgba(0,217,255,0.18);text-align:left;}
.data-tbl td{padding:0.35rem 0.6rem;border-bottom:1px solid rgba(0,217,255,0.07);font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:rgba(255,255,255,0.8);}
.data-tbl tr:hover td{background:rgba(0,217,255,0.04);}
.tbl-wrap{max-height:300px;overflow-y:auto;border:1px solid rgba(0,217,255,0.15);border-radius:8px;}

/* Modal */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:1000;align-items:center;justify-content:center;}
.modal-bg.open{display:flex;}
.modal{background:linear-gradient(135deg,#0a0e27,#1a1f3a);border:1px solid rgba(0,217,255,0.3);border-radius:14px;padding:2rem;max-width:720px;width:95%;max-height:85vh;overflow-y:auto;position:relative;box-shadow:0 0 50px rgba(0,217,255,0.15);}
.modal h2{font-family:'Orbitron',sans-serif;color:#00d9ff;font-size:1.1rem;letter-spacing:2px;margin-bottom:1rem;}
.modal h3{font-family:'Orbitron',sans-serif;color:#00d9ff;font-size:0.9rem;letter-spacing:1.5px;margin:1.2rem 0 0.5rem;}
.modal p,.modal li{color:rgba(255,255,255,0.82);font-size:0.88rem;line-height:1.7;margin-bottom:0.4rem;}
.modal li{margin-left:1.2rem;}
.modal-close{position:absolute;top:1rem;right:1rem;background:rgba(0,217,255,0.15);border:1px solid rgba(0,217,255,0.3);border-radius:6px;color:#00d9ff;font-size:1.1rem;cursor:pointer;padding:0.25rem 0.6rem;font-family:'Share Tech Mono',monospace;}
.modal-close:hover{background:rgba(0,217,255,0.3);}
.m-tab-bar{display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap;}
.m-tab{font-family:'Orbitron',sans-serif;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;padding:0.35rem 0.8rem;border:1px solid rgba(0,217,255,0.25);border-radius:6px;cursor:pointer;color:#8b9dc3;background:none;transition:all 0.2s;}
.m-tab.active,.m-tab:hover{color:#00d9ff;border-color:#00d9ff;background:rgba(0,217,255,0.08);}
.m-section{display:none;}
.m-section.active{display:block;}
.scale-row{background:rgba(255,255,255,0.03);border-left:4px solid;border-radius:0 8px 8px 0;padding:0.7rem 1rem;margin:0.45rem 0;}
.scale-row-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;}
.scale-name{font-family:'Orbitron',sans-serif;font-size:0.82rem;font-weight:700;}
.scale-meta{font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#8b9dc3;}
.scale-desc{font-size:0.82rem;color:rgba(255,255,255,0.8);}
.mtbl{width:100%;border-collapse:collapse;font-size:0.8rem;margin:0.5rem 0;}
.mtbl th{color:#00d9ff;font-family:'Orbitron',sans-serif;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;padding:0.4rem 0.6rem;border-bottom:1px solid rgba(0,217,255,0.2);}
.mtbl td{padding:0.35rem 0.6rem;border-bottom:1px solid rgba(0,217,255,0.08);color:rgba(255,255,255,0.82);}

/* Analytics */
.analytics-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:0.28rem 0;border-bottom:1px solid rgba(0,217,255,0.07);}
.stat-key{font-size:0.82rem;color:#8b9dc3;}
.stat-val{font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#00d9ff;}
</style>
</head>
<body>

<!-- MODALS -->
<div class="modal-bg" id="termsModal">
<div class="modal">
  <button class="modal-close" onclick="closeModal('termsModal')">✕</button>
  <h2>📡 Terms, Disclaimers & System Information</h2>
  <div class="m-tab-bar">
    <button class="m-tab active" onclick="mTab(this,'mAbout')">About</button>
    <button class="m-tab" onclick="mTab(this,'mData')">Data & Accuracy</button>
    <button class="m-tab" onclick="mTab(this,'mLegal')">Legal</button>
  </div>
  <div class="m-section active" id="mAbout">
    <h3>🌟 About This System</h3>
    <p><strong>SURYA2BOOMI</strong> (Sun-to-Earth) is a solar flare and HF radio blackout forecasting system built on real-time NOAA GOES satellite data.</p>
    <p>It predicts solar flare class (No-flare / C / M / X) for the next 72 hours and 7 days, and derives HF radio blackout probabilities (R1–R5) from those predictions.</p>
    <h3>👨‍💻 Developer</h3>
    <p>Project: Surya2Boomi v1.0<br>Developer: Sudeepa Weerasena<br>Institution: IIT Sri Lanka<br>Contact: sudeepa.20221986@iit.ac.lk<br>Purpose: Educational and Research</p>
    <h3>🛰 Data Sources</h3>
    <ul>
      <li>NOAA GOES X-ray flux — real-time 1-minute averages (0.05–0.4nm)</li>
      <li>NOAA GOES Magnetometer — geomagnetic field Hp component</li>
      <li>NOAA Sunspot Report — daily observed sunspot number</li>
      <li>NOAA HF blackout event archive — 2017–2025 historical records</li>
    </ul>
  </div>
  <div class="m-section" id="mHow">
    <h3>🔬 Solar Flare Prediction Model</h3>
    <p><strong>Algorithm:</strong> HistGradientBoostingClassifier (scikit-learn) — 86 decision trees built sequentially via gradient boosting. Histogram binning (256 bins) makes training ~10× faster than standard GBT.</p>
    <p><strong>122 Features:</strong> X-ray flux, magnetic field, sunspot number, lag features (t−1h to t−24h), rolling statistics (6h/12h/24h mean/max/std), rate-of-change deltas, cyclic time encodings.</p>
    <p><strong>Training:</strong> GOES observations 2017–2022. Validation: 2023. Test: 2024–2026.</p>
    <p><strong>Forecast method:</strong> Autoregressive rollout — model predicts t+1, feeds prediction back as lag feature, predicts t+2, etc.</p>
    <h3>📡 HF Blackout Prediction</h3>
    <p>Not a separate ML model. Uses the <strong>Law of Total Probability</strong> with empirical conditional rates from 8 years of NOAA blackout event records (2017-2025):</p>
    <p style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;background:rgba(0,217,255,0.05);padding:0.7rem;border-radius:6px;margin-top:0.5rem;">
    P(R1) = P(no-flare)×0.02% + P(C)×0.88% + P(M)×13.65% + P(X)×17.44%
    </p>
  </div>
  <div class="m-section" id="mData">
    <h3>📊 Model Performance (Test Set 2024)</h3>
    <table class="mtbl"><tr><th>Metric</th><th>Value</th></tr>
    <tr><td>F1 Score</td><td>0.949</td></tr>
    <tr><td>True Skill Statistic (TSS)</td><td>0.907</td></tr>
    <tr><td>AUC-ROC</td><td>0.98+</td></tr>
    <tr><td>Miss rate (flares)</td><td>&lt; 5%</td></tr></table>
    <h3>📈 Confidence Tiers</h3>
    <table class="mtbl"><tr><th>Tier</th><th>Days</th><th>Meaning</th></tr>
    <tr><td>High</td><td>d+1 to d+3</td><td>Strong — good feature context</td></tr>
    <tr><td>Medium</td><td>d+4 to d+5</td><td>Trend likely correct, magnitude may drift</td></tr>
    <tr><td>Low</td><td>d+6 to d+7</td><td>Background activity indicator only</td></tr></table>
  </div>
  <div class="m-section" id="mLegal">
    <h3>⚠️ Important Disclaimers</h3>
    <p><strong>Forecast Accuracy:</strong> Predictions are based on statistical models trained on historical patterns. Solar activity is inherently probabilistic. Actual conditions may differ significantly, particularly beyond 48 hours.</p>
    <p><strong>Not for Safety-Critical Use:</strong> This system must NOT be used as the sole basis for aviation, maritime, military, or emergency communications decisions.</p>
    <p><strong>Commercial Use:</strong> Educational and research purposes only. Commercial deployment requires independent validation and regulatory compliance.</p>
    <p><strong>Data Latency:</strong> Live forecasts depend on real-time NOAA API availability. Fallback to historical state occurs during outages — treat such forecasts with additional caution.</p>
    <p><strong>No Warranty:</strong> Provided "as is" without warranty. Developers accept no liability for decisions made based on outputs from this system.</p>
    <p><strong>Time Zone:</strong> All times in UTC. Local solar noon is approximately when blackout risk is highest.</p>
    <p><em>By using this system you acknowledge these terms and agree to use forecasts responsibly as supplementary guidance only.</em></p>
  </div>
</div>
</div>

<div class="modal-bg" id="rscaleModal">
<div class="modal">
  <button class="modal-close" onclick="closeModal('rscaleModal')">✕</button>
  <h2>📻 R-Scale & Flare Class Reference Guide</h2>
  <h3>NOAA R-Scale — HF Radio Blackout Severity</h3>
  <div class="scale-row" style="border-color:#ffd700">
    <div class="scale-row-hdr"><span class="scale-name" style="color:#ffd700">R1 Minor</span><span class="scale-meta">Triggered by M1+ | Duration: Minutes</span></div>
    <div class="scale-desc">Weak HF degradation on sunlit side, low-frequency nav signals degraded</div>
  </div>
  <div class="scale-row" style="border-color:#ff8c00">
    <div class="scale-row-hdr"><span class="scale-name" style="color:#ff8c00">R2 Moderate</span><span class="scale-meta">Triggered by M5+ | Duration: 10–30 min</span></div>
    <div class="scale-desc">Limited HF blackout on sunlit side, loss of contact for tens of minutes</div>
  </div>
  <div class="scale-row" style="border-color:#ff4500">
    <div class="scale-row-hdr"><span class="scale-name" style="color:#ff4500">R3 Strong</span><span class="scale-meta">Triggered by X1+ | Duration: ~1 hour</span></div>
    <div class="scale-desc">Wide-area blackout, HF radio contact lost about an hour, low-frequency nav errors</div>
  </div>
  <div class="scale-row" style="border-color:#dc143c">
    <div class="scale-row-hdr"><span class="scale-name" style="color:#dc143c">R4 Severe</span><span class="scale-meta">Triggered by X10+ | Duration: 1–2 hours</span></div>
    <div class="scale-desc">HF radio blackout on most of sunlit side, nav position errors</div>
  </div>
  <div class="scale-row" style="border-color:#8b0000">
    <div class="scale-row-hdr"><span class="scale-name" style="color:#8b0000">R5 Extreme</span><span class="scale-meta">Triggered by X20+ | Duration: Several hours</span></div>
    <div class="scale-desc">Complete HF blackout on entire sunlit side, no HF for several hours</div>
  </div>
  <h3>☀️ Solar Flare Classes</h3>
  <table class="mtbl"><tr><th>Class</th><th>Flux Range</th><th>HF Impact</th></tr>
  <tr><td style="color:#4ade80">A/B</td><td>&lt; 10⁻⁶ W/m²</td><td>None — quiet sun</td></tr>
  <tr><td style="color:#ffd700">C</td><td>10⁻⁶ – 10⁻⁵ W/m²</td><td>Occasional R1 on sunlit limb</td></tr>
  <tr><td style="color:#ff8c00">M</td><td>10⁻⁵ – 10⁻⁴ W/m²</td><td>R1–R2 blackout likely</td></tr>
  <tr><td style="color:#ff4500">X</td><td>&gt; 10⁻⁴ W/m²</td><td>R1–R3+ blackout, CME possible</td></tr></table>
</div>
</div>

<!-- LAYOUT -->
<div class="layout">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sb-logo">SURYA2BOOMI</div>
  <div class="sb-sub">Solar Flare Forecast v1.0</div>

  <div class="sb-sec">System Status</div>
  <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.3rem;">
    <span class="status-dot"></span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#4ade80;letter-spacing:1.5px;">OPERATIONAL</span>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#8b9dc3;" id="sidebarTime"></div>

  <div class="sb-sec">Live Conditions</div>
  <div class="sb-stat-grid">
    <div><div class="sb-stat-lbl">Class Now</div><div class="sb-stat-val" id="sb-class-now" style="color:#4ade80;">No-flare</div></div>
    <div><div class="sb-stat-lbl">P(M) t+1</div><div class="sb-stat-val" id="sb-pm-t1" style="color:#ff8c00;">0.49%</div></div>
    <div style="margin-top:0.5rem"><div class="sb-stat-lbl">72H Peak M</div><div class="sb-stat-val" id="sb-peak-m" style="color:#ff8c00;">0.49%</div></div>
    <div style="margin-top:0.5rem"><div class="sb-stat-lbl">72H Peak R1</div><div class="sb-stat-val" id="sb-peak-r1" style="color:#00d9ff;">0.25%</div></div>
  </div>

  <div class="sb-sec">R-Scale Reference</div>
  <div class="r-row"><div class="r-box" style="background:#ffd70022;border:1px solid #ffd700;color:#ffd700;">R1</div><span class="r-lbl">Minor</span></div>
  <div class="r-row"><div class="r-box" style="background:#ff8c0022;border:1px solid #ff8c00;color:#ff8c00;">R2</div><span class="r-lbl">Moderate</span></div>
  <div class="r-row"><div class="r-box" style="background:#ff450022;border:1px solid #ff4500;color:#ff4500;">R3</div><span class="r-lbl">Strong</span></div>
  <div class="r-row"><div class="r-box" style="background:#dc143c22;border:1px solid #dc143c;color:#dc143c;">R4</div><span class="r-lbl">Severe</span></div>

  <div class="sb-sec">Forecast Pipeline</div>
  <div class="pipe-row"><span class="pipe-lbl">Live NOAA Fetch</span><span class="pipe-ok">✓ LIVE</span></div>
  <div class="pipe-row"><span class="pipe-lbl">72h Flare</span><span class="pipe-ok">✓ In-Memory</span></div>
  <div class="pipe-row"><span class="pipe-lbl">72h Blackout</span><span class="pipe-ok">✓ In-Memory</span></div>
  <div class="pipe-row"><span class="pipe-lbl">7-Day Flare</span><span class="pipe-ok">✓ In-Memory</span></div>
  <div class="pipe-row"><span class="pipe-lbl">7-Day Blackout</span><span class="pipe-ok">✓ In-Memory</span></div>

  <div style="margin-top:0.8rem;display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;">
    <button class="sb-btn sb-btn-sm" onclick="openModal('termsModal')">📋 Terms</button>
    <button class="sb-btn sb-btn-sm" onclick="openModal('rscaleModal')">📻 R-Scale</button>
  </div>
  <div style="margin-top:0.4rem;display:grid;grid-template-columns:1fr;gap:0.4rem;">
    <button class="sb-btn sb-btn-sm" style="background:rgba(0,217,255,0.1); border:1px solid rgba(0,217,255,0.3); color:#00d9ff;" onclick="triggerRefresh()">🔄 Refresh Data</button>
  </div>

  <div class="sb-footer">Model: HistGradBoost · <br>122 features · GOES 2017-2026<br>© 2026 Sudeepa Weerasena · IIT</div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- Header -->
  <div class="hdr">
    <div class="hdr-left">
      <span class="hdr-icon">📡</span>
      <div>
        <div class="hdr-title">SURYA2BOOMI</div>
        <div class="hdr-sub">Solar Flare &amp; HF Radio Blackout Forecast System</div>
      </div>
    </div>
    <div class="hdr-right">
      <div class="hdr-time" id="hdrTime">--:--:-- UTC</div>
      <div class="hdr-date" id="hdrDate">---- -- -- | Solar Cycle 25</div>
      <div class="live-badge"><div class="live-dot"></div><span class="live-txt">LIVE FORECAST</span></div>
    </div>
  </div>
  <div class="divider"></div>

  <!-- KPI Strip -->
  <div class="kpi-strip">
    <div class="kpi">
      <div class="kpi-lbl">Current Class</div>
      <div class="kpi-val" style="color:#4ade80;">No-flare</div>
      <div class="kpi-sub">P = 85.45%</div>
      <div class="tooltip">Predicted flare class for the next hour based on live NOAA X-ray data</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">P(M-class) t+1</div>
      <div class="kpi-val" style="color:#ff8c00;">0.49%</div>
      <div class="kpi-sub">Moderate flare</div>
      <div class="tooltip">Probability of an M-class flare in the next hour. M-class flares trigger R1-R2 blackouts.</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">P(X-class) t+1</div>
      <div class="kpi-val" style="color:#ff4500;">0.23%</div>
      <div class="kpi-sub">Major flare</div>
      <div class="tooltip">Probability of an X-class flare in the next hour. X-class flares trigger R2-R3+ blackouts.</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">HF Risk t+1</div>
      <div class="kpi-val" style="color:#00d9ff;">0.250%</div>
      <div class="kpi-sub">R1 next hour</div>
      <div class="tooltip">Probability of an R1+ HF radio blackout next hour (law of total probability).</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">72h Peak M</div>
      <div class="kpi-val" style="color:#ff8c00;">0.49%</div>
      <div class="kpi-sub">Max in window</div>
      <div class="tooltip">Maximum P(M-class) across all 72 forecast hours. Indicates peak activity window.</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">72h Peak R1</div>
      <div class="kpi-val" style="color:#667eea;">0.250%</div>
      <div class="kpi-sub">Max blackout</div>
      <div class="tooltip">Maximum P(R1 blackout) across all 72 forecast hours.</div>
    </div>
  </div>

  <!-- Alert Banner -->
  <div class="alert-banner" style="background:linear-gradient(135deg,rgba(74,222,128,0.12) 0%,rgba(74,222,128,0.04) 100%);border-color:rgba(74,222,128,0.5);box-shadow:0 0 25px rgba(74,222,128,0.2);">
    <div>
      <div class="alert-lbl">ALERT STATUS — t+1</div>
      <div class="alert-code" style="color:#4ade80;">QUIET</div>
      <div class="alert-mono">HF communications nominal</div>
    </div>
    <div style="text-align:center;">
      <div class="alert-desc">NO SIGNIFICANT ACTIVITY</div>
      <div class="alert-mono" id="alert-flare-class">Predicted flare class → No-flare | Confidence: High</div>
      <div class="alert-mono" id="alert-probs">P(C)=13.83%&nbsp;&nbsp;P(M)=0.49%&nbsp;&nbsp;P(X)=0.23%</div>
    </div>
    <div style="text-align:right;">
      <div class="alert-lbl">HF BLACKOUT t+1</div>
      <div id="alert-blackout-label" style="font-family:'Share Tech Mono',monospace;font-size:1.05rem;color:#00d9ff;">No blackout</div>
      <div class="alert-mono" id="alert-r-probs">R1:0.250%&nbsp;&nbsp;R2:0.020%&nbsp;&nbsp;R3:0.020%</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab(this,'tab72h')">📈 72-Hour Forecast</button>
    <button class="tab" onclick="switchTab(this,'tab7d')">🌐 7-Day Outlook</button>
    <button class="tab" onclick="switchTab(this,'tabHF')">📡 HF Blackout Detail</button>
    <button class="tab" onclick="switchTab(this,'tabAnalytics')">📊 Analytics</button>
  </div>

  <!-- TAB: 72H FORECAST -->
  <div class="tab-content active" id="tab72h">
    <div class="two-col">
      <div>
        <div class="sec-hdr">Solar Flare Class Probabilities — 72 Hours <span>P(M) and P(X) are key risk indicators</span></div>
        <div class="chart-wrap"><canvas id="chartFlare" height="240"></canvas></div>

        <div class="sec-hdr">HF Radio Blackout Probability — 72 Hours <span>Derived from flare probs via empirical NOAA conditional rates</span></div>
        <div class="chart-wrap"><canvas id="chartBlackout" height="210"></canvas></div>

        <div class="sec-hdr">Solar Activity Gauge <span style="cursor:help;" title="Combined activity index based on flare class probabilities">?</span></div>
        <div class="glass" style="text-align:center; padding: 1.5rem 1rem 0.5rem 1rem; position: relative;">
          <div style="font-family:'Orbitron',sans-serif; font-size:0.8rem; color:#8b9dc3; margin-bottom:0.5rem;">Flare Activity</div>
          <div style="height: 180px; position: relative; max-width: 320px; margin: 0 auto;">
            <canvas id="chartGauge"></canvas>
            <div id="gaugeVal" style="position: absolute; bottom: 15%; left: 50%; transform: translateX(-50%); font-family:'Share Tech Mono',monospace; font-size: 2.2rem; color: #00d9ff;">16.4%</div>
            <div style="position: absolute; bottom: 12%; left: 5%; font-family:'Share Tech Mono',monospace; font-size: 0.7rem; color: #8b9dc3;">0</div>
            <div style="position: absolute; bottom: 12%; right: 5%; font-family:'Share Tech Mono',monospace; font-size: 0.7rem; color: #8b9dc3;">100</div>
            <div style="position: absolute; top: 0%; left: 50%; transform: translateX(-50%); font-family:'Share Tech Mono',monospace; font-size: 0.7rem; color: #8b9dc3;">50</div>
          </div>
        </div>

        <div class="sec-hdr">Predicted Class Distribution — 72 Hours <span>Hours per predicted class across full window</span></div>
        <div class="chart-wrap"><canvas id="chartDist" height="200"></canvas></div>
      </div>
      <div>
        <div class="sec-hdr">3-Day Daily Summary <span>Peak flare class per calendar day</span></div>
        <div id="fdCards"></div>

        <div class="sec-hdr">3-Day HF Blackout Summary <span>Max R-level risk per day</span></div>
        <div id="bdCards"></div>

        <div class="sec-hdr">Next 12-Hour Detail <span>Bar = class probability · R1% = blackout risk</span></div>
        <div id="hourRows"></div>
      </div>
    </div>
  </div>

  <!-- TAB: 7-DAY -->
  <div class="tab-content" id="tab7d">
    <div class="two-col">
      <div>
        <div class="sec-hdr">7-Day Solar Flare Outlook <span>d+1..d+3 = High conf. | d+4..d+5 = Med. | d+6..d+7 = Low</span></div>
        <div class="chart-wrap"><canvas id="chart7dFlare" height="120"></canvas></div>
        <div class="sec-hdr">7-Day HF Blackout Risk Outlook <span>Daily max R-level probability per day</span></div>
        <div class="chart-wrap"><canvas id="chart7dBlackout" height="100"></canvas></div>
      </div>
      <div>
        <div class="sec-hdr">7-Day Daily Cards <span>Peak flare class + blackout risk per day</span></div>
        <div id="f7Cards"></div>
      </div>
    </div>
  </div>

  <!-- TAB: HF DETAIL -->
  <div class="tab-content" id="tabHF">
    <div class="sec-hdr">Empirical Conditional Rates <span>Source: NOAA 2017–2025 blackout event records</span></div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-bottom:1rem;">
      <div class="glass" style="text-align:center;">
        <div style="font-family:'Orbitron',sans-serif;font-size:0.72rem;color:#4ade80;margin-bottom:0.5rem;">No-flare</div>
        <div style="font-size:0.6rem;color:#8b9dc3;">R1 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ffd700;">0.02%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R2 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff8c00;">0.00%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R3 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff4500;">0.00%</div>
      </div>
      <div class="glass" style="text-align:center;">
        <div style="font-family:'Orbitron',sans-serif;font-size:0.72rem;color:#ffd700;margin-bottom:0.5rem;">C-class</div>
        <div style="font-size:0.6rem;color:#8b9dc3;">R1 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ffd700;">0.88%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R2 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff8c00;">0.08%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R3 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff4500;">0.01%</div>
      </div>
      <div class="glass" style="text-align:center;">
        <div style="font-family:'Orbitron',sans-serif;font-size:0.72rem;color:#ff8c00;margin-bottom:0.5rem;">M-class</div>
        <div style="font-size:0.6rem;color:#8b9dc3;">R1 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ffd700;">13.65%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R2 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff8c00;">1.51%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R3 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff4500;">0.25%</div>
      </div>
      <div class="glass" style="text-align:center;">
        <div style="font-family:'Orbitron',sans-serif;font-size:0.72rem;color:#ff4500;margin-bottom:0.5rem;">X-class</div>
        <div style="font-size:0.6rem;color:#8b9dc3;">R1 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ffd700;">17.44%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R2 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff8c00;">2.56%</div>
        <div style="font-size:0.6rem;color:#8b9dc3;margin-top:0.3rem;">R3 rate</div><div style="font-family:'Share Tech Mono',monospace;color:#ff4500;">7.19%</div>
      </div>
    </div>
    <div class="sec-hdr" style="margin-top:1.2rem;">72-Hour Blackout Detail Table <span>All 72 hourly rows</span></div>
    <div class="tbl-wrap">
      <table class="data-tbl"><thead><tr><th>Offset</th><th>Timestamp</th><th>Flare Class</th><th>R1 (%)</th><th>R2 (%)</th><th>R3 (%)</th><th>Status</th></tr></thead>
      <tbody id="bhTable"></tbody></table>
    </div>
  </div>

  <!-- TAB: ANALYTICS -->
  <div class="tab-content" id="tabAnalytics">
    <div class="analytics-grid">
      <div>
        <div class="sec-hdr">72h Class Distribution <span>Proportion of hours per predicted class</span></div>
        <div class="chart-wrap"><canvas id="chartPie" height="180"></canvas></div>
        <div class="sec-hdr">Blackout Risk Heatmap <span>R1/R2/R3 probability across 72 hours</span></div>
        <div class="chart-wrap"><canvas id="chartHeat" height="100"></canvas></div>
      </div>
      <div>
        <div class="sec-hdr">7-Day Daily Peak Probabilities <span>M and X class with confidence markers</span></div>
        <div class="chart-wrap"><canvas id="chartScatter" height="180"></canvas></div>
        <div class="sec-hdr">Forecast Summary Statistics</div>
        <div class="glass">
          <div class="stat-row"><span class="stat-key">72h forecast window</span><span class="stat-val">72 hours</span></div>
          <div class="stat-row"><span class="stat-key">Peak P(M-class)</span><span class="stat-val" id="stat-peak-m">0.49%</span></div>
          <div class="stat-row"><span class="stat-key">Peak P(X-class)</span><span class="stat-val" id="stat-peak-x">0.23%</span></div>
          <div class="stat-row"><span class="stat-key">Peak R1 (72h)</span><span class="stat-val" id="stat-peak-r1">0.2500%</span></div>
          <div class="stat-row"><span class="stat-key">Peak R2 (72h)</span><span class="stat-val" id="stat-peak-r2">0.0200%</span></div>
          <div class="stat-row"><span class="stat-key">Hours M-class</span><span class="stat-val" id="stat-hours-m">0</span></div>
          <div class="stat-row"><span class="stat-key">Hours X-class</span><span class="stat-val" id="stat-hours-x">0</span></div>
          <div class="stat-row"><span class="stat-key">7-day outlook rows</span><span class="stat-val">7</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB: RAW DATA -->
  <div class="tab-content" id="tabData">
    <div class="sec-hdr">forecast_hourly.csv — 72h Flare (step7 output)</div>
    <div class="tbl-wrap" style="margin-bottom:1rem;">
      <table class="data-tbl"><thead><tr><th>Offset</th><th>Timestamp</th><th>Class</th><th>Prob</th><th>P(C)</th><th>P(M)</th><th>P(X)</th></tr></thead>
      <tbody id="fhTable"></tbody></table>
    </div>
    <div class="sec-hdr">forecast_7day.csv — 7-Day Flare (step10 output)</div>
    <div class="tbl-wrap" style="margin-bottom:1rem;">
      <table class="data-tbl"><thead><tr><th>Offset</th><th>Date</th><th>Peak Class</th><th>Peak Prob</th><th>P(C)</th><th>P(M)</th><th>P(X)</th><th>Confidence</th></tr></thead>
      <tbody id="f7Table"></tbody></table>
    </div>
    <div class="sec-hdr">blackout_7day.csv — 7-Day Blackout (step11 output)</div>
    <div class="tbl-wrap">
      <table class="data-tbl"><thead><tr><th>Offset</th><th>Date</th><th>Flare</th><th>R1%</th><th>R2%</th><th>R3%</th><th>Peak</th><th>Confidence</th></tr></thead>
      <tbody id="b7Table"></tbody></table>
    </div>
    <div class="sec-hdr" style="margin-top:1.2rem;">Pipeline Data Flow</div>
    <div class="formula">
      <span style="color:#00d9ff;">live_data.py</span>  →  Live NOAA API  →  Feature snapshot (122 features)<br>
      <span style="color:#4ade80;">step7</span>  →  72h flare forecast (in-memory)<br>
      <span style="color:#ffd700;">step8</span>  →  72h blackout probabilities (in-memory)<br>
      <span style="color:#4ade80;">step10</span> →  7-day flare forecast (in-memory)<br>
      <span style="color:#ffd700;">step11</span> →  7-day blackout probabilities (in-memory)<br>
      <span style="color:#8b9dc3;">app.py</span> →  pipeline.run_full_pipeline()  →  this dashboard
    </div>
  </div>

  <!-- Footer -->
  <div style="height:1px;background:rgba(0,217,255,0.1);margin:2rem 0 0.8rem;"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#4a5568;">
    <span>SURYA2BOOMI — SOLAR FLARE & HF BLACKOUT FORECAST SYSTEM</span>
    <span>Model: HistGradBoost · 122 features · GOES NOAA 2017-2026</span>
    <span>© 2026 Sudeepa Weerasena · IIT Sri Lanka · Educational Use Only</span>
  </div>

</div><!-- /main -->
</div><!-- /layout -->

<script>
const D={};

// Clock
function tick(){const n=new Date();const pad=x=>String(x).padStart(2,'0');
const ts=`${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())} UTC`;
const ds=`${n.getUTCFullYear()}-${pad(n.getUTCMonth()+1)}-${pad(n.getUTCDate())} | Solar Cycle 25`;
document.getElementById('hdrTime').textContent=ts;
document.getElementById('hdrDate').textContent=ds;
document.getElementById('sidebarTime').innerHTML=`${n.getUTCFullYear()}-${pad(n.getUTCMonth()+1)}-${pad(n.getUTCDate())}<br>${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())} UTC`;}
tick();setInterval(tick,1000);

// Tabs
function switchTab(btn,id){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active');}
function mTab(btn,id){document.querySelectorAll('.m-tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.m-section').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active');}

// Modals
function openModal(id){document.getElementById(id).classList.add('open');}
function closeModal(id){document.getElementById(id).classList.remove('open');}
document.querySelectorAll('.modal-bg').forEach(m=>m.addEventListener('click',e=>{if(e.target===m)m.classList.remove('open');}));

// Refresh: clears session_state TTL flag then reloads so Python re-fetches fresh data
function triggerRefresh(){
  try { window.parent.location.reload(); } catch(e1) {
    try { window.top.location.reload(); } catch(e2) {
      location.reload();
    }
  }
}

// Chart defaults
const cOpts={responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b9dc3',font:{family:"'Rajdhani',sans-serif",size:11}}},tooltip:{backgroundColor:'rgba(10,14,39,0.95)',borderColor:'rgba(0,217,255,0.3)',borderWidth:1,titleColor:'#00d9ff',bodyColor:'rgba(255,255,255,0.85)'}},scales:{x:{grid:{color:'rgba(255,255,255,0.06)'},ticks:{color:'#8b9dc3',font:{family:"'Rajdhani',sans-serif",size:10}}},y:{grid:{color:'rgba(255,255,255,0.06)'},ticks:{color:'#8b9dc3',font:{family:"'Rajdhani',sans-serif",size:10}}}}};

// Chart 1: Flare probs 72h
new Chart(document.getElementById('chartFlare'),{type:'line',data:{labels:D.fh_hours.map(h=>'t+'+h),datasets:[{label:'P(M-class)',data:D.fh_pm,borderColor:'#ff8c00',backgroundColor:'rgba(255,140,0,0.12)',fill:true,tension:0.3,pointRadius:0,borderWidth:2.5},{label:'P(X-class)',data:D.fh_px,borderColor:'#ff4500',backgroundColor:'rgba(255,69,0,0.09)',fill:true,tension:0.3,pointRadius:0,borderWidth:2},{label:'P(C-class)',data:D.fh_pc,borderColor:'#ffd700',backgroundColor:'transparent',tension:0.3,pointRadius:0,borderWidth:1.5,borderDash:[4,4]}]},options:{...cOpts,plugins:{...cOpts.plugins},scales:{x:{...cOpts.scales.x,ticks:{...cOpts.scales.x.ticks,maxTicksLimit:12}},y:{...cOpts.scales.y,title:{display:true,text:'Probability (%)',color:'#8b9dc3',font:{size:10}}}}}});

// Chart 2: Blackout 72h
new Chart(document.getElementById('chartBlackout'),{type:'line',data:{labels:D.fh_hours.map(h=>'t+'+h),datasets:[{label:'R1 Minor',data:D.bh_r1,borderColor:'#ffd700',backgroundColor:'rgba(255,215,0,0.1)',fill:true,tension:0.3,pointRadius:0,borderWidth:2.5},{label:'R2 Moderate',data:D.bh_r2,borderColor:'#ff8c00',backgroundColor:'rgba(255,140,0,0.07)',fill:true,tension:0.3,pointRadius:0,borderWidth:2},{label:'R3 Strong',data:D.bh_r3,borderColor:'#ff4500',backgroundColor:'rgba(255,69,0,0.05)',fill:true,tension:0.3,pointRadius:0,borderWidth:1.5}]},options:{...cOpts,scales:{x:{...cOpts.scales.x,ticks:{...cOpts.scales.x.ticks,maxTicksLimit:12}},y:{...cOpts.scales.y,title:{display:true,text:'Probability (%)',color:'#8b9dc3',font:{size:10}}}}}});

// Chart 3: Distribution bar
new Chart(document.getElementById('chartDist'),{type:'bar',data:{labels:['No-flare','C-class','M-class','X-class'],datasets:[{label:'Hours',data:D.class_dist,backgroundColor:['#4ade80','#ffd700','#ff8c00','#ff4500'],borderRadius:4}]},options:{...cOpts,plugins:{...cOpts.plugins,legend:{display:false}},scales:{x:{...cOpts.scales.x},y:{...cOpts.scales.y,title:{display:true,text:'Hours',color:'#8b9dc3',font:{size:10}}}}}});

// Chart 7d flare
new Chart(document.getElementById('chart7dFlare'),{type:'bar',data:{labels:D.f7.map(r=>r.day_offset),datasets:[{label:'P(M)',data:D.f7.map(r=>r.max_p_m),backgroundColor:'rgba(255,140,0,0.8)',borderRadius:3},{label:'P(X)',data:D.f7.map(r=>r.max_p_x),backgroundColor:'rgba(255,69,0,0.8)',borderRadius:3},{label:'P(C)',data:D.f7.map(r=>r.max_p_c),backgroundColor:'rgba(255,215,0,0.65)',borderRadius:3}]},options:{...cOpts,scales:{x:{...cOpts.scales.x},y:{...cOpts.scales.y,title:{display:true,text:'Probability (%)',color:'#8b9dc3',font:{size:10}}}}}});

// Chart 7d blackout
new Chart(document.getElementById('chart7dBlackout'),{type:'bar',data:{labels:D.b7.map(r=>r.day_offset),datasets:[{label:'R1',data:D.b7.map(r=>r.max_p_R1),backgroundColor:'rgba(255,215,0,0.8)',borderRadius:3},{label:'R2',data:D.b7.map(r=>r.max_p_R2),backgroundColor:'rgba(255,140,0,0.8)',borderRadius:3},{label:'R3',data:D.b7.map(r=>r.max_p_R3),backgroundColor:'rgba(255,69,0,0.8)',borderRadius:3}]},options:{...cOpts,scales:{x:{...cOpts.scales.x},y:{...cOpts.scales.y,title:{display:true,text:'Probability (%)',color:'#8b9dc3',font:{size:10}}}}}});

// Pie chart
new Chart(document.getElementById('chartPie'),{type:'doughnut',data:{labels:['No-flare','C-class','M-class','X-class'],datasets:[{data:D.class_dist,backgroundColor:['#4ade80','#ffd700','#ff8c00','#ff4500'],borderColor:'#0a0e27',borderWidth:2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#8b9dc3',font:{family:"'Rajdhani',sans-serif",size:11},padding:16}},tooltip:{backgroundColor:'rgba(10,14,39,0.95)',borderColor:'rgba(0,217,255,0.3)',borderWidth:1,titleColor:'#00d9ff',bodyColor:'rgba(255,255,255,0.85)'}}}});

// Scatter 7d
new Chart(document.getElementById('chartScatter'),{type:'line',data:{labels:D.f7.map(r=>r.day_offset),datasets:[{label:'P(M)',data:D.f7.map(r=>r.max_p_m),borderColor:'#ff8c00',pointBackgroundColor:D.f7.map(r=>r.confidence==='High'?'#4ade80':r.confidence==='Medium'?'#ffd700':'#ff8c00'),pointRadius:8,pointHoverRadius:10,fill:false,tension:0.3,borderWidth:2},{label:'P(X)',data:D.f7.map(r=>r.max_p_x),borderColor:'#ff4500',pointRadius:5,fill:false,tension:0.3,borderWidth:1.5,borderDash:[4,4]}]},options:{...cOpts,scales:{x:{...cOpts.scales.x},y:{...cOpts.scales.y,title:{display:true,text:'Probability (%)',color:'#8b9dc3',font:{size:10}}}}}});

// Heat chart
new Chart(document.getElementById('chartHeat'),{type:'bar',data:{labels:D.fh_hours.map(h=>'t+'+h),datasets:[{label:'R1',data:D.bh_r1,backgroundColor:'rgba(255,215,0,0.6)',borderRadius:0},{label:'R2',data:D.bh_r2,backgroundColor:'rgba(255,140,0,0.8)',borderRadius:0},{label:'R3',data:D.bh_r3,backgroundColor:'rgba(255,69,0,0.9)',borderRadius:0}]},options:{...cOpts,scales:{x:{...cOpts.scales.x,stacked:true,ticks:{...cOpts.scales.x.ticks,maxTicksLimit:12}},y:{...cOpts.scales.y,stacked:true,title:{display:true,text:'%',color:'#8b9dc3',font:{size:10}}}}}});

// Gauge chart
const gV = parseFloat(document.getElementById('gaugeVal').textContent);
new Chart(document.getElementById('chartGauge'), {
  type: 'doughnut',
  data: {
    datasets: [{
      data: [gV, 100 - gV],
      backgroundColor: [(gV > 50 ? '#ff4500' : gV > 20 ? '#ff8c00' : '#00d9ff'), 'rgba(255,255,255,0.05)'],
      borderWidth: 0,
      circumference: 180,
      rotation: 270,
      cutout: '80%',
      borderRadius: 10
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: 2,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false }
    }
  }
});

// Build dynamic content
const confC={'High':'#4ade80','Medium':'#ffd700','Low':'#ff8c00'};

// Helper: get peak label and color from R probs
function getRLabel(r1,r2,r3){if(r3>=1.0)return{lbl:'R3 Strong',col:'#ff4500'};if(r2>=0.5)return{lbl:'R2 Moderate',col:'#ff8c00'};if(r1>=0.5)return{lbl:'R1 Minor',col:'#ffd700'};return{lbl:'No blackout',col:'#4ade80'};}

// 3-day flare daily cards
const fdHTML=(D.fd||[]).map(r=>{
const cc=D.class_colors[r.peak_name]||'#4ade80';
return`<div class="day-card"><div class="day-hdr"><div><div class="day-offset">${r.day_offset}</div><div class="day-date">${r.date}</div></div><div><div class="day-class" style="color:${cc};">${r.peak_name}</div><div class="day-prob">${r.probability}%</div></div></div></div>`;}).join('');
document.getElementById('fdCards').innerHTML=fdHTML;

// 3-day blackout cards
const bdHTML=(D.bd||D.b7.slice(0,3)).map(r=>{
const rl=getRLabel(r.max_p_R1,r.max_p_R2,r.max_p_R3);
return`<div class="day-card"><div class="day-hdr">
<div><div class="day-offset">${r.day_offset}</div><div class="day-date">${r.date}</div></div>
<div style="text-align:right">
<span style="display:inline-block;padding:0.15rem 0.55rem;border-radius:20px;font-family:'Orbitron',sans-serif;font-size:0.6rem;font-weight:700;background:${rl.col}22;color:${rl.col};border:1px solid ${rl.col};">${rl.lbl}</span>
</div></div>
<div class="r-grid">
<div><div class="r-cell-lbl">R1</div><div class="r-cell-val" style="color:#ffd700">${r.max_p_R1.toFixed(4)}%</div></div>
<div><div class="r-cell-lbl">R2</div><div class="r-cell-val" style="color:#ff8c00">${r.max_p_R2.toFixed(4)}%</div></div>
<div><div class="r-cell-lbl">R3</div><div class="r-cell-val" style="color:#ff4500">${r.max_p_R3.toFixed(4)}%</div></div>
</div></div>`;}).join('');
document.getElementById('bdCards').innerHTML=bdHTML;

// 12-hour rows
const hrHTML=D.fh_hours.slice(0,12).map((h,i)=>{
const pName=D.fh_pred_name[i];const prob=D.fh_prob[i];const bw=Math.min(prob,100);const r1=D.bh_r1[i];
const cc=D.class_colors[pName]||'#4ade80';
return`<div class="hr-row">
<div class="hr-off">t+${h}</div>
<div class="hr-bar-wrap"><div class="hr-bar" style="width:${bw}%;background:${cc};"></div></div>
<div class="hr-class" style="color:${cc};">${pName}</div>
<div class="hr-r1" title="R1 blackout probability">R1:${r1.toFixed(3)}%</div>
</div>`;}).join('');
document.getElementById('hourRows').innerHTML=hrHTML;

// 7-day f7 cards — match b7 to get R-level labels
const f7HTML=D.f7.map((r,idx)=>{
const b7r=D.b7[idx]||{};
const rl=getRLabel(b7r.max_p_R1||0,b7r.max_p_R2||0,b7r.max_p_R3||0);
const peakCol=D.class_colors[r.peak_name]||'#4ade80';
return`<div class="day-card"><div class="day-hdr">
<div><div class="day-offset">${r.day_offset}</div><div class="day-date">${r.date}</div></div>
<div style="text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:0.2rem;">
<span style="display:inline-block;padding:0.15rem 0.55rem;border-radius:20px;font-family:'Orbitron',sans-serif;font-size:0.6rem;font-weight:700;background:${rl.col}22;color:${rl.col};border:1px solid ${rl.col};">${rl.lbl}</span>
<span style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:${confC[r.confidence]||'#8b9dc3'};">${r.confidence}</span>
</div></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.3rem;">
<div><div style="font-size:0.6rem;color:#8b9dc3;">P(M) ${r.max_p_m.toFixed(1)}%</div>
<div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(r.max_p_m*20,100)}%;background:#ff8c00;"></div></div></div>
<div><div style="font-size:0.6rem;color:#8b9dc3;">P(X) ${r.max_p_x.toFixed(1)}%</div>
<div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(r.max_p_x*50,100)}%;background:#ff4500;"></div></div></div>
</div></div>`;}).join('');
document.getElementById('f7Cards').innerHTML=f7HTML;

// BH table (all 72 rows)
const bhTHTML=D.fh_hours.map((h,i)=>{
const pName=D.fh_pred_name[i];const cc=D.class_colors[pName]||'#4ade80';const ll=D.bh_likely_label[i];
return`<tr>
<td>t+${h}</td><td>${D.fh_ts[i]}</td><td style="color:${cc};">${pName}</td>
<td>${D.bh_r1[i].toFixed(4)}</td><td>${D.bh_r2[i].toFixed(4)}</td><td>${D.bh_r3[i].toFixed(4)}</td>
<td>${ll}</td></tr>`;}).join('');
document.getElementById('bhTable').innerHTML=bhTHTML;

// FH table
const fhTHTML=D.fh_hours.map((h,i)=>{
const pName=D.fh_pred_name[i];const cc=D.class_colors[pName]||'#4ade80';
return`<tr>
<td>t+${h}</td><td>${D.fh_ts[i]}</td><td style="color:${cc};">${pName}</td>
<td>${D.fh_prob[i]}%</td><td>${D.fh_pc[i]}%</td><td>${D.fh_pm[i]}%</td><td>${D.fh_px[i]}%</td></tr>`;}).join('');
document.getElementById('fhTable').innerHTML=fhTHTML;

// F7 table
const f7THTML=D.f7.map(r=>{
const cc=D.class_colors[r.peak_name]||'#4ade80';
return`<tr>
<td>${r.day_offset}</td><td>${r.date}</td><td style="color:${cc};">${r.peak_name}</td>
<td>${r.peak_prob}%</td><td>${r.max_p_c}%</td><td>${r.max_p_m}%</td><td>${r.max_p_x}%</td>
<td style="color:${confC[r.confidence]||'#8b9dc3'};">${r.confidence}</td></tr>`;}).join('');
document.getElementById('f7Table').innerHTML=f7THTML;

// B7 table
const b7THTML=D.b7.map(r=>{
const cc=D.class_colors[r.dominant_flare]||'#4ade80';const rl=getRLabel(r.max_p_R1,r.max_p_R2,r.max_p_R3);
return`<tr>
<td>${r.day_offset}</td><td>${r.date}</td><td style="color:${cc};">${r.dominant_flare}</td>
<td>${r.max_p_R1.toFixed(4)}</td><td>${r.max_p_R2.toFixed(4)}</td><td>${r.max_p_R3.toFixed(4)}</td>
<td>${rl.lbl}</td><td style="color:${confC[r.confidence]||'#8b9dc3'};">${r.confidence}</td></tr>`;}).join('');
document.getElementById('b7Table').innerHTML=b7THTML;

// Update analytics stats
document.getElementById('stat-peak-m').textContent=Math.max(...D.fh_pm).toFixed(2)+'%';
document.getElementById('stat-peak-x').textContent=Math.max(...D.fh_px).toFixed(2)+'%';
document.getElementById('stat-peak-r1').textContent=Math.max(...D.bh_r1).toFixed(4)+'%';
document.getElementById('stat-peak-r2').textContent=Math.max(...D.bh_r2).toFixed(4)+'%';
document.getElementById('stat-hours-m').textContent=D.class_dist[2];
document.getElementById('stat-hours-x').textContent=D.class_dist[3];

// Update sidebar
document.getElementById('sb-pm-t1').textContent=D.fh_pm[0].toFixed(2)+'%';
document.getElementById('sb-peak-m').textContent=Math.max(...D.fh_pm).toFixed(2)+'%';
document.getElementById('sb-peak-r1').textContent=Math.max(...D.bh_r1).toFixed(3)+'%';

// Update alert banner details
document.getElementById('alert-flare-class').textContent='Predicted flare class → '+D.fh_pred_name[0]+' | Confidence: High';
document.getElementById('alert-probs').innerHTML='P(C)='+D.fh_pc[0].toFixed(2)+'%&nbsp;&nbsp;P(M)='+D.fh_pm[0].toFixed(2)+'%&nbsp;&nbsp;P(X)='+D.fh_px[0].toFixed(2)+'%';
document.getElementById('alert-blackout-label').textContent=D.bh_likely_label[0];
document.getElementById('alert-r-probs').innerHTML='R1:'+D.bh_r1[0].toFixed(3)+'%&nbsp;&nbsp;R2:'+D.bh_r2[0].toFixed(3)+'%&nbsp;&nbsp;R3:'+D.bh_r3[0].toFixed(3)+'%';

// Update sidebar class now
const sbClassNow=document.getElementById('sb-class-now');
sbClassNow.textContent=D.fh_pred_name[0];
sbClassNow.style.color=D.class_colors[D.fh_pred_name[0]]||'#4ade80';
</script>
</body>
</html>
"""

def main():

    data = load_data()
    if not data:
        st.stop()
    D, kpis = data
    
    html_content = HTML_TEMPLATE

    # Inject D object
    start_marker = "const D={"
    end_marker = "};"
    d_start_idx = html_content.find(start_marker)
    if d_start_idx != -1:
        d_end_idx = html_content.find(end_marker, d_start_idx)
        if d_end_idx != -1:
            json_data = json.dumps(D)
            inner_json = json_data[1:-1]
            html_content = html_content[:d_start_idx + len(start_marker)] + inner_json + html_content[d_end_idx:]

    # Dynamic KPI replacements
    html_content = html_content.replace('<div class="kpi-val" style="color:#4ade80;">No-flare</div>', f'<div class="kpi-val" style="color:{kpis["alert_color"]};">{kpis["current_class"]}</div>')
    html_content = html_content.replace('<div class="kpi-sub">P = 85.45%</div>', f'<div class="kpi-sub">P = {kpis["current_prob"]}</div>')
    html_content = html_content.replace('<div class="kpi-val" style="color:#ff8c00;">0.49%</div>', f'<div class="kpi-val" style="color:#ff8c00;">{kpis["p_m_t1"]}</div>', 1)
    html_content = html_content.replace('<div class="kpi-val" style="color:#ff4500;">0.23%</div>', f'<div class="kpi-val" style="color:#ff4500;">{kpis["p_x_t1"]}</div>')
    html_content = html_content.replace('<div class="kpi-val" style="color:#00d9ff;">0.250%</div>', f'<div class="kpi-val" style="color:#00d9ff;">{kpis["r1_t1"]}</div>', 1)
    html_content = html_content.replace('<div class="kpi-val" style="color:#ff8c00;">0.49%</div>', f'<div class="kpi-val" style="color:#ff8c00;">{kpis["peak_m_72h"]}</div>', 1)
    html_content = html_content.replace('<div class="kpi-val" style="color:#667eea;">0.250%</div>', f'<div class="kpi-val" style="color:#667eea;">{kpis["peak_r1_72h"]}</div>')
    html_content = html_content.replace('<div class="alert-code" style="color:#4ade80;">QUIET</div>', f'<div class="alert-code" style="color:{kpis["alert_color"]};">{kpis["alert_code"]}</div>')
    html_content = html_content.replace('<div class="alert-desc">NO SIGNIFICANT ACTIVITY</div>', f'<div class="alert-desc">{kpis["alert_desc"]}</div>')
    
    # Gauge update
    html_content = html_content.replace('<div id="gaugeVal" style="position: absolute; bottom: 15%; left: 50%; transform: translateX(-50%); font-family:\'Share Tech Mono\',monospace; font-size: 2.2rem; color: #00d9ff;">16.4%</div>', f'<div id="gaugeVal" style="position: absolute; bottom: 15%; left: 50%; transform: translateX(-50%); font-family:\'Share Tech Mono\',monospace; font-size: 2.2rem; color: #00d9ff;">{kpis["gauge_val"]}%</div>')

    # Render HTML in Streamlit
    st.components.v1.html(html_content, height=2000, scrolling=True)

if __name__ == "__main__":
    main()