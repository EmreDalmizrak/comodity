"""
Commodity Monte Carlo Simulation — Iran-Israel Conflict Risk
Streamlit App  |  Live prices via yfinance
Run: streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import streamlit as st
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Commodity Monte Carlo — Iran-Israel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #e24b4a, #f0993b, #185fa5, #1d9e75);
}
.hero-title { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:600; color:#e6edf3; margin:0 0 0.3rem; }
.hero-sub   { font-size:0.85rem; color:#8b949e; margin:0; font-family:'IBM Plex Mono',monospace; }
.live-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:#1a2e22; border:1px solid #1d9e75; border-radius:99px;
    padding:4px 12px; font-size:11px; font-family:'IBM Plex Mono',monospace;
    color:#5ecc8e; letter-spacing:0.5px; margin-bottom:1rem;
}
.live-dot { width:7px; height:7px; background:#1d9e75; border-radius:50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.metric-row { display:flex; gap:12px; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-card { flex:1; min-width:130px; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 16px; }
.metric-card.up   { border-top:2px solid #1d9e75; }
.metric-card.down { border-top:2px solid #e24b4a; }
.metric-card.info { border-top:2px solid #185fa5; }
.metric-card.warn { border-top:2px solid #f0993b; }
.metric-label { font-size:11px; color:#8b949e; font-family:'IBM Plex Mono',monospace; letter-spacing:0.5px; margin-bottom:4px; }
.metric-val   { font-size:1.4rem; font-weight:600; color:#e6edf3; font-family:'IBM Plex Mono',monospace; }
.metric-delta { font-size:11px; margin-top:2px; font-family:'IBM Plex Mono',monospace; }
.delta-up  { color:#1d9e75; } .delta-down { color:#e24b4a; } .delta-neu { color:#8b949e; }
.price-row { display:flex; gap:10px; margin-bottom:1.2rem; flex-wrap:wrap; }
.price-card {
    flex:1; min-width:160px; background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:12px 16px;
    display:flex; justify-content:space-between; align-items:center;
}
.price-name { font-size:12px; color:#8b949e; font-family:'IBM Plex Mono',monospace; }
.price-val  { font-size:1.2rem; font-weight:600; color:#e6edf3; font-family:'IBM Plex Mono',monospace; }
.price-chg-up   { font-size:11px; color:#1d9e75; font-family:'IBM Plex Mono',monospace; }
.price-chg-down { font-size:11px; color:#e24b4a; font-family:'IBM Plex Mono',monospace; }
.scenario-badge {
    display:inline-block; padding:4px 12px; border-radius:99px;
    font-size:12px; font-family:'IBM Plex Mono',monospace;
    font-weight:600; letter-spacing:0.5px; margin-bottom:1rem;
}
.sc-base   { background:#1c2e3a; color:#5ea8e0; border:1px solid #185fa5; }
.sc-mild   { background:#2e2418; color:#e0a84a; border:1px solid #ba7517; }
.sc-strait { background:#2e1a1a; color:#e07070; border:1px solid #a32d2d; }
.sc-deesc  { background:#1a2e22; color:#5ecc8e; border:1px solid #1d9e75; }
.section-hdr {
    font-family:'IBM Plex Mono',monospace; font-size:11px;
    letter-spacing:2px; color:#8b949e; text-transform:uppercase;
    margin:1.2rem 0 0.6rem; border-bottom:1px solid #21262d; padding-bottom:6px;
}
.stats-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.stats-tbl th { text-align:left; padding:7px 10px; color:#8b949e; font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:1px; border-bottom:1px solid #21262d; font-weight:500; }
.stats-tbl td { padding:7px 10px; color:#e6edf3; border-bottom:1px solid #161b22; font-family:'IBM Plex Mono',monospace; font-size:12px; }
.tbl-up   { color:#1d9e75 !important; } .tbl-down { color:#e24b4a !important; } .tbl-neu { color:#8b949e !important; }
section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
.stButton>button {
    background:#161b22 !important; border:1px solid #30363d !important;
    color:#e6edf3 !important; font-family:'IBM Plex Mono',monospace !important;
    font-size:13px !important; border-radius:6px !important; width:100%;
}
.stButton>button:hover { border-color:#58a6ff !important; color:#58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

COMMODITY_META = {
    "Crude Oil (WTI)": {
        "symbol": "OIL", "ticker": "CL=F",
        "unit": "$", "decimals": 2,
        "fallback_spot": 85.00,
        "fallback_drift": 0.05, "fallback_sigma": 0.28,
    },
    "Natural Gas": {
        "symbol": "GAS", "ticker": "NG=F",
        "unit": "$", "decimals": 3,
        "fallback_spot": 2.80,
        "fallback_drift": 0.02, "fallback_sigma": 0.55,
    },
    "Gold": {
        "symbol": "GOLD", "ticker": "GC=F",
        "unit": "$", "decimals": 0,
        "fallback_spot": 2340.0,
        "fallback_drift": 0.06, "fallback_sigma": 0.15,
    },
}

# Each scenario defines ABSOLUTE drift & sigma per commodity (not adjustments).
# This guarantees the ordering:
#   Strait Closure > Mild Escalation > Base Case > De-escalation
#
# OIL / GAS share one set; GOLD has its own (safe-haven, lower vol).

SCENARIOS = {
    "Base Case": {
        # --- core GBM params (absolute, per commodity key) ---
        "drift": {"OIL": 0.03, "GAS": 0.03, "GOLD": 0.03},
        "sigma": {"OIL": 0.23, "GAS": 0.30, "GOLD": 0.23},
        # --- jump process ---
        "jump_prob": 0.01,   # daily probability of a geopolitical jump
        "jump_mean": 0.02,   # mean fractional jump size
        # --- strait closure ---
        "strait_prob": 0.005,
        "mean_rev":    0.05,
        # --- UI ---
        "badge_class": "sc-base", "emoji": "🔵",
        "description": "Low-level tension, current market dynamics prevail.",
    },
    "Mild Escalation": {
        "drift": {"OIL": 0.06, "GAS": 0.06, "GOLD": 0.06},
        "sigma": {"OIL": 0.30, "GAS": 0.40, "GOLD": 0.30},
        "jump_prob": 0.03,
        "jump_mean": 0.05,
        "strait_prob": 0.02,
        "mean_rev":    0.04,
        "badge_class": "sc-mild", "emoji": "🟡",
        "description": "Increased airstrikes, regional risk premium builds.",
    },
    "Strait Closure": {
        "drift": {"OIL": 0.12, "GAS": 0.12, "GOLD": 0.12},
        "sigma": {"OIL": 0.45, "GAS": 0.60, "GOLD": 0.45},
        "jump_prob": 0.06,
        "jump_mean": 0.12,
        "strait_prob": 0.15,
        "mean_rev":    0.02,
        "badge_class": "sc-strait", "emoji": "🔴",
        "description": "Hormuz blocked — severe supply disruption scenario.",
    },
    "De-escalation": {
        "drift": {"OIL": -0.02, "GAS": -0.02, "GOLD": -0.02},
        "sigma": {"OIL": 0.18,  "GAS": 0.22,  "GOLD": 0.18},
        "jump_prob": 0.00,
        "jump_mean": -0.01,
        "strait_prob": 0.001,
        "mean_rev":    0.12,
        "badge_class": "sc-deesc", "emoji": "🟢",
        "description": "Ceasefire / diplomacy — risk premium deflates.",
    },
}

# One-time strait closure price impact per commodity
STRAIT_IMPACT = {"OIL": 0.25, "GAS": 0.35, "GOLD": 0.08}

# Chart colours
BG   = "#0d1117"
BG2  = "#161b22"
GRID = "#21262d"
TEXT = "#e6edf3"
MUT  = "#8b949e"
BLUE = "#185FA5"
RED  = "#e24b4a"
GRN  = "#1d9e75"
AMB  = "#f0993b"


# ─────────────────────────────────────────────
#  Live data fetch  (cached 15 min)
# ─────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)   # refresh every 15 minutes
def fetch_live_data(ticker: str, fallback_spot: float,
                    fallback_drift: float, fallback_sigma: float):
    """
    Fetch latest price + calibrate drift & sigma from 1 year of history.
    Returns (spot, drift, sigma, prev_close, pct_change, history_closes, last_updated, source)
    Falls back to hard-coded defaults if yfinance fails.
    """
    try:
        import yfinance as yf
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1y")

        if hist.empty or len(hist) < 5:
            raise ValueError("Not enough data")

        closes      = hist["Close"].dropna()
        spot        = float(closes.iloc[-1])
        prev_close  = float(closes.iloc[-2])
        pct_chg     = (spot / prev_close - 1) * 100

        log_ret     = np.log(closes / closes.shift(1)).dropna()
        ann_sigma   = float(log_ret.std()  * np.sqrt(252))
        ann_drift   = float(log_ret.mean() * 252)

        last_updated = datetime.now().strftime("%H:%M:%S")
        return spot, ann_drift, ann_sigma, prev_close, pct_chg, closes.tolist(), last_updated, "live"

    except Exception:
        # silent fallback — app still works without internet
        return (fallback_spot, fallback_drift, fallback_sigma,
                fallback_spot, 0.0, [], "N/A (offline)", "fallback")


# ─────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_mc(spot, drift, sigma, jump_prob, jump_mean,
           strait_prob, mean_rev, strait_impact,
           n_paths, horizon, seed, symbol):
    """
    GBM + mean-reversion + Poisson jump process + strait closure shock.

    jump_prob : daily probability of a geopolitical jump event
    jump_mean : mean fractional price change on a jump (can be negative)
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 252.0
    S0  = spot

    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = S0
    S = np.full(n_paths, S0, dtype=float)
    strait_closed = np.zeros(n_paths, dtype=bool)

    for t in range(horizon):
        Z   = rng.standard_normal(n_paths)

        # ── GBM step ──────────────────────────────────────────
        dS  = S * ((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        # ── Mean reversion toward S0 ──────────────────────────
        dS += -mean_rev * (S - S0) * dt

        # ── Poisson jumps (geopolitical shocks) ───────────────
        # jump_prob is a daily probability; scale to dt
        if jump_prob > 0:
            jump_mask = rng.random(n_paths) < jump_prob * dt * 252
            # jump size ~ N(jump_mean, jump_mean/2) clipped to avoid absurdity
            jump_sizes = rng.normal(jump_mean, abs(jump_mean) * 0.5 + 0.01, n_paths)
            jump_sizes = np.clip(jump_sizes, -0.40, 0.60)
            dS += S * jump_sizes * jump_mask

        # ── One-time Strait of Hormuz closure ─────────────────
        new_close      = (~strait_closed) & (rng.random(n_paths) < strait_prob * dt * 252)
        strait_closed |= new_close
        dS            += S * (strait_impact + rng.random(n_paths) * 0.05) * new_close

        S = np.maximum(0.01, S + dS)
        paths[:, t + 1] = S

    return paths


def compute_stats(paths, spot):
    final = paths[:, -1]
    T     = paths.shape[1] - 1
    p5    = np.percentile(final, 5)
    tail  = final[final <= p5]
    return {
        "T": T, "spot": spot, "N": paths.shape[0],
        "mean":      final.mean(),
        "median":    np.median(final),
        "std":       final.std(),
        "p5":        p5,
        "p25":       np.percentile(final, 25),
        "p75":       np.percentile(final, 75),
        "p95":       np.percentile(final, 95),
        "var95":     p5,
        "cvar95":    tail.mean() if len(tail) else p5,
        "prob_up":   (final > spot).mean() * 100,
        "prob_up10": (final > spot * 1.10).mean() * 100,
        "prob_dn10": (final < spot * 0.90).mean() * 100,
        "exp_ret":   (final.mean() / spot - 1) * 100,
    }


# ─────────────────────────────────────────────
#  Chart helpers
# ─────────────────────────────────────────────

def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": GRID, "axes.labelcolor": MUT,
        "axes.titlecolor": TEXT, "xtick.color": MUT, "ytick.color": MUT,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "text.color": TEXT, "font.family": "monospace",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_history_and_paths(hist_closes, paths, spot, name, unit, dec, horizon):
    mpl_dark()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=BG,
                             gridspec_kw={"width_ratios": [1, 1.6]})

    # ── Left: historical prices ──
    ax0 = axes[0]
    if hist_closes:
        ax0.plot(hist_closes, color=BLUE, lw=1.5, alpha=0.9)
        ax0.axhline(spot, color=AMB, lw=1, ls="--", alpha=0.8)
        ax0.set_title("1-Year Historical Price", fontsize=10)
        ax0.set_xlabel("Trading days", fontsize=9)
        ax0.set_ylabel(f"Price ({unit})", fontsize=9)
        ax0.grid(True, alpha=0.3)
        # shade last point
        ax0.scatter(len(hist_closes)-1, spot, color=AMB, s=40, zorder=5)
        ax0.text(len(hist_closes)-1, spot, f"  {unit}{spot:,.{dec}f}",
                 color=AMB, fontsize=8, va="bottom")
    else:
        ax0.text(0.5, 0.5, "Live data\nunavailable\n(fallback mode)",
                 ha="center", va="center", transform=ax0.transAxes,
                 fontsize=10, color="#555")
        ax0.set_title("Historical Price", fontsize=10)

    # ── Right: simulated paths ──
    ax1 = axes[1]
    days = np.arange(horizon + 1)
    p5   = np.percentile(paths, 5,  axis=0)
    p25  = np.percentile(paths, 25, axis=0)
    med  = np.median(paths, axis=0)
    p75  = np.percentile(paths, 75, axis=0)
    p95  = np.percentile(paths, 95, axis=0)

    ax1.fill_between(days, p5,  p95, alpha=0.10, color=BLUE)
    ax1.fill_between(days, p25, p75, alpha=0.24, color=BLUE)
    ax1.plot(days, med, color=BLUE, lw=2.2)
    ax1.plot(days, p95, color=RED,  lw=1.2, ls="--", alpha=0.9)
    ax1.plot(days, p5,  color=GRN,  lw=1.2, ls="--", alpha=0.9)
    ax1.axhline(spot, color="#555", lw=1, ls=":", alpha=0.8)

    ax1.set_xlabel("Trading days (forecast)", fontsize=9)
    ax1.set_ylabel(f"Price ({unit})", fontsize=9)
    ax1.set_title(f"Monte Carlo — {paths.shape[0]:,} Paths", fontsize=10)
    ax1.grid(True, alpha=0.4)

    legend_els = [
        Line2D([0],[0], color=BLUE, lw=2,   label="Median"),
        Line2D([0],[0], color=RED,  lw=1.2, ls="--", label="95th pct"),
        Line2D([0],[0], color=GRN,  lw=1.2, ls="--", label="5th pct"),
        Patch(fc=BLUE, alpha=0.24, label="50% CI"),
        Patch(fc=BLUE, alpha=0.10, label="90% CI"),
    ]
    ax1.legend(handles=legend_els, fontsize=8, framealpha=0.3,
               facecolor=BG2, edgecolor=GRID, loc="upper left")

    fig.suptitle(name, fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


def plot_distribution(paths, spot, name, unit, dec, horizon):
    mpl_dark()
    final   = paths[:, -1]
    returns = (final / spot - 1) * 100

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG)
    counts, edges = np.histogram(returns, bins=55)
    centers = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]
    bar_colors = [RED if c < -10 else GRN if c > 10 else BLUE for c in centers]

    ax.bar(centers, counts, width=bw * 0.88, color=bar_colors, alpha=0.82, edgecolor="none")
    ax.axvline(0,                        color="#555", lw=1,   ls=":",  alpha=0.8)
    ax.axvline(np.median(returns),       color=BLUE,  lw=1.5, ls="--", alpha=0.9)
    ax.axvline(np.percentile(returns,5), color=RED,   lw=1.2, ls=":",  alpha=0.8)

    ax.legend(handles=[
        Patch(fc=RED,  alpha=0.82, label="< -10%  (tail risk)"),
        Patch(fc=BLUE, alpha=0.82, label="-10% to +10%  (neutral)"),
        Patch(fc=GRN,  alpha=0.82, label="> +10%  (upside)"),
    ], fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID, loc="upper left")

    ax.set_xlabel(f"Return at day {horizon} (%)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title(f"{name} — Return Distribution", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scenario_compare(com_name, spot, symbol,
                          strait_impact, n_paths, horizon, seed):
    mpl_dark()
    sc_colors = {
        "Base Case": BLUE, "Mild Escalation": AMB,
        "Strait Closure": RED, "De-escalation": GRN,
    }
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    for sc_name, sc in SCENARIOS.items():
        d = sc["drift"][symbol]
        s = sc["sigma"][symbol]
        p = run_mc(spot, d, s,
                   sc["jump_prob"], sc["jump_mean"],
                   sc["strait_prob"], sc["mean_rev"],
                   strait_impact, min(n_paths, 500), horizon, seed, symbol)
        ax.plot(np.arange(p.shape[1]), np.median(p, axis=0),
                lw=2, color=sc_colors[sc_name], label=sc_name)

    ax.axhline(spot, color="#555", lw=1, ls=":", alpha=0.8, label="Spot")
    ax.set_xlabel("Trading days", fontsize=9)
    ax.set_ylabel("Price ($)", fontsize=9)
    ax.set_title(f"{com_name} — All Scenarios (median paths)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## SIMULATION")
    st.markdown("---")

    commodity_name = st.selectbox("Commodity", list(COMMODITY_META.keys()), index=0)
    meta = COMMODITY_META[commodity_name]

    st.markdown("### Scenario")
    scenario_name = st.radio(
        "scenario_radio", list(SCENARIOS.keys()),
        format_func=lambda x: f"{SCENARIOS[x]['emoji']}  {x}",
        label_visibility="collapsed",
    )
    sc = SCENARIOS[scenario_name]

    st.markdown("---")
    st.markdown("### Simulation Params")
    n_paths      = st.slider("Paths (N)",       100, 2000,  800, step=100)
    horizon_days = st.slider("Horizon (days)",   30,  365,   90, step=5)
    seed         = st.slider("Random seed",       0,  999,   42, step=1)

    st.markdown("---")
    if st.button("🔄  Refresh live prices"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Override Market Params")
    st.caption("Auto-filled from scenario + live vol. Adjust if needed.")

    _sc_drift = sc["drift"][meta["symbol"]]
    _sc_sigma = sc["sigma"][meta["symbol"]]
    override_drift = st.slider("Annual drift (%)",      -30, 50,  int(_sc_drift * 100)) / 100.0
    override_sigma = st.slider("Annual volatility (%)",   5, 120, int(_sc_sigma * 100)) / 100.0

    st.markdown("---")
    st.markdown("### Geopolitical Shocks")
    jump_prob   = st.slider("Jump prob. (daily %)",     0.0, 10.0, sc["jump_prob"] * 100, step=0.1) / 100
    jump_mean   = st.slider("Jump mean size (%)",      -5.0, 20.0, sc["jump_mean"] * 100, step=0.5) / 100
    strait_prob = st.slider("Strait closure prob. (%)", 0.0, 20.0, sc["strait_prob"]* 100, step=0.5) / 100
    mean_rev    = st.slider("Mean reversion (k)",        0.0,  0.3, sc["mean_rev"],         step=0.01)

    strait_impact = STRAIT_IMPACT[meta["symbol"]]


# ─────────────────────────────────────────────
#  Fetch live data
# ─────────────────────────────────────────────

with st.spinner("Fetching live prices…"):
    spot, live_drift, live_sigma, prev_close, pct_chg, hist_closes, last_updated, data_source = \
        fetch_live_data(
            meta["ticker"],
            meta["fallback_spot"],
            meta["fallback_drift"],
            meta["fallback_sigma"],
        )

# Blend: scenario absolute params + override from sidebar
# (live drift used only if override wasn't touched, but sidebar always wins)
drift = override_drift
sigma = max(0.05, override_sigma)

unit = meta["unit"]
dec  = meta["decimals"]


# ─────────────────────────────────────────────
#  Run simulation
# ─────────────────────────────────────────────

paths = run_mc(
    spot=spot, drift=drift, sigma=sigma,
    jump_prob=jump_prob, jump_mean=jump_mean,
    strait_prob=strait_prob, mean_rev=mean_rev,
    strait_impact=strait_impact,
    n_paths=n_paths, horizon=horizon_days,
    seed=seed, symbol=meta["symbol"],
)

sim      = compute_stats(paths, spot)
spot_val = spot


# ─────────────────────────────────────────────
#  Hero banner
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <p class="hero-title">COMMODITY MONTE CARLO</p>
  <p class="hero-sub">Iran-Israel Conflict · Geopolitical Risk Simulation · GBM + Jump Process</p>
</div>
""", unsafe_allow_html=True)

# Live / fallback badge
if data_source == "live":
    st.markdown(f"""
    <div class="live-badge">
      <span class="live-dot"></span>
      LIVE PRICES &nbsp;·&nbsp; Last updated {last_updated} &nbsp;·&nbsp; Auto-refresh 15 min
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️  Could not reach Yahoo Finance — using fallback prices. Check internet connection.")

# Scenario badge
bc = sc["badge_class"]
st.markdown(f"""
<span class="scenario-badge {bc}">{sc['emoji']}  {scenario_name.upper()}</span>
<p style="color:#8b949e;font-size:13px;margin-top:-6px;font-family:'IBM Plex Mono',monospace;">
  {sc['description']}
</p>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Live price ticker row
# ─────────────────────────────────────────────

def fmt(v, d=None):
    d = d if d is not None else dec
    return f"{unit}{v:,.{d}f}"

chg_class = "price-chg-up" if pct_chg >= 0 else "price-chg-down"
chg_sign  = "+" if pct_chg >= 0 else ""

# Fetch all three for the ticker strip
@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_spots():
    results = {}
    try:
        import yfinance as yf
        for name, m in COMMODITY_META.items():
            hist = yf.Ticker(m["ticker"]).history(period="2d")
            if not hist.empty and len(hist) >= 2:
                c  = float(hist["Close"].iloc[-1])
                pc = float(hist["Close"].iloc[-2])
                results[name] = (c, (c/pc-1)*100, m["unit"], m["decimals"])
            else:
                results[name] = (m["fallback_spot"], 0.0, m["unit"], m["decimals"])
    except Exception:
        for name, m in COMMODITY_META.items():
            results[name] = (m["fallback_spot"], 0.0, m["unit"], m["decimals"])
    return results

all_spots = fetch_all_spots()
price_cards_html = ""
for cname, (cprice, cchg, cunit, cdec) in all_spots.items():
    cc = "price-chg-up" if cchg >= 0 else "price-chg-down"
    cs = "+" if cchg >= 0 else ""
    price_cards_html += f"""
    <div class="price-card">
      <div>
        <div class="price-name">{cname}</div>
        <div class="price-val">{cunit}{cprice:,.{cdec}f}</div>
      </div>
      <div class="{cc}">{cs}{cchg:.2f}%</div>
    </div>"""

st.markdown(f'<div class="price-row">{price_cards_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Simulation metric cards
# ─────────────────────────────────────────────

exp_ret    = sim["exp_ret"]
dcls       = "delta-up"   if exp_ret >= 0 else "delta-down"
sign       = "+"           if exp_ret >= 0 else ""
card_class = "up"          if exp_ret >= 0 else "down"

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card info">
    <div class="metric-label">LIVE SPOT</div>
    <div class="metric-val">{fmt(spot_val)}</div>
    <div class="metric-delta {chg_class}">{chg_sign}{pct_chg:.2f}% today</div>
  </div>
  <div class="metric-card {card_class}">
    <div class="metric-label">MEDIAN · DAY {sim['T']}</div>
    <div class="metric-val">{fmt(sim['median'])}</div>
    <div class="metric-delta {dcls}">{sign}{exp_ret:.1f}% vs spot</div>
  </div>
  <div class="metric-card up">
    <div class="metric-label">95TH PCT</div>
    <div class="metric-val">{fmt(sim['p95'])}</div>
    <div class="metric-delta delta-up">+{(sim['p95']/spot_val-1)*100:.1f}%</div>
  </div>
  <div class="metric-card down">
    <div class="metric-label">5TH PCT</div>
    <div class="metric-val">{fmt(sim['p5'])}</div>
    <div class="metric-delta delta-down">{(sim['p5']/spot_val-1)*100:.1f}%</div>
  </div>
  <div class="metric-card warn">
    <div class="metric-label">VAR 95%</div>
    <div class="metric-val">{fmt(abs(spot_val-sim['var95']))}</div>
    <div class="metric-delta delta-down">Potential loss</div>
  </div>
  <div class="metric-card info">
    <div class="metric-label">PROB. PRICE UP</div>
    <div class="metric-val">{sim['prob_up']:.1f}%</div>
    <div class="metric-delta delta-neu">{n_paths:,} paths</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  History + Paths",
    "📊  Return Distribution",
    "🔀  Scenario Comparison",
    "📋  Statistics",
])

with tab1:
    fig1 = plot_history_and_paths(hist_closes, paths, spot_val,
                                  commodity_name, unit, dec, horizon_days)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    if data_source == "live":
        st.caption(f"Left panel: 1-year real market data from Yahoo Finance ({meta['ticker']}). "
                   f"Right panel: {n_paths:,}-path Monte Carlo forecast.")

with tab2:
    fig2 = plot_distribution(paths, spot_val, commodity_name, unit, dec, horizon_days)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab3:
    fig3 = plot_scenario_compare(
        commodity_name, spot_val, meta["symbol"],
        strait_impact, n_paths, horizon_days, seed,
    )
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

with tab4:
    st.markdown('<p class="section-hdr">Simulation Statistics</p>', unsafe_allow_html=True)

    def tbl_color(v, threshold=50, good="up"):
        if v > threshold: return "tbl-up" if good == "up" else "tbl-down"
        if v < 20:        return "tbl-down" if good == "up" else "tbl-up"
        return "tbl-neu"

    stat_rows = [
        ("Paths simulated",         f"{sim['N']:,}",                             "tbl-neu"),
        ("Horizon",                 f"{sim['T']} trading days",                  "tbl-neu"),
        ("Live spot price",         fmt(sim['spot']),                             "tbl-neu"),
        ("Today's change",          f"{chg_sign}{pct_chg:.2f}%",                "tbl-up" if pct_chg>=0 else "tbl-down"),
        ("Mean final price",        fmt(sim['mean']),                             "tbl-up" if sim['mean']>spot_val else "tbl-down"),
        ("Median final price",      fmt(sim['median']),                           "tbl-up" if sim['median']>spot_val else "tbl-down"),
        ("Std deviation",           fmt(sim['std']),                              "tbl-neu"),
        ("5th percentile",          fmt(sim['p5']),                               "tbl-down"),
        ("25th percentile",         fmt(sim['p25']),                              "tbl-neu"),
        ("75th percentile",         fmt(sim['p75']),                              "tbl-neu"),
        ("95th percentile",         fmt(sim['p95']),                              "tbl-up"),
        ("VaR 95% (loss)",          fmt(abs(spot_val-sim['var95'])),              "tbl-down"),
        ("CVaR 95%",                fmt(sim['cvar95']),                           "tbl-down"),
        ("Expected return",         f"{sim['exp_ret']:+.2f}%",                   "tbl-up" if sim['exp_ret']>=0 else "tbl-down"),
        ("Probability price rises", f"{sim['prob_up']:.1f}%",                    tbl_color(sim['prob_up'])),
        ("Probability +10% spike",  f"{sim['prob_up10']:.1f}%",                  "tbl-neu"),
        ("Probability -10% drop",   f"{sim['prob_dn10']:.1f}%",                  tbl_color(sim['prob_dn10'], good="down")),
    ]
    rows_html = "".join(f'<tr><td>{l}</td><td class="{c}">{v}</td></tr>' for l,v,c in stat_rows)
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>METRIC</th><th>VALUE</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Calibrated Model Parameters</p>', unsafe_allow_html=True)
    param_rows = [
        ("Data source",              "Yahoo Finance (yfinance)" if data_source=="live" else "Fallback (offline)"),
        ("Ticker",                   meta["ticker"]),
        ("Last updated",             last_updated),
        ("Live annual drift (mu)",   f"{live_drift*100:+.1f}%"),
        ("Live annual vol (sigma)",  f"{live_sigma*100:.1f}%"),
        ("Applied drift",            f"{drift*100:+.1f}%  (scenario override)"),
        ("Applied sigma",            f"{sigma*100:.1f}%"),
        ("Mean reversion (kappa)",   f"{mean_rev:.3f}"),
        ("Jump prob. (daily)",       f"{jump_prob*100:.2f}%"),
        ("Jump mean size",           f"{jump_mean*100:+.1f}%"),
        ("Strait closure prob.",     f"{strait_prob*100:.2f}% / day"),
        ("Strait price impact",      f"{strait_impact*100:.0f}%"),
    ]
    params_html = "".join(f'<tr><td>{l}</td><td class="tbl-neu">{v}</td></tr>' for l,v in param_rows)
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>PARAMETER</th><th>VALUE</th></tr></thead>
      <tbody>{params_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────

st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #21262d;
            text-align:center;font-family:'IBM Plex Mono',monospace;
            font-size:11px;color:#484f58;letter-spacing:0.5px;">
  LIVE DATA · YAHOO FINANCE · GBM + POISSON JUMP PROCESS · FOR EDUCATIONAL PURPOSES ONLY
</div>
""", unsafe_allow_html=True)
