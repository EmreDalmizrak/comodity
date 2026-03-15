"""
Commodity Monte Carlo Simulation — Iran-Israel Conflict Risk
Streamlit App
Run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import streamlit as st
from dataclasses import dataclass
from typing import Optional

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
#  Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #0d1117; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #e24b4a, #f0993b, #185fa5, #1d9e75);
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.85rem;
    color: #8b949e;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
}
.metric-card.up   { border-top: 2px solid #1d9e75; }
.metric-card.down { border-top: 2px solid #e24b4a; }
.metric-card.info { border-top: 2px solid #185fa5; }
.metric-card.warn { border-top: 2px solid #f0993b; }
.metric-label { font-size: 11px; color: #8b949e; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; margin-bottom: 4px; }
.metric-val   { font-size: 1.4rem; font-weight: 600; color: #e6edf3; font-family: 'IBM Plex Mono', monospace; }
.metric-delta { font-size: 11px; margin-top: 2px; font-family: 'IBM Plex Mono', monospace; }
.delta-up   { color: #1d9e75; }
.delta-down { color: #e24b4a; }
.delta-neu  { color: #8b949e; }

/* Scenario badge */
.scenario-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}
.sc-base     { background:#1c2e3a; color:#5ea8e0; border:1px solid #185fa5; }
.sc-mild     { background:#2e2418; color:#e0a84a; border:1px solid #ba7517; }
.sc-strait   { background:#2e1a1a; color:#e07070; border:1px solid #a32d2d; }
.sc-deesc    { background:#1a2e22; color:#5ecc8e; border:1px solid #1d9e75; }

/* Section header */
.section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #8b949e;
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
}

/* Stats table */
.stats-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.stats-tbl th { text-align:left; padding:7px 10px; color:#8b949e; font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:1px; border-bottom:1px solid #21262d; font-weight:500; }
.stats-tbl td { padding:7px 10px; color:#e6edf3; border-bottom:1px solid #161b22; font-family:'IBM Plex Mono',monospace; font-size:12px; }
.stats-tbl tr:hover td { background:#161b22; }
.tbl-up   { color: #1d9e75 !important; }
.tbl-down { color: #e24b4a !important; }
.tbl-neu  { color: #8b949e !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] .stSlider label { color: #8b949e !important; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
section[data-testid="stSidebar"] h2 { color: #e6edf3; font-family: 'IBM Plex Mono', monospace; font-size: 14px; letter-spacing: 1px; }

/* Streamlit overrides */
.stButton>button {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    width: 100%;
    transition: all 0.15s;
}
.stButton>button:hover { border-color: #58a6ff !important; color: #58a6ff !important; }

div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Data & Model
# ─────────────────────────────────────────────

COMMODITIES = {
    "Crude Oil (WTI)": {"symbol": "OIL",  "spot": 85.00,  "unit": "$", "decimals": 2, "default_drift": 0.05,  "default_sigma": 0.28},
    "Natural Gas":     {"symbol": "GAS",  "spot": 2.80,   "unit": "$", "decimals": 3, "default_drift": 0.02,  "default_sigma": 0.55},
    "Gold":            {"symbol": "GOLD", "spot": 2340.0, "unit": "$", "decimals": 0, "default_drift": 0.06,  "default_sigma": 0.15},
}

SCENARIOS = {
    "Base Case": {
        "esc_prob": 0.03, "shock_mag": 0.08, "strait_prob": 0.02, "mean_rev": 0.05,
        "drift_adj": 0.0, "sigma_adj": 0.0,
        "badge_class": "sc-base", "emoji": "🔵",
        "description": "Low-level tension, current market dynamics prevail.",
    },
    "Mild Escalation": {
        "esc_prob": 0.08, "shock_mag": 0.15, "strait_prob": 0.05, "mean_rev": 0.04,
        "drift_adj": 0.03, "sigma_adj": 0.10,
        "badge_class": "sc-mild", "emoji": "🟡",
        "description": "Increased airstrikes, regional risk premium builds.",
    },
    "Strait Closure": {
        "esc_prob": 0.15, "shock_mag": 0.28, "strait_prob": 0.15, "mean_rev": 0.02,
        "drift_adj": 0.10, "sigma_adj": 0.25,
        "badge_class": "sc-strait", "emoji": "🔴",
        "description": "Hormuz blocked — severe supply disruption scenario.",
    },
    "De-escalation": {
        "esc_prob": 0.01, "shock_mag": 0.04, "strait_prob": 0.005, "mean_rev": 0.12,
        "drift_adj": -0.05, "sigma_adj": -0.08,
        "badge_class": "sc-deesc", "emoji": "🟢",
        "description": "Ceasefire / diplomacy — risk premium deflates.",
    },
}

STRAIT_IMPACT = {"OIL": 0.20, "GAS": 0.25, "GOLD": 0.08}


def randn_batch(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng, rng


@st.cache_data(show_spinner=False)
def run_mc(
    spot: float, drift: float, sigma: float,
    esc_prob: float, shock_mag: float, strait_prob: float, mean_rev: float,
    strait_impact: float,
    n_paths: int, horizon: int, seed: int, symbol: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    S0 = spot

    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = S0
    S = np.full(n_paths, S0, dtype=float)
    strait_closed = np.zeros(n_paths, dtype=bool)

    for t in range(horizon):
        Z = rng.standard_normal(n_paths)
        dS  = S * ((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        dS += -mean_rev * (S - S0) * dt

        esc_mask = rng.random(n_paths) < esc_prob * dt
        dS += S * shock_mag * (0.5 + rng.random(n_paths)) * esc_mask

        new_close = (~strait_closed) & (rng.random(n_paths) < strait_prob * dt)
        strait_closed |= new_close
        dS += S * (strait_impact + rng.random(n_paths) * 0.05) * new_close

        S = np.maximum(0.01, S + dS)
        paths[:, t + 1] = S

    return paths


def compute_stats(paths, spot):
    final = paths[:, -1]
    T = paths.shape[1] - 1
    p5  = np.percentile(final, 5)
    tail = final[final <= p5]
    return {
        "T": T, "spot": spot, "N": paths.shape[0],
        "mean":   final.mean(),
        "median": np.median(final),
        "std":    final.std(),
        "p5":     p5,
        "p25":    np.percentile(final, 25),
        "p75":    np.percentile(final, 75),
        "p95":    np.percentile(final, 95),
        "var95":  p5,
        "cvar95": tail.mean() if len(tail) else p5,
        "prob_up":    (final > spot).mean() * 100,
        "prob_up10":  (final > spot * 1.10).mean() * 100,
        "prob_dn10":  (final < spot * 0.90).mean() * 100,
        "exp_ret":    (final.mean() / spot - 1) * 100,
    }


# ─────────────────────────────────────────────
#  Matplotlib dark style
# ─────────────────────────────────────────────

BG   = "#0d1117"
BG2  = "#161b22"
GRID = "#21262d"
TEXT = "#e6edf3"
MUT  = "#8b949e"
BLUE = "#185FA5"
RED  = "#e24b4a"
GRN  = "#1d9e75"
AMB  = "#f0993b"

def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG2,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   MUT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUT,
        "ytick.color":       MUT,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "text.color":        TEXT,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plot_paths(paths, spot, name, unit, decimals, horizon):
    mpl_dark()
    days = np.arange(horizon + 1)
    p5   = np.percentile(paths, 5,  axis=0)
    p25  = np.percentile(paths, 25, axis=0)
    med  = np.median(paths, axis=0)
    p75  = np.percentile(paths, 75, axis=0)
    p95  = np.percentile(paths, 95, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.fill_between(days, p5,  p95, alpha=0.10, color=BLUE)
    ax.fill_between(days, p25, p75, alpha=0.24, color=BLUE)
    ax.plot(days, med, color=BLUE,  lw=2.2, label="Median")
    ax.plot(days, p95, color=RED,   lw=1.2, ls="--", alpha=0.9, label="95th pct")
    ax.plot(days, p5,  color=GRN,   lw=1.2, ls="--", alpha=0.9, label="5th pct")
    ax.axhline(spot, color="#555", lw=1, ls=":", alpha=0.8, label=f"Spot {unit}{spot:,.{decimals}f}")

    ax.set_xlabel("Trading days", fontsize=9)
    ax.set_ylabel(f"Price ({unit})", fontsize=9)
    ax.set_title(f"{name} — Simulated Price Paths ({paths.shape[0]:,} paths)", fontsize=11)
    ax.grid(True, alpha=0.4)

    legend_els = [
        Line2D([0],[0], color=BLUE, lw=2, label="Median"),
        Line2D([0],[0], color=RED,  lw=1.2, ls="--", label="95th pct"),
        Line2D([0],[0], color=GRN,  lw=1.2, ls="--", label="5th pct"),
        plt.Rectangle((0,0),1,1, fc=BLUE, alpha=0.24, label="50% CI"),
        plt.Rectangle((0,0),1,1, fc=BLUE, alpha=0.10, label="90% CI"),
    ]
    ax.legend(handles=legend_els, fontsize=8, framealpha=0.3,
              facecolor=BG2, edgecolor=GRID, loc="upper left")
    fig.tight_layout()
    return fig


def plot_distribution(paths, spot, name, unit, decimals, horizon):
    mpl_dark()
    final   = paths[:, -1]
    returns = (final / spot - 1) * 100

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG)
    n_bins = 55
    counts, edges = np.histogram(returns, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]

    colors = [RED if c < -10 else GRN if c > 10 else BLUE for c in centers]
    ax.bar(centers, counts, width=bw * 0.88, color=colors, alpha=0.82, edgecolor="none")
    ax.axvline(0,                  color="#555", lw=1,   ls=":",  alpha=0.8)
    ax.axvline(np.median(returns), color=BLUE,  lw=1.5, ls="--", alpha=0.9, label="Median return")
    ax.axvline(np.percentile(returns, 5), color=RED, lw=1.2, ls=":", alpha=0.8, label="VaR 95%")

    ax.set_xlabel(f"Return at day {horizon} (%)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title(f"{name} — Return Distribution", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID)

    # colour legend
    from matplotlib.patches import Patch
    patches = [
        Patch(fc=RED,  alpha=0.82, label="< −10%  (tail risk)"),
        Patch(fc=BLUE, alpha=0.82, label="−10% to +10%  (neutral)"),
        Patch(fc=GRN,  alpha=0.82, label="> +10%  (upside)"),
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.3,
              facecolor=BG2, edgecolor=GRID, loc="upper left")

    fig.tight_layout()
    return fig


def plot_scenario_compare(commodity_name, spot, drift, sigma, strait_impact,
                          n_paths, horizon, seed, symbol):
    mpl_dark()
    sc_colors = {
        "Base Case":       BLUE,
        "Mild Escalation": AMB,
        "Strait Closure":  RED,
        "De-escalation":   GRN,
    }
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    for sc_name, sc in SCENARIOS.items():
        d = drift + sc["drift_adj"]
        s = max(0.05, sigma + sc["sigma_adj"])
        paths = run_mc(
            spot, d, s,
            sc["esc_prob"], sc["shock_mag"], sc["strait_prob"], sc["mean_rev"],
            strait_impact, n_paths, horizon, seed, symbol,
        )
        med  = np.median(paths, axis=0)
        days = np.arange(len(med))
        ax.plot(days, med, lw=2, color=sc_colors[sc_name], label=sc_name)

    ax.axhline(spot, color="#555", lw=1, ls=":", alpha=0.8, label="Spot")
    ax.set_xlabel("Trading days", fontsize=9)
    ax.set_ylabel("Price ($)", fontsize=9)
    ax.set_title(f"{commodity_name} — All Scenarios (median paths)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ SIMULATION")
    st.markdown("---")

    # Commodity
    commodity_name = st.selectbox("Commodity", list(COMMODITIES.keys()), index=0)
    com = COMMODITIES[commodity_name]

    # Scenario
    st.markdown("### Scenario")
    scenario_name = st.radio(
        "label",
        list(SCENARIOS.keys()),
        format_func=lambda x: f"{SCENARIOS[x]['emoji']}  {x}",
        label_visibility="collapsed",
    )
    sc = SCENARIOS[scenario_name]

    st.markdown("---")
    st.markdown("### Simulation Params")
    n_paths      = st.slider("Paths (N)",         100,  2000, 800,  step=100)
    horizon_days = st.slider("Horizon (days)",     30,   365,  90,   step=5)
    seed         = st.slider("Random seed",         0,   999,  42,   step=1)

    st.markdown("---")
    st.markdown("### Market Params")
    drift = st.slider("Annual drift (%)",      -30, 50,
                      int((com["default_drift"] + sc["drift_adj"]) * 100)) / 100.0
    sigma = st.slider("Annual volatility (%)",   5, 100,
                      int(max(0.05, com["default_sigma"] + sc["sigma_adj"]) * 100)) / 100.0

    st.markdown("---")
    st.markdown("### Geopolitical Shocks")
    esc_prob   = st.slider("Escalation prob. (daily %)", 0.0, 30.0, sc["esc_prob"] * 100,  step=0.5) / 100
    shock_mag  = st.slider("Shock magnitude (%)",         0.0, 50.0, sc["shock_mag"] * 100, step=0.5) / 100
    strait_prob = st.slider("Strait closure prob. (%)",   0.0, 20.0, sc["strait_prob"]* 100,step=0.5) / 100
    mean_rev   = st.slider("Mean reversion (κ)",          0.0,  0.3, sc["mean_rev"],         step=0.01)

    strait_impact = STRAIT_IMPACT[com["symbol"]]
    run_btn = st.button("▶  RUN SIMULATION", use_container_width=True)


# ─────────────────────────────────────────────
#  Main layout
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="hero-banner">
  <p class="hero-title">📊 COMMODITY MONTE CARLO</p>
  <p class="hero-sub">Iran-Israel Conflict · Geopolitical Risk Simulation · GBM + Jump Process</p>
</div>
""", unsafe_allow_html=True)

# Scenario badge + description
bc = sc["badge_class"]
st.markdown(f"""
<span class="scenario-badge {bc}">{sc['emoji']}  {scenario_name.upper()}</span>
<p style="color:#8b949e; font-size:13px; margin-top:-6px; font-family:'IBM Plex Mono',monospace;">
  {sc['description']}
</p>
""", unsafe_allow_html=True)

# ─── Run simulation ───────────────────────────
paths = run_mc(
    spot=com["spot"],
    drift=drift,
    sigma=sigma,
    esc_prob=esc_prob,
    shock_mag=shock_mag,
    strait_prob=strait_prob,
    mean_rev=mean_rev,
    strait_impact=strait_impact,
    n_paths=n_paths,
    horizon=horizon_days,
    seed=seed,
    symbol=com["symbol"],
)
st = compute_stats(paths, com["spot"])
spot   = com["spot"]
unit   = com["unit"]
dec    = com["decimals"]

# ─── Metric cards ────────────────────────────
exp_ret = st["exp_ret"]
delta_class = "delta-up" if exp_ret >= 0 else "delta-down"
sign = "+" if exp_ret >= 0 else ""

def fmt(v): return f"{unit}{v:,.{dec}f}"

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card info">
    <div class="metric-label">SPOT PRICE</div>
    <div class="metric-val">{fmt(spot)}</div>
    <div class="metric-delta delta-neu">{commodity_name}</div>
  </div>
  <div class="metric-card {'up' if exp_ret >= 0 else 'down'}">
    <div class="metric-label">MEDIAN · DAY {st['T']}</div>
    <div class="metric-val">{fmt(st['median'])}</div>
    <div class="metric-delta {delta_class}">{sign}{exp_ret:.1f}% vs spot</div>
  </div>
  <div class="metric-card up">
    <div class="metric-label">95TH PCT</div>
    <div class="metric-val">{fmt(st['p95'])}</div>
    <div class="metric-delta delta-up">+{(st['p95']/spot-1)*100:.1f}%</div>
  </div>
  <div class="metric-card down">
    <div class="metric-label">5TH PCT</div>
    <div class="metric-val">{fmt(st['p5'])}</div>
    <div class="metric-delta delta-down">{(st['p5']/spot-1)*100:.1f}%</div>
  </div>
  <div class="metric-card warn">
    <div class="metric-label">VAR 95%</div>
    <div class="metric-val">{fmt(abs(spot - st['var95']))}</div>
    <div class="metric-delta delta-down">Potential loss</div>
  </div>
  <div class="metric-card info">
    <div class="metric-label">PROB. PRICE ↑</div>
    <div class="metric-val">{st['prob_up']:.1f}%</div>
    <div class="metric-delta delta-neu">{n_paths:,} paths</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Price Paths",
    "📊  Return Distribution",
    "🔀  Scenario Comparison",
    "📋  Statistics",
])

with tab1:
    fig1 = plot_paths(paths, spot, commodity_name, unit, dec, horizon_days)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with tab2:
    fig2 = plot_distribution(paths, spot, commodity_name, unit, dec, horizon_days)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab3:
    fig3 = plot_scenario_compare(
        commodity_name, spot, com["default_drift"], com["default_sigma"],
        strait_impact, min(n_paths, 500), horizon_days, seed, com["symbol"],
    )
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

with tab4:
    st.markdown('<p class="section-hdr">Simulation Statistics</p>', unsafe_allow_html=True)

    def pct_color(v, good="up"):
        if v > 50: return "tbl-up" if good == "up" else "tbl-down"
        if v < 20: return "tbl-down" if good == "up" else "tbl-up"
        return "tbl-neu"

    rows = [
        ("Paths simulated",        f"{st['N']:,}",                          "tbl-neu"),
        ("Horizon",                f"{st['T']} trading days",               "tbl-neu"),
        ("Spot price today",       fmt(st['spot']),                         "tbl-neu"),
        ("Mean final price",       fmt(st['mean']),                         "tbl-up" if st['mean'] > spot else "tbl-down"),
        ("Median final price",     fmt(st['median']),                       "tbl-up" if st['median'] > spot else "tbl-down"),
        ("Std deviation",          fmt(st['std']),                          "tbl-neu"),
        ("5th percentile",         fmt(st['p5']),                           "tbl-down"),
        ("25th percentile",        fmt(st['p25']),                          "tbl-neu"),
        ("75th percentile",        fmt(st['p75']),                          "tbl-neu"),
        ("95th percentile",        fmt(st['p95']),                          "tbl-up"),
        ("VaR 95% (loss)",         fmt(abs(spot - st['var95'])),            "tbl-down"),
        ("CVaR 95%",               fmt(st['cvar95']),                       "tbl-down"),
        ("Expected return",        f"{st['exp_ret']:+.2f}%",               "tbl-up" if st['exp_ret'] >= 0 else "tbl-down"),
        ("Probability price rises",f"{st['prob_up']:.1f}%",                pct_color(st['prob_up'])),
        ("Probability +10% spike", f"{st['prob_up10']:.1f}%",              "tbl-neu"),
        ("Probability −10% drop",  f"{st['prob_dn10']:.1f}%",              pct_color(st['prob_dn10'], good="down")),
    ]
    rows_html = "".join(
        f'<tr><td>{lbl}</td><td class="{cls}">{val}</td></tr>'
        for lbl, val, cls in rows
    )
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>METRIC</th><th>VALUE</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Model Parameters</p>', unsafe_allow_html=True)
    params_html = "".join(
        f'<tr><td>{lbl}</td><td class="tbl-neu">{val}</td></tr>'
        for lbl, val in [
            ("Annual drift (μ)",          f"{drift*100:+.1f}%"),
            ("Annual volatility (σ)",     f"{sigma*100:.1f}%"),
            ("Mean reversion (κ)",        f"{mean_rev:.3f}"),
            ("Escalation prob. (daily)",  f"{esc_prob*100:.2f}%"),
            ("Shock magnitude",           f"{shock_mag*100:.1f}%"),
            ("Strait closure prob.",      f"{strait_prob*100:.2f}%"),
            ("Strait price impact",       f"{strait_impact*100:.0f}%"),
        ]
    )
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>PARAMETER</th><th>VALUE</th></tr></thead>
      <tbody>{params_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────
st.markdown("""
<div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #21262d;
            text-align:center; font-family:'IBM Plex Mono',monospace;
            font-size:11px; color:#484f58; letter-spacing:0.5px;">
  GBM + POISSON JUMP PROCESS · STRAIT OF HORMUZ CLOSURE OVERLAY · FOR EDUCATIONAL PURPOSES ONLY
</div>
""", unsafe_allow_html=True)
