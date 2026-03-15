"""
Commodity Monte Carlo — Iran-Israel Conflict Risk
Streamlit App  |  Live prices · Multiple models · Statistical tests
Run: streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from scipy.stats import norm, jarque_bera, kstest, shapiro
import streamlit as st
from datetime import datetime

# ══════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Commodity Monte Carlo — Iran-Israel",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.hero-banner{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #30363d;border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#e24b4a,#f0993b,#185fa5,#1d9e75);}
.hero-title{font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:600;color:#e6edf3;margin:0 0 .3rem;}
.hero-sub{font-size:.85rem;color:#8b949e;margin:0;font-family:'IBM Plex Mono',monospace;}
.live-badge{display:inline-flex;align-items:center;gap:6px;background:#1a2e22;border:1px solid #1d9e75;border-radius:99px;padding:4px 12px;font-size:11px;font-family:'IBM Plex Mono',monospace;color:#5ecc8e;letter-spacing:.5px;margin-bottom:1rem;}
.live-dot{width:7px;height:7px;background:#1d9e75;border-radius:50%;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.metric-row{display:flex;gap:12px;margin-bottom:1.2rem;flex-wrap:wrap;}
.metric-card{flex:1;min-width:130px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px;}
.metric-card.up{border-top:2px solid #1d9e75;}.metric-card.down{border-top:2px solid #e24b4a;}.metric-card.info{border-top:2px solid #185fa5;}.metric-card.warn{border-top:2px solid #f0993b;}
.metric-label{font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;letter-spacing:.5px;margin-bottom:4px;}
.metric-val{font-size:1.4rem;font-weight:600;color:#e6edf3;font-family:'IBM Plex Mono',monospace;}
.metric-delta{font-size:11px;margin-top:2px;font-family:'IBM Plex Mono',monospace;}
.delta-up{color:#1d9e75;}.delta-down{color:#e24b4a;}.delta-neu{color:#8b949e;}
.scenario-badge{display:inline-block;padding:4px 12px;border-radius:99px;font-size:12px;font-family:'IBM Plex Mono',monospace;font-weight:600;letter-spacing:.5px;margin-bottom:1rem;}
.sc-base{background:#1c2e3a;color:#5ea8e0;border:1px solid #185fa5;}
.sc-mild{background:#2e2418;color:#e0a84a;border:1px solid #ba7517;}
.sc-strait{background:#2e1a1a;color:#e07070;border:1px solid #a32d2d;}
.sc-deesc{background:#1a2e22;color:#5ecc8e;border:1px solid #1d9e75;}
.section-hdr{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:2px;color:#8b949e;text-transform:uppercase;margin:1.2rem 0 .6rem;border-bottom:1px solid #21262d;padding-bottom:6px;}
.stats-tbl{width:100%;border-collapse:collapse;font-size:13px;}
.stats-tbl th{text-align:left;padding:7px 10px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:1px;border-bottom:1px solid #21262d;font-weight:500;}
.stats-tbl td{padding:7px 10px;color:#e6edf3;border-bottom:1px solid #161b22;font-family:'IBM Plex Mono',monospace;font-size:12px;}
.tbl-up{color:#1d9e75!important;}.tbl-down{color:#e24b4a!important;}.tbl-neu{color:#8b949e!important;}
.test-pass{color:#1d9e75;font-weight:600;}.test-fail{color:#e24b4a;font-weight:600;}.test-warn{color:#f0993b;font-weight:600;}
.model-badge{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;font-family:'IBM Plex Mono',monospace;font-weight:600;letter-spacing:.5px;margin-left:8px;background:#1c2e3a;color:#5ea8e0;border:1px solid #185fa5;}
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
.stButton>button{background:#161b22!important;border:1px solid #30363d!important;color:#e6edf3!important;font-family:'IBM Plex Mono',monospace!important;font-size:13px!important;border-radius:6px!important;width:100%;}
.stButton>button:hover{border-color:#58a6ff!important;color:#58a6ff!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  Constants & Scenarios
# ══════════════════════════════════════════════

COMMODITY_META = {
    "Crude Oil (WTI)": {
        "symbol":"OIL","ticker":"CL=F","unit":"$","decimals":2,
        "fallback_spot":83.00,"fallback_drift":0.04,"fallback_sigma":0.27,
    },
    "Natural Gas": {
        "symbol":"GAS","ticker":"NG=F","unit":"$","decimals":3,
        "fallback_spot":2.75,"fallback_drift":0.01,"fallback_sigma":0.52,
    },
    "Gold": {
        "symbol":"GOLD","ticker":"GC=F","unit":"$","decimals":0,
        "fallback_spot":2320.0,"fallback_drift":0.07,"fallback_sigma":0.14,
    },
}

# Absolute scenario params — guarantees Strait > Mild > Base > De-esc
SCENARIOS = {
    "Base Case": {
        "drift":{"OIL":0.03,"GAS":0.03,"GOLD":0.03},
        "sigma":{"OIL":0.23,"GAS":0.30,"GOLD":0.23},
        "jump_prob":0.01,"jump_mean":0.02,"jump_std":0.02,
        "strait_prob":0.005,"mean_rev":0.05,
        # Heston stochastic vol params
        "heston_kappa":2.0,"heston_theta":0.053,"heston_xi":0.30,"heston_rho":-0.65,
        "badge_class":"sc-base","emoji":"🔵",
        "description":"Low-level tension, current market dynamics prevail.",
    },
    "Mild Escalation": {
        "drift":{"OIL":0.06,"GAS":0.06,"GOLD":0.06},
        "sigma":{"OIL":0.30,"GAS":0.40,"GOLD":0.30},
        "jump_prob":0.03,"jump_mean":0.05,"jump_std":0.03,
        "strait_prob":0.02,"mean_rev":0.04,
        "heston_kappa":1.5,"heston_theta":0.090,"heston_xi":0.40,"heston_rho":-0.60,
        "badge_class":"sc-mild","emoji":"🟡",
        "description":"Increased airstrikes, regional risk premium builds.",
    },
    "Strait Closure": {
        "drift":{"OIL":0.12,"GAS":0.12,"GOLD":0.12},
        "sigma":{"OIL":0.45,"GAS":0.60,"GOLD":0.45},
        "jump_prob":0.06,"jump_mean":0.12,"jump_std":0.05,
        "strait_prob":0.15,"mean_rev":0.02,
        "heston_kappa":1.0,"heston_theta":0.202,"heston_xi":0.60,"heston_rho":-0.50,
        "badge_class":"sc-strait","emoji":"🔴",
        "description":"Hormuz blocked — severe supply disruption scenario.",
    },
    "De-escalation": {
        "drift":{"OIL":-0.02,"GAS":-0.02,"GOLD":-0.02},
        "sigma":{"OIL":0.18,"GAS":0.22,"GOLD":0.18},
        "jump_prob":0.00,"jump_mean":-0.01,"jump_std":0.01,
        "strait_prob":0.001,"mean_rev":0.12,
        "heston_kappa":3.0,"heston_theta":0.032,"heston_xi":0.20,"heston_rho":-0.70,
        "badge_class":"sc-deesc","emoji":"🟢",
        "description":"Ceasefire / diplomacy — risk premium deflates.",
    },
}

STRAIT_IMPACT = {"OIL":0.25,"GAS":0.35,"GOLD":0.08}

BG="#0d1117"; BG2="#161b22"; GRID="#21262d"; TEXT="#e6edf3"
MUT="#8b949e"; BLUE="#185FA5"; RED="#e24b4a"; GRN="#1d9e75"; AMB="#f0993b"

# ══════════════════════════════════════════════
#  Live data
# ══════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_data(ticker, fallback_spot, fallback_drift, fallback_sigma):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty or len(hist) < 20:
            raise ValueError("insufficient data")
        closes    = hist["Close"].dropna()
        spot      = float(closes.iloc[-1])
        prev      = float(closes.iloc[-2])
        pct_chg   = (spot/prev - 1)*100
        log_ret   = np.log(closes/closes.shift(1)).dropna()
        ann_sigma = float(log_ret.std()*np.sqrt(252))
        ann_drift = float(log_ret.mean()*252)
        ts = datetime.now().strftime("%H:%M:%S")
        return spot, ann_drift, ann_sigma, prev, pct_chg, closes.values.tolist(), ts, "live", log_ret.values.tolist()
    except Exception:
        return fallback_spot, fallback_drift, fallback_sigma, fallback_spot, 0.0, [], "N/A", "fallback", []

# ══════════════════════════════════════════════
#  Model 1 — GBM + Merton Jump-Diffusion
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_gbm_jump(spot, drift, sigma, jump_prob, jump_mean, jump_std,
                 strait_prob, mean_rev, strait_impact,
                 n_paths, horizon, seed, symbol):
    """
    Merton (1976) Jump-Diffusion:
      dS/S = (μ - λk) dt + σ dW + J dN
      J ~ LogNormal(jump_mean, jump_std)
      N ~ Poisson(λ dt)
    Plus mean-reversion and strait closure shock.
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0/252.0
    S0  = spot
    lam = jump_prob          # daily jump arrival rate
    k   = np.exp(jump_mean + 0.5*jump_std**2) - 1   # mean jump size

    paths = np.zeros((n_paths, horizon+1))
    paths[:,0] = S0
    S = np.full(n_paths, S0, dtype=float)
    strait_closed = np.zeros(n_paths, dtype=bool)

    for t in range(horizon):
        Z = rng.standard_normal(n_paths)

        # Compensated GBM drift
        comp_drift = drift - lam*k - 0.5*sigma**2
        dS = S*(comp_drift*dt + sigma*np.sqrt(dt)*Z)

        # Mean reversion
        dS += -mean_rev*(S - S0)*dt

        # Merton jumps — Poisson arrivals
        n_jumps = rng.poisson(lam*dt*252, n_paths)   # number of jumps this step
        jump_sizes = np.zeros(n_paths)
        for i in np.where(n_jumps > 0)[0]:
            j = rng.lognormal(jump_mean, jump_std, int(n_jumps[i]))
            jump_sizes[i] = np.prod(j) - 1.0         # compound jump return
        dS += S*jump_sizes

        # Strait closure
        new_close = (~strait_closed) & (rng.random(n_paths) < strait_prob*dt*252)
        strait_closed |= new_close
        dS += S*(strait_impact + rng.random(n_paths)*0.05)*new_close

        S = np.maximum(0.01, S + dS)
        paths[:,t+1] = S

    return paths

# ══════════════════════════════════════════════
#  Model 2 — Heston Stochastic Volatility
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_heston(spot, drift, sigma, kappa, theta, xi, rho,
               jump_prob, jump_mean, jump_std,
               strait_prob, strait_impact,
               n_paths, horizon, seed, symbol):
    """
    Heston (1993) stochastic volatility model:
      dS = μ S dt + √V S dW₁
      dV = κ(θ - V) dt + ξ √V dW₂
      corr(dW₁, dW₂) = ρ
    Plus jump overlay and strait closure.
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0/252.0
    S0  = spot
    V0  = sigma**2   # initial variance

    paths = np.zeros((n_paths, horizon+1))
    paths[:,0] = S0
    S = np.full(n_paths, S0, dtype=float)
    V = np.full(n_paths, V0, dtype=float)
    strait_closed = np.zeros(n_paths, dtype=bool)

    lam = jump_prob
    k   = np.exp(jump_mean + 0.5*jump_std**2) - 1

    for t in range(horizon):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho*Z1 + np.sqrt(1-rho**2)*rng.standard_normal(n_paths)

        V_pos = np.maximum(V, 1e-8)
        vol   = np.sqrt(V_pos)

        # Price step
        comp_drift = drift - lam*k - 0.5*V_pos
        dS = S*(comp_drift*dt + vol*np.sqrt(dt)*Z1)

        # Variance step (full truncation scheme for stability)
        dV = kappa*(theta - V_pos)*dt + xi*vol*np.sqrt(dt)*Z2
        V  = np.maximum(0.0, V + dV)

        # Jump overlay
        if jump_prob > 0:
            n_jumps = rng.poisson(lam*dt*252, n_paths)
            jump_sizes = np.zeros(n_paths)
            for i in np.where(n_jumps > 0)[0]:
                j = rng.lognormal(jump_mean, jump_std, int(n_jumps[i]))
                jump_sizes[i] = np.prod(j) - 1.0
            dS += S*jump_sizes

        # Strait closure
        new_close = (~strait_closed) & (rng.random(n_paths) < strait_prob*dt*252)
        strait_closed |= new_close
        dS += S*(strait_impact + rng.random(n_paths)*0.05)*new_close

        S = np.maximum(0.01, S + dS)
        paths[:,t+1] = S

    return paths

# ══════════════════════════════════════════════
#  Statistics
# ══════════════════════════════════════════════

def compute_stats(paths, spot):
    final = paths[:,-1]
    T     = paths.shape[1]-1
    p5    = np.percentile(final,5)
    tail  = final[final <= p5]
    log_r = np.log(paths[:,1:]/paths[:,:-1]).flatten()
    skew  = float(scipy_stats.skew(final))
    kurt  = float(scipy_stats.kurtosis(final))   # excess kurtosis
    return {
        "T":T,"spot":spot,"N":paths.shape[0],
        "mean":   final.mean(),
        "median": np.median(final),
        "std":    final.std(),
        "skew":   skew,
        "kurt":   kurt,
        "p1":     np.percentile(final,1),
        "p5":     p5,
        "p25":    np.percentile(final,25),
        "p75":    np.percentile(final,75),
        "p95":    np.percentile(final,95),
        "p99":    np.percentile(final,99),
        "var95":  p5,
        "var99":  np.percentile(final,1),
        "cvar95": tail.mean() if len(tail) else p5,
        "prob_up":    (final>spot).mean()*100,
        "prob_up10":  (final>spot*1.10).mean()*100,
        "prob_up20":  (final>spot*1.20).mean()*100,
        "prob_dn10":  (final<spot*0.90).mean()*100,
        "prob_dn20":  (final<spot*0.80).mean()*100,
        "exp_ret":    (final.mean()/spot-1)*100,
        "log_returns": log_r,
    }

# ══════════════════════════════════════════════
#  Statistical Tests
# ══════════════════════════════════════════════

def run_statistical_tests(paths, spot, hist_returns=None):
    """
    Run a battery of statistical tests on simulated final prices and returns.
    Returns a dict of test results with statistic, p-value, verdict.
    """
    final   = paths[:,-1]
    returns = (final/spot - 1)*100
    log_ret = np.log(paths[:,1:]/paths[:,:-1]).flatten()

    results = {}

    # ── 1. Z-score of mean return (H0: mean return = 0) ──────
    n   = len(returns)
    mu  = returns.mean()
    se  = returns.std(ddof=1)/np.sqrt(n)
    z   = mu/se if se > 0 else 0.0
    p_z = 2*(1 - norm.cdf(abs(z)))
    results["z_mean_return"] = {
        "name":    "Z-test: Mean Return ≠ 0",
        "stat":    round(z,4),
        "p_val":   round(p_z,4),
        "verdict": "Reject H₀" if p_z < 0.05 else "Fail to reject H₀",
        "pass":    p_z < 0.05,
        "interp":  f"Mean return ({mu:+.2f}%) is {'statistically significant' if p_z<0.05 else 'not significant'} at 5% level",
    }

    # ── 2. Jarque-Bera normality test on simulated log-returns ──
    if len(log_ret) > 8:
        jb_stat, jb_p = jarque_bera(log_ret[:5000])   # cap for speed
        results["jarque_bera"] = {
            "name":    "Jarque-Bera: Normality of Returns",
            "stat":    round(float(jb_stat),4),
            "p_val":   round(float(jb_p),4),
            "verdict": "Non-normal" if jb_p < 0.05 else "Cannot reject normality",
            "pass":    jb_p < 0.05,
            "interp":  "Fat tails / skew detected — consistent with geopolitical shocks" if jb_p<0.05 else "Returns appear approximately normal",
        }

    # ── 3. Kolmogorov-Smirnov test (simulated vs normal) ──────
    standardised = (returns - returns.mean())/returns.std()
    ks_stat, ks_p = kstest(standardised, "norm")
    results["ks_normality"] = {
        "name":    "KS-test: Final Price vs Normal",
        "stat":    round(float(ks_stat),4),
        "p_val":   round(float(ks_p),4),
        "verdict": "Significant deviation" if ks_p < 0.05 else "No significant deviation",
        "pass":    ks_p < 0.05,
        "interp":  "Distribution has heavier tails than normal (expected with jumps)" if ks_p<0.05 else "Distribution is close to normal",
    }

    # ── 4. VaR Back-test Z-score (Kupiec proportion of failures) ──
    var95    = np.percentile(final, 5)
    failures = (final < var95).sum()
    expected = 0.05 * len(final)
    # Kupiec LR statistic (approximate z-score)
    if expected > 0:
        kupiec_z = (failures - expected) / np.sqrt(expected*(1-0.05))
        kupiec_p = 2*(1 - norm.cdf(abs(kupiec_z)))
    else:
        kupiec_z, kupiec_p = 0.0, 1.0
    results["kupiec_var"] = {
        "name":    "Kupiec Test: VaR 95% Accuracy",
        "stat":    round(float(kupiec_z),4),
        "p_val":   round(float(kupiec_p),4),
        "verdict": "VaR model rejected" if kupiec_p < 0.05 else "VaR model not rejected",
        "pass":    kupiec_p >= 0.05,
        "interp":  f"Observed failures: {failures} vs expected: {expected:.0f} ({failures/len(final)*100:.1f}% vs 5.0%)",
    }

    # ── 5. T-test: scenario return vs zero ────────────────────
    t_stat, t_p = scipy_stats.ttest_1samp(returns, 0)
    results["t_test_return"] = {
        "name":    "T-test: Return Significance",
        "stat":    round(float(t_stat),4),
        "p_val":   round(float(t_p),4),
        "verdict": "Significant" if t_p < 0.05 else "Not significant",
        "pass":    t_p < 0.05,
        "interp":  f"Scenario drift is {'statistically distinguishable' if t_p<0.05 else 'not distinguishable'} from zero",
    }

    # ── 6. Tail ratio (95th / 5th pct symmetry) ──────────────
    p95_r = np.percentile(returns, 95)
    p05_r = np.percentile(returns, 5)
    tail_ratio = abs(p95_r) / abs(p05_r) if abs(p05_r) > 0 else np.nan
    results["tail_ratio"] = {
        "name":    "Tail Ratio (upside/downside)",
        "stat":    round(float(tail_ratio),4) if not np.isnan(tail_ratio) else "N/A",
        "p_val":   "—",
        "verdict": "Upside skewed" if tail_ratio > 1.1 else ("Downside skewed" if tail_ratio < 0.9 else "Symmetric"),
        "pass":    None,
        "interp":  f"95th pct return ({p95_r:+.1f}%) vs 5th pct ({p05_r:+.1f}%)",
    }

    # ── 7. Z-score of individual paths (outlier detection) ────
    path_final_means = paths[:,-1]
    z_scores_paths   = (path_final_means - path_final_means.mean()) / path_final_means.std()
    extreme_paths    = (np.abs(z_scores_paths) > 2.5).sum()
    results["path_zscore"] = {
        "name":    "Path Z-score: Outlier Paths (|z|>2.5)",
        "stat":    round(float(z_scores_paths.mean()),4),
        "p_val":   "—",
        "verdict": f"{extreme_paths} paths ({extreme_paths/len(paths)*100:.1f}%)",
        "pass":    None,
        "interp":  f"Expected ~1.2% of paths beyond 2.5σ; observed {extreme_paths/len(paths)*100:.1f}%",
    }

    # ── 8. Compare simulated vs historical returns (if available) ──
    if hist_returns and len(hist_returns) > 20:
        hr = np.array(hist_returns)*100
        sr = log_ret[:len(hr)]*100 if len(log_ret) > len(hr) else log_ret*100
        ks2_stat, ks2_p = scipy_stats.ks_2samp(hr, sr[:min(1000,len(sr))])
        results["ks2_hist_sim"] = {
            "name":    "KS-2samp: Simulated vs Historical Returns",
            "stat":    round(float(ks2_stat),4),
            "p_val":   round(float(ks2_p),4),
            "verdict": "Distributions differ" if ks2_p < 0.05 else "Distributions consistent",
            "pass":    ks2_p >= 0.05,
            "interp":  "Simulation captures a different regime than historical (expected under stress scenarios)" if ks2_p<0.05 else "Simulation is consistent with historical return distribution",
        }

    return results

# ══════════════════════════════════════════════
#  Chart helpers
# ══════════════════════════════════════════════

def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":BG2,"axes.edgecolor":GRID,
        "axes.labelcolor":MUT,"axes.titlecolor":TEXT,
        "xtick.color":MUT,"ytick.color":MUT,"grid.color":GRID,"grid.linewidth":0.6,
        "text.color":TEXT,"font.family":"monospace",
        "axes.spines.top":False,"axes.spines.right":False,
    })

def _bands(paths):
    return (np.percentile(paths,5,axis=0), np.percentile(paths,25,axis=0),
            np.median(paths,axis=0),
            np.percentile(paths,75,axis=0), np.percentile(paths,95,axis=0))


def plot_model_comparison(paths_gbm, paths_heston, spot, name, unit, dec, horizon):
    mpl_dark()
    fig, axes = plt.subplots(1, 2, figsize=(13,4), facecolor=BG)
    days = np.arange(horizon+1)

    for ax, paths, model_name, color in [
        (axes[0], paths_gbm,    "Merton Jump-Diffusion", BLUE),
        (axes[1], paths_heston, "Heston Stoch. Vol",     AMB),
    ]:
        p5,p25,med,p75,p95 = _bands(paths)
        ax.fill_between(days,p5,p95,  alpha=0.10, color=color)
        ax.fill_between(days,p25,p75, alpha=0.24, color=color)
        ax.plot(days, med, color=color, lw=2.2)
        ax.plot(days, p95, color=RED,  lw=1.0, ls="--", alpha=0.8)
        ax.plot(days, p5,  color=GRN,  lw=1.0, ls="--", alpha=0.8)
        ax.axhline(spot, color="#555", lw=1, ls=":", alpha=0.7)
        ax.set_title(f"{name} — {model_name}", fontsize=10)
        ax.set_xlabel("Trading days", fontsize=9)
        ax.set_ylabel(f"Price ({unit})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[
            Line2D([0],[0],color=color,lw=2,label="Median"),
            Line2D([0],[0],color=RED,lw=1,ls="--",label="95th"),
            Line2D([0],[0],color=GRN,lw=1,ls="--",label="5th"),
            Patch(fc=color,alpha=0.24,label="50% CI"),
        ], fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    fig.tight_layout()
    return fig


def plot_history_and_dist(hist_closes, paths_gbm, paths_heston, spot, name, unit, dec, horizon):
    mpl_dark()
    fig, axes = plt.subplots(1, 3, figsize=(15,4), facecolor=BG,
                             gridspec_kw={"width_ratios":[1,1,1]})

    # ── Historical ──
    ax0 = axes[0]
    if hist_closes:
        ax0.plot(hist_closes[-252:], color=BLUE, lw=1.4, alpha=0.9)
        ax0.axhline(spot, color=AMB, lw=1, ls="--", alpha=0.8)
        ax0.scatter(len(hist_closes[-252:])-1, spot, color=AMB, s=40, zorder=5)
        ax0.set_title("1-Year Historical", fontsize=10)
    else:
        ax0.text(0.5,0.5,"No live data",ha="center",va="center",
                 transform=ax0.transAxes,fontsize=10,color="#555")
    ax0.set_xlabel("Trading days",fontsize=9)
    ax0.set_ylabel(f"Price ({unit})",fontsize=9)
    ax0.grid(True,alpha=0.3)

    # ── Return distributions overlay ──
    ax1 = axes[1]
    final_gbm    = paths_gbm[:,-1]
    final_heston = paths_heston[:,-1]
    ret_gbm    = (final_gbm/spot-1)*100
    ret_heston = (final_heston/spot-1)*100
    bins = np.linspace(min(ret_gbm.min(),ret_heston.min()),
                       max(ret_gbm.max(),ret_heston.max()), 60)
    ax1.hist(ret_gbm,    bins=bins, alpha=0.55, color=BLUE, label="Jump-Diffusion", density=True)
    ax1.hist(ret_heston, bins=bins, alpha=0.55, color=AMB,  label="Heston",         density=True)
    ax1.axvline(0,              color="#555",lw=1,ls=":",alpha=0.8)
    ax1.axvline(ret_gbm.mean(), color=BLUE,  lw=1.5,ls="--",alpha=0.9)
    ax1.axvline(ret_heston.mean(),color=AMB, lw=1.5,ls="--",alpha=0.9)
    ax1.set_xlabel(f"Return at day {horizon} (%)",fontsize=9)
    ax1.set_ylabel("Density",fontsize=9)
    ax1.set_title("Model Return Distributions",fontsize=10)
    ax1.legend(fontsize=8,framealpha=0.3,facecolor=BG2,edgecolor=GRID)
    ax1.grid(True,axis="y",alpha=0.3)

    # ── Volatility smile proxy — rolling vol of path medians ──
    ax2 = axes[2]
    med_gbm    = np.median(paths_gbm,    axis=0)
    med_heston = np.median(paths_heston, axis=0)
    window = 10
    if horizon >= window:
        roll_vol_gbm    = [np.std(np.log(med_gbm[max(0,i-window):i+1]/med_gbm[max(0,i-window)]))*np.sqrt(252)
                           for i in range(1,len(med_gbm))]
        roll_vol_heston = [np.std(np.log(med_heston[max(0,i-window):i+1]/med_heston[max(0,i-window)]))*np.sqrt(252)
                           for i in range(1,len(med_heston))]
        ax2.plot(roll_vol_gbm,    color=BLUE, lw=1.4, label="Jump-Diff rolling vol")
        ax2.plot(roll_vol_heston, color=AMB,  lw=1.4, label="Heston rolling vol")
        ax2.set_title("Rolling Realised Vol (median path)", fontsize=10)
        ax2.set_xlabel("Trading days", fontsize=9)
        ax2.set_ylabel("Annualised vol", fontsize=9)
        ax2.legend(fontsize=8,framealpha=0.3,facecolor=BG2,edgecolor=GRID)
        ax2.grid(True,alpha=0.3)
    else:
        ax2.text(0.5,0.5,"Horizon too short\nfor rolling vol",
                 ha="center",va="center",transform=ax2.transAxes,color=MUT,fontsize=9)

    fig.suptitle(name, fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


def plot_scenario_comparison(spot, symbol, strait_impact, n_paths, horizon, seed, name):
    mpl_dark()
    sc_colors = {"Base Case":BLUE,"Mild Escalation":AMB,"Strait Closure":RED,"De-escalation":GRN}
    fig, axes = plt.subplots(1,2,figsize=(13,4),facecolor=BG)

    for ax, model_label, use_heston in [(axes[0],"Merton Jump-Diffusion",False),(axes[1],"Heston Stoch. Vol",True)]:
        for sc_name, sc in SCENARIOS.items():
            d = sc["drift"][symbol]; s = sc["sigma"][symbol]
            if use_heston:
                p = run_heston(spot,d,s,sc["heston_kappa"],sc["heston_theta"],
                               sc["heston_xi"],sc["heston_rho"],
                               sc["jump_prob"],sc["jump_mean"],sc["jump_std"],
                               sc["strait_prob"],strait_impact,min(n_paths,400),horizon,seed,symbol)
            else:
                p = run_gbm_jump(spot,d,s,sc["jump_prob"],sc["jump_mean"],sc["jump_std"],
                                 sc["strait_prob"],sc["mean_rev"],strait_impact,
                                 min(n_paths,400),horizon,seed,symbol)
            ax.plot(np.arange(p.shape[1]),np.median(p,axis=0),
                    lw=2,color=sc_colors[sc_name],label=sc_name)
        ax.axhline(spot,color="#555",lw=1,ls=":",alpha=0.7,label="Spot")
        ax.set_title(f"{model_label}",fontsize=10)
        ax.set_xlabel("Trading days",fontsize=9); ax.set_ylabel("Price ($)",fontsize=9)
        ax.legend(fontsize=8,framealpha=0.3,facecolor=BG2,edgecolor=GRID)
        ax.grid(True,alpha=0.3)
    fig.suptitle(f"{name} — Scenario Comparison",fontsize=12,fontweight="bold",color=TEXT)
    fig.tight_layout()
    return fig


def plot_statistical_summary(paths_gbm, paths_heston, spot, horizon):
    """QQ-plot + autocorrelation of log-returns for both models."""
    mpl_dark()
    fig, axes = plt.subplots(1,4,figsize=(16,4),facecolor=BG)

    for col, (paths, label, color) in enumerate([
        (paths_gbm,    "Jump-Diffusion", BLUE),
        (paths_heston, "Heston",         AMB),
    ]):
        log_ret = np.log(paths[:,1:]/paths[:,:-1]).flatten()
        sample  = log_ret[np.isfinite(log_ret)][:3000]

        # QQ plot
        ax = axes[col]
        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(sample, dist="norm")
        ax.scatter(osm, osr, color=color, s=3, alpha=0.5)
        ax.plot(osm, slope*np.array(osm)+intercept, color=RED, lw=1.5, ls="--")
        ax.set_title(f"QQ-Plot ({label})", fontsize=10)
        ax.set_xlabel("Theoretical quantiles", fontsize=9)
        ax.set_ylabel("Sample quantiles",      fontsize=9)
        ax.grid(True, alpha=0.3)

        # Return histogram with normal overlay
        ax2 = axes[col+2]
        ax2.hist(sample, bins=80, density=True, color=color, alpha=0.7, edgecolor="none")
        xr = np.linspace(sample.min(), sample.max(), 300)
        ax2.plot(xr, norm.pdf(xr, sample.mean(), sample.std()), color=RED, lw=1.5, ls="--", label="Normal fit")
        ax2.set_title(f"Log-Return Dist ({label})", fontsize=10)
        ax2.set_xlabel("Log-return", fontsize=9)
        ax2.set_ylabel("Density",    fontsize=9)
        ax2.legend(fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
        ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("## SIMULATION")
    st.markdown("---")

    commodity_name = st.selectbox("Commodity", list(COMMODITY_META.keys()), index=0)
    meta = COMMODITY_META[commodity_name]

    st.markdown("### Scenario")
    scenario_name = st.radio("sc", list(SCENARIOS.keys()),
        format_func=lambda x: f"{SCENARIOS[x]['emoji']}  {x}",
        label_visibility="collapsed")
    sc = SCENARIOS[scenario_name]

    st.markdown("---")
    st.markdown("### Model")
    model_choice = st.radio("model", ["Both models","Merton Jump-Diffusion only","Heston only"],
                            label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Simulation Params")
    n_paths      = st.slider("Paths (N)",        200, 2000, 800, step=100)
    horizon_days = st.slider("Horizon (days)",    30,  365,  90, step=5)
    seed         = st.slider("Random seed",        0,  999,  42, step=1)

    st.markdown("---")
    st.markdown("### Override Params")
    st.caption("Scenario defaults shown. Adjust to override.")
    _sym = meta["symbol"]
    override_drift = st.slider("Drift (%)",       -30, 50,  int(sc["drift"][_sym]*100)) / 100.0
    override_sigma = st.slider("Volatility (%)",    5, 120, int(sc["sigma"][_sym]*100)) / 100.0
    override_jp    = st.slider("Jump prob (daily %)",0.0,10.0, sc["jump_prob"]*100,step=0.1)/100
    override_jm    = st.slider("Jump mean (%)",    -5.0,20.0, sc["jump_mean"]*100,step=0.5)/100

    st.markdown("---")
    if st.button("🔄  Refresh live prices"):
        st.cache_data.clear(); st.rerun()

    strait_impact = STRAIT_IMPACT[meta["symbol"]]

# ══════════════════════════════════════════════
#  Fetch live data
# ══════════════════════════════════════════════

with st.spinner("Fetching live prices from Yahoo Finance…"):
    spot, live_drift, live_sigma, prev_close, pct_chg, hist_closes, last_updated, data_source, hist_log_rets = \
        fetch_live_data(meta["ticker"], meta["fallback_spot"],
                        meta["fallback_drift"], meta["fallback_sigma"])

drift = override_drift
sigma = max(0.05, override_sigma)
unit  = meta["unit"]
dec   = meta["decimals"]
sym   = meta["symbol"]

# ══════════════════════════════════════════════
#  Run models
# ══════════════════════════════════════════════

with st.spinner("Running Monte Carlo simulation…"):
    paths_gbm = run_gbm_jump(
        spot, drift, sigma,
        override_jp, override_jm, sc["jump_std"],
        sc["strait_prob"], sc["mean_rev"], strait_impact,
        n_paths, horizon_days, seed, sym,
    )
    paths_heston = run_heston(
        spot, drift, sigma,
        sc["heston_kappa"], sc["heston_theta"], sc["heston_xi"], sc["heston_rho"],
        override_jp, override_jm, sc["jump_std"],
        sc["strait_prob"], strait_impact,
        n_paths, horizon_days, seed, sym,
    )

sim_gbm    = compute_stats(paths_gbm,    spot)
sim_heston = compute_stats(paths_heston, spot)

# ══════════════════════════════════════════════
#  Hero
# ══════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
  <p class="hero-title">COMMODITY MONTE CARLO</p>
  <p class="hero-sub">Iran-Israel Conflict · Merton Jump-Diffusion · Heston Stochastic Vol · Statistical Tests</p>
</div>
""", unsafe_allow_html=True)

if data_source == "live":
    st.markdown(f'<div class="live-badge"><span class="live-dot"></span>LIVE · {last_updated} · auto-refresh 15 min</div>', unsafe_allow_html=True)
else:
    st.warning("⚠️  Yahoo Finance unavailable — using fallback prices.")

bc = sc["badge_class"]
st.markdown(f"""
<span class="scenario-badge {bc}">{sc['emoji']}  {scenario_name.upper()}</span>
<span class="model-badge">{model_choice}</span>
<p style="color:#8b949e;font-size:13px;margin-top:-4px;font-family:'IBM Plex Mono',monospace;">{sc['description']}</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  Metric cards — show both model medians
# ══════════════════════════════════════════════

def fmt(v, d=None):
    d = d if d is not None else dec
    return f"{unit}{v:,.{d}f}"

chg_cls  = "delta-up" if pct_chg >= 0 else "delta-down"
chg_sign = "+" if pct_chg >= 0 else ""

g = sim_gbm; h = sim_heston
g_ret  = g["exp_ret"]; h_ret = h["exp_ret"]

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card info">
    <div class="metric-label">LIVE SPOT</div>
    <div class="metric-val">{fmt(spot)}</div>
    <div class="metric-delta {chg_cls}">{chg_sign}{pct_chg:.2f}% today</div>
  </div>
  <div class="metric-card {'up' if g_ret>=0 else 'down'}">
    <div class="metric-label">JUMP-DIFF MEDIAN · D{g['T']}</div>
    <div class="metric-val">{fmt(g['median'])}</div>
    <div class="metric-delta {'delta-up' if g_ret>=0 else 'delta-down'}">{'+' if g_ret>=0 else ''}{g_ret:.1f}%</div>
  </div>
  <div class="metric-card {'up' if h_ret>=0 else 'down'}">
    <div class="metric-label">HESTON MEDIAN · D{h['T']}</div>
    <div class="metric-val">{fmt(h['median'])}</div>
    <div class="metric-delta {'delta-up' if h_ret>=0 else 'delta-down'}">{'+' if h_ret>=0 else ''}{h_ret:.1f}%</div>
  </div>
  <div class="metric-card down">
    <div class="metric-label">VAR 95% (Jump-Diff)</div>
    <div class="metric-val">{fmt(abs(spot-g['var95']))}</div>
    <div class="metric-delta delta-down">Max expected loss</div>
  </div>
  <div class="metric-card warn">
    <div class="metric-label">CVAR 95% (Jump-Diff)</div>
    <div class="metric-val">{fmt(g['cvar95'])}</div>
    <div class="metric-delta delta-down">Expected tail loss</div>
  </div>
  <div class="metric-card info">
    <div class="metric-label">PROB UP (both models)</div>
    <div class="metric-val">{(g['prob_up']+h['prob_up'])/2:.1f}%</div>
    <div class="metric-delta delta-neu">avg · {n_paths:,} paths each</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Model Comparison",
    "📊  Distributions & Vol",
    "🔀  Scenario Comparison",
    "🧪  Statistical Tests",
    "📋  Full Statistics",
])

with tab1:
    fig1 = plot_model_comparison(paths_gbm, paths_heston, spot, commodity_name, unit, dec, horizon_days)
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)
    st.caption("Left: Merton Jump-Diffusion (log-normally distributed jumps, Poisson arrivals). "
               "Right: Heston model (stochastic variance process, correlated with price).")

with tab2:
    fig2 = plot_history_and_dist(hist_closes, paths_gbm, paths_heston, spot, commodity_name, unit, dec, horizon_days)
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

with tab3:
    fig3 = plot_scenario_comparison(spot, sym, strait_impact, n_paths, horizon_days, seed, commodity_name)
    st.pyplot(fig3, use_container_width=True); plt.close(fig3)

with tab4:
    st.markdown('<p class="section-hdr">Statistical Test Battery</p>', unsafe_allow_html=True)

    fig4 = plot_statistical_summary(paths_gbm, paths_heston, spot, horizon_days)
    st.pyplot(fig4, use_container_width=True); plt.close(fig4)

    st.markdown('<p class="section-hdr">Jump-Diffusion Tests</p>', unsafe_allow_html=True)
    tests_gbm = run_statistical_tests(paths_gbm, spot, hist_log_rets)

    st.markdown('<p class="section-hdr">Heston Tests</p>', unsafe_allow_html=True)
    tests_heston = run_statistical_tests(paths_heston, spot, hist_log_rets)

    col1, col2 = st.columns(2)
    for col, tests, label in [(col1, tests_gbm, "Merton Jump-Diffusion"), (col2, tests_heston, "Heston")]:
        with col:
            st.markdown(f'<p class="section-hdr">{label}</p>', unsafe_allow_html=True)
            rows = ""
            for key, t in tests.items():
                if t["pass"] is True:
                    verdict_cls = "test-pass"
                elif t["pass"] is False:
                    verdict_cls = "test-fail"
                else:
                    verdict_cls = "test-warn"
                pval_str = f"{t['p_val']}" if t['p_val'] != "—" else "—"
                rows += f"""
                <tr>
                  <td>{t['name']}</td>
                  <td class="tbl-neu">{t['stat']}</td>
                  <td class="tbl-neu">{pval_str}</td>
                  <td class="{verdict_cls}">{t['verdict']}</td>
                </tr>
                <tr><td colspan="4" style="color:#484f58;font-size:11px;padding:2px 10px 8px;">{t['interp']}</td></tr>
                """
            st.markdown(f"""
            <table class="stats-tbl">
              <thead><tr><th>TEST</th><th>STATISTIC</th><th>P-VALUE</th><th>RESULT</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

with tab5:
    st.markdown('<p class="section-hdr">Model comparison — key statistics</p>', unsafe_allow_html=True)

    def tbl_color(v, threshold=50, good="up"):
        if v > threshold: return "tbl-up" if good=="up" else "tbl-down"
        if v < 20:        return "tbl-down" if good=="up" else "tbl-up"
        return "tbl-neu"

    stat_rows = [
        ("Live spot price",          fmt(spot),                                          fmt(spot),                                         "tbl-neu"),
        ("Today's change",           f"{chg_sign}{pct_chg:.2f}%",                       f"{chg_sign}{pct_chg:.2f}%",                       "tbl-neu"),
        ("Mean final price",         fmt(g['mean']),                                     fmt(h['mean']),                                    "tbl-neu"),
        ("Median final price",       fmt(g['median']),                                   fmt(h['median']),                                  "tbl-neu"),
        ("Std deviation",            fmt(g['std']),                                      fmt(h['std']),                                     "tbl-neu"),
        ("Skewness",                 f"{g['skew']:+.3f}",                               f"{h['skew']:+.3f}",                               "tbl-neu"),
        ("Excess kurtosis",          f"{g['kurt']:+.3f}",                               f"{h['kurt']:+.3f}",                               "tbl-neu"),
        ("1st percentile",           fmt(g['p1']),                                       fmt(h['p1']),                                      "tbl-down"),
        ("5th percentile",           fmt(g['p5']),                                       fmt(h['p5']),                                      "tbl-down"),
        ("25th percentile",          fmt(g['p25']),                                      fmt(h['p25']),                                     "tbl-neu"),
        ("75th percentile",          fmt(g['p75']),                                      fmt(h['p75']),                                     "tbl-neu"),
        ("95th percentile",          fmt(g['p95']),                                      fmt(h['p95']),                                     "tbl-up"),
        ("99th percentile",          fmt(g['p99']),                                      fmt(h['p99']),                                     "tbl-up"),
        ("VaR 95% (loss)",           fmt(abs(spot-g['var95'])),                          fmt(abs(spot-h['var95'])),                         "tbl-down"),
        ("VaR 99% (loss)",           fmt(abs(spot-g['var99'])),                          fmt(abs(spot-h['var99'])),                         "tbl-down"),
        ("CVaR 95%",                 fmt(g['cvar95']),                                   fmt(h['cvar95']),                                  "tbl-down"),
        ("Expected return",          f"{g['exp_ret']:+.2f}%",                           f"{h['exp_ret']:+.2f}%",                           "tbl-neu"),
        ("P(price rises)",           f"{g['prob_up']:.1f}%",                            f"{h['prob_up']:.1f}%",                            "tbl-neu"),
        ("P(+10% spike)",            f"{g['prob_up10']:.1f}%",                          f"{h['prob_up10']:.1f}%",                          "tbl-neu"),
        ("P(+20% spike)",            f"{g['prob_up20']:.1f}%",                          f"{h['prob_up20']:.1f}%",                          "tbl-neu"),
        ("P(-10% drop)",             f"{g['prob_dn10']:.1f}%",                          f"{h['prob_dn10']:.1f}%",                          "tbl-neu"),
        ("P(-20% drop)",             f"{g['prob_dn20']:.1f}%",                          f"{h['prob_dn20']:.1f}%",                          "tbl-neu"),
    ]

    rows_html = "".join(
        f'<tr><td>{l}</td><td class="{c}">{v1}</td><td class="{c}">{v2}</td></tr>'
        for l,v1,v2,c in stat_rows
    )
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>METRIC</th><th>JUMP-DIFFUSION</th><th>HESTON</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Applied Parameters</p>', unsafe_allow_html=True)
    param_rows = [
        ("Data source",        "Yahoo Finance" if data_source=="live" else "Fallback"),
        ("Ticker",             meta["ticker"]),
        ("Drift (applied)",    f"{drift*100:+.1f}%"),
        ("Sigma (applied)",    f"{sigma*100:.1f}%"),
        ("Jump prob.",         f"{override_jp*100:.2f}% / day"),
        ("Jump mean",          f"{override_jm*100:+.1f}%"),
        ("Jump std",           f"{sc['jump_std']*100:.1f}%"),
        ("Heston κ (kappa)",   f"{sc['heston_kappa']:.2f}"),
        ("Heston θ (theta)",   f"{sc['heston_theta']:.4f}  (long-run var = {sc['heston_theta']**0.5*100:.1f}% vol)"),
        ("Heston ξ (xi)",      f"{sc['heston_xi']:.2f}  (vol-of-vol)"),
        ("Heston ρ (rho)",     f"{sc['heston_rho']:.2f}  (price-vol correlation)"),
        ("Strait closure",     f"{sc['strait_prob']*100:.2f}% / day"),
        ("Strait impact",      f"{strait_impact*100:.0f}%"),
        ("Mean reversion κ",   f"{sc['mean_rev']:.3f}"),
    ]
    ph = "".join(f'<tr><td>{l}</td><td class="tbl-neu">{v}</td></tr>' for l,v in param_rows)
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>PARAMETER</th><th>VALUE</th></tr></thead>
      <tbody>{ph}</tbody>
    </table>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #21262d;
text-align:center;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#484f58;letter-spacing:.5px;">
MERTON JUMP-DIFFUSION · HESTON STOCH. VOL · Z-TEST · JARQUE-BERA · KS-TEST · KUPIEC VAR · FOR EDUCATIONAL PURPOSES ONLY
</div>
""", unsafe_allow_html=True)
