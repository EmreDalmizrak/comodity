"""
Commodity Monte Carlo — Iran-Israel Conflict Risk
Streamlit App  |  Live prices · GBM + Jump · Log-return analysis · Z-score tests
Run: streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from scipy.stats import norm, jarque_bera, kstest
import streamlit as st
from datetime import datetime

# ══════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Commodity Monte Carlo — Iran-Israel",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════
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
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
.stButton>button{background:#161b22!important;border:1px solid #30363d!important;color:#e6edf3!important;font-family:'IBM Plex Mono',monospace!important;font-size:13px!important;border-radius:6px!important;width:100%;}
.stButton>button:hover{border-color:#58a6ff!important;color:#58a6ff!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  Constants & Scenarios
# ══════════════════════════════════════════════════════

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
        "badge_class":"sc-base","emoji":"🔵",
        "description":"Low-level tension, current market dynamics prevail.",
    },
    "Mild Escalation": {
        "drift":{"OIL":0.06,"GAS":0.06,"GOLD":0.06},
        "sigma":{"OIL":0.30,"GAS":0.40,"GOLD":0.30},
        "jump_prob":0.03,"jump_mean":0.05,"jump_std":0.03,
        "strait_prob":0.02,"mean_rev":0.04,
        "badge_class":"sc-mild","emoji":"🟡",
        "description":"Increased airstrikes, regional risk premium builds.",
    },
    "Strait Closure": {
        "drift":{"OIL":0.12,"GAS":0.12,"GOLD":0.12},
        "sigma":{"OIL":0.45,"GAS":0.60,"GOLD":0.45},
        "jump_prob":0.06,"jump_mean":0.12,"jump_std":0.05,
        "strait_prob":0.15,"mean_rev":0.02,
        "badge_class":"sc-strait","emoji":"🔴",
        "description":"Hormuz blocked — severe supply disruption scenario.",
    },
    "De-escalation": {
        "drift":{"OIL":-0.02,"GAS":-0.02,"GOLD":-0.02},
        "sigma":{"OIL":0.18,"GAS":0.22,"GOLD":0.18},
        "jump_prob":0.00,"jump_mean":-0.01,"jump_std":0.01,
        "strait_prob":0.001,"mean_rev":0.12,
        "badge_class":"sc-deesc","emoji":"🟢",
        "description":"Ceasefire / diplomacy — risk premium deflates.",
    },
}

STRAIT_IMPACT = {"OIL":0.25,"GAS":0.35,"GOLD":0.08}

BG="#0d1117"; BG2="#161b22"; GRID="#21262d"; TEXT="#e6edf3"
MUT="#8b949e"; BLUE="#185FA5"; RED="#e24b4a"; GRN="#1d9e75"; AMB="#f0993b"

# ══════════════════════════════════════════════════════
#  Live data
# ══════════════════════════════════════════════════════

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
        ts        = datetime.now().strftime("%H:%M:%S")
        return (spot, ann_drift, ann_sigma, prev, pct_chg,
                closes.values.tolist(), ts, "live",
                log_ret.values.tolist())
    except Exception:
        return (fallback_spot, fallback_drift, fallback_sigma,
                fallback_spot, 0.0, [], "N/A", "fallback", [])

# ══════════════════════════════════════════════════════
#  Simulation — GBM + Merton Jump-Diffusion
# ══════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_mc(spot, drift, sigma, jump_prob, jump_mean, jump_std,
           strait_prob, mean_rev, strait_impact,
           n_paths, horizon, seed, symbol):
    """
    Merton (1976) Jump-Diffusion:
      dS/S = (μ − λk) dt + σ dWt + (J−1) dNt
      J     ~ LogNormal(jump_mean, jump_std)
      Nt    ~ Poisson(λ dt)
    + mean-reversion toward S0
    + one-time Strait of Hormuz closure shock
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 252.0
    S0  = spot
    lam = jump_prob
    # compensator: E[J-1] = exp(mu_J + 0.5*sig_J^2) - 1
    k   = np.exp(jump_mean + 0.5*jump_std**2) - 1

    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = S0
    S = np.full(n_paths, S0, dtype=float)
    strait_closed = np.zeros(n_paths, dtype=bool)

    for t in range(horizon):
        Z = rng.standard_normal(n_paths)

        # compensated GBM
        comp_drift = drift - lam*k - 0.5*sigma**2
        dS = S * (comp_drift*dt + sigma*np.sqrt(dt)*Z)

        # mean reversion
        dS += -mean_rev*(S - S0)*dt

        # Poisson jump arrivals
        if lam > 0:
            n_jumps = rng.poisson(lam * dt * 252, n_paths)
            jump_ret = np.zeros(n_paths)
            for i in np.where(n_jumps > 0)[0]:
                j = rng.lognormal(jump_mean, jump_std, int(n_jumps[i]))
                jump_ret[i] = np.prod(j) - 1.0
            dS += S * jump_ret

        # Strait closure (one-time per path)
        new_close      = (~strait_closed) & (rng.random(n_paths) < strait_prob * dt * 252)
        strait_closed |= new_close
        dS            += S * (strait_impact + rng.random(n_paths)*0.05) * new_close

        S = np.maximum(0.01, S + dS)
        paths[:, t+1] = S

    return paths

# ══════════════════════════════════════════════════════
#  Summary statistics
# ══════════════════════════════════════════════════════

def compute_stats(paths, spot):
    final   = paths[:, -1]
    T       = paths.shape[1] - 1
    # log-returns for every step, every path
    log_ret = np.log(paths[:, 1:] / paths[:, :-1])   # shape (N, T)
    lr_flat = log_ret.flatten()
    lr_flat = lr_flat[np.isfinite(lr_flat)]

    p5  = np.percentile(final, 5)
    tail = final[final <= p5]

    return {
        "T": T, "spot": spot, "N": paths.shape[0],
        "mean":   final.mean(),
        "median": np.median(final),
        "std":    final.std(),
        "skew":   float(scipy_stats.skew(lr_flat)),
        "kurt":   float(scipy_stats.kurtosis(lr_flat)),   # excess kurtosis
        "p1":     np.percentile(final, 1),
        "p5":     p5,
        "p25":    np.percentile(final, 25),
        "p75":    np.percentile(final, 75),
        "p95":    np.percentile(final, 95),
        "p99":    np.percentile(final, 99),
        "var95":  p5,
        "var99":  np.percentile(final, 1),
        "cvar95": tail.mean() if len(tail) else p5,
        "prob_up":    (final > spot).mean()*100,
        "prob_up10":  (final > spot*1.10).mean()*100,
        "prob_up20":  (final > spot*1.20).mean()*100,
        "prob_dn10":  (final < spot*0.90).mean()*100,
        "prob_dn20":  (final < spot*0.80).mean()*100,
        "exp_ret":    (final.mean()/spot - 1)*100,
        # per-step log-return series (median path) for charting
        "lr_median_path": np.log(np.median(paths, axis=0)[1:] /
                                  np.median(paths, axis=0)[:-1]),
        "lr_flat": lr_flat,                   # all paths flattened
        "log_ret_matrix": log_ret,            # (N, T) for cross-path z-scores
    }

# ══════════════════════════════════════════════════════
#  Statistical tests  —  all on LOG-RETURNS
# ══════════════════════════════════════════════════════

def run_stat_tests(paths, spot, hist_log_rets=None):
    """
    Every test uses log-returns (not percentage returns of final prices).

    Z-score methodology
    -------------------
    We compute the Z-score of the MEAN daily log-return against the
    null hypothesis that the true mean is zero:

        Z = (r̄ − 0) / (s / √n)

    where r̄ = sample mean log-return, s = sample std, n = sample size.
    This is the standard one-sample Z-test, valid when n is large (CLT).
    A large |Z| means the scenario's drift is statistically detectable.
    """
    log_ret = np.log(paths[:, 1:] / paths[:, :-1]).flatten()
    log_ret = log_ret[np.isfinite(log_ret)]
    # cap at 20 000 for speed; still statistically representative
    sample  = log_ret[:20_000] if len(log_ret) > 20_000 else log_ret

    final   = paths[:, -1]
    n       = len(sample)
    r_bar   = sample.mean()
    s       = sample.std(ddof=1)
    se      = s / np.sqrt(n)           # standard error of the mean

    results = {}

    # ── 1. Z-test on mean daily log-return ─────────────────────
    # H₀: μ_log = 0  (no drift)
    # Under H₀, Z ~ N(0,1) for large n
    z_stat = r_bar / se if se > 0 else 0.0
    p_z    = 2 * (1 - norm.cdf(abs(z_stat)))
    # Annualise the mean for interpretability
    ann_drift_pct = r_bar * 252 * 100
    results["z_drift"] = {
        "name":    "Z-test — Mean Log-Return ≠ 0",
        "stat":    f"{z_stat:+.4f}",
        "p_val":   f"{p_z:.4f}",
        "verdict": "Reject H₀  (drift detected)" if p_z < 0.05 else "Fail to reject H₀  (no significant drift)",
        "pass":    p_z < 0.05,
        "interp":  (f"Daily mean log-return = {r_bar*100:+.4f}%  "
                    f"(≈ {ann_drift_pct:+.1f}% annualised). "
                    f"SE = {se*100:.5f}%.  Z = {z_stat:+.2f}.  "
                    f"{'Drift is statistically significant' if p_z<0.05 else 'Drift is not statistically distinguishable from zero'} "
                    f"at the 5% level."),
    }

    # ── 2. Z-score of the scenario vs historical drift ──────────
    # If historical log-returns are available, test whether the
    # simulated mean differs from the historical mean (two-sample Z).
    if hist_log_rets and len(hist_log_rets) > 20:
        hr     = np.array(hist_log_rets)
        hr     = hr[np.isfinite(hr)]
        n2     = len(hr)
        r2     = hr.mean()
        s2     = hr.std(ddof=1)
        # Two-sample Z: (r̄₁ − r̄₂) / √(s₁²/n₁ + s₂²/n₂)
        se_2s  = np.sqrt(s**2/n + s2**2/n2)
        z2     = (r_bar - r2) / se_2s if se_2s > 0 else 0.0
        p_z2   = 2 * (1 - norm.cdf(abs(z2)))
        results["z_vs_hist"] = {
            "name":    "Z-test — Simulated vs Historical Drift",
            "stat":    f"{z2:+.4f}",
            "p_val":   f"{p_z2:.4f}",
            "verdict": "Significantly different from history" if p_z2 < 0.05 else "Consistent with historical drift",
            "pass":    p_z2 < 0.05,
            "interp":  (f"Simulated mean = {r_bar*100:+.4f}%/day, "
                        f"Historical mean = {r2*100:+.4f}%/day. "
                        f"Difference {'is' if p_z2<0.05 else 'is not'} statistically significant — "
                        f"{'scenario adds a real risk premium above historical norms' if p_z2<0.05 else 'scenario consistent with historical behaviour'}."),
        }

    # ── 3. Jarque-Bera normality test ──────────────────────────
    # H₀: log-returns are normally distributed
    # Fat tails / skew → JB statistic large → reject normality
    jb_stat, jb_p = jarque_bera(sample)
    sk  = float(scipy_stats.skew(sample))
    krt = float(scipy_stats.kurtosis(sample))   # excess kurtosis
    results["jarque_bera"] = {
        "name":    "Jarque-Bera — Normality of Log-Returns",
        "stat":    f"{jb_stat:.4f}",
        "p_val":   f"{jb_p:.4f}",
        "verdict": "Non-normal (fat tails / skew)" if jb_p < 0.05 else "Cannot reject normality",
        "pass":    jb_p < 0.05,
        "interp":  (f"Skewness = {sk:+.3f}  (normal = 0), "
                    f"Excess kurtosis = {krt:+.3f}  (normal = 0). "
                    f"{'Heavy tails and/or asymmetry detected — consistent with jump process' if jb_p<0.05 else 'Log-returns appear approximately normal'}."),
    }

    # ── 4. KS-test: simulated log-returns vs normal ─────────────
    std_sample = (sample - sample.mean()) / sample.std()
    ks_stat, ks_p = kstest(std_sample, "norm")
    results["ks_test"] = {
        "name":    "KS-test — Log-Returns vs Normal CDF",
        "stat":    f"{ks_stat:.4f}",
        "p_val":   f"{ks_p:.4f}",
        "verdict": "Significant deviation from normal" if ks_p < 0.05 else "No significant deviation",
        "pass":    ks_p < 0.05,
        "interp":  ("Tails are heavier than a normal distribution — geopolitical jumps are visible in the shape."
                    if ks_p < 0.05 else "Log-return distribution is close to normal."),
    }

    # ── 5. Kupiec proportion-of-failures VaR back-test ─────────
    # H₀: true exceedance rate = 5%
    # Uses a normal approximation of the binomial
    var95     = np.percentile(final, 5)
    failures  = int((final < var95).sum())
    exp_fail  = 0.05 * len(final)
    kup_z     = (failures - exp_fail) / np.sqrt(exp_fail * 0.95) if exp_fail > 0 else 0.0
    kup_p     = 2 * (1 - norm.cdf(abs(kup_z)))
    results["kupiec"] = {
        "name":    "Kupiec Test — VaR 95% Accuracy",
        "stat":    f"{kup_z:+.4f}",
        "p_val":   f"{kup_p:.4f}",
        "verdict": "VaR model rejected" if kup_p < 0.05 else "VaR model not rejected",
        "pass":    kup_p >= 0.05,
        "interp":  (f"Observed breaches: {failures} ({failures/len(final)*100:.2f}%) "
                    f"vs expected: {exp_fail:.0f} (5.00%). "
                    f"VaR 95% = {var95:,.2f}.  "
                    f"Model {'over-estimates' if failures/len(final)<0.05 else 'under-estimates'} tail risk."
                    if kup_p < 0.05 else
                    f"Observed breaches: {failures} ({failures/len(final)*100:.2f}%) within acceptable range of 5%."),
    }

    # ── 6. Tail asymmetry Z-score ───────────────────────────────
    # Compare magnitude of best 5% vs worst 5% of log-returns
    top5  = np.percentile(sample,  95)
    bot5  = np.percentile(sample,   5)
    tail_z = (top5 + bot5) / (s * np.sqrt(2/n))   # asymmetry relative to noise
    results["tail_asym"] = {
        "name":    "Tail Asymmetry Z-score",
        "stat":    f"{tail_z:+.4f}",
        "p_val":   "—",
        "verdict": "Upside skewed" if tail_z > 1 else ("Downside skewed" if tail_z < -1 else "Symmetric tails"),
        "pass":    None,
        "interp":  (f"95th pct log-ret = {top5*100:+.3f}%,  "
                    f"5th pct log-ret = {bot5*100:+.3f}%. "
                    f"Positive Z → more upside mass; negative → more downside mass."),
    }

    # ── 7. Cross-path Z-score (outlier path detection) ──────────
    # For each path, compute its total log-return; then Z-score across paths
    total_log_ret = np.log(paths[:, -1] / paths[:, 0])
    z_paths = (total_log_ret - total_log_ret.mean()) / total_log_ret.std()
    extreme = int((np.abs(z_paths) > 2.576).sum())   # 99% threshold
    results["path_outlier"] = {
        "name":    "Cross-path Z-score (|z| > 2.576)",
        "stat":    f"{z_paths.mean():+.4f}",
        "p_val":   "—",
        "verdict": f"{extreme} paths  ({extreme/len(paths)*100:.1f}%)",
        "pass":    None,
        "interp":  (f"Expected ~1% of paths beyond 2.576σ at 99% level; "
                    f"observed {extreme/len(paths)*100:.1f}%. "
                    f"Excess indicates heavy tails or jump clustering."),
    }

    return results

# ══════════════════════════════════════════════════════
#  Matplotlib helpers
# ══════════════════════════════════════════════════════

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


# ── Chart 1: History + simulated paths ────────────────

def plot_history_and_paths(hist_closes, paths, spot, name, unit, dec, horizon):
    mpl_dark()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=BG,
                             gridspec_kw={"width_ratios":[1, 1.6]})
    # historical
    ax0 = axes[0]
    if hist_closes:
        ax0.plot(hist_closes[-252:], color=BLUE, lw=1.4, alpha=0.9)
        ax0.axhline(spot, color=AMB, lw=1, ls="--", alpha=0.8)
        ax0.scatter(len(hist_closes[-252:])-1, spot, color=AMB, s=40, zorder=5)
        ax0.set_title("1-Year Historical Price", fontsize=10)
    else:
        ax0.text(0.5,0.5,"No live data",ha="center",va="center",
                 transform=ax0.transAxes,fontsize=10,color="#555")
        ax0.set_title("Historical Price", fontsize=10)
    ax0.set_xlabel("Trading days", fontsize=9)
    ax0.set_ylabel(f"Price ({unit})", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # simulated fan
    ax1 = axes[1]
    days = np.arange(horizon+1)
    p5,p25,med,p75,p95 = _bands(paths)
    ax1.fill_between(days, p5,  p95, alpha=0.10, color=BLUE)
    ax1.fill_between(days, p25, p75, alpha=0.24, color=BLUE)
    ax1.plot(days, med, color=BLUE, lw=2.2)
    ax1.plot(days, p95, color=RED,  lw=1.2, ls="--", alpha=0.9)
    ax1.plot(days, p5,  color=GRN,  lw=1.2, ls="--", alpha=0.9)
    ax1.axhline(spot, color="#555", lw=1, ls=":", alpha=0.7)
    ax1.set_xlabel("Trading days (forecast)", fontsize=9)
    ax1.set_ylabel(f"Price ({unit})", fontsize=9)
    ax1.set_title(f"Monte Carlo — {paths.shape[0]:,} Paths", fontsize=10)
    ax1.grid(True, alpha=0.4)
    ax1.legend(handles=[
        Line2D([0],[0],color=BLUE,lw=2,label="Median"),
        Line2D([0],[0],color=RED,lw=1.2,ls="--",label="95th pct"),
        Line2D([0],[0],color=GRN,lw=1.2,ls="--",label="5th pct"),
        Patch(fc=BLUE,alpha=0.24,label="50% CI"),
        Patch(fc=BLUE,alpha=0.10,label="90% CI"),
    ], fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID, loc="upper left")
    fig.suptitle(name, fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


# ── Chart 2: Return distribution ──────────────────────

def plot_distribution(paths, spot, name, unit, dec, horizon):
    mpl_dark()
    final   = paths[:,-1]
    returns = (final/spot - 1)*100
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG)
    counts, edges = np.histogram(returns, bins=55)
    centers = (edges[:-1]+edges[1:])/2
    bw = edges[1]-edges[0]
    bar_colors = [RED if c<-10 else GRN if c>10 else BLUE for c in centers]
    ax.bar(centers, counts, width=bw*0.88, color=bar_colors, alpha=0.82, edgecolor="none")
    ax.axvline(0,                        color="#555",lw=1,ls=":",alpha=0.8)
    ax.axvline(np.median(returns),       color=BLUE, lw=1.5,ls="--",alpha=0.9)
    ax.axvline(np.percentile(returns,5), color=RED,  lw=1.2,ls=":",alpha=0.8, label="VaR 95%")
    ax.legend(handles=[
        Patch(fc=RED,  alpha=0.82, label="< −10%  (tail risk)"),
        Patch(fc=BLUE, alpha=0.82, label="−10% to +10%  (neutral)"),
        Patch(fc=GRN,  alpha=0.82, label="> +10%  (upside)"),
    ], fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID, loc="upper left")
    ax.set_xlabel(f"Return at day {horizon} (%)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title(f"{name} — Return Distribution", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Chart 3: Log-return analysis ─────────────────────
# NEW: Three panels — (a) log-return time series of median path,
#      (b) log-return histogram vs normal, (c) rolling Z-score

def plot_log_returns(paths, spot, name, hist_log_rets=None):
    mpl_dark()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=BG)

    # median path log-returns over time
    med_path  = np.median(paths, axis=0)
    lr_series = np.log(med_path[1:] / med_path[:-1])
    days      = np.arange(1, len(lr_series)+1)

    ax0 = axes[0]
    colors_bar = [GRN if v >= 0 else RED for v in lr_series]
    ax0.bar(days, lr_series, color=colors_bar, alpha=0.75, width=0.8)
    ax0.axhline(0, color="#555", lw=0.8, ls=":")
    ax0.axhline( lr_series.std()*2,  color=AMB, lw=1, ls="--", alpha=0.7, label="+2σ")
    ax0.axhline(-lr_series.std()*2,  color=AMB, lw=1, ls="--", alpha=0.7, label="−2σ")
    ax0.set_title("Log-Returns — Median Path", fontsize=10)
    ax0.set_xlabel("Trading day", fontsize=9)
    ax0.set_ylabel("Daily log-return", fontsize=9)
    ax0.legend(fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    ax0.grid(True, alpha=0.3)

    # histogram: simulated vs historical vs normal fit
    ax1 = axes[1]
    lr_all = np.log(paths[:,1:]/paths[:,:-1]).flatten()
    lr_all = lr_all[np.isfinite(lr_all)]
    sample = lr_all[:10_000]
    ax1.hist(sample, bins=80, density=True, color=BLUE, alpha=0.65,
             label=f"Simulated (n={len(sample):,})", edgecolor="none")
    if hist_log_rets and len(hist_log_rets) > 10:
        hr = np.array(hist_log_rets)
        ax1.hist(hr, bins=40, density=True, color=AMB, alpha=0.55,
                 label=f"Historical (n={len(hr)})", edgecolor="none")
    xr = np.linspace(sample.min(), sample.max(), 300)
    ax1.plot(xr, scipy_stats.norm.pdf(xr, sample.mean(), sample.std()),
             color=RED, lw=1.8, ls="--", label="Normal fit")
    ax1.set_title("Log-Return Distribution", fontsize=10)
    ax1.set_xlabel("Daily log-return", fontsize=9)
    ax1.set_ylabel("Density", fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    ax1.grid(True, axis="y", alpha=0.3)

    # rolling Z-score of median path log-returns
    ax2 = axes[2]
    window = max(10, len(lr_series)//10)
    roll_mean = np.array([lr_series[max(0,i-window):i+1].mean() for i in range(len(lr_series))])
    roll_std  = np.array([lr_series[max(0,i-window):i+1].std()  for i in range(len(lr_series))])
    roll_std  = np.where(roll_std < 1e-10, 1e-10, roll_std)
    roll_z    = (lr_series - roll_mean) / roll_std

    ax2.plot(days, roll_z, color=BLUE, lw=1.2, alpha=0.9, label="Rolling Z-score")
    ax2.fill_between(days, -2, 2, alpha=0.08, color=GRN, label="±2σ band")
    ax2.axhline( 2, color=AMB, lw=1, ls="--", alpha=0.7)
    ax2.axhline(-2, color=AMB, lw=1, ls="--", alpha=0.7)
    ax2.axhline( 0, color="#555", lw=0.8, ls=":")
    # mark extreme Z events
    extreme_mask = np.abs(roll_z) > 2.5
    ax2.scatter(days[extreme_mask], roll_z[extreme_mask],
                color=RED, s=20, zorder=5, label=f"Extreme (|z|>2.5): {extreme_mask.sum()}")
    ax2.set_title(f"Rolling Z-score of Log-Returns (w={window}d)", fontsize=10)
    ax2.set_xlabel("Trading day", fontsize=9)
    ax2.set_ylabel("Z-score", fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{name} — Log-Return Analysis", fontsize=12,
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


# ── Chart 4: Scenario comparison ──────────────────────

def plot_scenario_compare(com_name, spot, symbol, strait_impact, n_paths, horizon, seed):
    mpl_dark()
    sc_colors = {"Base Case":BLUE,"Mild Escalation":AMB,"Strait Closure":RED,"De-escalation":GRN}
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    for sc_name, sc in SCENARIOS.items():
        d = sc["drift"][symbol]; s = sc["sigma"][symbol]
        p = run_mc(spot, d, s,
                   sc["jump_prob"], sc["jump_mean"], sc["jump_std"],
                   sc["strait_prob"], sc["mean_rev"], strait_impact,
                   min(n_paths,500), horizon, seed, symbol)
        ax.plot(np.arange(p.shape[1]), np.median(p, axis=0),
                lw=2, color=sc_colors[sc_name], label=sc_name)
    ax.axhline(spot, color="#555", lw=1, ls=":", alpha=0.7, label="Spot")
    ax.set_xlabel("Trading days", fontsize=9)
    ax.set_ylabel("Price ($)", fontsize=9)
    ax.set_title(f"{com_name} — All Scenarios (median paths)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=BG2, edgecolor=GRID)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## SIMULATION")
    st.markdown("---")

    commodity_name = st.selectbox("Commodity", list(COMMODITY_META.keys()), index=0)
    meta = COMMODITY_META[commodity_name]

    st.markdown("### Scenario")
    scenario_name = st.radio(
        "sc", list(SCENARIOS.keys()),
        format_func=lambda x: f"{SCENARIOS[x]['emoji']}  {x}",
        label_visibility="collapsed",
    )
    sc = SCENARIOS[scenario_name]

    st.markdown("---")
    st.markdown("### Simulation Params")
    n_paths      = st.slider("Paths (N)",       200, 2000, 800, step=100)
    horizon_days = st.slider("Horizon (days)",   30,  365,  90, step=5)
    seed         = st.slider("Random seed",       0,  999,  42, step=1)

    st.markdown("---")
    st.markdown("### Override Market Params")
    st.caption("Scenario defaults shown. Adjust to override.")
    _sym = meta["symbol"]
    override_drift = st.slider("Drift (%)",            -30, 50,  int(sc["drift"][_sym]*100)) / 100.0
    override_sigma = st.slider("Volatility (%)",         5, 120, int(sc["sigma"][_sym]*100)) / 100.0
    override_jp    = st.slider("Jump prob (daily %)",  0.0, 10.0, sc["jump_prob"]*100, step=0.1) / 100
    override_jm    = st.slider("Jump mean (%)",        -5.0,20.0, sc["jump_mean"]*100, step=0.5) / 100

    st.markdown("---")
    if st.button("🔄  Refresh live prices"):
        st.cache_data.clear(); st.rerun()

    strait_impact = STRAIT_IMPACT[meta["symbol"]]

# ══════════════════════════════════════════════════════
#  Fetch + simulate
# ══════════════════════════════════════════════════════

with st.spinner("Fetching live prices…"):
    (spot, live_drift, live_sigma, prev_close, pct_chg,
     hist_closes, last_updated, data_source, hist_log_rets) = fetch_live_data(
        meta["ticker"], meta["fallback_spot"],
        meta["fallback_drift"], meta["fallback_sigma"],
    )

drift = override_drift
sigma = max(0.05, override_sigma)
unit  = meta["unit"]
dec   = meta["decimals"]
sym   = meta["symbol"]

with st.spinner("Running simulation…"):
    paths = run_mc(
        spot, drift, sigma,
        override_jp, override_jm, sc["jump_std"],
        sc["strait_prob"], sc["mean_rev"], strait_impact,
        n_paths, horizon_days, seed, sym,
    )

sim = compute_stats(paths, spot)

# ══════════════════════════════════════════════════════
#  Hero
# ══════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
  <p class="hero-title">COMMODITY MONTE CARLO</p>
  <p class="hero-sub">Iran-Israel Conflict · GBM + Merton Jump-Diffusion · Log-Return Analysis · Z-score Tests</p>
</div>
""", unsafe_allow_html=True)

if data_source == "live":
    st.markdown(f'<div class="live-badge"><span class="live-dot"></span>LIVE · {last_updated} · auto-refresh 15 min</div>', unsafe_allow_html=True)
else:
    st.warning("⚠️  Yahoo Finance unavailable — using fallback prices.")

bc = sc["badge_class"]
st.markdown(f"""
<span class="scenario-badge {bc}">{sc['emoji']}  {scenario_name.upper()}</span>
<p style="color:#8b949e;font-size:13px;margin-top:-4px;font-family:'IBM Plex Mono',monospace;">
  {sc['description']}
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  Metric cards
# ══════════════════════════════════════════════════════

def fmt(v, d=None):
    d = d if d is not None else dec
    return f"{unit}{v:,.{d}f}"

chg_cls  = "delta-up" if pct_chg >= 0 else "delta-down"
chg_sign = "+" if pct_chg >= 0 else ""
exp_ret  = sim["exp_ret"]
d_cls    = "delta-up" if exp_ret >= 0 else "delta-down"
sign     = "+" if exp_ret >= 0 else ""

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card info">
    <div class="metric-label">LIVE SPOT</div>
    <div class="metric-val">{fmt(spot)}</div>
    <div class="metric-delta {chg_cls}">{chg_sign}{pct_chg:.2f}% today</div>
  </div>
  <div class="metric-card {'up' if exp_ret>=0 else 'down'}">
    <div class="metric-label">MEDIAN · DAY {sim['T']}</div>
    <div class="metric-val">{fmt(sim['median'])}</div>
    <div class="metric-delta {d_cls}">{sign}{exp_ret:.1f}%</div>
  </div>
  <div class="metric-card up">
    <div class="metric-label">95TH PCT</div>
    <div class="metric-val">{fmt(sim['p95'])}</div>
    <div class="metric-delta delta-up">+{(sim['p95']/spot-1)*100:.1f}%</div>
  </div>
  <div class="metric-card down">
    <div class="metric-label">5TH PCT</div>
    <div class="metric-val">{fmt(sim['p5'])}</div>
    <div class="metric-delta delta-down">{(sim['p5']/spot-1)*100:.1f}%</div>
  </div>
  <div class="metric-card warn">
    <div class="metric-label">VAR 95%</div>
    <div class="metric-val">{fmt(abs(spot-sim['var95']))}</div>
    <div class="metric-delta delta-down">Potential loss</div>
  </div>
  <div class="metric-card info">
    <div class="metric-label">PROB. UP</div>
    <div class="metric-val">{sim['prob_up']:.1f}%</div>
    <div class="metric-delta delta-neu">{n_paths:,} paths</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Price Paths",
    "📊  Return Distribution",
    "📉  Log-Return Analysis",
    "🔀  Scenario Comparison",
    "🧪  Statistical Tests + Stats",
])

with tab1:
    fig1 = plot_history_and_paths(hist_closes, paths, spot,
                                  commodity_name, unit, dec, horizon_days)
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)
    if data_source == "live":
        st.caption(f"Left: 1-year real data ({meta['ticker']}). Right: {n_paths:,}-path Monte Carlo forecast.")

with tab2:
    fig2 = plot_distribution(paths, spot, commodity_name, unit, dec, horizon_days)
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

with tab3:
    fig3 = plot_log_returns(paths, spot, commodity_name, hist_log_rets)
    st.pyplot(fig3, use_container_width=True); plt.close(fig3)
    st.caption(
        "Left: Daily log-return of the median simulated path. "
        "Centre: Log-return histogram — simulated vs historical (if live) vs normal fit. "
        "Right: Rolling Z-score detects abnormal return days; red dots = events beyond 2.5σ."
    )

with tab4:
    fig4 = plot_scenario_compare(commodity_name, spot, sym,
                                  strait_impact, n_paths, horizon_days, seed)
    st.pyplot(fig4, use_container_width=True); plt.close(fig4)

with tab5:
    # ── Statistical tests ─────────────────────────────────
    st.markdown('<p class="section-hdr">Statistical Test Battery — on Log-Returns</p>', unsafe_allow_html=True)
    tests = run_stat_tests(paths, spot, hist_log_rets)

    rows = ""
    for key, t in tests.items():
        if   t["pass"] is True:  vc = "test-pass"
        elif t["pass"] is False: vc = "test-fail"
        else:                    vc = "test-warn"
        rows += (
            f'<tr><td>{t["name"]}</td>'
            f'<td class="tbl-neu">{t["stat"]}</td>'
            f'<td class="tbl-neu">{t["p_val"]}</td>'
            f'<td class="{vc}">{t["verdict"]}</td></tr>'
            f'<tr><td colspan="4" style="color:#484f58;font-size:11px;'
            f'padding:2px 10px 10px;">{t["interp"]}</td></tr>'
        )
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>TEST</th><th>STATISTIC</th><th>P-VALUE</th><th>RESULT</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── Summary stats ─────────────────────────────────────
    st.markdown('<p class="section-hdr">Simulation Statistics</p>', unsafe_allow_html=True)

    def tc(v, threshold=50, good="up"):
        if v > threshold: return "tbl-up" if good=="up" else "tbl-down"
        if v < 20:        return "tbl-down" if good=="up" else "tbl-up"
        return "tbl-neu"

    stat_rows = [
        ("Live spot price",          fmt(sim["spot"]),                                       "tbl-neu"),
        ("Today's change",           f"{chg_sign}{pct_chg:.2f}%",                          "tbl-up" if pct_chg>=0 else "tbl-down"),
        ("Mean final price",         fmt(sim["mean"]),                                       "tbl-up" if sim["mean"]>spot else "tbl-down"),
        ("Median final price",       fmt(sim["median"]),                                     "tbl-up" if sim["median"]>spot else "tbl-down"),
        ("Std deviation",            fmt(sim["std"]),                                        "tbl-neu"),
        ("Log-return skewness",      f"{sim['skew']:+.4f}",                                 "tbl-neu"),
        ("Log-return excess kurt.",  f"{sim['kurt']:+.4f}",                                 "tbl-neu"),
        ("1st percentile",           fmt(sim["p1"]),                                        "tbl-down"),
        ("5th percentile",           fmt(sim["p5"]),                                        "tbl-down"),
        ("25th percentile",          fmt(sim["p25"]),                                       "tbl-neu"),
        ("75th percentile",          fmt(sim["p75"]),                                       "tbl-neu"),
        ("95th percentile",          fmt(sim["p95"]),                                       "tbl-up"),
        ("99th percentile",          fmt(sim["p99"]),                                       "tbl-up"),
        ("VaR 95%  (loss)",          fmt(abs(spot-sim["var95"])),                           "tbl-down"),
        ("VaR 99%  (loss)",          fmt(abs(spot-sim["var99"])),                           "tbl-down"),
        ("CVaR 95%",                 fmt(sim["cvar95"]),                                    "tbl-down"),
        ("Expected return",          f"{sim['exp_ret']:+.2f}%",                            "tbl-up" if sim["exp_ret"]>=0 else "tbl-down"),
        ("Prob. price rises",        f"{sim['prob_up']:.1f}%",                             tc(sim["prob_up"])),
        ("Prob. +10% spike",         f"{sim['prob_up10']:.1f}%",                           "tbl-neu"),
        ("Prob. +20% spike",         f"{sim['prob_up20']:.1f}%",                           "tbl-neu"),
        ("Prob. −10% drop",          f"{sim['prob_dn10']:.1f}%",                           tc(sim["prob_dn10"], good="down")),
        ("Prob. −20% drop",          f"{sim['prob_dn20']:.1f}%",                           tc(sim["prob_dn20"], good="down")),
    ]
    rows_html = "".join(
        f'<tr><td>{l}</td><td class="{c}">{v}</td></tr>'
        for l,v,c in stat_rows
    )
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>METRIC</th><th>VALUE</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── Model params ──────────────────────────────────────
    st.markdown('<p class="section-hdr">Applied Model Parameters</p>', unsafe_allow_html=True)
    pr = [
        ("Data source",           "Yahoo Finance" if data_source=="live" else "Fallback"),
        ("Ticker",                meta["ticker"]),
        ("Last updated",          last_updated),
        ("Live annualised drift", f"{live_drift*100:+.1f}%"),
        ("Live annualised vol",   f"{live_sigma*100:.1f}%"),
        ("Applied drift (μ)",     f"{drift*100:+.1f}%"),
        ("Applied vol (σ)",       f"{sigma*100:.1f}%"),
        ("Jump arrival (λ)",      f"{override_jp*100:.2f}% / day"),
        ("Jump mean (μ_J)",       f"{override_jm*100:+.1f}%"),
        ("Jump std (σ_J)",        f"{sc['jump_std']*100:.1f}%"),
        ("Strait closure prob.",  f"{sc['strait_prob']*100:.3f}% / day"),
        ("Strait price impact",   f"{strait_impact*100:.0f}%"),
        ("Mean reversion (κ)",    f"{sc['mean_rev']:.3f}"),
    ]
    ph = "".join(f'<tr><td>{l}</td><td class="tbl-neu">{v}</td></tr>' for l,v in pr)
    st.markdown(f"""
    <table class="stats-tbl">
      <thead><tr><th>PARAMETER</th><th>VALUE</th></tr></thead>
      <tbody>{ph}</tbody>
    </table>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #21262d;
text-align:center;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#484f58;letter-spacing:.5px;">
GBM + MERTON JUMP-DIFFUSION · LOG-RETURN Z-SCORES · JARQUE-BERA · KS-TEST · KUPIEC VaR · FOR EDUCATIONAL PURPOSES ONLY
</div>
""", unsafe_allow_html=True)
