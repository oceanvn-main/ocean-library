"""
=============================================================================
Experiment: MA(q) Modelling on Real Data — S&P 500 Daily Log-Returns
=============================================================================

Pipeline:
  Phase 0  — Download S&P 500 data & compute log-returns
  Phase 1  — Stationarity tests (ADF + KPSS)
  Phase 2  — ACF plot + AIC/BIC grid search to identify order q
  Phase 3  — Fit MA(q_hat) via MLE + invertibility check
  Phase 4  — Residual white-noise diagnostics (Ljung-Box + ACF plots)
  Phase 5  — Theoretical note: MA(q) observational equivalence

=============================================================================
"""

import os
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (save to file)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")   # suppress convergence noise

# ---------------------------------------------------------------------------
# Resolve output directory
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("  EXPERIMENT: MA(q) on Real Data — S&P 500 Daily Log-Returns")
print("=" * 70)

# ===========================================================================
# PHASE 0 — Setup & Data Download
# ===========================================================================
print("\n" + "=" * 70)
print("  PHASE 0 — Data Download & Log-Return Construction")
print("=" * 70)

import yfinance as yf

TICKER     = "^GSPC"
START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

print(f"\n  Downloading {TICKER} from {START_DATE} to {END_DATE} ...")
raw = yf.download(TICKER, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)

prices = raw["Close"].dropna().squeeze()
print(f"  Downloaded {len(prices):,} daily close prices.")

# Log-returns: r_t = log(P_t / P_{t-1})
log_returns = np.log(prices / prices.shift(1)).dropna()
r = log_returns.values          # plain numpy array
T = len(r)
print(f"  Log-return series length: T = {T:,}")
print(f"  Mean   : {r.mean():.6f}")
print(f"  Std dev: {r.std():.6f}")
print(f"  Min    : {r.min():.4f}  |  Max: {r.max():.4f}")

# --- Plot 1: Price & Return Series ---
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("S&P 500 — Daily Close Price & Log-Returns (2000-2024)",
             fontsize=14, fontweight="bold")

axes[0].plot(prices.index, prices.values, color="#1f77b4", linewidth=0.8)
axes[0].set_title("Close Price")
axes[0].set_ylabel("Price (USD)")
axes[0].grid(alpha=0.3)

axes[1].plot(log_returns.index, r, color="#d62728", linewidth=0.5, alpha=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Daily Log-Returns  r_t = log(P_t / P_{t-1})")
axes[1].set_ylabel("Log-Return")
axes[1].grid(alpha=0.3)

plt.tight_layout()
path0 = os.path.join(RESULTS_DIR, "maq_phase0_price_returns.png")
plt.savefig(path0, dpi=150)
plt.close()
print(f"\n  Plot saved: {path0}")


# ===========================================================================
# PHASE 1 — Stationarity Tests
# ===========================================================================
print("\n" + "=" * 70)
print("  PHASE 1 — Stationarity Tests (ADF + KPSS)")
print("=" * 70)

from statsmodels.tsa.stattools import adfuller, kpss

# --- ADF test ---
adf_stat, adf_pval, adf_lags, _, adf_cv, _ = adfuller(r, autolag="AIC")
print(f"\n  ADF Test:")
print(f"    Statistic : {adf_stat:.4f}")
print(f"    p-value   : {adf_pval:.6f}")
print(f"    Lags used : {adf_lags}")
print(f"    Critical values: 1%={adf_cv['1%']:.3f}  5%={adf_cv['5%']:.3f}  10%={adf_cv['10%']:.3f}")
adf_pass = adf_pval < 0.05
print(f"    => {'STATIONARY (reject unit root)' if adf_pass else 'FAIL — unit root not rejected'}")

# --- KPSS test ---
kpss_stat, kpss_pval, kpss_lags, kpss_cv = kpss(r, regression="c", nlags="auto")
print(f"\n  KPSS Test (H0: stationary):")
print(f"    Statistic : {kpss_stat:.4f}")
print(f"    p-value   : {kpss_pval:.4f}")
print(f"    Lags used : {kpss_lags}")
print(f"    Critical values: 1%={kpss_cv['1%']:.3f}  5%={kpss_cv['5%']:.3f}  10%={kpss_cv['10%']:.3f}")
kpss_pass = kpss_pval > 0.05
print(f"    => {'STATIONARY (fail to reject H0)' if kpss_pass else 'NOT stationary (reject H0)'}")

if adf_pass and kpss_pass:
    print("\n  => CONFIRMED: Log-returns are I(0) — suitable for MA(q) modelling.")
else:
    print("\n  => WARNING: Mixed stationarity evidence — proceed with caution.")


# ===========================================================================
# PHASE 2 — ACF Analysis + AIC/BIC Order Selection
# ===========================================================================
print("\n" + "=" * 70)
print("  PHASE 2 — ACF Plot + AIC/BIC Grid Search (q = 0 ... 10)")
print("=" * 70)

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA

MAX_LAGS = 20
acf_vals, acf_confint = acf(r, nlags=MAX_LAGS, alpha=0.05)

# Confidence bands (Bartlett): +/- 1.96/sqrt(T)
ci_band = 1.96 / math.sqrt(T)
print(f"\n  Sample ACF (Bartlett 95% band = +/- {ci_band:.4f}):")
print(f"  {'Lag':>4}  {'ACF':>10}  {'Significant?':>14}")
print(f"  {'-'*4}  {'-'*10}  {'-'*14}")
for lag in range(1, MAX_LAGS + 1):
    sig = "YES ***" if abs(acf_vals[lag]) > ci_band else "no"
    print(f"  {lag:>4}  {acf_vals[lag]:>10.4f}  {sig:>14}")

# ACF-based q estimate: last lag with significant ACF
sig_lags = [k for k in range(1, MAX_LAGS + 1) if abs(acf_vals[k]) > ci_band]
q_acf = max(sig_lags) if sig_lags else 0
print(f"\n  ACF-suggested order: q_acf = {q_acf}")

# --- AIC/BIC grid search ---
print(f"\n  Fitting ARIMA(0,0,q) for q = 0 ... 10 ...")
print(f"  {'q':>4}  {'AIC':>12}  {'BIC':>12}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}")

aic_scores = {}
bic_scores = {}
for q in range(0, 11):
    try:
        fit = ARIMA(r, order=(0, 0, q)).fit(method_kwargs={"warn_convergence": False})
        aic_scores[q] = fit.aic
        bic_scores[q] = fit.bic
        print(f"  {q:>4}  {fit.aic:>12.2f}  {fit.bic:>12.2f}")
    except Exception as e:
        print(f"  {q:>4}  {'FAILED':>12}  {'FAILED':>12}   ({e})")

q_aic = min(aic_scores, key=aic_scores.get)
q_bic = min(bic_scores, key=bic_scores.get)
print(f"\n  AIC-optimal order: q_aic = {q_aic}")
print(f"  BIC-optimal order: q_bic = {q_bic}")

# Final selection: prefer BIC (penalises more; more parsimonious)
q_hat = q_bic
print(f"\n  => Selected order: q_hat = {q_hat}  (BIC-optimal, most parsimonious)")

# --- Plot 2: Sample ACF ---
fig, ax = plt.subplots(figsize=(12, 5))
lags_arr = np.arange(0, MAX_LAGS + 1)
markerline, stemlines, baseline = ax.stem(lags_arr, acf_vals,
                                           linefmt="C0-", markerfmt="C0o",
                                           basefmt="k-")
plt.setp(stemlines, linewidth=1.5)
plt.setp(markerline, markersize=5)
ax.axhline( ci_band, color="red", linestyle="--", linewidth=1.2,
            label=f"95% CI (+/-{ci_band:.4f})")
ax.axhline(-ci_band, color="red", linestyle="--", linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.set_title(f"Sample ACF — S&P 500 Daily Log-Returns\n"
             f"ACF cutoff at q={q_acf} | AIC q={q_aic} | BIC q={q_bic}")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path2 = os.path.join(RESULTS_DIR, "maq_phase2_acf.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"\n  ACF plot saved: {path2}")


# ===========================================================================
# PHASE 3 — Fit MA(q_hat) Model
# ===========================================================================
print("\n" + "=" * 70)
print(f"  PHASE 3 — Fitting MA({q_hat}) Model via MLE")
print("=" * 70)

import pandas as pd

model  = ARIMA(r, order=(0, 0, q_hat))
result = model.fit(method_kwargs={"warn_convergence": False})

print(f"\n  Model: ARIMA(0, 0, {q_hat}) — i.e. pure MA({q_hat})")

# Normalise all param arrays to pandas Series regardless of statsmodels version
_pnames  = result.model.param_names
params   = pd.Series(np.asarray(result.params),  index=_pnames)
std_errs = pd.Series(np.asarray(result.bse),     index=_pnames)
tvalues  = pd.Series(np.asarray(result.tvalues), index=_pnames)
pvalues  = pd.Series(np.asarray(result.pvalues), index=_pnames)

print(f"\n  Fitted parameters:")
print(f"  {'Parameter':>20}  {'Estimate':>12}  {'Std Err':>10}  {'t-stat':>10}  {'p-value':>10}")
print(f"  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
for name in _pnames:
    est, se, tv, pv = params[name], std_errs[name], tvalues[name], pvalues[name]
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
    print(f"  {name:>20}  {est:>12.6f}  {se:>10.6f}  {tv:>10.4f}  {pv:>10.6f} {sig}")

# Extract innovation variance
sigma2_fit = params.get("sigma2", params.iloc[-1])   # sigma2 is always last param
print(f"\n  Fitted sigma^2 = {sigma2_fit:.8f}  (sigma = {math.sqrt(sigma2_fit):.6f})")
print(f"  Log-likelihood = {result.llf:.2f}")
print(f"  AIC = {result.aic:.2f}  |  BIC = {result.bic:.2f}")

# --- Invertibility Check ---
print(f"\n  Invertibility Check (all MA roots must lie OUTSIDE unit circle):")
if q_hat == 0:
    print("  q=0 => MA(0) is just white noise, trivially invertible.")
    all_invertible = True
    roots = []
else:
    # Extract MA lag coefficients: ma.L1, ma.L2, ..., ma.Lq
    ma_coefs = [params[f"ma.L{i}"] for i in range(1, q_hat + 1)
                if f"ma.L{i}" in params.index]
    # Polynomial: 1 + theta_1*z + ... + theta_q*z^q
    poly_coefs = np.array([1.0] + ma_coefs)
    roots = np.roots(poly_coefs[::-1])   # np.roots wants highest-degree first
    all_invertible = True
    print(f"  {'Root':>6}  {'|root|':>10}  {'Outside unit circle?':>22}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*22}")
    for i, root in enumerate(roots):
        mod = abs(root)
        outside = mod > 1.0
        all_invertible = all_invertible and outside
        print(f"  {i+1:>6}  {mod:>10.6f}  {'YES' if outside else 'NO  <-- non-invertible!':>22}")
    print(f"\n  Invertibility: {'PASS — model is invertible.' if all_invertible else 'FAIL — some roots inside unit circle.'}") 


# ===========================================================================
# PHASE 4 — Residual White-Noise Diagnostics
# ===========================================================================
print("\n" + "=" * 70)
print("  PHASE 4 — Residual White-Noise Diagnostics")
print("=" * 70)

residuals = result.resid.values     # fitted innovations eps_hat_t

# --- Statistical helper (same as MA(1) experiment) ---
resid_mean = residuals.mean()
resid_var  = residuals.var(ddof=1)
r_cen      = residuals - resid_mean
cov1_resid = np.mean(r_cen[1:]  * r_cen[:-1])
cov2_resid = np.mean(r_cen[2:]  * r_cen[:-2])

print(f"\n  Residual Statistics:")
print(f"    Mean              = {resid_mean:.8f}   (should be ~= 0)")
print(f"    Variance          = {resid_var:.8f}   (fitted sigma^2 = {sigma2_fit:.8f})")
print(f"    Lag-1 autocovar.  = {cov1_resid:.8f}   (should be ~= 0)")
print(f"    Lag-2 autocovar.  = {cov2_resid:.8f}   (should be ~= 0)")

tol_cov = 0.0001
print(f"\n  Quick checks (tolerance = {tol_cov}):")
print(f"    Mean ~= 0   : {'PASS' if abs(resid_mean) < 1e-6 else 'FAIL'}")
print(f"    Var  ~= sigma^2 : {'PASS' if abs(resid_var - sigma2_fit) / sigma2_fit < 0.02 else 'FAIL'}")
print(f"    Cov1 ~= 0   : {'PASS' if abs(cov1_resid) < tol_cov else 'MARGINAL (small MA structure may remain)'}")
print(f"    Cov2 ~= 0   : {'PASS' if abs(cov2_resid) < tol_cov else 'MARGINAL'}")

# --- Ljung-Box Test ---
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(residuals, lags=range(1, 11), return_df=True)
print(f"\n  Ljung-Box Test on Residuals (H0: residuals are white noise):")
print(f"  {'Lag':>4}  {'LB Stat':>10}  {'p-value':>10}  {'Conclusion':>20}")
print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*20}")
lb_all_pass = True
for lag, row in lb_result.iterrows():
    ok = row["lb_pvalue"] > 0.05
    lb_all_pass = lb_all_pass and ok
    print(f"  {lag:>4}  {row['lb_stat']:>10.4f}  {row['lb_pvalue']:>10.4f}  "
          f"  {'PASS (WN)' if ok else 'FAIL (not WN)'}")

if lb_all_pass:
    print(f"\n  => CONFIRMED: Residuals are white noise.")
    print(f"     MA({q_hat}) adequately captures all linear serial dependence.")
else:
    print(f"\n  => Some Ljung-Box lags are significant — consider higher q.")

# --- Plot 3: ACF comparison (raw returns vs residuals) ---
acf_raw,   ci_raw   = acf(r,         nlags=MAX_LAGS, alpha=0.05)
acf_resid, ci_resid = acf(residuals, nlags=MAX_LAGS, alpha=0.05)

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Raw returns series
ax_ret = fig.add_subplot(gs[0, :])
ax_ret.plot(log_returns.index, r, color="#d62728", linewidth=0.4, alpha=0.7)
ax_ret.axhline(0, color="black", linewidth=0.6)
ax_ret.set_title("S&P 500 Daily Log-Returns")
ax_ret.set_ylabel("Log-Return")
ax_ret.grid(alpha=0.3)

# ACF of raw returns
ax_acf_raw = fig.add_subplot(gs[1, 0])
lags_arr   = np.arange(0, MAX_LAGS + 1)
ml, sl, bl = ax_acf_raw.stem(lags_arr, acf_raw, linefmt="C0-",
                               markerfmt="C0o", basefmt="k-")
plt.setp(sl, linewidth=1.2); plt.setp(ml, markersize=4)
ax_acf_raw.axhline( ci_band, color="red", linestyle="--", linewidth=1.0)
ax_acf_raw.axhline(-ci_band, color="red", linestyle="--", linewidth=1.0)
ax_acf_raw.set_title("ACF — Raw Log-Returns")
ax_acf_raw.set_xlabel("Lag"); ax_acf_raw.set_ylabel("ACF")
ax_acf_raw.grid(alpha=0.3)

# ACF of residuals
ax_acf_res = fig.add_subplot(gs[1, 1])
ml2, sl2, bl2 = ax_acf_res.stem(lags_arr, acf_resid, linefmt="C2-",
                                  markerfmt="C2o", basefmt="k-")
plt.setp(sl2, linewidth=1.2); plt.setp(ml2, markersize=4)
ax_acf_res.axhline( ci_band, color="red", linestyle="--", linewidth=1.0)
ax_acf_res.axhline(-ci_band, color="red", linestyle="--", linewidth=1.0)
ax_acf_res.set_title(f"ACF — MA({q_hat}) Residuals")
ax_acf_res.set_xlabel("Lag"); ax_acf_res.set_ylabel("ACF")
ax_acf_res.grid(alpha=0.3)

fig.suptitle(f"MA({q_hat}) Fit on S&P 500 Log-Returns — Diagnostic Summary",
             fontsize=13, fontweight="bold")
path4 = os.path.join(RESULTS_DIR, "maq_phase4_diagnostics.png")
plt.savefig(path4, dpi=150)
plt.close()
print(f"\n  Diagnostic plot saved: {path4}")

# --- Plot 4: AIC/BIC vs q ---
if len(aic_scores) > 1:
    qs   = sorted(aic_scores.keys())
    aics = [aic_scores[q] for q in qs]
    bics = [bic_scores[q] for q in qs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qs, aics, "o-", color="#1f77b4", label="AIC", linewidth=2, markersize=7)
    ax.plot(qs, bics, "s-", color="#ff7f0e", label="BIC", linewidth=2, markersize=7)
    ax.axvline(q_aic, color="#1f77b4", linestyle=":", alpha=0.7,
               label=f"AIC min (q={q_aic})")
    ax.axvline(q_bic, color="#ff7f0e", linestyle=":", alpha=0.7,
               label=f"BIC min (q={q_bic})")
    ax.set_xlabel("MA order q")
    ax.set_ylabel("Information Criterion")
    ax.set_title("Model Selection: AIC and BIC vs MA Order q")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path4b = os.path.join(RESULTS_DIR, "maq_phase4_aic_bic.png")
    plt.savefig(path4b, dpi=150)
    plt.close()
    print(f"  AIC/BIC plot saved: {path4b}")


# ===========================================================================
# PHASE 5 — Theoretical Note: MA(q) Observational Equivalence
# ===========================================================================
print("\n" + "=" * 70)
print("  PHASE 5 — Theoretical Note: MA(q) Observational Equivalence")
print("=" * 70)

print(f"""
  For a general MA(q) process with polynomial:

    theta(L) = 1 + theta_1*L + theta_2*L^2 + ... + theta_q*L^q

  whose roots are  z_1, z_2, ..., z_q (in the complex plane), there exist

                      2^q

  observationally equivalent representations — one for every subset of
  roots reflected through the unit circle  (z_k  ->  1/z_k*).

  Each reflection multiplies the innovation variance by |z_k|^2 to keep
  the autocovariance structure unchanged.

  Example for this MA({q_hat}) fit:
""")

if q_hat > 0 and "roots" in dir():
    for i, root in enumerate(roots):
        mod  = abs(root)
        twin = 1.0 / np.conj(root)
        sigma2_twin = sigma2_fit * (mod ** 2)
        print(f"    Root z_{i+1} = {root:.4f}  |z_{i+1}| = {mod:.4f}")
        print(f"      Reflected twin: 1/z_{i+1}* = {twin:.4f}  |twin| = {abs(twin):.4f}")
        print(f"      Implied sigma_twin^2 = {sigma2_fit:.6f} x {mod**2:.4f} = {sigma2_twin:.6f}")
        print()

    n_twins = 2 ** q_hat
    print(f"  Total observationally equivalent representations: 2^{q_hat} = {n_twins}")
    print(f"  MLE always returns the INVERTIBLE one (all roots outside unit circle).")
else:
    print(f"  MA(0) is just white noise: Y_t = eps_t.")
    print(f"  No MA polynomial => no alternative representations.")
    print(f"  The 'twin' concept is vacuous for q=0.")

print(f"""
  KEY INSIGHT:
    Standard MLE cannot determine WHICH of the 2^q representations
    generated the observed data — this is the MA(q) identification problem.
    Additional restrictions (invertibility, or economic priors) are required
    to uniquely pin down the model.
""")


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("=" * 70)
print("  EXPERIMENT COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
  Dataset   : S&P 500 daily log-returns ({START_DATE} to {END_DATE})
  Obs.      : T = {T:,}

  Phase 1 — Stationarity
    ADF  : {'PASS' if adf_pass  else 'FAIL'}  (p={adf_pval:.4f})
    KPSS : {'PASS' if kpss_pass else 'FAIL'}  (p={kpss_pval:.4f})

  Phase 2 — Order Selection
    ACF suggested q = {q_acf}
    AIC  optimal  q = {q_aic}
    BIC  optimal  q = {q_bic}  <-- selected

  Phase 3 — MA({q_hat}) Fit
    sigma^2  = {sigma2_fit:.8f}
    Invertible: {'YES' if all_invertible else 'NO'}

  Phase 4 — Residual Diagnostics
    Ljung-Box (all lags 1-10): {'ALL PASS' if lb_all_pass else 'SOME FAILURES'}

  Plots saved to: {RESULTS_DIR}
    - maq_phase0_price_returns.png
    - maq_phase2_acf.png
    - maq_phase4_diagnostics.png
    - maq_phase4_aic_bic.png
""")
