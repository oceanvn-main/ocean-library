"""
Stages 9-12: white noise, MA(1), and AR(1) -- population spectrum vs
averaged periodogram, on processes with a KNOWN closed-form spectrum
(Python cross-check of exp_synth_stochastic.m).

PURPOSE
    Stages 1-8 were entirely deterministic. This script moves from finite
    Fourier decomposition to stochastic processes, and demonstrates:
        population spectrum  !=  periodogram from one finite realization
    A single periodogram realization is a noisy, inconsistent estimator;
    AVERAGING periodograms across many independent realizations converges
    toward the theoretical population spectrum s_Y(omega).

DATASET
    Simulated processes only:
        Stage 9  : mu + 3 known sinusoids (as in exp_synth_core.py) + white noise
        Stage 10 : pure white noise, sigma^2 = 1
        Stage 11 : MA(1), Y_t = eps_t + theta*eps_{t-1}, theta in {0.71, -0.63}
        Stage 12 : AR(1), Y_t = phi*Y_{t-1} + eps_t, phi in {0.84, -0.69}

ASSUMPTIONS
    1. sigma_eps^2 = 0.83 for Stage 9 (as specified); sigma^2 = 1 for
       Stages 10-12 (population spectrum formulas are linear in sigma^2,
       so the unit-variance choice keeps the numbers directly readable).
    2. AR(1)/MA(1) realizations use a burn-in (discarded initial samples)
       to reach the stationary distribution before the N=127 window used
       for estimation.
    3. n_realizations = 300 independent realizations per averaged
       periodogram (Monte Carlo estimate of E[periodogram]).

REPRODUCIBILITY
    np.random.default_rng(42) fixes all randomness; re-running reproduces
    identical figures and numbers.

See also: exp_synth_stochastic.m, exp_synth_core.py, exp_synth_leakage.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)

print("=== STAGE 9-12: STOCHASTIC PROCESSES -- POPULATION VS SAMPLE SPECTRUM (Python) ===\n")

N = 127
n = np.arange(N)
dOmega = 2 * np.pi / N
w_grid = 2 * np.pi * np.arange(N) / N
n_realizations = 300


def biased_gamma(x, max_lag):
    N_ = len(x)
    return np.array([np.sum(x[:N_ - k] * x[k:]) / N_ for k in range(max_lag + 1)])


# =========================================================================
# STAGE 9 -- Known sinusoids + white noise
# =========================================================================
print("--- Stage 9: 3 known sinusoids + white noise (sigma^2=0.83) ---")
mu = 3.7
A = np.array([2.1, 1.35, 0.72])
phi = np.array([0.43, -0.91, 1.27])
j_true = np.array([9, 27, 49])
w_true = 2 * np.pi * j_true / N
sigma2_9 = 0.83

signal_clean = (mu
                 + A[0] * np.cos(w_true[0] * n + phi[0])
                 + A[1] * np.cos(w_true[1] * n + phi[1])
                 + A[2] * np.cos(w_true[2] * n + phi[2]))
signal_clean = signal_clean - np.mean(signal_clean)

eps_one = np.sqrt(sigma2_9) * rng.standard_normal(N)
y_one = signal_clean + (eps_one - np.mean(eps_one))
P_one = np.abs(np.fft.fft(y_one)) ** 2 / (2 * np.pi * N)

P_stack9 = np.zeros((n_realizations, N))
for r in range(n_realizations):
    eps = np.sqrt(sigma2_9) * rng.standard_normal(N)
    yr = signal_clean + (eps - np.mean(eps))
    P_stack9[r] = np.abs(np.fft.fft(yr)) ** 2 / (2 * np.pi * N)
P_avg9 = P_stack9.mean(axis=0)

noise_floor_theory = sigma2_9 / (2 * np.pi)
mask_far = np.ones(N, dtype=bool)
for j in j_true:
    idxs = np.mod(np.arange(j - 2, j + 3), N)
    mask_far[idxs] = False
    idxs2 = np.mod(np.arange(N - j - 2, N - j + 3), N)
    mask_far[idxs2] = False
print(f"Averaged-periodogram noise floor (away from peaks) = {P_avg9[mask_far].mean():.4f}, "
      f"theory sigma^2/(2*pi) = {noise_floor_theory:.4f}")

fig9, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
half = N // 2
ax1.plot(w_grid[:half], P_one[:half], linewidth=1.0)
ax1.set_title("Stage 9: single realization (noisy periodogram)")
ax1.set_xlabel(r"$\omega$ (rad/sample)"); ax1.set_ylabel(r"$s(\omega)$")
ax2.plot(w_grid[:half], P_avg9[:half], linewidth=1.0)
ax2.axhline(noise_floor_theory, linestyle="--", label=r"noise floor $\sigma^2/(2\pi)$")
ax2.set_title(f"Stage 9: average of {n_realizations} realizations (smoother, floor -> theory)")
ax2.set_xlabel(r"$\omega$ (rad/sample)"); ax2.set_ylabel(r"$s(\omega)$")
ax2.legend(loc="best")
fig9.tight_layout()
fig9.savefig(os.path.join(RESULTS_DIR, "stage9_white_noise_plus_signal_py.png"), dpi=150)
plt.close(fig9)

# =========================================================================
# STAGE 10 -- Pure white noise: population vs sample autocovariance
# =========================================================================
print("\n--- Stage 10: Pure white noise, sigma^2=1 ---")
sigma2_10 = 1.0

eps_one10 = np.sqrt(sigma2_10) * rng.standard_normal(N)
gamma_one = biased_gamma(eps_one10, 5)
print("Single-realization hat_gamma_k (k=0..5):", np.round(gamma_one, 3))
print("  (population gamma_0=1, gamma_k=0 for k!=0 -- one realization will NOT match exactly)")

gamma_avg = np.zeros(6)
P_stack10 = np.zeros((n_realizations, N))
for r in range(n_realizations):
    eps = np.sqrt(sigma2_10) * rng.standard_normal(N)
    gamma_avg += biased_gamma(eps, 5)
    P_stack10[r] = np.abs(np.fft.fft(eps)) ** 2 / (2 * np.pi * N)
gamma_avg /= n_realizations
P_avg10 = P_stack10.mean(axis=0)
flat_theory10 = sigma2_10 / (2 * np.pi)

print(f"Averaged hat_gamma_k over {n_realizations} realizations (k=0..5):", np.round(gamma_avg, 3))
print(f"Averaged periodogram mean level = {P_avg10.mean():.5f}, theory s_Y(w)=sigma^2/(2*pi) = {flat_theory10:.5f}")

fig10, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(w_grid, P_avg10, linewidth=1.0, label=f"avg of {n_realizations} periodograms")
ax.axhline(flat_theory10, linestyle="--", linewidth=1.5, label=r"$s_Y(\omega) = \sigma^2/(2\pi)$")
ax.set_xlabel(r"$\omega$ (rad/sample)"); ax.set_ylabel(r"$s(\omega)$")
ax.set_title("Stage 10: Averaged periodogram -> flat population spectrum of white noise")
ax.set_xlim(0, np.pi)
ax.legend(loc="best")
fig10.tight_layout()
fig10.savefig(os.path.join(RESULTS_DIR, "stage10_white_noise_theory_py.png"), dpi=150)
plt.close(fig10)

# =========================================================================
# STAGE 11 -- MA(1) as a spectral filter
# =========================================================================
print("\n--- Stage 11: MA(1), sigma^2=1 ---")
thetas = [0.71, -0.63]
burn_ma = 200

fig11, axes11 = plt.subplots(1, len(thetas), figsize=(11, 4.5))
ma_endpoints = []
for i, theta in enumerate(thetas):
    s0 = (1 / (2 * np.pi)) * (1 + theta) ** 2
    s_pi = (1 / (2 * np.pi)) * (1 - theta) ** 2
    ma_endpoints.append((theta, s0, s_pi))
    print(f"theta={theta:+.2f}: s(0)={s0:.4f}, s(pi)={s_pi:.4f}  [theory: sigma^2/(2*pi)*(1+/-theta)^2]")

    P_stack = np.zeros((n_realizations, N))
    for r in range(n_realizations):
        eps_full = rng.standard_normal(N + burn_ma + 1)
        y_ma = eps_full[1:] + theta * eps_full[:-1]
        y_ma = y_ma[burn_ma:]
        P_stack[r] = np.abs(np.fft.fft(y_ma)) ** 2 / (2 * np.pi * N)
    P_avg = P_stack.mean(axis=0)
    s_theory = (1 / (2 * np.pi)) * (1 + theta ** 2 + 2 * theta * np.cos(w_grid))

    ax = axes11[i]
    ax.plot(w_grid[:half], P_avg[:half], linewidth=1.2, label="averaged periodogram")
    ax.plot(w_grid[:half], s_theory[:half], "--", linewidth=1.5, label=r"theory $s_Y(\omega)$")
    ax.set_title(f"MA(1), theta={theta:+.2f}")
    ax.set_xlabel(r"$\omega$ (rad/sample)"); ax.set_ylabel(r"$s(\omega)$")
    ax.legend(loc="best")
fig11.tight_layout()
fig11.savefig(os.path.join(RESULTS_DIR, "stage11_ma1_spectrum_py.png"), dpi=150)
plt.close(fig11)

# =========================================================================
# STAGE 12 -- AR(1)
# =========================================================================
print("\n--- Stage 12: AR(1), sigma^2=1 ---")
phis = [0.84, -0.69]
burn_ar = 500

fig12, axes12 = plt.subplots(2, len(phis), figsize=(11, 8))
ar_gamma0 = []
for i, ph in enumerate(phis):
    gamma0_theory = 1 / (1 - ph ** 2)
    ar_gamma0.append((ph, gamma0_theory))
    print(f"phi={ph:+.2f}: theoretical gamma_0={gamma0_theory:.4f}, gamma_k=gamma_0*phi^k")

    gamma_avg_ar = np.zeros(8)
    P_stack = np.zeros((n_realizations, N))
    for r in range(n_realizations):
        eps_full = rng.standard_normal(N + burn_ar)
        y_ar = np.zeros(N + burn_ar)
        for t in range(1, N + burn_ar):
            y_ar[t] = ph * y_ar[t - 1] + eps_full[t]
        y_ar = y_ar[burn_ar:]
        gamma_avg_ar += biased_gamma(y_ar, 7)
        P_stack[r] = np.abs(np.fft.fft(y_ar)) ** 2 / (2 * np.pi * N)
    gamma_avg_ar /= n_realizations
    P_avg = P_stack.mean(axis=0)
    s_theory = 1 / (2 * np.pi * (1 + ph ** 2 - 2 * ph * np.cos(w_grid)))

    print("  averaged hat_gamma_k (k=0..7):", np.round(gamma_avg_ar, 3))

    ax_acf = axes12[0, i]
    ax_acf.stem(np.arange(8), gamma_avg_ar, basefmt=" ")
    ax_acf.plot(np.arange(8), gamma0_theory * ph ** np.arange(8), "--", linewidth=1.2, label="theory")
    ax_acf.set_title(f"AR(1) phi={ph:+.2f}: avg autocovariance")
    ax_acf.set_xlabel("lag k"); ax_acf.set_ylabel(r"$\hat\gamma_k$")
    ax_acf.legend(loc="best")

    ax_psd = axes12[1, i]
    ax_psd.plot(w_grid[:half], P_avg[:half], linewidth=1.2, label="averaged periodogram")
    ax_psd.plot(w_grid[:half], s_theory[:half], "--", linewidth=1.5, label=r"theory $s_Y(\omega)$")
    ax_psd.set_title(f"AR(1) phi={ph:+.2f}: spectrum")
    ax_psd.set_xlabel(r"$\omega$ (rad/sample)"); ax_psd.set_ylabel(r"$s(\omega)$")
    ax_psd.legend(loc="best")
fig12.tight_layout()
fig12.savefig(os.path.join(RESULTS_DIR, "stage12_ar1_spectrum_py.png"), dpi=150)
plt.close(fig12)

# =========================================================================
# Compact diagnostic report
# =========================================================================
report_path = os.path.join(RESULTS_DIR, "report_stochastic_py.md")
with open(report_path, "w") as f:
    f.write("# Stage 9-12 Diagnostic Report (exp_synth_stochastic.py)\n\n")
    f.write("| Experiment | Main diagnostic | Result |\n")
    f.write("|---|---|---|\n")
    f.write(f"| White noise + signal (Stage 9) | avg noise floor vs theory | "
            f"{P_avg9[mask_far].mean():.4f} vs {noise_floor_theory:.4f} |\n")
    f.write(f"| Pure white noise (Stage 10) | avg periodogram mean vs sigma^2/(2pi) | "
            f"{P_avg10.mean():.5f} vs {flat_theory10:.5f} |\n")
    for theta, s0, s_pi in ma_endpoints:
        f.write(f"| MA(1) theta={theta:.2f} (Stage 11) | s(0), s(pi) theory | {s0:.4f}, {s_pi:.4f} |\n")
    for ph, g0 in ar_gamma0:
        f.write(f"| AR(1) phi={ph:.2f} (Stage 12) | gamma_0 theory | {g0:.4f} |\n")
print(f"\nReport written to {report_path}")

print("\n=== STAGE 9-12 COMPLETE ===")
