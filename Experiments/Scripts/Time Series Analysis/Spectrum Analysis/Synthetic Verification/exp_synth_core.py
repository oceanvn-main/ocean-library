"""
Stages 1-6: Fourier basis, variance decomposition, and the two periodogram
routes, verified on a KNOWN synthetic process (Python cross-check of
exp_synth_core.m).

PURPOSE
    Verify the full textbook chain on a signal whose true frequencies,
    amplitudes, and variance decomposition are known in advance:
        time samples -> Fourier basis -> frequency coefficients
                      -> variance decomposition -> periodogram
    and, independently:
        time samples -> sample autocovariances -> Fourier transform -> periodogram
    Both routes must agree exactly (up to floating-point error).

DATASET
    None -- a deterministic signal y[n] = mu + sum_j A_j*cos(w_j*n+phi_j),
    n = 0..N-1, N = 127 (odd), with w_j placed exactly on the Fourier grid
    (j = 9, 27, 49) so each component's true variance contribution is
    known analytically: V_j = A_j^2 / 2.

ASSUMPTIONS
    1. Ts = 0.4 s/sample; every frequency axis here is discrete-time
       omega (rad/sample), never mixed with physical Hz.
    2. The Fourier grid frequencies omega_j = 2*pi*j/N, j=1..(N-1)/2, plus
       the constant term, form a COMPLETE orthogonal basis of R^N.
    3. hat_gamma_k = (1/N) * sum_n x[n]*x[n-k] (biased autocovariance).
       CRITICAL: divide by N, not by the number of overlapping terms
       (N-k) -- e.g. np.mean(x[:N-k]*x[k:]) silently divides by (N-k) and
       breaks the periodogram = FT(autocovariance) identity below.

REPRODUCIBILITY
    Fully deterministic (no randomness).

See also: exp_synth_core.m, exp_synth_leakage.py, exp_synth_stochastic.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== STAGE 1-6: SYNTHETIC FOURIER / PERIODOGRAM VERIFICATION (Python) ===\n")

# --- Core settings ---
N = 127                    # odd length: eq. 6.2.6 applies with no Nyquist special case
Ts = 0.4                   # s/sample
fs = 1.0 / Ts               # samples/s
dOmega = 2 * np.pi / N       # Fourier-grid angular-frequency spacing (rad/sample)

n = np.arange(N)

mu = 3.7
A = np.array([2.1, 1.35, 0.72])
phi = np.array([0.43, -0.91, 1.27])
j_true = np.array([9, 27, 49])          # deliberately separated, non-adjacent grid bins
w_true = 2 * np.pi * j_true / N

y = (mu
     + A[0] * np.cos(w_true[0] * n + phi[0])
     + A[1] * np.cos(w_true[1] * n + phi[1])
     + A[2] * np.cos(w_true[2] * n + phi[2]))

print(f"N={N}, Ts={Ts} s, fs={fs} samples/s, dOmega=2*pi/N={dOmega:.6f} rad/sample\n")

# =========================================================================
# STAGE 1 -- Fourier basis reconstructs the sample exactly
# =========================================================================
print("--- Stage 1: Fourier regression (eq. 6.2.6), full basis ---")

J = (N - 1) // 2                      # 63 non-DC Fourier-grid frequencies
j_grid = np.arange(1, J + 1)
omega_grid = 2 * np.pi * j_grid / N   # (J,)

C = np.cos(np.outer(n, omega_grid))   # (N, J)
S = np.sin(np.outer(n, omega_grid))   # (N, J)
X = np.hstack([np.ones((N, 1)), C, S])  # (N, N) -- complete orthogonal basis of R^N

beta, *_ = np.linalg.lstsq(X, y, rcond=None)
a0_hat = beta[0]
alpha_hat = beta[1:1 + J]
delta_hat = beta[1 + J:1 + 2 * J]

y_hat = X @ beta
recon_err = np.max(np.abs(y_hat - y))
print(f"Reconstruction max |y_hat - y| = {recon_err:.3e}  (expect ~0, machine precision)")
print(f"a0_hat = {a0_hat:.10f} vs mean(y) = {np.mean(y):.10f}\n")

# =========================================================================
# STAGE 2 -- Orthogonality of the Fourier design columns
# =========================================================================
print("--- Stage 2: Orthogonality of X (X^T X) ---")
XtX = X.T @ X
off_diag_max = np.max(np.abs(XtX - np.diag(np.diag(XtX))))
print(f"max |off-diagonal(X^T X)| = {off_diag_max:.3e}  (expect ~0)")
print(f"1'1 = {XtX[0,0]:.4f} (expect N={N}); c_1'c_1 = {XtX[1,1]:.4f} (expect N/2={N/2}); "
      f"s_1's_1 = {XtX[1+J,1+J]:.4f}\n")

# =========================================================================
# STAGE 3 -- Sample variance decomposition across frequencies
# =========================================================================
print("--- Stage 3: Variance decomposition ---")
y_bar = np.mean(y)
x = y - y_bar
gamma0_hat = np.mean(x ** 2)          # hat_gamma_0 = (1/N) sum x[n]^2

z_components = C * alpha_hat + S * delta_hat   # (N, J), column j = z_j[n]
V = np.mean(z_components ** 2, axis=0)          # (J,)
V_theory = 0.5 * (alpha_hat ** 2 + delta_hat ** 2)

print(f"max |V - V_theory| = {np.max(np.abs(V - V_theory)):.3e}")
print(f"gamma0_hat = {gamma0_hat:.6f}, sum(V_j) = {np.sum(V):.6f}, diff = {gamma0_hat - np.sum(V):.3e}")

expected_total_V = 0.5 * np.sum(A ** 2)
print(f"Expected total variance from known sinusoids = 0.5*sum(A_j^2) = {expected_total_V:.5f}")
print(f"{'j':<6}{'V_j (hat)':<14}{'A_j^2/2 (true)':<14}")
for k in range(3):
    idx = j_true[k] - 1
    print(f"{j_true[k]:<6}{V[idx]:<14.6f}{A[k]**2/2:<14.6f}")
print()

# =========================================================================
# STAGE 4 -- DFT coefficients equal the regression coefficients
# =========================================================================
print("--- Stage 4: DFT vs regression-coefficient equivalence ---")
Xw = np.fft.fft(x)          # bin j <-> omega_j = 2*pi*j/N

print(f"{'j':<4}{'alpha_j':<10}{'delta_j':<10}{'X(w_j)':<24}{'predicted N/2*(a-id)':<24}{'|diff|':<10}")
stage4_rows = []
for j in j_true:
    idx = j - 1
    Xw_j = Xw[j]
    pred = (N / 2) * (alpha_hat[idx] - 1j * delta_hat[idx])
    d = abs(Xw_j - pred)
    print(f"{j:<4}{alpha_hat[idx]:<10.4f}{delta_hat[idx]:<10.4f}"
          f"{f'{Xw_j.real:.3f}{Xw_j.imag:+.3f}i':<24}{f'{pred.real:.3f}{pred.imag:+.3f}i':<24}{d:<10.3e}")
    stage4_rows.append((j, alpha_hat[idx], delta_hat[idx], Xw_j, pred, d))
print()

# =========================================================================
# STAGE 5 -- Periodogram from two independent routes
# =========================================================================
print("--- Stage 5: Periodogram -- autocovariance route vs DFT route ---")

# Biased sample autocovariance, lags k = 0..N-1.
# CRITICAL: divide by N (not N-k) -- see module docstring ASSUMPTIONS(3).
gamma_hat = np.array([np.sum(x[:N - k] * x[k:]) / N for k in range(N)])

ks = np.arange(1, N)


def s_A(w):
    return (gamma_hat[0] + 2 * np.sum(gamma_hat[1:] * np.cos(w * ks))) / (2 * np.pi)


grid_w = 2 * np.pi * np.arange(N) / N
sA_grid = np.array([s_A(w) for w in grid_w])
sB_grid = np.abs(Xw) ** 2 / (2 * np.pi * N)

max_diff_grid = np.max(np.abs(sA_grid - sB_grid))
print(f"max |s_A(w_r) - s_B(w_r)| over full Fourier grid = {max_diff_grid:.3e}\n")

# =========================================================================
# STAGE 6 -- Periodogram area equals sample variance
# =========================================================================
print("--- Stage 6: Periodogram area = sample variance ---")
s_positive = sB_grid[1:J + 1]              # s(w_j) for j=1..J (positive frequencies only)
area_terms = 2 * s_positive * dOmega

print(f"max |2*s(w_j)*dOmega - V_j| = {np.max(np.abs(area_terms - V)):.3e}")
print(f"sum_j 2*s(w_j)*dOmega = {np.sum(area_terms):.6f}  vs  gamma0_hat = {gamma0_hat:.6f}  "
      f"(diff = {np.sum(area_terms) - gamma0_hat:.3e})\n")

# =========================================================================
# Visualization
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(n, y, "o", label="y[n] (original)", markersize=4)
ax1.plot(n, y_hat, "-", linewidth=1.2, label="reconstructed")
ax1.set_xlabel("n (sample index)")
ax1.set_ylabel("y[n]")
ax1.set_title(f"Stage 1: Fourier reconstruction (max |err| = {recon_err:.2e})")
ax1.legend(loc="best")
fig1.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, "stage1_reconstruction_py.png"), dpi=150)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 5.5))
im = ax2.imshow(XtX, aspect="equal")
fig2.colorbar(im, ax=ax2)
ax2.set_title("Stage 2: X^T X (near-diagonal => orthogonal Fourier directions)")
ax2.set_xlabel("basis column index")
ax2.set_ylabel("basis column index")
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, "stage2_orthogonality_py.png"), dpi=150)
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 4.5))
ax3.bar(j_grid, V)
ax3.set_xlabel("Fourier-grid index j")
ax3.set_ylabel("V_j (variance contribution)")
ax3.set_title(f"Stage 3: V_j per frequency (sum = {np.sum(V):.4f}, gamma0_hat = {gamma0_hat:.4f})")
fig3.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, "stage3_variance_decomposition_py.png"), dpi=150)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(8, 4.5))
ax4.plot(grid_w, sA_grid, "o", markersize=3, label="Route A: FT(autocovariance)")
ax4.plot(grid_w, sB_grid, "-", linewidth=1.2, label=r"Route B: $|X(\omega)|^2/(2\pi N)$")
ax4.set_xlabel(r"$\omega$ (rad/sample)")
ax4.set_ylabel(r"$s(\omega)$")
ax4.set_title(f"Stage 5: max route diff = {max_diff_grid:.2e}")
ax4.set_xlim(0, np.pi)
ax4.legend(loc="best")
fig4.tight_layout()
fig4.savefig(os.path.join(RESULTS_DIR, "stage5_periodogram_routes_py.png"), dpi=150)
plt.close(fig4)

# =========================================================================
# Compact diagnostic report
# =========================================================================
report_path = os.path.join(RESULTS_DIR, "report_core_py.md")
with open(report_path, "w") as f:
    f.write("# Stage 1-6 Diagnostic Report (exp_synth_core.py)\n\n")
    f.write("| Experiment | Main diagnostic | Result |\n")
    f.write("|---|---|---|\n")
    f.write(f"| Fourier reconstruction | max\\|y_hat-y\\| | {recon_err:.3e} |\n")
    f.write(f"| Orthogonality | max off-diag(X^T X) | {off_diag_max:.3e} |\n")
    f.write(f"| Variance decomposition | gamma0_hat vs sum(V_j) | {gamma0_hat:.6f} vs {np.sum(V):.6f} |\n")
    max_dft_diff = max(row[5] for row in stage4_rows)
    f.write(f"| DFT equivalence | max\\|X(w_j) - N/2(a-id)\\| (j=9,27,49) | {max_dft_diff:.3e} |\n")
    f.write(f"| Two periodogram routes | max\\|s_A - s_B\\| on grid | {max_diff_grid:.3e} |\n")
    f.write(f"| Variance/area relation | sum(2 s(w_j) dOmega) vs gamma0_hat | "
            f"{np.sum(area_terms):.6f} vs {gamma0_hat:.6f} |\n")
print(f"Report written to {report_path}")

print("\n=== STAGE 1-6 COMPLETE ===")
