"""
Stages 7-8: spectral leakage from finite observation, window comparison
(boxcar vs Hann), and frequency-resolution vs N (Python cross-check of
exp_synth_leakage.m).

PURPOSE
    Stage 1-6 used only ON-GRID frequencies, where the finite Fourier
    basis reconstructs the signal exactly. This script deliberately
    breaks that ideal condition:
        finite observation -> rectangular window -> frequency convolution -> leakage
    by shifting one component off the Fourier grid, and separately studies
    how observation length N controls frequency-grid RESOLUTION (a related
    but distinct concept from window-shape leakage).

DATASET
    Same closed-form 3-tone signal as exp_synth_core.py, with omega_2
    perturbed from an exact grid frequency (j=27) to an off-grid value
    (j=27.37), amplitude and phase unchanged.

ASSUMPTIONS
    1. Ts = 0.4 s/sample fixed throughout; frequency axes are omega
       (rad/sample), never mixed with Hz in the same figure.
    2. "Boxcar" window == no window == the periodogram used in
       exp_synth_core.py.
    3. Window power normalization U = mean(win**2) makes periodogram
       height comparable across window choices.

REPRODUCIBILITY
    Fully deterministic (no randomness).

See also: exp_synth_leakage.m, exp_synth_core.py, exp_synth_stochastic.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== STAGE 7-8: LEAKAGE, WINDOWING, AND RESOLUTION (Python) ===\n")

Ts = 0.4
mu = 3.7
A = np.array([2.1, 1.35, 0.72])
phi = np.array([0.43, -0.91, 1.27])

j1, j3 = 9, 49
j2_on = 27
j2_off = 27.37   # Stage 7: deliberately off the Fourier grid


def make_signal(N, j2):
    n = np.arange(N)
    return (mu
            + A[0] * np.cos(2 * np.pi * j1 / N * n + phi[0])
            + A[1] * np.cos(2 * np.pi * j2 / N * n + phi[1])
            + A[2] * np.cos(2 * np.pi * j3 / N * n + phi[2]))


def dense_periodogram(x, n, w_vec, win):
    U = np.mean(win ** 2)
    xw = x * win
    P = np.zeros_like(w_vec)
    for i, w in enumerate(w_vec):
        Xw = np.sum(xw * np.exp(-1j * w * n))
        P[i] = np.abs(Xw) ** 2 / (2 * np.pi * len(n) * U)
    return P


def half_max_width(P, w_vec):
    half = np.max(P) / 2
    idx = np.where(P >= half)[0]
    return w_vec[idx[-1]] - w_vec[idx[0]]


# =========================================================================
# STAGE 7 -- On-grid vs off-grid: spectral leakage
# =========================================================================
print("--- Stage 7: On-grid vs off-grid leakage (N=127) ---")
N = 127
n = np.arange(N)
w2_on = 2 * np.pi * j2_on / N

x_on = make_signal(N, j2_on); x_on = x_on - np.mean(x_on)
x_off = make_signal(N, j2_off); x_off = x_off - np.mean(x_off)

X_on_fft = np.fft.fft(x_on)
X_off_fft = np.fft.fft(x_off)
bins_near = np.arange(25, 30)
print("|X|^2 near bin 27 (on-grid) :", np.round(np.abs(X_on_fft[bins_near]) ** 2, 1))
print("|X|^2 near bin 27 (off-grid):", np.round(np.abs(X_off_fft[bins_near]) ** 2, 1))

w_dense = np.linspace(w2_on - 15 * (2 * np.pi / N), w2_on + 15 * (2 * np.pi / N), 3000)
P_on = dense_periodogram(x_on, n, w_dense, np.ones(N))
P_off = dense_periodogram(x_off, n, w_dense, np.ones(N))

fig7a, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(w_dense, 10 * np.log10(P_on), linewidth=1.3, label="on-grid (j=27)")
ax.plot(w_dense, 10 * np.log10(P_off), linewidth=1.3, label="off-grid (j=27.37)")
ax.axvline(w2_on, linestyle=":", label="omega grid bin 27")
ax.set_xlabel(r"$\omega$ (rad/sample)")
ax.set_ylabel(r"$s(\omega)$ [dB]")
ax.set_title("Stage 7: Boxcar periodogram near j=27 -- on-grid concentrates, off-grid leaks")
ax.legend(loc="best")
fig7a.tight_layout()
fig7a.savefig(os.path.join(RESULTS_DIR, "stage7_leakage_ongrid_vs_offgrid_py.png"), dpi=150)
plt.close(fig7a)

# --- Stage 7b: window comparison (boxcar vs Hann), off-grid signal only ---
print("\n--- Stage 7b: Boxcar vs Hann window (off-grid signal) ---")
win_box = np.ones(N)
win_hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

P_box = dense_periodogram(x_off, n, w_dense, win_box)
P_hann = dense_periodogram(x_off, n, w_dense, win_hann)

w_box = half_max_width(P_box, w_dense)
w_hann = half_max_width(P_hann, w_dense)
print(f"Boxcar main-lobe half-max width : {w_box:.5f} rad/sample")
print(f"Hann   main-lobe half-max width : {w_hann:.5f} rad/sample")
print("(Hann should be WIDER but should suppress distant sidelobes more)")

fig7b, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(w_dense, 10 * np.log10(P_box), linewidth=1.3, label=f"boxcar (width={w_box:.4f})")
ax.plot(w_dense, 10 * np.log10(P_hann), linewidth=1.3, label=f"Hann (width={w_hann:.4f})")
ax.set_xlabel(r"$\omega$ (rad/sample)")
ax.set_ylabel(r"$s(\omega)$ [dB]")
ax.set_title("Stage 7b: Window effect on off-grid leakage (same signal, different window)")
ax.legend(loc="best")
fig7b.tight_layout()
fig7b.savefig(os.path.join(RESULTS_DIR, "stage7b_boxcar_vs_hann_py.png"), dpi=150)
plt.close(fig7b)

# =========================================================================
# STAGE 8 -- Frequency resolution vs sample length N
# =========================================================================
print(f"\n--- Stage 8: Resolution vs N (Ts fixed = {Ts}) ---")
Ns = [63, 127, 251]
widths = []

fig8, ax = plt.subplots(figsize=(8, 4.5))
for Ni in Ns:
    ni = np.arange(Ni)
    xi = make_signal(Ni, j2_off); xi = xi - np.mean(xi)
    wi = np.linspace(w2_on - 15 * (2 * np.pi / Ni), w2_on + 15 * (2 * np.pi / Ni), 3000)
    Pi = dense_periodogram(xi, ni, wi, np.ones(Ni))
    wid = half_max_width(Pi, wi)
    widths.append(wid)
    dOmega_i = 2 * np.pi / Ni
    print(f"N={Ni:4d}: dOmega=2*pi/N={dOmega_i:.5f} rad/sample, half-max lobe width={wid:.5f} rad/sample")
    ax.plot(wi, 10 * np.log10(Pi), linewidth=1.3, label=f"N={Ni} (dOmega={dOmega_i:.4f})")

ax.set_xlabel(r"$\omega$ (rad/sample)")
ax.set_ylabel(r"$s(\omega)$ [dB]")
ax.set_title("Stage 8: Longer N => narrower Fourier-grid spacing => better resolution")
ax.legend(loc="best")
fig8.tight_layout()
fig8.savefig(os.path.join(RESULTS_DIR, "stage8_resolution_vs_N_py.png"), dpi=150)
plt.close(fig8)

# =========================================================================
# Compact diagnostic report
# =========================================================================
report_path = os.path.join(RESULTS_DIR, "report_leakage_py.md")
with open(report_path, "w") as f:
    f.write("# Stage 7-8 Diagnostic Report (exp_synth_leakage.py)\n\n")
    f.write("| Experiment | Main diagnostic | Result |\n")
    f.write("|---|---|---|\n")
    f.write(f"| Leakage (on-grid) | \\|X\\|^2 near bin 27 | {np.round(np.abs(X_on_fft[bins_near])**2,1).tolist()} |\n")
    f.write(f"| Leakage (off-grid) | \\|X\\|^2 near bin 27 | {np.round(np.abs(X_off_fft[bins_near])**2,1).tolist()} |\n")
    f.write(f"| Window effect | half-max width, boxcar vs Hann | {w_box:.5f} vs {w_hann:.5f} rad/sample |\n")
    f.write(f"| Resolution vs N | half-max width, N=63/127/251 | "
            f"{widths[0]:.5f} / {widths[1]:.5f} / {widths[2]:.5f} rad/sample |\n")
print(f"\nReport written to {report_path}")

print("\n=== STAGE 7-8 COMPLETE ===")
