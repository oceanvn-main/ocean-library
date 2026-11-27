# Synthetic Spectrum Verification — Compact Report

A controlled, ground-truth experiment verifying the full chain

```
time samples -> Fourier basis -> frequency coefficients -> variance decomposition -> periodogram
time samples -> sample autocovariance -> Fourier transform -> periodogram
```

before ever touching real data (e.g. the sunspot series in [`../exp_population_spectrum.m`](../exp_population_spectrum.m)).
Every number below comes from a real run of the Python scripts in this folder
(`.venv` — numpy/scipy/matplotlib); the MATLAB scripts (`exp_synth_core.m`,
`exp_synth_leakage.m`, `exp_synth_stochastic.m`) implement identical math and
are ready to run in the MATLAB desktop (not executed in this session — no
attachable MATLAB instance was available here).

**Ground truth signal** (Stages 1–9): `N=127` (odd — no Nyquist special case),
`Ts=0.4 s` (`fs=2.5` samples/s), `dOmega = 2*pi/127 = 0.049474` rad/sample.

```
y[n] = 3.7 + 2.1*cos(w1*n+0.43) + 1.35*cos(w2*n-0.91) + 0.72*cos(w3*n+1.27)
w1 = 2*pi*9/127,  w2 = 2*pi*27/127,  w3 = 2*pi*49/127
```

Expected total variance from the three components: `0.5*sum(A_j^2) = 3.37545`.

## Results table

| Experiment | Main diagnostic | Result |
|---|---|---|
| Fourier reconstruction | max&#124;y_hat − y&#124; over full 127-column basis | **3.3e-14** (machine precision) |
| Orthogonality | max off-diagonal of X'X (127×127) | **8.5e-13**; diag: `1'1=127`, `c_j'c_j=s_j's_j=63.5=N/2` |
| Variance decomposition | `gamma0_hat` vs `sum(V_j)`; `V_j` at j=9/27/49 vs `A_j^2/2` | `3.375450` vs `3.375450`; `2.205000/0.911250/0.259200` — exact match |
| DFT equivalence | max&#124;X(w_j) − N/2·(alpha_j−i·delta_j)&#124; (j=9,27,49) | **9.4e-13** |
| Two periodogram routes | max&#124;s_A(w) − s_B(w)&#124; over the full Fourier grid | **4.3e-14** |
| Variance/area relation | `sum_j 2*s(w_j)*dOmega` vs `gamma0_hat` | `3.375450` vs `3.375450` |
| Leakage (on-grid, j=27) | &#124;X&#124;² at bins 25–29 | `[0, 0, 7348.8, 0, 0]` — energy in exactly one bin |
| Leakage (off-grid, j=27.37) | &#124;X&#124;² at bins 25–29 | `[121.2, 350.1, 4638.4, 1547.4, 223.8]` — spread across neighbors |
| Window effect | boxcar vs Hann half-max lobe width | `0.04207` vs `0.07176` rad/sample — Hann is wider, suppresses distant sidelobes |
| Resolution vs N | half-max lobe width, N=63/127/251 | `0.08680 / 0.04207 / 0.02204` rad/sample — shrinks with N as `dOmega=2*pi/N` does |
| White noise + signal (σ²=0.83) | averaged-periodogram noise floor vs `σ²/(2π)` | `0.1327` vs `0.1321` (300 realizations) |
| Pure white noise (σ²=1) | avg `hat_gamma_k` (k=0..5) vs `[1,0,0,0,0,0]`; avg periodogram mean vs `1/(2π)` | `[1.005,-0.003,0.002,-0.003,-0.004,0.002]`; `0.15991` vs `0.15915` |
| MA(1), θ=+0.71 | `s(0)`, `s(π)` theory | `0.4654`, `0.0134` — power shifted to low frequency |
| MA(1), θ=−0.63 | `s(0)`, `s(π)` theory | `0.0218`, `0.4229` — power shifted to high frequency |
| AR(1), φ=+0.84 | `gamma_0` theory vs avg `hat_gamma_0`; spectral concentration | `3.3967` vs `3.400`; concentrated at ω=0 |
| AR(1), φ=−0.69 | `gamma_0` theory vs avg `hat_gamma_0`; ACF sign alternation | `1.9088` vs `1.924`; alternating sign, concentrated at ω=π |

## Files

| Stage(s) | MATLAB | Python | Key figures (`Results/`) |
|---|---|---|---|
| 1–6 (core deterministic chain) | `exp_synth_core.m` | `exp_synth_core.py` | `stage1_reconstruction*`, `stage2_orthogonality*`, `stage3_variance_decomposition*`, `stage5_periodogram_routes*` |
| 7–8 (leakage, windowing, resolution) | `exp_synth_leakage.m` | `exp_synth_leakage.py` | `stage7_leakage_ongrid_vs_offgrid*`, `stage7b_boxcar_vs_hann*`, `stage8_resolution_vs_N*` |
| 9–12 (stochastic: white noise, MA(1), AR(1)) | `exp_synth_stochastic.m` | `exp_synth_stochastic.py` | `stage9_white_noise_plus_signal*`, `stage10_white_noise_theory*`, `stage11_ma1_spectrum*`, `stage12_ar1_spectrum*` |

Per-stage-group raw diagnostics (auto-generated on each run):
`Results/report_core*.md`, `Results/report_leakage*.md`, `Results/report_stochastic*.md`.

## Axis convention

Every frequency-domain figure in this experiment uses discrete-time angular
frequency **omega, in rad/sample** on the horizontal axis — never physical Hz,
and never mixed in the same figure. To convert: `Omega = omega / Ts` (rad/s),
or `f = omega / (2*pi*Ts)` (Hz), using `Ts = 0.4 s/sample` throughout.

## Key gotcha found while building this

The biased sample autocovariance `hat_gamma_k = (1/N) * sum_n x[n]*x[n-k]`
**must divide by `N`, not by the number of overlapping terms `N-k`**. A naive
`np.mean(x[:N-k] * x[k:])` (or a MATLAB `mean(...)` over the same slice)
silently divides by `(N-k)` instead, which is the *unbiased* estimator —
and it breaks the Stage 5 identity `s_A(omega) == s_B(omega)` for every
`omega != 0`. Both `exp_synth_core.m` and `exp_synth_core.py` divide by `N`
explicitly (`sum(...)/N`), not via `mean()`.

## What Stage 1–6 established before adding any randomness

1. With a **complete** Fourier basis (`1 + 2*63 = 127` columns for `N=127`
   samples), eq. 6.2.6 is an exact change of coordinates, not a fitted
   approximation — reconstruction error is at floating-point noise level.
2. The basis columns are exactly orthogonal (`X'X` is diagonal to ~1e-13),
   giving concrete meaning to "different Fourier frequencies are orthogonal
   directions."
3. `hat_gamma_0 = sum_j V_j`: the sample variance is *exactly* partitioned
   across frequencies, and for the three inserted tones `V_j = A_j^2/2`
   recovers the true amplitudes.
4. The OLS Fourier coefficients and the raw DFT coefficients are the same
   projection expressed two ways: `X(w_j) = (N/2)*(alpha_j - i*delta_j)`.
5. The periodogram computed from the Fourier transform of the sample
   autocovariance (Route A) and from `|DFT|^2` directly (Route B) agree
   to floating-point precision — confirming the Wiener–Khinchin identity
   holds exactly at finite N, not just asymptotically.
6. Periodogram area and sample variance are the same quantity viewed two
   ways: `sum_j 2*s(w_j)*dOmega = gamma0_hat`.

Only after all six of these held to machine precision did the experiment
move on to windowing/leakage (7–8) and stochastic processes (9–12), per the
requested conceptual progression.
