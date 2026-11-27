# Gibbs Phenomenon & Ideal Lowpass Filter Guide

This guide documents the theoretical framework, mathematical formulation, and implementation of the Gibbs Phenomenon and Square-Summability analysis for the Ideal Lowpass Filter.

---

## 1. Mathematical Framework

### A. The Ideal Lowpass Filter
An ideal lowpass filter is a frequency-selective system that passes all frequency components below a cutoff frequency $w_c$ (where $0 < w_c < \pi$) without attenuation, and completely attenuates all components above $w_c$. Symmetrically extended to the periodic interval $[-\pi, \pi]$, the ideal frequency response $H_{\text{lp}}(e^{j\omega})$ is defined as:

$$
H_{\text{lp}}(e^{j\omega}) = \begin{cases} 
1, & |\omega| < \omega_c \\
0, & \omega_c < |\omega| \le \pi 
\end{cases}
$$

$H_{\text{lp}}(e^{j\omega})$ is periodic with a period of $2\pi$.

### B. Analytical Impulse Response
The discrete-time impulse response $h_{\text{lp}}[n]$ is the inverse discrete-time Fourier transform (IDFT) of the ideal frequency response:

$$
h_{\text{lp}}[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} H_{\text{lp}}(e^{j\omega}) e^{j\omega n} d\omega = \frac{1}{2\pi} \int_{-\omega_c}^{\omega_c} 1 \cdot e^{j\omega n} d\omega
$$

Evaluating this integral yields:
* **For $n = 0$** (applying L'Hôpital's rule or direct integration):
  $$
  h_{\text{lp}}[0] = \frac{\omega_c}{\pi}
  $$
* **For $n \neq 0$**:
  $$
  h_{\text{lp}}[n] = \frac{\sin(\omega_c n)}{\pi n}
  $$

This is the classic symmetric sinc function. It is infinitely long and not absolutely summable, meaning the ideal filter is non-causal and unstable in practice.

### C. Truncated Fourier Series Approximation
To realize a practical approximation, we truncate the infinite impulse response using a rectangular window of length $2M + 1$, which centers $M$ terms on each side of the origin. The frequency response of this truncated approximation is:

$$
H_M(e^{j\omega}) = \sum_{n=-M}^{M} h_{\text{lp}}[n] e^{-j\omega n}
$$

Because $h_{\text{lp}}[n]$ is real and perfectly symmetric ($h_{\text{lp}}[n] = h_{\text{lp}}[-n]$), the complex exponentials pair up into cosines, simplifying the calculation to a real-valued cosine sum:

$$
H_M(e^{j\omega}) = h_{\text{lp}}[0] + 2 \sum_{n=1}^{M} h_{\text{lp}}[n] \cos(\omega n)
$$

### D. The Gibbs Phenomenon and Overshoot
As $M \to \infty$, the Fourier series $H_M(e^{j\omega})$ converges to the ideal response $H_{\text{lp}}(e^{j\omega})$ in a mean-square sense. However, because of the sharp discontinuity at $\omega = \pm\omega_c$, the approximation exhibits oscillatory behavior (ripples) near the transition. 

The peak of the first overshoot occurs at the local maximum of $H_M(e^{j\omega})$ immediately to the left of the discontinuity $\omega_c$:

$$
\text{Peak}_M = \max_{\omega \in [0, \omega_c]} H_M(e^{j\omega})
$$

The percentage overshoot relative to the step transition height of $1.0$ is defined as:

$$
\text{Overshoot}_M = (\text{Peak}_M - 1.0) \times 100\%
$$

As $M \to \infty$, the width of the ripples decreases to zero, but their height does not decay. Instead, the overshoot percentage converges asymptotically to the theoretical Gibbs constant:

$$
\lim_{M \to \infty} \text{Overshoot}_M \approx 8.95\%
$$

### E. Parseval's Theorem and Mean-Square Error
While pointwise convergence fails at the point of discontinuity, mean-square convergence holds. The total energy of the ideal filter in the frequency domain is:

$$
E_{\text{total}} = \frac{1}{2\pi} \int_{-\pi}^{\pi} |H_{\text{lp}}(e^{j\omega})|^2 d\omega = \frac{\omega_c}{\pi}
$$

The energy of the truncated approximation $H_M(e^{j\omega})$ is:

$$
E_{\text{truncated}, M} = h_{\text{lp}}[0]^2 + 2 \sum_{n=1}^{M} \left(h_{\text{lp}}[n]\right)^2
$$

The mean-squared approximation error (which represents the energy of the difference between the ideal response and the approximation) is defined as:

$$
\text{Error\_Energy}_M = \frac{1}{2\pi} \int_{-\pi}^{\pi} |H_{\text{lp}}(e^{j\omega}) - H_M(e^{j\omega})|^2 d\omega
$$

By Parseval's relation, this frequency-domain integral is exactly equal to the time-domain tail energy (the energy of the discarded coefficients):

$$
\text{Error\_Energy}_M = E_{\text{total}} - E_{\text{truncated}, M} = \frac{\omega_c}{\pi} - \left[ h_{\text{lp}}[0]^2 + 2 \sum_{n=1}^{M} \left(h_{\text{lp}}[n]\right)^2 \right]
$$

As $M \to \infty$, the tail energy converges to $0$, proving that the approximation converges in the mean-square (L2) sense.

---

## 2. API Reference: `OceanMath.Signal.IdealLowpass`

The core mathematical engine is implemented as a reusable class in the namespace `OceanMath.Signal.IdealLowpass`.

### Constructor
* **`obj = IdealLowpass(w_c)`**
  Instantiates the filter object with a specified cutoff frequency $w_c$.
  * **Input**: `w_c` (double) - Cutoff frequency in radians/sample. Must satisfy $0 < w_c < \pi$.
  * **Validation**: Performs strict error handling for non-scalar, non-double, or out-of-bounds parameters.

### Public Methods
* **`h = impulseResponse(obj, n)`**
  Computes the analytical impulse response $h_{\text{lp}}[n]$ for a vector of sample indices `n`.
  * **Input**: `n` (double vector) - Sample indices.
  * **Output**: `h` (double vector) - Calculated impulse response. Uses L'Hôpital's limit $\omega_c / \pi$ at $n=0$ to prevent division by zero.
  
* **`H = frequencyResponseApproximation(obj, w, M)`**
  Computes the truncated Fourier approximation $H_M(e^{j\omega})$ for a vector of frequencies `w` and truncation limit `M`.
  * **Inputs**:
    * `w` (double vector) - Frequencies in radians/sample.
    * `M` (scalar integer) - Truncation parameter ($M \ge 0$).
  * **Output**: `H` (double vector) - Real-valued frequency response approximation.

* **`[peakVal, peakFreq, pctOvershoot] = calculateOvershoot(obj, M)`**
  Performs a high-precision grid search over 10,000 points spanning $[0, w_c]$ to locate the peak overshoot.
  * **Input**: `M` (scalar integer) - Truncation parameter ($M \ge 0$).
  * **Outputs**:
    * `peakVal` (double) - Peak amplitude value of the overshoot.
    * `peakFreq` (double) - Frequency (rad/sample) where the first overshoot occurs.
    * `pctOvershoot` (double) - Overshoot percentage relative to $1.0$.

* **`[errFreq, errTime] = verifyParseval(obj, M, numGridPoints)`**
  Validates Parseval's identity by computing the error energy via both numerical frequency-domain integration and analytical time-domain tail-energy calculation.
  * **Inputs**:
    * `M` (scalar integer) - Truncation parameter.
    * `numGridPoints` (scalar integer, optional) - Grid points for numerical integration (default: 10,000).
  * **Outputs**:
    * `errFreq` (double) - L2 energy error computed in the frequency domain via trapezoidal integration over $[-\pi, \pi]$.
    * `errTime` (double) - L2 energy error computed analytically in the time domain.

---

## 3. Copy-Paste Recipe

You can run the full scientific experiment, verify Parseval's theorem, and visualize the Gibbs phenomenon using the following copy-paste script:

```matlab
% exp_gibbs_phenomenon_recipe.m
% Copy-paste recipe to run the Gibbs Phenomenon & Parseval verification.

clear; clc; close all;

% 1. Set cutoff frequency and instantiate filter object
w_c = pi / 3;
fprintf('Instantiating IdealLowpass filter with cutoff = %.4f (pi/3)...\n', w_c);
filterObj = OceanMath.Signal.IdealLowpass(w_c);

% 2. Verify L'Hopital Continuity and Symmetry
h0 = filterObj.impulseResponse(0);
h_sym_pos = filterObj.impulseResponse(1:10);
h_sym_neg = filterObj.impulseResponse(-1:-1:-10);
sym_diff = max(abs(h_sym_pos - h_sym_neg));
fprintf('Impulse Response at n=0 (L''Hopital limit): %.6f (Expected: %.6f)\n', h0, w_c/pi);
fprintf('Symmetry check (max difference h[n] vs h[-n]): %e\n\n', sym_diff);

% 3. Verify Parseval's Theorem for a sweep of M
fprintf('=========================================================\n');
fprintf('         PARSEVAL L2 ERROR VERIFICATION SUMMARY          \n');
fprintf('=========================================================\n');
fprintf('%-5s | %-20s | %-20s | %-12s\n', 'M', 'Frequency Error (L2)', 'Time Error (L2)', 'Difference');
fprintf('---------------------------------------------------------\n');
selected_M = [1, 2, 5, 10, 20, 50, 80, 100];
for M = selected_M
    [errFreq, errTime] = filterObj.verifyParseval(M, 10000);
    diffVal = abs(errFreq - errTime);
    fprintf('%5d | %20.15e | %20.15e | %12.3e\n', M, errFreq, errTime, diffVal);
end
fprintf('---------------------------------------------------------\n\n');

% 4. Run the experiment and open the visualizations
fprintf('Launching visualizer and plotting convergence profiles...\n');

% Create frequency grids
w_grid = linspace(-pi, pi, 1000);
M_values = [1, 3, 7, 19, 100];

% Setup dark visualizer theme
viz = OceanViz.Core('Theme', 'Dark');

% Figure 1: Pointwise Convergence (Replicating Oppenheim & Schafer Fig 2.21)
fig1 = viz.createFigure('Pointwise Fourier Series Convergence');
w_ideal = [-pi, -w_c, -w_c, w_c, w_c, pi];
H_ideal = [0, 0, 1, 1, 0, 0];
colors = viz.getColorSequence(length(M_values));
subplot_indices = {1, 2, 3, 4, [5, 6]};

for i = 1:length(M_values)
    M = M_values(i);
    ax = subplot(3, 2, subplot_indices{i});
    
    % Plot ideal response
    plot(ax, w_ideal, H_ideal, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.8);
    hold(ax, 'on');
    
    % Compute and plot approximation
    H_approx = filterObj.frequencyResponseApproximation(w_grid, M);
    plot(ax, w_grid, H_approx, 'Color', colors(i, :), 'LineWidth', 1.8);
    
    title(ax, sprintf('Fourier Series Approximation (M = %d)', M));
    xlabel(ax, 'Frequency \omega (rad/sample)');
    ylabel(ax, 'H_M(e^{j\omega})');
    
    viz.styleAxes(ax);
    lgd = legend(ax, {'Ideal Filter', sprintf('M = %d Approx', M)}, 'Location', 'best');
    set(lgd, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);
end

% Figure 2: Gibbs Overshoot & Parseval Decay Sweeps
fig2 = viz.createFigure('Mathematical Characterization');

% Left Plot: Overshoot % vs M
ax_left = subplot(1, 2, 1);
M_sweep = 1:200;
overshoots = zeros(size(M_sweep));
for k = 1:length(M_sweep)
    [~, ~, overshoots(k)] = filterObj.calculateOvershoot(M_sweep(k));
end
plot(ax_left, M_sweep, overshoots, 'Color', viz.Colors.Primary, 'LineWidth', 2);
hold(ax_left, 'on');
% Overlay asymptotic limit (8.95%)
plot(ax_left, [1, 200], [8.948967, 8.948967], '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.8);
title(ax_left, 'Gibbs Overshoot Percentage vs. M');
xlabel(ax_left, 'Truncation Limit M');
ylabel(ax_left, 'Overshoot (%)');
viz.styleAxes(ax_left);
lgd1 = legend(ax_left, {'Empirical Overshoot', 'Asymptotic Limit (8.95%)'}, 'Location', 'best');
set(lgd1, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);

% Right Plot: L2 Error Energy Decay (Semilogy)
ax_right = subplot(1, 2, 2);
M_sweep_err = 1:100;
err_freq = zeros(size(M_sweep_err));
err_time = zeros(size(M_sweep_err));
for k = 1:length(M_sweep_err)
    [err_freq(k), err_time(k)] = filterObj.verifyParseval(M_sweep_err(k));
end
semilogy(ax_right, M_sweep_err, err_freq, 'Color', viz.Colors.Primary, 'LineWidth', 2);
hold(ax_right, 'on');
semilogy(ax_right, M_sweep_err, err_time, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 2);
title(ax_right, 'L2 Error Energy vs. M');
xlabel(ax_right, 'Truncation Limit M');
ylabel(ax_right, 'L2 Error Energy');
viz.styleAxes(ax_right);
lgd2 = legend(ax_right, {'Frequency Domain Error', 'Time Domain Error'}, 'Location', 'best');
set(lgd2, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);

fprintf('\nGibbs Phenomenon Experiment finished successfully.\n');
```
