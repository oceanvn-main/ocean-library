# Hamilton Ch. 6 Selective Seasonal Filtering & Isolation Guide

## Overview

This guide explains how to replicate **Hamilton Chapter 6 (Figures 6.3 & 6.4)** using a synthetic monthly time series containing:
1. **Low-Frequency Trend ($\omega \approx 0$)**: Large non-stationary trend / random walk dominating total variance.
2. **12-Month Annual Seasonal Cycle**: Period $T = 12$ months ($\omega = 2\pi / 12$).
3. **3-Month Quarterly Seasonal Cycle**: Period $T = 3$ months ($\omega = 2\pi / 3$).
4. **6-Month Semi-Annual Seasonal Cycle**: Period $T = 6$ months ($\omega = 2\pi / 6$).

---

## Filter Construction

To eliminate the trend ($\omega = 0$), the 12-month cycle, and the 3-month cycle **without removing the 6-month cycle**, we construct a composite notch polynomial $F(B)$:

| Target Component | Frequency $\omega$ | Polynomial Factor |
|---|---|---|
| Trend / DC ($\omega = 0$) | $0$ | $H_{\text{dc}}(B) = 1 - B$ |
| 12-Month Cycle ($T = 12$) | $2\pi / 12 = \pi/6$ | $H_{12}(B) = 1 - 2\cos(\pi/6)B + B^2$ |
| 3-Month Cycle ($T = 3$) | $2\pi / 3$ | $H_{3}(B) = 1 + B + B^2$ |

The composite FIR filter polynomial is:
```
F(B) = H_dc(B) * H_12(B) * H_3(B)
```

### Zero-Phase Passband Property at 6-Month Period

Evaluating $F(e^{-i \omega})$ at $\omega_6 = 2\pi / 6 = \pi/3$:
- $H_{\text{dc}}(\pi/3) \neq 0$ (Pass)
- $H_{12}(\pi/3) \neq 0$ (Pass)
- $H_{3}(\pi/3) \neq 0$ (Pass)

When applied via zero-phase filtering (`filtfilt`), all components at $\omega = 0$, $T = 12$, and $T = 3$ are completely cancelled, leaving **only the 6-month periodic component** in the time-domain output.

---

## Usage & Verification

```matlab
% Run the experiment script
run('Experiments/Scripts/Time Series Analysis/exp_hamilton_ch6_selective_filter.m');
```

Output plot saved to:
`Experiments/Scripts/Time Series Analysis/Results/exp_hamilton_ch6_selective_filter.png`

---

## See Also

- `OceanPipeline.Transforms.DifferenceSpectrumFilter` — nth-order backshift spectrum filter
- Experiment: `exp_binance_fetch_and_diff_spectrum.m` — real Binance market data experiment
- Synthetic Experiment: `exp_synthetic_diff_spectrum_verify.m` — general spectrum filter verification
