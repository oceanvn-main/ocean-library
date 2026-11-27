# Hamilton Chapter 6 Spectral Analysis Guide (Real Binance BTC Data)

## Overview

This guide documents the implementation and theoretical verification of **Hamilton's Chapter 6 Spectral Analysis (Figures 6.3–6.6)** using real-world cryptocurrency data (**Binance BTC/USDT daily close prices**).

The experiment compares three representations of a non-stationary time series:
1. **Raw Log Level $y_t = \ln(\text{Price}_t)$**: Non-stationary level series.
2. **First-Difference Log Return $x_t = 100 \cdot (\ln y_t - \ln y_{t-1})$**: Stationary series filtered by high-pass differencing $h_x(\omega) = 4\sin^2(\omega/2)$.
3. **30-Day Growth Rate $w_t = 100 \cdot (\ln y_t - \ln y_{t-30})$**: Long-lag differenced series exhibiting filter ripples and zero nulls at frequencies $\omega_j = 2\pi j / 30$.

---

## Theoretical Formulations

### 1. Sample Periodogram Definition
Given a zero-mean realization $\{z_t\}_{t=1}^N$, the sample periodogram is defined as:

$$\hat{I}_z(\omega) = \frac{1}{2\pi N} \left| \sum_{t=1}^N z_t e^{-i\omega (t-1)} \right|^2, \quad \omega \in [0, \pi]$$

### 2. Spectral Transfer Functions of Detrending Filters
- **First-Difference Filter**: $(1 - L)$
  $$h_x(\omega) = |1 - e^{-i\omega}|^2 = 4 \sin^2\left(\frac{\omega}{2}\right)$$
- **30-Day Difference Filter**: $(1 - L^{30})$
  $$h_w(\omega) = |1 - e^{-i 30 \omega}|^2 = 4 \sin^2\left(\frac{30\omega}{2}\right)$$

### 3. Parseval's Energy Conservation Theorem

$$\int_{0}^{\pi} \hat{I}_z(\omega) d\omega = \text{Var}(z_t)$$

---

## MATLAB Usage Example

```matlab
% Run the complete end-to-end Hamilton spectral pipeline
run('Experiments/Scripts/Time Series Analysis/Spectrum Analysis/exp_hamilton_ch6_spectral_binance.m');
```

---

## Visualizations Generated

The experiment script `exp_hamilton_ch6_spectral_binance.m` generates 5 OceanViz dark-theme figures:

| Figure | Contents |
|--------|----------|
| **Figure 1** | Raw Binance Daily OHLCV Candlestick (MA7 / MA25 / MA99) + Volume subpanel via `OceanViz.Charts.CandlestickChart` |
| **Figure 2** | Hamilton Fig 6.3/6.4 Analog: Log Level series $y_t$ and zero-frequency dominated periodogram $I_y(\omega)$ |
| **Figure 3** | Hamilton Fig 6.5 Analog: 1-Day Log Return series $x_t$, sample periodogram $I_x(\omega)$, Welch PSD, and theoretical AR(p) spectrum |
| **Figure 4** | Hamilton Fig 6.6 Analog: 30-Day Growth Rate series $w_t$, periodogram $I_w(\omega)$, and theoretical zero null markers ($\omega_j = 2\pi j / 30$) |
| **Figure 5** | Theoretical Spectral Filter Gain comparison: 1-day differencing gain $h_x(\omega)$ vs. 30-day differencing gain $h_w(\omega)$ |

---

## Python Phase 2 MLOps Execution

```python
# Execute Python Phase 2 verification script using exported Parquet DataLake
import os
os.system('py "Experiments/Scripts/Time Series Analysis/Spectrum Analysis/exp_hamilton_ch6_spectral_binance.py"')
```

---

## Key Insights & Empirical Findings

- **Zero-Frequency Dominance**: Non-stationary level series $y_t$ concentrate massive spectral energy at $\omega \to 0$ (periods $T \to \infty$).
- **High-Pass Filtering**: First-differencing $x_t$ completely removes the DC trend spike and scales high-frequency variance.
- **Filter Nulls & Spurious Cycles**: 30-day differencing introduces artificial spectral zero nulls at $\omega = 2\pi j / 30$ ($j=1, 2, \dots$), creating spurious periodic oscillations in periodogram interpretations.

---

## See Also

- [Binance TSA Pipeline Guide](binance_tsa_pipeline.md)
- [Time Series Analysis Guide](time_series.md)
- [Sampling & Spectral Signal Processing Guide](ch4_sampling.md)
