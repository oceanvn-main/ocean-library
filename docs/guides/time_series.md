# Time Series Analysis Guide

This guide introduces the `+OceanMath/+TimeSeries` package, which provides classes and utilities for modeling, simulating, and analyzing discrete-time stochastic processes.

## Overview

Time series analysis deals with sequences of data points recorded at successive, equally spaced points in time. The package is organized as follows:
1. **`+Models/`**: Holds process model definitions (e.g., Autoregressive Moving Average `ARMA`, and Autoregressive `AR` with fitting and forecasting support).
2. **`+Diagnostics/`**: Holds statistical tools, including empirical Autocorrelation Functions (ACF) and Partial Autocorrelation Functions (PACF) using Durbin-Levinson recursion.
3. **`+Filters/`**: *Planned* trend and seasonal extraction utilities.

---

## Usage

### 1. Constructing a Model
The `OceanMath.TimeSeries.Models.ARMA` and `OceanMath.TimeSeries.Models.AR` classes represent univariate mean models.
- **ARMA(p, q)**:
  `Y(t) = Constant + phi_1 * Y(t-1) + ... + phi_p * Y(t-p) + e(t) + theta_1 * e(t-1) + ... + theta_q * e(t-q)`
- **AR(p)**:
  `Y(t) = Constant + phi_1 * Y(t-1) + ... + phi_p * Y(t-p) + e(t)`

```matlab
% Construct an AR(1) process: Y_t = 0.8 * Y_{t-1} + e_t
modelAR1 = OceanMath.TimeSeries.Models.AR('ARCoeffs', 0.8);

% Construct an MA(1) process: Y_t = e_t + 0.5 * e_{t-1}
modelMA1 = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 0.5);

% Construct an ARMA(1,1) process with drift: Y_t = 1.0 + 0.5 * Y_{t-1} + e_t + 0.3 * e_{t-1}
modelARMA = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', 0.5, 'MACoeffs', 0.3, 'Constant', 1.0);
```

### 2. Checking Model Properties
You can verify if the process is stationary (roots of AR characteristic polynomial modulus < 1) and invertible (roots of MA characteristic polynomial modulus < 1).

```matlab
% Stationarity check
isStat = modelAR1.isStationary(); % returns true

% Invertibility check
isInvert = modelMA1.isInvertible(); % returns true
```

### 3. Simulation
To generate a synthetic realization of the process:

```matlab
N = 300;     % Sequence length
sigma = 1.0; % White noise standard deviation
y = modelAR1.simulate(N, sigma);
```
*Note: The simulation automatically applies a burn-in window of 100 samples to ensure the series reaches a steady-state before returning.*

### 4. Parameter Estimation (Fitting)
Given a vector of empirical observations `y`, you can fit an autoregressive process using the Yule-Walker equations.
You can either specify a target order `p` or let the model select the optimal order using Akaike (AIC) or Bayesian (BIC) Information Criteria sweeps:

```matlab
% Fit an AR(2) model directly
modelFitted = OceanMath.TimeSeries.Models.AR.fit(y, 'Order', 2);

% Automatically select optimal order between 1 and 10 using BIC
modelBest = OceanMath.TimeSeries.Models.AR.fit(y, 'MaxOrder', 10, 'Criterion', 'BIC');
fprintf('Selected AR order: %d\n', length(modelBest.ARCoeffs));
```

### 5. Multi-Step Forecasting
You can generate recursive point forecasts and analytical 95% uncertainty bands:

```matlab
horizon = 20; % Forecast 20 steps ahead
[yHat, bands] = modelBest.forecast(y, horizon);

% bands(:, 1) is the lower 95% limit, bands(:, 2) is the upper 95% limit
```

### 6. Visualization
Visualize the time series dashboard (realization, ACF, and PACF) or standalone publication-quality ACF plots using `OceanViz`:

```matlab
maxLag = 20;
acfVals = OceanMath.TimeSeries.Diagnostics.autocorr(y, maxLag);
pacfVals = OceanMath.TimeSeries.Diagnostics.parcorr(y, maxLag);

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Time Series Analysis');
ax = axes(fig);

% Standalone ACF plot with translucent 95% confidence ribbon & TeX formatting
dataACF = struct();
dataACF.ACF = acfVals;
dataACF.Lags = 0:maxLag;
dataACF.ConfBound = 1.96 / sqrt(length(y));
dataACF.Title = "Autocorrelation Function (ACF)";

viz.plot(ax, 'ACF', dataACF);
```

---

## Examples

### The MA(1) Duality
In Section 3.7 of the textbook, it is shown that two MA(1) processes:
- `Y_t = e_t + 0.5 * e_{t-1}` (invertible)
- `Y_t = e_t + 2.0 * e_{t-1}` (non-invertible)
share the exact same autocorrelation function:
- `rho_1 = 0.4`
- `rho_k = 0.0` for `k > 1`

We can verify this equivalence by computing their theoretical ACFs:

```matlab
modelA = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 0.5);
modelB = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 2.0);

acfA = modelA.theoreticalACF(5); % [1.0, 0.4, 0, 0, 0, 0]
acfB = modelB.theoreticalACF(5); % [1.0, 0.4, 0, 0, 0, 0]

assert(all(abs(acfA - acfB) < 1e-12));
disp('Duality Verified!');
```

---

## See Also
- [OceanMath.TimeSeries.Models.AR](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanMath/+TimeSeries/+Models/AR.m)
- [OceanMath.TimeSeries.Models.ARMA](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanMath/+TimeSeries/+Models/ARMA.m)
- [OceanMath.TimeSeries.Diagnostics.autocorr](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanMath/+TimeSeries/+Diagnostics/autocorr.m)
- [OceanMath.TimeSeries.Diagnostics.parcorr](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanMath/+TimeSeries/+Diagnostics/parcorr.m)
- [OceanViz.Strategies.TimeSeriesVizStrategy](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanViz/+Strategies/TimeSeriesVizStrategy.m)
- [OceanViz.Strategies.ACFStrategy](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanViz/+Strategies/ACFStrategy.m)
