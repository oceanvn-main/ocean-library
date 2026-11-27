# Time Series Analysis Recipes

Quick, copy-pasteable snippets for using the `+OceanMath/+TimeSeries` package.

## Recipe 1: Simulate and Analyze an AR(1) Process

```matlab
% Set seed for reproducibility
rng(123);

% 1. Create model (AR coefficient phi = 0.75, Constant = 0.5)
model = OceanMath.TimeSeries.Models.AR('ARCoeffs', 0.75, 'Constant', 0.5);

% 2. Simulate sequence (N = 400 steps, noise std = 1.0)
N = 400;
sigma = 1.0;
y = model.simulate(N, sigma);

% 3. Calculate empirical and theoretical ACF/PACF
maxLag = 20;
acfEmp = OceanMath.TimeSeries.Diagnostics.autocorr(y, maxLag);
pacfEmp = OceanMath.TimeSeries.Diagnostics.parcorr(y, maxLag);
acfTheo = model.theoreticalACF(maxLag);
pacfTheo = model.theoreticalPACF(maxLag);

% 4. Render 3-panel dashboard via OceanViz
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('AR(1) Analysis');
ax = axes(fig);

plotData = struct();
plotData.Y = y;
plotData.ACF = acfEmp;
plotData.PACF = pacfEmp;
plotData.TheoreticalACF = acfTheo;
plotData.TheoreticalPACF = pacfTheo;
plotData.Lags = maxLag;
plotData.ModelName = "AR(1) phi=0.75, c=0.5";

viz.plot(ax, 'TimeSeries', plotData);
```

---

## Recipe 2: Simulate and Analyze an MA(1) Process

```matlab
% Set seed for reproducibility
rng(456);

% 1. Create model (MA coefficient theta = -0.6)
model = OceanMath.TimeSeries.Models.ARMA('MACoeffs', -0.6);

% 2. Simulate sequence (N = 300 steps, noise std = 1.0)
N = 300;
sigma = 1.0;
y = model.simulate(N, sigma);

% 3. Calculate empirical and theoretical ACF/PACF
maxLag = 15;
acfEmp = OceanMath.TimeSeries.Diagnostics.autocorr(y, maxLag);
pacfEmp = OceanMath.TimeSeries.Diagnostics.parcorr(y, maxLag);
acfTheo = model.theoreticalACF(maxLag);
pacfTheo = model.theoreticalPACF(maxLag);

% 4. Render 3-panel dashboard via OceanViz
viz = OceanViz.Core('Theme', 'Light');
fig = viz.createFigure('MA(1) Analysis');
ax = axes(fig);

plotData = struct();
plotData.Y = y;
plotData.ACF = acfEmp;
plotData.PACF = pacfEmp;
plotData.TheoreticalACF = acfTheo;
plotData.TheoreticalPACF = pacfTheo;
plotData.Lags = maxLag;
plotData.ModelName = "MA(1) theta=-0.6";

viz.plot(ax, 'TimeSeries', plotData);
```

---

## Recipe 3: Verify MA(1) Duality (Textbook Section 3.7)

Verify that the invertible and non-invertible MA(1) processes share the exact same theoretical autocorrelation structure.

```matlab
% 1. Instantiate the two dual models
modelInv = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 0.5);   % Invertible (modulus < 1)
modelNonInv = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 2.0);% Non-invertible (modulus > 1)

% 2. Verify properties
fprintf('Model Inv (theta = 0.5) Invertible:  %s\n', mat2str(modelInv.isInvertible()));
fprintf('Model NonInv (theta = 2.0) Invertible: %s\n', mat2str(modelNonInv.isInvertible()));

% 3. Calculate ACF up to 10 lags
acfInv = modelInv.theoreticalACF(10);
acfNonInv = modelNonInv.theoreticalACF(10);

% 4. Assert equivalence
assert(all(abs(acfInv - acfNonInv) < 1e-12));
fprintf('ACF matches exactly! Lag 1 value: %.4f (Expected: 0.4)\n', acfInv(2));
```

---

## Recipe 4: Autoregressive Model Fitting & Forecasting

This recipe demonstrates loading data, fitting an AR(p) model with automatic BIC order selection, and generating a 30-step ahead forecast with 95% uncertainty bands.

```matlab
% Load some sample return data
load stocks.mat;
y = stocks(:, 1);

% 1. Fit an AR model to the historical data, selecting order by BIC
fittedModel = OceanMath.TimeSeries.Models.AR.fit(y, 'MaxOrder', 8, 'Criterion', 'BIC');
fprintf('Selected AR Order: %d\n', length(fittedModel.ARCoeffs));
fprintf('Fitted Equation:   %s\n', fittedModel.getEquation());

% 2. Generate 30-step ahead forecast with 95% confidence bands
horizon = 30;
[yHat, bands] = fittedModel.forecast(y, horizon);

% 3. Plot history, forecast, and uncertainty bands
figure('Name', 'AR Forecasting');
hold on;
plot(1:length(y), y, 'b-', 'LineWidth', 1.0, 'DisplayName', 'History');
t_fut = (length(y) + 1) : (length(y) + horizon);

% Shaded confidence bands
fill([t_fut, fliplr(t_fut)], [bands(:, 2)', fliplr(bands(:, 1)')], ...
     [1 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '95% CI');

plot(t_fut, yHat, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Forecast');
xlabel('Time step');
ylabel('Value');
title('Autoregressive Model Forecast');
legend('Location', 'best');
grid on;
```
