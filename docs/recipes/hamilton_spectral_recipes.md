# Hamilton Chapter 6 Spectral Analysis Recipes

Copy-pasteable code snippets for executing Hamilton Chapter 6 spectral analysis on financial time series.

## 1. Compute FFT Sample Periodogram & Parseval Verification

```matlab
% Ingest Binance Daily BTC Data
reader = OceanPipeline.Ingestion.BinanceDataReader('Symbol', 'BTCUSDT', 'Interval', '1d', 'Limit', 1000);
TT = reader.fetchKlinesTimetable();

% Compute 1-day log returns
r = 100 * [NaN; diff(log(TT.Close))];
r = r(~isnan(r));
r_centered = r - mean(r);

% FFT Sample Periodogram
N = length(r_centered);
N_fft = 2^nextpow2(2 * N);
Z_fft = fft(r_centered, N_fft);
numPts = floor(N_fft / 2) + 1;
w_grid = linspace(0, pi, numPts)';
I_sample = (1 / (2 * pi * N)) * (abs(Z_fft(1:numPts)).^2);
I_sample(2:end-1) = 2 * I_sample(2:end-1);

% Verify Parseval's Spectral Energy Theorem
dw = w_grid(2) - w_grid(1);
integrated_area = sum(I_sample) * dw;
sample_variance = var(r_centered, 1);
fprintf('Parseval Error: %.4e\n', abs(integrated_area - sample_variance));
```

## 2. Theoretical AR(p) Population Spectrum & OceanViz Plotting

```matlab
% Fit AR(4) process via Yule-Walker
arModel = OceanMath.TimeSeries.Models.AR.fit(r_centered, 'Order', 4);

% Compute theoretical population spectrum
aPoly = [1, -arModel.ARCoeffs];
popSpec = (arModel.NoiseStd^2 / (2 * pi)) ./ (abs(polyval(aPoly, exp(-1i * w_grid))).^2);

% Render via OceanViz Engine (Dark Theme)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Sample vs Theoretical AR Spectrum');
ax = axes(fig);
viz.styleAxes(ax);
hold(ax, 'on');

plot(ax, w_grid, 10*log10(I_sample + 1e-12), 'Color', viz.Colors.Primary, 'DisplayName', 'Sample Periodogram');
plot(ax, w_grid, 10*log10(popSpec + 1e-12), '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.8, 'DisplayName', 'AR(4) Population Spectrum');

title(ax, 'Hamilton Spectral Density Comparison');
xlabel(ax, 'Frequency \omega (rad/sample)');
ylabel(ax, 'Power (dB)');
legend(ax, 'Location', 'northeast', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax, 'off');
```

## 3. Theoretical Gain Transfer Functions

```matlab
% 1-day differencing gain vs 30-day growth gain
h_1d  = 4 * (sin(w_grid / 2)).^2;
h_30d = 4 * (sin(30 * w_grid / 2)).^2;

% Plot filter gain comparison
fig2 = viz.createFigure('Filter Gain Response');
ax2 = axes(fig2);
viz.styleAxes(ax2);
hold(ax2, 'on');

plot(ax2, w_grid, h_1d, 'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'DisplayName', '1-Day Differencing (High-Pass)');
plot(ax2, w_grid, h_30d, 'Color', viz.Colors.Secondary, 'LineWidth', 1.2, 'DisplayName', '30-Day Differencing (Filter Ripples)');

title(ax2, 'Hamilton Spectral Transfer Function Comparison');
xlabel(ax2, 'Frequency \omega (rad/sample)');
ylabel(ax2, 'Power Gain |H(\omega)|^2');
legend(ax2, 'Location', 'northeast', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none');
```
