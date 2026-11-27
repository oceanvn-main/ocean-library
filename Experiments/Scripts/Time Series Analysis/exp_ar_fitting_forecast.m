% EXPERIMENT: Autoregressive (AR) Processes — Fitting, Lag Selection, and Forecasting
%
% PURPOSE:
%   This script runs a scientific experiment evaluating Autoregressive (AR) models.
%   It simulates a known second-order AR(2) process, estimates its parameters from 
%   data using Yule-Walker equations, performs automatic lag order selection using
%   Akaike and Bayesian Information Criteria (AIC & BIC), and runs recursive 
%   multi-step forecasts with analytical 95% confidence intervals.
%
% ASSUMPTIONS:
%   1. The white noise sequence e(t) is i.i.d. Gaussian with zero mean and variance 1.0.
%   2. The process is stationary, meaning the roots of the AR characteristic polynomial
%      lie inside the unit circle.
%   3. In forecasting, the future noise terms are independent of historical data,
%      allowing analytical forecast variance propagation.
%
% REPRODUCIBILITY:
%   A fixed random seed (rng(100)) is set at the start of the experiment.
%
% See also: OceanMath.TimeSeries.Models.AR, OceanViz.Core

clear;
clc;
close all;

% Set seed for reproducibility
rng(100);

% --- Simulation Parameters ---
N = 400;              % Total length of realization
N_train = 350;        % Training sequence length (for estimation)
N_test = N - N_train; % Test sequence length (held out for forecasting validation)
sigma = 1.0;          % White noise standard deviation

% Define true AR(2) process parameters
truePhi = [0.8, -0.3];
trueConstant = 0.5;

fprintf('=== PART 1: SIMULATION & PARAMETER RECOVERY ===\n\n');

% 1. Create known model and simulate
trueModel = OceanMath.TimeSeries.Models.AR(...
    'ARCoeffs', truePhi, ...
    'Constant', trueConstant, ...
    'NoiseStd', sigma ...
);

y = trueModel.simulate(N, sigma);

% Split train/test
yTrain = y(1:N_train);
yTest = y(N_train + 1 : end);

% 2. Fit AR(2) model directly to train set
fittedModel = OceanMath.TimeSeries.Models.AR.fit(yTrain, 'Order', 2);

fprintf('True Model Equation:      %s\n', trueModel.getEquation());
fprintf('Fitted Model Equation:    %s\n', fittedModel.getEquation());
fprintf('Estimated Noise Std:      %.4f (True: %.4f)\n\n', fittedModel.NoiseStd, sigma);

% Check parameter recovery
fprintf('True Coefficients:        [%.2f, %.2f]\n', truePhi(1), truePhi(2));
fprintf('Estimated Coefficients:   [%.4f, %.4f]\n\n', fittedModel.ARCoeffs(1), fittedModel.ARCoeffs(2));

%% Part 2: Lag Order Selection (AIC & BIC)
fprintf('=== PART 2: LAG ORDER SELECTION ===\n\n');

maxOrder = 10;
aicVals = zeros(maxOrder, 1);
bicVals = zeros(maxOrder, 1);

for p = 1:maxOrder
    m = OceanMath.TimeSeries.Models.AR.fit(yTrain, 'Order', p);
    sig_sq = m.NoiseStd^2;
    aicVals(p) = N_train * log(sig_sq) + 2 * (p + 1);
    bicVals(p) = N_train * log(sig_sq) + (p + 1) * log(N_train);
end

[~, bestAIC] = min(aicVals);
[~, bestBIC] = min(bicVals);

fprintf('Optimal order selected by AIC: %d\n', bestAIC);
fprintf('Optimal order selected by BIC: %d\n', bestBIC);
fprintf('Both criteria correctly identify the true order p=2? %s\n\n', ...
    logicalStr(bestAIC == 2 && bestBIC == 2));

%% Part 3: Multi-Step Forecasting
fprintf('=== PART 3: RECURSIVE FORECASTING ===\n\n');

% Generate 50-step ahead forecast from training history
horizon = N_test;
[yHat, bands] = fittedModel.forecast(yTrain, horizon);

% Compute forecast accuracy metrics
errors = yTest - yHat;
rmse = sqrt(mean(errors.^2));
mae = mean(abs(errors));
fprintf('Forecast horizon:     %d steps\n', horizon);
fprintf('Forecast RMSE:        %.4f\n', rmse);
fprintf('Forecast MAE:         %.4f\n', mae);

% Verify coverage: how many test samples fall inside the 95% confidence bands
inside = (yTest >= bands(:, 1)) & (yTest <= bands(:, 2));
coverage = sum(inside) / length(yTest) * 100;
fprintf('Coverage percentage:  %.1f%% (Expected: ~95%%)\n\n', coverage);

%% Part 4: Visualization
% Setting up a premium dark-themed dashboard layout using OceanViz
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;

fig = viz.createFigure('AR Model Fitting and Forecasting');
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% A. Time Domain Forecasting Plot (Spanning Top Row)
axForecast = nexttile(layout, [1, 2]);
viz.styleAxes(axForecast);
hold(axForecast, 'on');

% Historical training path (last 100 steps for context)
historyPlotRange = (N_train - 100 + 1) : N_train;
plot(axForecast, historyPlotRange, yTrain(historyPlotRange), ...
    'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'History (Train)');

% Forecast steps indices
forecastIdx = (N_train + 1) : N;

% Shaded 95% confidence bands
fill(axForecast, [forecastIdx, fliplr(forecastIdx)], [bands(:, 2)', fliplr(bands(:, 1)')], ...
    colors.Secondary, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', '95% Confidence Band');

% Actual future path (test)
plot(axForecast, forecastIdx, yTest, ...
    'Color', colors.Text * 0.8, 'LineWidth', 1.0, 'LineStyle', ':', 'DisplayName', 'Actual (Test)');

% Predicted path
plot(axForecast, forecastIdx, yHat, ...
    'Color', colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Forecast (Mean)');

title(axForecast, sprintf('AR(2) Multi-step Forecasting (RMSE = %.3f)', rmse), ...
    'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(axForecast, 'Time step (t)', 'Color', colors.Text);
ylabel(axForecast, 'Value Y(t)', 'Color', colors.Text);
legend(axForecast, 'Location', 'northwest', 'TextColor', colors.Text, 'EdgeColor', 'none', 'Color', 'none');
hold(axForecast, 'off');

% B. Order Selection Metric Curves (Bottom-Left)
axMetric = nexttile(layout);
viz.styleAxes(axMetric);
hold(axMetric, 'on');

plot(axMetric, 1:maxOrder, aicVals, '-o', 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'AIC');
plot(axMetric, 1:maxOrder, bicVals, '-x', 'Color', colors.Boundary, 'LineWidth', 1.2, 'DisplayName', 'BIC');

% Draw indicator at best BIC (p=2)
xline(axMetric, bestBIC, '--', 'Color', colors.Secondary, 'LineWidth', 1.0, 'HandleVisibility', 'off');

title(axMetric, 'Information Criteria order Selection', 'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
xlabel(axMetric, 'Model Order (p)', 'Color', colors.Text);
ylabel(axMetric, 'Metric Value', 'Color', colors.Text);
legend(axMetric, 'TextColor', colors.Text, 'EdgeColor', 'none', 'Color', 'none');
xlim(axMetric, [0.8, maxOrder + 0.2]);
xticks(axMetric, 1:maxOrder);
hold(axMetric, 'off');

% C. Autocorrelation of Residuals (Bottom-Right)
axACF = nexttile(layout);
viz.styleAxes(axACF);
hold(axACF, 'on');

% Compute residuals of fitted model on training data
% Y_resid = yTrain(t) - Constant - sum(phi * yTrain(t-i))
residuals = zeros(N_train - 2, 1);
for t = 3:N_train
    residuals(t-2) = yTrain(t) - fittedModel.Constant - ...
        sum(fittedModel.ARCoeffs .* flip(yTrain(t-2:t-1)'));
end

% Compute empirical ACF of residuals
maxLag = 15;
acfResid = OceanMath.TimeSeries.Diagnostics.autocorr(residuals, maxLag);
confBound = 1.96 / sqrt(length(residuals));

% Shaded area for white noise bounds
fill(axACF, [0, maxLag, maxLag, 0], [confBound, confBound, -confBound, -confBound], ...
    colors.Secondary, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Plot stems
stem(axACF, 0:maxLag, acfResid, 'Filled', 'Color', colors.Primary, 'MarkerFaceColor', colors.Primary, ...
    'MarkerSize', 4, 'LineWidth', 1.2);
yline(axACF, 0, 'Color', colors.Text * 0.5, 'LineWidth', 0.8);

title(axACF, 'ACF of Residuals (Model Fit Diagnostics)', 'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
xlabel(axACF, 'Lag (k)', 'Color', colors.Text);
ylabel(axACF, 'ACF', 'Color', colors.Text);
xlim(axACF, [-0.5, maxLag + 0.5]);
ylim(axACF, [-0.5, 1.05]);
hold(axACF, 'off');

fprintf('=== EXPERIMENT COMPLETE ===\n');

function str = logicalStr(val)
    if val
        str = 'True';
    else
        str = 'False';
    end
end
