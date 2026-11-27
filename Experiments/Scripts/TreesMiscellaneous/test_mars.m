% test_mars.m
% Simple script to verify the MARS implementation on synthetic 1D data.

% Add path if needed (assuming Ocean Library is on path)
import OceanStats.Supervised.MARS.MARS;
import OceanViz.Core;

fprintf('--- Starting MARS Test ---\n');

%% 1. Generate Synthetic Data
N = 200;
rng(42);

% Uniform X between 0 and 10
X = rand(N, 1) * 10;
X = sort(X); % Sort for easier plotting

% True non-linear function: a sine wave with a linear trend
true_func = @(x) sin(1.5 * x) + 0.3 * x;
Y_true = true_func(X);

% Add Gaussian noise
Y = Y_true + randn(N, 1) * 0.3;

%% 2. Instantiate and Fit MARS
% We expect MARS to fit a piecewise linear approximation of the sine wave
mars_model = MARS('MaxTerms', 15, 'PenaltyCost', 2.0, 'MaxInteractionDegree', 1);

tic;
mars_model.fit(X, Y);
t = toc;

fprintf('\nFitting completed in %.3f seconds.\n\n', t);

% Print the discovered model
mars_model.printModel({'X1'});

%% 3. Evaluate and Plot
Y_pred = mars_model.predict(X);

% Calculate R-squared
rss = sum((Y - Y_pred).^2);
tss = sum((Y - mean(Y)).^2);
r2 = 1 - (rss/tss);

fprintf('\nR-Squared: %.4f\n', r2);

% Plot using VizEngine for style
viz = Core('Theme', 'Dark');
fig = viz.createFigure('MARS 1D Regression Test');
ax = axes(fig);
hold(ax, 'on');

% Plot raw data
scatter(ax, X, Y, 20, viz.Colors.Scatter(:)', 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Noisy Data');

% Plot true underlying function
plot(ax, X, Y_true, 'Color', viz.Colors.Secondary(:)', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', 'True Function');

% Plot MARS Prediction
plot(ax, X, Y_pred, 'Color', viz.Colors.Primary(:)', 'LineWidth', 2.5, 'DisplayName', 'MARS Prediction');

xlabel(ax, 'X', 'Interpreter', 'none');
ylabel(ax, 'Y', 'Interpreter', 'none');
title(ax, sprintf('MARS Approximation (R^2 = %.2f)', r2));
legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);

viz.styleAxes(ax);
exportgraphics(fig, 'mars_test_plot.png', 'Resolution', 300);
fprintf('Saved plot to mars_test_plot.png\n');
fprintf('--- MARS Test Complete ---\n');
