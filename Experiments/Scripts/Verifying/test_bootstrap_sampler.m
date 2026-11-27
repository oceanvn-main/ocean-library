% test_bootstrap_sampler_regression.m
% Reproduction of "Figure 8.2" style Bootstrap visualization
% 
% 1. Data Generation (Non-linear)
% 2. B-Spline/Poly Smooth
% 3. Residual Bootstrap
% 4. 4-Panel Visualization

clear; clc;

%% 1. Data Generation
rng(101); % Reproducibility
N = 80;
x = sort(3 * rand(N, 1));
% True function: Sine wave + Linear trend
y_true = sin(4*x) + 0.5*x; 
% Noise
sigma_true = 0.5;
y = y_true + randn(N, 1) * sigma_true;

% Grid for plotting
x_grid = linspace(min(x), max(x), 200)';

%% 2. Smoothing Function (Polynomial for simplicity/robustness)
% Using a high-degree polynomial to mimic a "Smooth"
deg = 6; 
fit_fn = @(x,y,xq) polyval(polyfit(x,y,deg), xq);

% Original Fit
y_hat = fit_fn(x, y, x);        % Fitted values at data points
y_smooth = fit_fn(x, y, x_grid); % Fitted curve for plotting

% Standard Error of the Regression (Noise Sigma)
residuals = y - y_hat;
MSE = sum(residuals.^2) / (N - deg - 1);
SE_reg = sqrt(MSE); 

% --- Correct Bayesian/Freq Confidence Interval for the Curve ---
% We need the SE of the *Mean Response*, not the Prediction Interval.
% Var(f_hat(x)) = sigma^2 * x_vec * (X'X)^-1 * x_vec'

% 1. Design Matrix for Observed Data
X_des = zeros(N, deg+1);
for i = 0:deg
    X_des(:, deg+1-i) = x.^i;
end

% 2. Covariance of Parameters
Cov_beta = MSE * inv(X_des' * X_des);

% 3. Calculate SE for the Grid (Prediction)
X_grid_des = zeros(length(x_grid), deg+1);
for i = 0:deg
    X_grid_des(:, deg+1-i) = x_grid.^i;
end

% SE_curve = sqrt(diag(X_grid * Cov_beta * X_grid'))
% Vectorized calculation for efficiency: sum((X*L).^2, 2) where L=chol(Cov)
% or simply:
SE_curve = zeros(length(x_grid), 1);
for i = 1:length(x_grid)
    row = X_grid_des(i, :);
    SE_curve(i) = sqrt(row * Cov_beta * row');
end
 

%% 3. Bootstrap (Residual Method)
B = 200;
boot_preds = zeros(length(x_grid), B);

fprintf('Running Bootstrap (B=%d)...\n', B);
for b = 1:B
    % Resample residuals
    res_boot = randsample(residuals, N, true);
    y_boot = y_hat + res_boot;
    
    % Re-fit
    boot_preds(:, b) = fit_fn(x, y_boot, x_grid);
end

% Compute Statistics
mean_boot = mean(boot_preds, 2);
se_boot = std(boot_preds, 0, 2);
ci_low = quantile(boot_preds, 0.025, 2);
ci_high = quantile(boot_preds, 0.975, 2);

%% 4. Visualization (Figure 8.2 Style)
import OceanViz.Core

viz = Core('Theme', 'Dark'); % Use Dark theme for "Premium" look
fig = viz.createFigure('Bootstrap Regression Analysis');
set(fig, 'Position', [100 100 1200 900]);

% Colors
cData = viz.Colors.Scatter(:)';   % Data points
cFit  = [0 1 0];                  % Green
cBand = [0.5 0.5 0.5];            % Grey dashed
cBoot = [0 1 1];                  % Cyan

% Ensure Theme Colors are Row Vectors [1x3] for fill/plot
cPrim = viz.Colors.Primary(:)';
cSec = viz.Colors.Secondary(:)';

% --- Panel 1: The Truth (Signal vs Noise) ---
ax1 = subplot(2,2,1);
hold(ax1, 'on');
plot(ax1, x_grid, sin(4*x_grid) + 0.5*x_grid, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True Function');
plot(ax1, x, y, '.', 'Color', cData, 'MarkerSize', 8, 'DisplayName', 'Noisy Data');
title(ax1, '1. The Truth (Signal + Noise)');
legend(ax1, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none');
text(ax1, 0.2, 2.3, {'y = f(x) + \epsilon', '\epsilon \sim N(0, \sigma^2)'}, 'Color', viz.Colors.Text);
viz.styleAxes(ax1);

% --- Panel 2: Bayesian Posterior (Analytical) ---
% Under flat prior, Posterior P(beta|y) approx N(beta_hat, sigma^2*(X'X)^-1)
ax2 = subplot(2,2,2);
hold(ax2, 'on');
% Plot Bands (Parametric/Bayesian)
% Plot Bands (Parametric/Bayesian - Confidence Interval of Mean)
fill(ax2, [x_grid(:); flipud(x_grid(:))], ...
    [y_smooth(:) + 1.96*SE_curve(:); flipud(y_smooth(:) - 1.96*SE_curve(:))], ...
    cPrim, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '95% Posterior (Mean)');
plot(ax2, x_grid, y_smooth, '-', 'Color', cPrim, 'LineWidth', 2, 'DisplayName', 'Posterior Mean');
plot(ax2, x, y, '.', 'Color', cData, 'MarkerSize', 6);
title(ax2, '2. Bayesian Methods (Posterior of Mean)');
% Explicit Condition Annotation
text(ax2, 0.2, 2.3, {'\bf{Conditions:}', '- Gaussian Errors', '- Flat/Non-informative Prior'}, ...
    'Color', cPrim, 'FontSize', 8, 'BackgroundColor', [0 0 0 0.5], 'EdgeColor', viz.Colors.Text);
viz.styleAxes(ax2);

% --- Panel 3: Bootstrap Methods (Empirical) ---
ax3 = subplot(2,2,3);
hold(ax3, 'on');
% Plot Bands (Bootstrap)
fill(ax3, [x_grid(:); flipud(x_grid(:))], ...
    [ci_high(:); flipud(ci_low(:))], ...
    cSec, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '95% Bootstrap CI');
plot(ax3, x_grid, mean_boot, '--', 'Color', cSec, 'LineWidth', 2, 'DisplayName', 'Bootstrap Mean');
plot(ax3, x, y, '.', 'Color', cData, 'MarkerSize', 6);
title(ax3, '3. Bootstrap Methods');
% Explicit Condition Annotation
text(ax3, 0.2, 2.3, {'\bf{Conditions:}', '- Sample \approx Population', '- Residuals are i.i.d'}, ...
    'Color', cSec, 'FontSize', 8, 'BackgroundColor', [0 0 0 0.5], 'EdgeColor', viz.Colors.Text);
viz.styleAxes(ax3);

% --- Panel 4: Comparison (Convergence) ---
ax4 = subplot(2,2,4);
hold(ax4, 'on');
% Overlay True
plot(ax4, x_grid, sin(4*x_grid) + 0.5*x_grid, 'k-', 'LineWidth', 2, 'DisplayName', 'True');
% Overlay Bayes (Dashed Blue)
plot(ax4, x_grid, y_smooth + 1.96*SE_curve, '--', 'Color', cPrim, 'DisplayName', 'Bayesian Upper');
plot(ax4, x_grid, y_smooth - 1.96*SE_curve, '--', 'Color', cPrim, 'HandleVisibility', 'off');
% Overlay Boot (Dotted Red)
plot(ax4, x_grid, ci_high, ':', 'Color', cSec, 'LineWidth', 2, 'DisplayName', 'Bootstrap Upper');
plot(ax4, x_grid, ci_low, ':', 'Color', cSec, 'LineWidth', 2, 'HandleVisibility', 'off');

title(ax4, '4. Method Comparison');
legend(ax4, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none');
text(ax4, 0.5, -1.0, 'Bootstrap \rightarrow Bayesian', 'HorizontalAlignment', 'center', 'Color', viz.Colors.Text);
viz.styleAxes(ax4);


fprintf('Visualization Complete.\n');
