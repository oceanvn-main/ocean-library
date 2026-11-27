% EXP_GRADIENT_BOOSTING  Gradient Boosted Trees — OceanViz premium visualization.
%
%   OVERVIEW:
%     Demonstrates Algorithm 10.3 (Friedman 2001) with three loss strategies.
%     Uses OceanViz.Core for professional dark-themed aesthetics.
%
%   EXPERIMENTS:
%     1. Clean Fit   — MSE, MAE, Huber on a noisy sine curve.
%     2. Convergence — Training loss curves via MultiComparison strategy.
%     3. Robustness  — 10% label corruption: Huber vs MSE comparison.

clc; clear; rng(42);

% =========================================================================
% Initialize OceanViz Engine
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');
C   = viz.Colors;   % shorthand alias: C.Primary, C.Secondary, C.Text, etc.

% =========================================================================
% GBM Hyperparameters
% =========================================================================
M   = 80;
nu  = 0.15;   % learning rate (shrinkage)
dep = 3;

% =========================================================================
% Dataset — Noisy Sine Function
% =========================================================================
N      = 300;
X      = linspace(0, 4*pi, N)';
Y      = sin(X) + 0.3 * randn(N, 1);
X_fine = linspace(0, 4*pi, 500)';   % dense grid for smooth prediction curves

% =========================================================================
% Loss Strategies
% =========================================================================
losses = { ...
    OceanStats.Loss.Regression.MSELoss(), ...
    OceanStats.Loss.Regression.MAELoss(), ...
    OceanStats.Loss.Regression.PseudoHuberLoss('Delta', 1.0) };
names      = {'MSE', 'MAE', 'Huber ($\delta$=1)'};   % LaTeX — for ax_fit legend
plainNames = {'MSE', 'MAE', 'Huber (d=1)'};           % Plain  — for MultiComparison
nLoss  = numel(losses);
pal    = viz.getColorSequence(nLoss);   % N x 3 branded color palette

% =========================================================================
% EXPERIMENT 1 + 2 — Clean Fit & Convergence (side-by-side)
% =========================================================================
fig1 = viz.createFigure('GBM — Fitted Models and Convergence');
fig1.Position = [40 80 1280 500];

ax_fit  = subplot(1, 2, 1, 'Parent', fig1);
ax_loss = subplot(1, 2, 2, 'Parent', fig1);

% Style both axes using the engine
viz.styleAxes(ax_fit);
viz.styleAxes(ax_loss);

% ---- Left: Fitted Curves ----
hold(ax_fit, 'on');

% True underlying function
plot(ax_fit, X_fine, sin(X_fine), '--', ...
    'Color', C.Text, 'LineWidth', 1.2, 'DisplayName', 'True $\sin(x)$');

% Raw training data
scatter(ax_fit, X, Y, 12, 'filled', ...        
    'MarkerFaceColor', C.Scatter, ...     
    'MarkerFaceAlpha', 0.45, 'DisplayName', 'Training data');

% ---- Right: Convergence (using MultiComparison strategy) ----
lossData       = struct();
lossData.X     = (1:M)';
lossData.Lines = cell(1, nLoss);

% Train all models
fprintf('Training %d GBMs on clean data...\n', nLoss);
for c = 1:nLoss
    gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine(...
        'Loss', losses{c}, 'NumTrees', M, 'LearningRate', nu, 'MaxDepth', dep);
    gbm.fit(X, Y);

    % Fitted curve
    plot(ax_fit, X_fine, gbm.predict(X_fine), '-', ...
        'Color', pal(c, :), 'LineWidth', 2.2, 'DisplayName', names{c});

    % Store loss curve for MultiComparison (plain text only — no LaTeX)
    lossData.Lines{c} = struct('Y', gbm.trainLoss(), 'Name', plainNames{c});
end

% Axis decoration — Left
title(ax_fit, 'GBM: Fitted Models vs True Function', ...
    'Interpreter', 'latex', 'Color', C.Text, 'FontSize', 12);
xlabel(ax_fit, 'Input $x$',          'Interpreter', 'latex', 'Color', C.Text);
ylabel(ax_fit, 'Prediction $f(x)$',  'Interpreter', 'latex', 'Color', C.Text);
legend(ax_fit, 'Interpreter', 'latex', 'Location', 'northwest', ...
    'TextColor', C.Text, 'EdgeColor', C.Grid, 'Color', C.Background);
hold(ax_fit, 'off');

% Axis decoration — Right (via MultiComparison strategy)
viz.plot(ax_loss, 'MultiComparison', lossData, ...
    'Title',      'Training Loss vs. Boosting Rounds', ...
    'XLabel',     'Boosting Round (m)', ...
    'YLabel',     'Training Loss', ...
    'ShowLegend', true);

% Log Y-scale: separates the three curves in the high-overlap early region
set(ax_loss, 'YScale', 'log');

% =========================================================================
% EXPERIMENT 3 — Robustness: MSE vs Huber on corrupted labels
% =========================================================================
Y_corrupt  = Y;
nCorrupt   = round(0.10 * N);
outIdx     = randperm(N, nCorrupt);
Y_corrupt(outIdx) = Y_corrupt(outIdx) + 8 * randn(nCorrupt, 1);

fig2 = viz.createFigure('GBM — Robustness Analysis');
fig2.Position = [40 630 1280 420];

ax_r = axes('Parent', fig2);
viz.styleAxes(ax_r);
hold(ax_r, 'on');

% Background: clean + corrupted data
scatter(ax_r, X, Y, 10, 'filled', ...                         
    'MarkerFaceColor', C.Scatter, ...
    'MarkerFaceAlpha', 0.35, 'DisplayName', 'Clean data');
scatter(ax_r, X(outIdx), Y_corrupt(outIdx), 45, 'filled', ...  
    'MarkerFaceColor', C.Secondary, ...                    
    'MarkerEdgeColor', 'w', 'LineWidth', 0.5, 'DisplayName', 'Corrupted labels ($+8\sigma$)');

% True function
plot(ax_r, X_fine, sin(X_fine), '--', ...
    'Color', C.Text, 'LineWidth', 1.2, 'DisplayName', 'True $\sin(x)$');

% MSE vs Huber
robustNames   = {'MSE (sensitive)', 'Huber (robust)'};
robustLosses  = {losses{1}, losses{3}};
robustColors  = {C.Primary, C.Boundary};   % Blue vs Gold

fprintf('\nTraining Robustness Comparison...\n');
for i = 1:2
    gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine(...
        'Loss', robustLosses{i}, 'NumTrees', M, 'LearningRate', nu, 'MaxDepth', dep);
    gbm.fit(X, Y_corrupt);
    plot(ax_r, X_fine, gbm.predict(X_fine), '-', ...
        'Color', robustColors{i}, 'LineWidth', 3, 'DisplayName', robustNames{i});
end

% Axis decoration
title(ax_r, 'Robustness: $\mathcal{L}_2$ (MSE) vs Huber Loss Under 10\% Label Corruption', ...
    'Interpreter', 'latex', 'Color', C.Text, 'FontSize', 12);
xlabel(ax_r, 'Input $x$',  'Interpreter', 'latex', 'Color', C.Text);
ylabel(ax_r, '$f(x)$',     'Interpreter', 'latex', 'Color', C.Text);
legend(ax_r, 'Interpreter', 'latex', 'Location', 'northwest', ...
    'TextColor', C.Text, 'EdgeColor', C.Grid, 'Color', C.Background);
hold(ax_r, 'off');

fprintf('\nDone. Two premium figures generated.\n');
fprintf('Figure 1: Clean data fit + convergence\n');
fprintf('Figure 2: Robustness analysis (10%% corruption)\n');
