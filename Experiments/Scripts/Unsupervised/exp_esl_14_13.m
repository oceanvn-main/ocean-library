% EXP_ESL_14_13 ESL Exercise 14.13 - Principal Curves vs. SOM
%
% Generates a 3D helix with Gaussian noise and fits both a Principal Curve
% and a 1D Self-Organizing Map (SOM) to discover the underlying 1D manifold.
%
% Reference: Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
% Section 14.4 & 14.5 (Exercise 14.13).
%
% Author: Ocean Library
% Date: 2026-05-05

clc; clear; rng(42, 'twister');

import OceanStats.Unsupervised.*
import OceanStats.NonParametric.*
import OceanViz.*

% =========================================================================
% 1. Data Generation (3D Helix)
% =========================================================================
N = 200;
s = linspace(0, 2*pi, N)';

Z1 = randn(N, 1);
Z2 = randn(N, 1);
Z3 = randn(N, 1);

X1 = cos(s) + 0.1 * Z1;
X2 = sin(s) + 0.1 * Z2;
X3 = s + 0.1 * Z3;

X = [X1, X2, X3];

% =========================================================================
% 2. Fit Principal Curve
% =========================================================================
% A slightly larger bandwidth helps smooth out the noise along the arc
bw = 1.27;
fprintf('Fitting Principal Curve (Kernel Smoother, Bandwidth=%.2f)...\n', bw);
smoother = KernelSmoother('Bandwidth', bw, 'KernelType', 'gaussian');
pc = PrincipalCurve('Smoother', smoother, 'MaxIter', 30, 'Tol', 1e-4);

model_pc = pc.fit(X);

% =========================================================================
% 3. Fit Self-Organizing Map (1D Grid)
% =========================================================================
fprintf('\nFitting 1D Self-Organizing Map (GridSize=[50, 1])...\n');
% A 1D SOM is mathematically related to the Principal Curve
som = SOM('GridSize', [50, 1], 'NumPasses', 50, ...
          'InitRadius', 15, 'FinalRadius', 1);
som.fit(X, 'Mode', 'batch', 'Verbose', true);

% =========================================================================
% 4. Visualization (OceanViz)
% =========================================================================
viz = Core('Theme', 'Scientific');
bg = viz.Colors.Background;

fig = figure('Name', 'ESL 14.13: Principal Curves vs SOM', 'Color', bg, 'Position', [100, 100, 1400, 500]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Tile 1: 3D Data & Principal Curve ---
ax1 = nexttile(tl);
hold(ax1, 'on');
h_data1 = scatter3(ax1, X(:,1), X(:,2), X(:,3), 15, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5);

% Plot Principal Curve
pc_curve = model_pc.Curve;
h_pc = plot3(ax1, pc_curve(:,1), pc_curve(:,2), pc_curve(:,3), '-', 'Color', viz.Colors.Primary, 'LineWidth', 3);

title(ax1, 'Principal Curve', 'Color', viz.Colors.Text);
xlabel(ax1, 'X_1', 'Color', viz.Colors.Text);
ylabel(ax1, 'X_2', 'Color', viz.Colors.Text);
zlabel(ax1, 'X_3', 'Color', viz.Colors.Text);
legend(ax1, [h_data1, h_pc], {'Original Data', 'Principal Curve'}, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
view(ax1, 3);
grid(ax1, 'on');
viz.styleAxes(ax1);

% --- Tile 2: 3D Data & SOM ---
ax2 = nexttile(tl);
hold(ax2, 'on');
h_data2 = scatter3(ax2, X(:,1), X(:,2), X(:,3), 15, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5);

% Plot SOM Nodes and Edges
W = som.Weights;
h_som = plot3(ax2, W(:,1), W(:,2), W(:,3), '-o', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'MarkerFaceColor', 'w');

title(ax2, 'Self-Organizing Map (1D Grid)', 'Color', viz.Colors.Text);
xlabel(ax2, 'X_1', 'Color', viz.Colors.Text);
ylabel(ax2, 'X_2', 'Color', viz.Colors.Text);
zlabel(ax2, 'X_3', 'Color', viz.Colors.Text);
legend(ax2, [h_data2, h_som], {'Original Data', 'SOM Nodes'}, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
view(ax2, 3);
grid(ax2, 'on');
viz.styleAxes(ax2);

% --- Tile 3: Coordinate Functions ---
% Plot the true coordinate functions vs S, and the estimated curve vs ArcLength
ax3 = nexttile(tl);
hold(ax3, 'on');

% Since ArcLength and S might have different absolute scales, we normalize 
% them both to [0, 1] for a clean comparison plot.
s_norm = s / max(s);
lam_norm = model_pc.ArcLengths / max(model_pc.ArcLengths);

% True Functions
h_true = plot(ax3, s_norm, cos(s), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
plot(ax3, s_norm, sin(s), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
plot(ax3, s_norm, s, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 2);

% Estimated PC Functions
h_pc_x = plot(ax3, lam_norm, pc_curve(:,1), '-', 'Color', viz.Colors.Primary, 'LineWidth', 2);
h_pc_y = plot(ax3, lam_norm, pc_curve(:,2), '-', 'Color', viz.Colors.Secondary, 'LineWidth', 2);
h_pc_z = plot(ax3, lam_norm, pc_curve(:,3), '-', 'Color', [0.4940, 0.1840, 0.5560], 'LineWidth', 2);

title(ax3, 'Coordinate Functions (True vs PC)', 'Color', viz.Colors.Text);
xlabel(ax3, 'Normalized Arc-Length / Parameter S', 'Color', viz.Colors.Text);
ylabel(ax3, 'Coordinate Value', 'Color', viz.Colors.Text);
legend(ax3, [h_true, h_pc_x, h_pc_y, h_pc_z], ...
    {'True Manifold (s)', 'PC X_1(\lambda)', 'PC X_2(\lambda)', 'PC X_3(\lambda)'}, ...
    'Location', 'northwest', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
viz.styleAxes(ax3);

sgtitle(fig, sprintf('ESL 14.13: Discovering a 1D Helix Manifold (BW = %.2f)', bw), 'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% Export figure
res_dir = fullfile('c:\', 'Ocean Library', 'Matlab code practice', 'Results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
viz.export(fig, fullfile(res_dir, 'exp_esl_14_13.png'));
