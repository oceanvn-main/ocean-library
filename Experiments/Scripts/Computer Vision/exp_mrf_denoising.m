% EXP_MRF_DENOISING  Markov Random Field (MRF) Spatial Prior Denoising: L1 vs L2 Comparison.
%
%   Procedural Workflow using Suitable Synthetic Geometric Data:
%     1. Dynamic Synthetic Data Generation (No Hardcoding):
%        Algorithmically instantiate a 64x64 matrix of baseline 0.1. Dynamically calculate
%        boundary indices for multiple geometric primitives (randomized circular disk and
%        rectangular block) to form piecewise-constant ground truth X_true. Apply randomized
%        Gaussian noise function (+ 0.25 * randn) to produce measurement matrix Y.
%     2. Initialization:
%        Initialize working state matrix X strictly equal to measurement Y (X_0 = Y).
%     3. MRF Energy Formulation:
%        - L1 Regularization (Total Variation): E_L1(X) = 0.5 * ||X - Y||^2 + lambda * sum |D * X|
%        - L2 Regularization (Quadratic Tikhonov): E_L2(X) = 0.5 * ||X - Y||^2 + lambda * sum (D * X)^2
%     4. Optimization Iteration via OceanMath IRLSSolver:
%        Solve both L1 and L2 formulations using OceanMath IRLSSolver.
%     5. OceanViz Engine Plotting & Comparative Analysis:
%        Render 8-panel comparison showing Ground Truth, Corrupted Y, L1 Restored, L2 Restored,
%        Absolute Errors, 1D Cross-Section Profile, and Energy Minimization Curves.
%
%   Author: Ocean Library  |  2026-08-05
%   Reference: Rudin-Osher-Fatemi (1992), Szeliski (2021) §4.2

clear; clc; close all;

import OceanMath.Operators.DifferenceOperator2D
import OceanMath.Regularization.L1Penalty
import OceanMath.Regularization.QuadraticPenalty
import OceanMath.Variational.VariationalEnergy
import OceanMath.Image.Variational.Solvers.IRLSSolver
import OceanViz.Core

fprintf('=== MRF Spatial Prior Denoising: L1 (Total Variation) vs L2 (Quadratic) ===\n\n');

% ── Step 1: Dynamic Synthetic Data Generation (No Hardcoding) ─────────────
rng(42);

N_grid = 64;
X_true = 0.1 * ones(N_grid, N_grid);   % Background baseline = 0.1

% 1. Dynamic Circular Disk (Value = 1.0)
disk_center_x = round(N_grid * 0.35 + (rand() - 0.5) * 4);
disk_center_y = round(N_grid * 0.35 + (rand() - 0.5) * 4);
disk_radius   = round(12 + rand() * 3);

[Xgrid, Ygrid] = meshgrid(1:N_grid, 1:N_grid);
disk_mask = ((Xgrid - disk_center_x).^2 + (Ygrid - disk_center_y).^2) <= disk_radius^2;
X_true(disk_mask) = 1.0;

% 2. Dynamic Rectangular Block (Value = 0.7)
rect_x1 = round(N_grid * 0.55);
rect_x2 = round(N_grid * 0.85);
rect_y1 = round(N_grid * 0.50);
rect_y2 = round(N_grid * 0.85);
X_true(rect_y1:rect_y2, rect_x1:rect_x2) = 0.7;

% 3. Apply randomized Gaussian noise function
noise_sigma = 0.25;
Y_measurement = X_true + noise_sigma * randn(N_grid, N_grid);

fprintf('Ground Truth Matrix: %dx%d Piecewise-Constant Synthetic Benchmark\n', N_grid, N_grid);
fprintf('Measurement Matrix Y created with Gaussian Noise sigma=%.2f\n\n', noise_sigma);

% ── Step 2 & 3: L1 vs L2 Energy Formulation & Solver Execution ───────────
op2d   = DifferenceOperator2D();
solver = IRLSSolver('MaxIter', 30, 'Tol', 1e-6);

% --- A. L1 Regularization (Total Variation: Edge-Preserving) ---
fprintf('Solving L1 (Total Variation) Regularization...\n');
pen_l1    = L1Penalty('Epsilon', 1e-5);
lambda_l1 = 0.6;
energy_l1 = VariationalEnergy(op2d, pen_l1, 'Lambda', lambda_l1);

tic;
X_l1     = solver.solve(energy_l1, Y_measurement);
t_l1     = toc;
history_l1 = solver.getHistory();

% --- B. L2 Regularization (Quadratic: Isotropic Blurring) ---
fprintf('Solving L2 (Quadratic Tikhonov) Regularization...\n');
pen_l2    = QuadraticPenalty();
lambda_l2 = 0.8;
energy_l2 = VariationalEnergy(op2d, pen_l2, 'Lambda', lambda_l2);

tic;
X_l2     = solver.solve(energy_l2, Y_measurement);
t_l2     = toc;
history_l2 = solver.getHistory();

% Calculate Reconstruction Error Metrics (RMSE & MAE)
rmse_l1 = sqrt(mean((X_l1(:) - X_true(:)).^2));
rmse_l2 = sqrt(mean((X_l2(:) - X_true(:)).^2));
mae_l1  = mean(abs(X_l1(:) - X_true(:)));
mae_l2  = mean(abs(X_l2(:) - X_true(:)));

fprintf('\n=== Quantitative Comparison ===\n');
fprintf('L1 (Total Variation) -> RMSE: %.4f, MAE: %.4f (Time: %.3fs)\n', rmse_l1, mae_l1, t_l1);
fprintf('L2 (Quadratic)       -> RMSE: %.4f, MAE: %.4f (Time: %.3fs)\n\n', rmse_l2, mae_l2, t_l2);

% ── Step 4: OceanViz Engine Plotting (Dark Mode & Neon Styling) ───────────
fprintf('Rendering L1 vs L2 Comparison using OceanViz Engine...\n');

viz = Core('Theme', 'Dark');
colors = viz.Colors;

% Define Neon Cyan [0 1 1] to Neon Magenta [1 0 1] custom colormap
num_colors = 256;
r_channel = linspace(0.0, 1.0, num_colors)';
g_channel = linspace(1.0, 0.0, num_colors)';
b_channel = ones(num_colors, 1);
neon_cmap = [r_channel, g_channel, b_channel];

fig = figure('Name', 'MRF Denoising: L1 (Total Variation) vs L2 (Quadratic)', ...
    'Color', colors.Background, ...
    'Position', [50 50 1350 750]);

% ── Row 1: Ground Truth, Measurement, L1 & L2 Restored Images ─────────────
% Subplot 1: Ground Truth X_true
ax1 = subplot(2, 4, 1, 'Parent', fig);
data1 = struct('Image', X_true, 'Title', 'Ground Truth X_{true}', 'Colormap', neon_cmap);
viz.plot(ax1, "Image", data1);

% Subplot 2: Corrupted Measurement Y
ax2 = subplot(2, 4, 2, 'Parent', fig);
data2 = struct('Image', Y_measurement, 'Title', 'Corrupted Measurement Y', 'Colormap', neon_cmap);
viz.plot(ax2, "Image", data2);

% Subplot 3: L1 Restored (Edge-Preserving)
ax3 = subplot(2, 4, 3, 'Parent', fig);
data3 = struct('Image', X_l1, 'Title', sprintf('L1 Restored (RMSE=%.3f)', rmse_l1), 'Colormap', neon_cmap);
viz.plot(ax3, "Image", data3);

% Subplot 4: L2 Restored (Blurred Edges)
ax4 = subplot(2, 4, 4, 'Parent', fig);
data4 = struct('Image', X_l2, 'Title', sprintf('L2 Restored (RMSE=%.3f)', rmse_l2), 'Colormap', neon_cmap);
viz.plot(ax4, "Image", data4);

% ── Row 2: Error Maps, Cross-Section Profile & Energy Minimization ────────
% Subplot 5: L1 Absolute Error |X_L1 - X_true|
ax5 = subplot(2, 4, 5, 'Parent', fig);
data5 = struct('Image', abs(X_l1 - X_true), 'Title', 'L1 Absolute Error', 'Colormap', neon_cmap);
viz.plot(ax5, "Image", data5);

% Subplot 6: L2 Absolute Error |X_L2 - X_true|
ax6 = subplot(2, 4, 6, 'Parent', fig);
data6 = struct('Image', abs(X_l2 - X_true), 'Title', 'L2 Absolute Error', 'Colormap', neon_cmap);
viz.plot(ax6, "Image", data6);

% Subplot 7: 1D Horizontal Cross-Section Profile Slice
ax7 = subplot(2, 4, 7, 'Parent', fig);
slice_row = disk_center_y;
plot(ax7, 1:N_grid, X_true(slice_row, :), 'k--', 'LineWidth', 2.0, 'DisplayName', 'Ground Truth');
hold(ax7, 'on');
plot(ax7, 1:N_grid, Y_measurement(slice_row, :), 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8, 'DisplayName', 'Noisy Y');
plot(ax7, 1:N_grid, X_l1(slice_row, :), 'Color', [0.0 1.0 1.0], 'LineWidth', 2.2, 'DisplayName', 'L1 (TV Sharp)');
plot(ax7, 1:N_grid, X_l2(slice_row, :), 'Color', [1.0 0.0 1.0], 'LineWidth', 2.0, 'DisplayName', 'L2 (Blurred)');
hold(ax7, 'off');
grid(ax7, 'on');
ax7.GridColor = colors.Grid; ax7.GridAlpha = 0.4;
title(ax7, sprintf('Horizontal Slice (Row %d)', slice_row), 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax7, 'Pixel X', 'Color', colors.Text);
ylabel(ax7, 'Intensity', 'Color', colors.Text);
set(ax7, 'Color', colors.Background, 'XColor', colors.Text, 'YColor', colors.Text);
leg7 = legend(ax7, 'Location', 'northeast');
set(leg7, 'Color', colors.Background * 1.2, 'TextColor', colors.Text, 'EdgeColor', colors.Grid);

% Subplot 8: Energy Minimization Curves Comparison
ax8 = subplot(2, 4, 8, 'Parent', fig);
plot(ax8, 0:(length(history_l1.Energy)-1), history_l1.Energy, '-o', ...
    'LineWidth', 2.0, 'Color', [0.0 1.0 1.0], 'MarkerFaceColor', [0.0 1.0 1.0], 'MarkerSize', 4, 'DisplayName', 'L1 Energy');
hold(ax8, 'on');
plot(ax8, 0:(length(history_l2.Energy)-1), history_l2.Energy, '-s', ...
    'LineWidth', 2.0, 'Color', [1.0 0.0 1.0], 'MarkerFaceColor', [1.0 0.0 1.0], 'MarkerSize', 4, 'DisplayName', 'L2 Energy');
hold(ax8, 'off');
grid(ax8, 'on');
ax8.GridColor = colors.Grid; ax8.GridAlpha = 0.4;
title(ax8, 'Energy Minimization Curves', 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax8, 'Iteration', 'Color', colors.Text);
ylabel(ax8, 'Energy E(X)', 'Color', colors.Text);
set(ax8, 'Color', colors.Background, 'XColor', colors.Text, 'YColor', colors.Text);
leg8 = legend(ax8, 'Location', 'northeast');
set(leg8, 'Color', colors.Background * 1.2, 'TextColor', colors.Text, 'EdgeColor', colors.Grid);

sgtitle(fig, 'MRF Spatial Prior Denoising: L1 (Total Variation) vs L2 (Quadratic) Comparison', ...
    'Color', colors.Text, 'FontSize', 15, 'FontWeight', 'bold');
