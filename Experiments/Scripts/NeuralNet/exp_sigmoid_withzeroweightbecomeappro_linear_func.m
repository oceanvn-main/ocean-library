%% Experiment: Linearity of Sigmoid with Near-Zero Weights
clear; clc; close all;
% 1. Generate objective, randomized input data
n_samples = 1000;
X = randn(n_samples, 1); %1000x1;

W_small = 0.001 * randn(1, 1);
b_small = 0.001 * randn(1, 1);

W_large = 3.0 * randn(1, 1);
b_large = randn(1, 1);

%Pre-activation Z
Z_small = X * W_small + b_small;
sigmoid_small = 1 ./ (1 + exp(-Z_small));

Z_large = X * W_large + b_large;
sigmoid_large = 1 ./ (1 + exp(-Z_large));

mdl_small = fitlm(Z_small, sigmoid_small);
r2_small = mdl_small.Rsquared.Ordinary;

mdl_large = fitlm(Z_large, sigmoid_large);
r2_large = mdl_large.Rsquared.Ordinary;

fprintf('R-squared for near-zero weights: %.4f\n', r2_small);
fprintf('R-squared for larger weights: %.4f\n', r2_large);

% 5. Plot the data (Dark Mode Styling)
fig = figure('Color', [0.15 0.15 0.15], 'Name', 'Sigmoid Linearity Experiment');

% Subplot 1: Near-Zero Weights
ax1 = subplot(1, 2, 1);
scatter(Z_small, sigmoid_small, 15, 'filled', 'MarkerFaceColor', [0.3 0.8 0.9]);
title(sprintf('Near-Zero Weights\nLinearity (R^2) = %.4f', r2_small), 'Color', 'w');
xlabel('Pre-activation (Z)', 'Color', 'w');
ylabel('Activation Output (A)', 'Color', 'w');
set(ax1, 'Color', [0.2 0.2 0.2], 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.5 0.5 0.5]);
grid on;

% Subplot 2: Larger Weights
ax2 = subplot(1, 2, 2);
scatter(Z_large, sigmoid_large, 15, 'filled', 'MarkerFaceColor', [0.9 0.4 0.4]);
title(sprintf('Larger Weights\nLinearity (R^2) = %.4f', r2_large), 'Color', 'w');
xlabel('Pre-activation (Z)', 'Color', 'w');
ylabel('Activation Output (A)', 'Color', 'w');
set(ax2, 'Color', [0.2 0.2 0.2], 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.5 0.5 0.5]);
grid on;