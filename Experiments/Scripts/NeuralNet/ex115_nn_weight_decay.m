% =============================================================================
% Exercise 11.5 — Single Hidden Layer Neural Network with Weight Decay
% ESL §11.5 — Back-propagation & Weight Decay (Regularization)
%
%   Model:   Y = sigma(a1' * X) + (a2' * X)^2 + 0.30 * Z
%   a1 = [3;3],  a2 = [3;-3],  X ~ N(0,I_2),  Z ~ N(0,1)
%
%   Part (b): Plot train/test learning curves for multiple weight-decay values.
%   Part (c): Determine minimum hidden units needed to perform well.
%
% MATLAB Built-ins used: fitnet, train, csaps, fnval
% Plotting             : OceanViz Engine (LearningCurve + MultiComparison)
% =============================================================================
clear; clc; close all;

addpath('c:\Ocean Library\Matlab code practice');

% ── 0. OceanViz Engine ────────────────────────────────────────────────────
viz = OceanViz.Core('Theme', 'Dark');

% ── 1. Data Generation ────────────────────────────────────────────────────
rng(42);

a1 = [3; 3];
a2 = [3; -3];

N_train = 100;
X_train = randn(N_train, 2);
Y_train = 1./(1 + exp(-X_train*a1)) + (X_train*a2).^2 + 0.30 * randn(N_train, 1);

N_test  = 1000;
X_test  = randn(N_test,  2);
Y_test  = 1./(1 + exp(-X_test*a1))  + (X_test*a2).^2  + 0.30 * randn(N_test,  1);

% Combine for MATLAB's divideind (features × samples)
X_all = [X_train; X_test]';
Y_all = [Y_train; Y_test]';
train_idx = 1:N_train;
test_idx  = (N_train + 1):(N_train + N_test);

% ── 2. Part (b) — Learning Curves for Multiple Weight Decay Values ─────────
weight_decays = [0, 0.01, 0.1, 0.5];
n_decays      = numel(weight_decays);

fig1 = viz.createFigure('Ex 11.5 — Learning Curves vs Weight Decay');
fig1.Position = [80, 80, 1100, 850];

for i = 1:n_decays
    % Build network
    net = fitnet(10, 'trainscg');
    net.trainParam.showWindow      = false;
    net.trainParam.epochs          = 500;
    net.performParam.regularization = weight_decays(i);  % Weight Decay

    % Strict train / test split (no validation set)
    net.divideFcn          = 'divideind';
    net.divideParam.trainInd = train_idx;
    net.divideParam.valInd   = [];
    net.divideParam.testInd  = test_idx;

    [~, tr] = train(net, X_all, Y_all);

    % Trim to valid epochs
    ep  = tr.epoch(1:tr.num_epochs+1);
    pf  = tr.perf(1:tr.num_epochs+1);
    tpf = tr.tperf(1:tr.num_epochs+1);

    % ── Plot via OceanViz LearningCurve ──────────────────────────────────
    ax = subplot(2, 2, i);

    data_lc.X = ep';
    data_lc.Lines = { ...
        struct('Y', log10(pf'),  'Name', 'Train MSE'), ...
        struct('Y', log10(tpf'), 'Name', 'Test MSE') };

    viz.plot(ax, 'LearningCurve', data_lc, ...
        'Title',      sprintf('Weight Decay \\lambda = %.2f', weight_decays(i)), ...
        'XLabel',     'Epochs', ...
        'YLabel',     'log_{10}(MSE)', ...
        'ShowLegend', true);
end

sgtitle('Ex 11.5(b) — Effect of Weight Decay on Convergence', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% ── 3. Part (c) — Minimum Hidden Units Needed ─────────────────────────────
fprintf('\n[Ex 11.5c] Training networks with H = 1..10 hidden units...\n');
hidden_range      = 1:10;
train_err_units   = zeros(numel(hidden_range), 1);
test_err_units    = zeros(numel(hidden_range), 1);
fixed_decay       = 0.05;   % A modest value to prevent severe overfitting

for h = hidden_range
    net = fitnet(h, 'trainscg');
    net.trainParam.showWindow       = false;
    net.trainParam.epochs           = 500;
    net.performParam.regularization = fixed_decay;

    net.divideFcn            = 'divideind';
    net.divideParam.trainInd = train_idx;
    net.divideParam.valInd   = [];
    net.divideParam.testInd  = test_idx;

    [~, tr] = train(net, X_all, Y_all);

    train_err_units(h) = tr.perf(end);
    test_err_units(h)  = tr.tperf(end);
    fprintf('  H = %2d  |  Train MSE = %.4f  |  Test MSE = %.4f\n', ...
        h, train_err_units(h), test_err_units(h));
end

% Find the "elbow" — minimum H where test error stops meaningfully improving
[~, best_h] = min(test_err_units);
fprintf('\n  --> Minimum hidden units for best test error: H = %d\n', best_h);

% ── Plot via OceanViz MultiComparison ─────────────────────────────────────
fig2 = viz.createFigure('Ex 11.5 — Performance vs Hidden Units');
fig2.Position = [200, 200, 700, 460];
ax2 = axes('Parent', fig2);

data_mc.X = hidden_range';
data_mc.Lines = { ...
    struct('Y', train_err_units, 'Name', 'Train MSE', 'PickX', []), ...
    struct('Y', test_err_units,  'Name', 'Test MSE',  'PickX', best_h) };
data_mc.TrueModelK = best_h;

viz.plot(ax2, 'MultiComparison', data_mc, ...
    'Title',      'Ex 11.5(c) — MSE vs. Number of Hidden Units', ...
    'XLabel',     'Number of Hidden Units (H)', ...
    'YLabel',     'Final MSE', ...
    'ShowLegend', true);

fprintf('\n[Ex 11.5] Done.\n');
