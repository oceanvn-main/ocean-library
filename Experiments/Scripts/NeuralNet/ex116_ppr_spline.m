% =============================================================================
% Exercise 11.6 — Projection Pursuit Regression (PPR) with Cubic Splines
% ESL §11.6 — Compare PPR against the NN from Ex 11.5
%
%   Model:   Y = sigma(a1' * X) + (a2' * X)^2 + 0.30 * Z
%   a1 = [3;3],  a2 = [3;-3],  X ~ N(0,I_2),  Z ~ N(0,1)
%
%   Tasks:
%     1. Vary smoothing parameter p and number of PPR terms M.
%     2. Find the minimum M where PPR performs well.
%     3. Compare final PPR test MSE with Neural Network test MSE from Ex 11.5.
%     4. Visualize the "ridge traces" of each PPR term.
%
% MATLAB Built-ins : csaps, fnval, fminbnd
% Plotting         : OceanViz Engine (Surface, LearningCurve, MultiComparison)
% =============================================================================
clear; clc; close all;

addpath('c:\Ocean Library\Matlab code practice');

% ── 0. OceanViz Engine ────────────────────────────────────────────────────
viz = OceanViz.Core('Theme', 'Dark');

% ── 1. Data Generation (SAME seed as Ex 11.5 for a fair comparison) ───────
rng(42);

a1 = [3; 3];
a2 = [3; -3];

N_train = 100;
X_train = randn(N_train, 2);
Z_train = randn(N_train, 1);
Y_train = 1./(1 + exp(-X_train*a1)) + (X_train*a2).^2 + 0.30 * Z_train;

N_test  = 1000;
X_test  = randn(N_test,  2);
Z_test  = randn(N_test,  1);
Y_test  = 1./(1 + exp(-X_test*a1))  + (X_test*a2).^2  + 0.30 * Z_test;

options_fmin = optimset('Display', 'off');

% ── 2. Task A — Vary Smoothing Parameter p (M = 2 fixed) ──────────────────
p_values  = [0.5, 0.7, 0.9, 1.0];    % 1.0 = full interpolation
M_fixed   = 2;
test_mse_p = zeros(numel(p_values), 1);

fprintf('[Ex 11.6] Sweeping smoothing parameter p ...\n');
for k = 1:numel(p_values)
    [~, Splines_k, W_k] = fit_ppr(X_train, Y_train, M_fixed, p_values(k), options_fmin);
    test_mse_p(k) = eval_ppr_mse(X_test, Y_test, Splines_k, W_k);
    fprintf('  p = %.2f  |  Test MSE = %.4f\n', p_values(k), test_mse_p(k));
end

% Pick best p
[~, best_p_idx] = min(test_mse_p);
p_best = p_values(best_p_idx);
fprintf('\n  --> Best p = %.2f  (Test MSE = %.4f)\n\n', p_best, test_mse_p(best_p_idx));

% ── Plot: p sweep via OceanViz MultiComparison ────────────────────────────
fig1 = viz.createFigure('Ex 11.6 — Effect of Smoothing Parameter p');
fig1.Position = [80, 80, 700, 460];
ax1 = axes('Parent', fig1);

data_p.X = p_values';
data_p.Lines = { struct('Y', test_mse_p, 'Name', 'Test MSE', 'PickX', p_best) };

viz.plot(ax1, 'MultiComparison', data_p, ...
    'Title',      'Ex 11.6 — Test MSE vs Smoothing Parameter p  (M=2)', ...
    'XLabel',     'Smoothing Parameter p', ...
    'YLabel',     'Test MSE', ...
    'ShowLegend', false);

% ── 3. Task B — Vary Number of Terms M (p = p_best) ───────────────────────
M_range       = 1:6;
test_mse_M    = zeros(numel(M_range), 1);
train_mse_M   = zeros(numel(M_range), 1);

fprintf('[Ex 11.6] Sweeping number of PPR terms M ...\n');
for m_idx = 1:numel(M_range)
    M = M_range(m_idx);
    [R_final, Splines_m, W_m] = fit_ppr(X_train, Y_train, M, p_best, options_fmin);
    train_mse_M(m_idx) = mean(R_final.^2);   % Residual MSE on training set
    test_mse_M(m_idx)  = eval_ppr_mse(X_test, Y_test, Splines_m, W_m);
    fprintf('  M = %d  |  Train MSE = %.4f  |  Test MSE = %.4f\n', ...
        M, train_mse_M(m_idx), test_mse_M(m_idx));
end

[~, best_M_idx] = min(test_mse_M);
best_M = M_range(best_M_idx);
fprintf('\n  --> Minimum M for best test error: M = %d  (Test MSE = %.4f)\n\n', ...
    best_M, test_mse_M(best_M_idx));

% ── Plot: M sweep via OceanViz MultiComparison ────────────────────────────
fig2 = viz.createFigure('Ex 11.6 — Performance vs Number of PPR Terms');
fig2.Position = [200, 120, 700, 460];
ax2 = axes('Parent', fig2);

data_M.X = M_range';
data_M.Lines = { ...
    struct('Y', train_mse_M, 'Name', 'Train MSE', 'PickX', []), ...
    struct('Y', test_mse_M,  'Name', 'Test MSE',  'PickX', best_M) };
data_M.TrueModelK = best_M;

viz.plot(ax2, 'MultiComparison', data_M, ...
    'Title',      'Ex 11.6 — MSE vs Number of PPR Terms  (best p)', ...
    'XLabel',     'Number of PPR Terms (M)', ...
    'YLabel',     'MSE', ...
    'ShowLegend', true);

% ── 4. Task C — Ridge Trace Visualization (best M, best p) ───────────────
fprintf('[Ex 11.6] Fitting final PPR model (M=%d, p=%.2f) for visualization...\n', best_M, p_best);
[~, Splines_final, W_final] = fit_ppr(X_train, Y_train, best_M, p_best, options_fmin);

% Dense grid for surface rendering
x1_range = linspace(-2.5, 2.5, 60);
x2_range = linspace(-2.5, 2.5, 60);
[X1g, X2g] = meshgrid(x1_range, x2_range);
X_grid     = [X1g(:), X2g(:)];

% True Signal surface
Y_true_grid = 1./(1 + exp(-X_grid*a1)) + (X_grid*a2).^2;
Z_true      = reshape(Y_true_grid, size(X1g));

% PPR Full Approximation
Z_ppr_full = zeros(size(X1g));
for m = 1:best_M
    Vm = X_grid * W_final(:, m);
    Ym = fnval(Splines_final{m}, Vm')';
    Z_ppr_full = Z_ppr_full + reshape(Ym, size(X1g));
end

% ── Figure 3: True surface vs PPR full approximation (side by side) ───────
fig3 = viz.createFigure('Ex 11.6 — PPR Approximation vs True Signal');
fig3.Position = [80, 300, 1100, 460];

ax_true = subplot(1, 2, 1);
surf_data_true.X = X1g; surf_data_true.Y = X2g; surf_data_true.Z = Z_true;
viz.plot(ax_true, 'Surface', surf_data_true, ...
    'Title',      'True Signal Surface', ...
    'XLabel',     'X_1', 'YLabel', 'X_2', 'ZLabel', 'Y', ...
    'Projection', '3D',  'FaceAlpha', 0.85, ...
    'ShowScatter', false, 'ViewAngle', [-35, 40]);

ax_ppr = subplot(1, 2, 2);
surf_data_ppr.X = X1g; surf_data_ppr.Y = X2g; surf_data_ppr.Z = Z_ppr_full;
viz.plot(ax_ppr, 'Surface', surf_data_ppr, ...
    'Title',      sprintf('PPR Approximation (M=%d, p=%.2f)', best_M, p_best), ...
    'XLabel',     'X_1', 'YLabel', 'X_2', 'ZLabel', 'Y', ...
    'Projection', '3D',  'FaceAlpha', 0.85, ...
    'ShowScatter', false, 'ViewAngle', [-35, 40]);

colormap(fig3, parula);

% ── Figure 4: Individual Ridge Traces per term ────────────────────────────
fig4 = viz.createFigure('Ex 11.6 — Individual PPR Ridge Traces');
n_cols = ceil(best_M / 2);
n_rows = ceil(best_M / n_cols);
fig4.Position = [300, 300, min(1100, 550*n_cols), min(900, 450*n_rows)];

for m = 1:best_M
    Vm_grid = X_grid * W_final(:, m);
    Ym_grid = fnval(Splines_final{m}, Vm_grid')';
    Zm      = reshape(Ym_grid, size(X1g));

    ax_m = subplot(n_rows, n_cols, m);
    surf_data_m.X = X1g; surf_data_m.Y = X2g; surf_data_m.Z = Zm;

    angle_deg = atan2d(W_final(2, m), W_final(1, m));
    viz.plot(ax_m, 'Surface', surf_data_m, ...
        'Title',      sprintf('Ridge Term %d  (\\theta = %.1f°)', m, angle_deg), ...
        'XLabel',     'X_1', 'YLabel', 'X_2', 'ZLabel', sprintf('g_%d', m), ...
        'Projection', '3D',  'FaceAlpha', 0.85, ...
        'ShowScatter', false, 'ViewAngle', [-35, 40]);
end

colormap(fig4, parula);

% ── 5. Summary: Compare with Neural Network from Ex 11.5 ──────────────────
fprintf('\n======================================================\n');
fprintf('  Ex 11.5 vs Ex 11.6 — Final Test MSE Comparison\n');
fprintf('======================================================\n');
fprintf('  PPR  (M=%d,  p=%.2f)   Test MSE = %.4f\n', best_M, p_best, test_mse_M(best_M_idx));
fprintf('  Note: Compare against Neural Network Test MSE from ex115_nn_weight_decay.m\n');
fprintf('======================================================\n');
fprintf('[Ex 11.6] Done.\n');

% =============================================================================
% LOCAL HELPER FUNCTIONS
% =============================================================================

function [R_final, Splines, W] = fit_ppr(X, Y, M, p, opts)
    % FIT_PPR  Forward-stagewise Projection Pursuit Regression.
    %   Uses fminbnd to find the best 1D projection angle in [0, pi],
    %   then fits a cubic smoothing spline (csaps) to the projected data.
    %
    %   Inputs:
    %     X — (N x 2) input matrix
    %     Y — (N x 1) target vector
    %     M — number of PPR terms
    %     p — csaps smoothing parameter (0=linear, 1=interpolate)
    %     opts — optimset options for fminbnd
    %
    %   Outputs:
    %     R_final  — (N x 1) final residual after M terms
    %     Splines  — M x 1 cell array of csaps objects
    %     W        — 2 x M matrix of learned projection directions
    
    N = size(X, 1);
    R = Y;                      % Current residual
    W = zeros(2, M);
    Splines = cell(M, 1);

    for m = 1:M
        % 1. Find optimal projection angle via bounded minimization
        obj = @(theta) ppr_mse(X, R, theta, p);
        theta_opt = fminbnd(obj, 0, pi, opts);
        w = [cos(theta_opt); sin(theta_opt)];
        W(:, m) = w;

        % 2. Project & fit smoothing spline to residuals
        V = X * w;
        [V_sort, sidx] = sort(V);
        R_sort = R(sidx);
        sp = csaps(V_sort, R_sort, p);
        Splines{m} = sp;

        % 3. Subtract the fitted term from the residual
        pred = fnval(sp, V')';
        R = R - pred;
    end
    R_final = R;
end

function mse = ppr_mse(X, R, theta, p)
    % Objective: MSE of csaps fit at angle theta
    w     = [cos(theta); sin(theta)];
    V     = X * w;
    [Vs, sidx] = sort(V);
    Rs    = R(sidx);
    sp    = csaps(Vs, Rs, p);
    pred  = fnval(sp, V')';
    mse   = mean((R - pred).^2);
end

function mse = eval_ppr_mse(X, Y, Splines, W)
    % Evaluate final PPR model on (X, Y)
    M    = size(W, 2);
    Yhat = zeros(size(Y));
    for m = 1:M
        Vm    = X * W(:, m);
        Yhat  = Yhat + fnval(Splines{m}, Vm')';
    end
    mse = mean((Y - Yhat).^2);
end
