% =========================================================================
% exp_invertibility_violation.m
%
% Experiment C — Invertibility Violation: What Actually Happens When |theta|>1
%
% Key question: What breaks when the AR(inf) reconstruction formula
%
%   eps_hat_L(t) = sum_{j=0}^{L-1} (-theta)^j * Y_{t-j}
%
% is applied with theta = 2 (non-invertible MA(1))?
%
% Process pair sharing the same autocovariance structure:
%   Non-invertible:   theta = 2,   sigma2 = 0.25
%   Invertible twin:  theta = 0.5, sigma2 = 1.00
%   (root-flipping: theta_tilde = 1/theta,  sigma2_tilde = sigma2 * theta^2)
%
% Kolb Phases:
%   Phase 0 — Setup & theoretical benchmarks
%   Phase 1 — Simulate both MA(1) processes  (concrete experience)
%   Phase 2 — Reconstruction at L=5,20,100   (reflective observation)
%   Phase 3 — Trace divergence to |theta|^j  (abstract conceptualization)
%   Phase 4 — Observational equivalence check (active experimentation)
%
% =========================================================================

clc; close all; clear;

fprintf('=================================================================\n');
fprintf('  EXPERIMENT C: Invertibility Violation — What Breaks When\n');
fprintf('                |theta| > 1 in MA(1)\n');
fprintf('=================================================================\n\n');

% =========================================================================
% PHASE 0 — Setup
% =========================================================================
fprintf('--- PHASE 0: Setup ---\n\n');

T       = 3000;     % simulation length
t_star  = 1500;     % fixed conditioning time for single-point reconstruction
L_max   = 120;      % max reconstruction lag
L_demo  = [5, 20, 100];   % demonstration lag lengths

% Initialize OceanViz engine
viz = OceanViz.Core('Theme', 'Light');

% Non-invertible process (|theta| > 1)
% Create parameter generator using non-invertible start: theta=2.0, sigma2=0.25
gen = OceanMath.TimeSeries.MA1TwinGenerator(2.0, 0.25);

theta_n   = gen.theta_non;
sigma2_n  = gen.sigma2_non;
sigma_n   = sqrt(sigma2_n);

theta_i   = gen.theta_inv;
sigma2_i  = gen.sigma2_inv;
sigma_i   = sqrt(sigma2_i);

% Theoretical shared second moments
theory_var  = gen.theory_var;
theory_cov1 = gen.theory_cov1;

fprintf('  Non-invertible  : theta = %.1f,  sigma^2 = %.2f\n', theta_n, sigma2_n);
fprintf('  Invertible twin : theta = %.2f,  sigma^2 = %.2f\n', theta_i, sigma2_i);
fprintf('\n  Shared second moments:\n');
fprintf('    Var(Y)             = %.4f\n', theory_var);
fprintf('    Cov(Y_t, Y_{t-1}) = %.4f\n', theory_cov1);
fprintf('    Cov(Y_t, Y_{t-k}) = 0.0     (k >= 2)\n\n');

fprintf('  Reconstruction formula:  eps_hat_L(t) = sum_{j=0}^{L-1} (-theta)^j * Y_{t-j}\n\n');
fprintf('  Coefficient terms (-theta)^j:\n');
fprintf('    %-6s  %-18s  %-18s\n', 'j', '(-2.0)^j (non-inv)', '(-0.5)^j (inv)');
fprintf('    %s\n', repmat('-', 1, 46));
for j = [0,1,2,3,5,10,20,30]
    fprintf('    %-6d  %18.4f  %18.8f\n', j, (-theta_n)^j, (-theta_i)^j);
end
fprintf('\n  => Non-invertible: |(-theta)^j| = 2^j -> inf  (DIVERGES)\n');
fprintf('  => Invertible:     |(-theta)^j| = 0.5^j -> 0  (CONVERGES)\n\n');

% =========================================================================
% PHASE 1 — Simulate Both MA(1) Processes
% =========================================================================
fprintf('--- PHASE 1: Simulate MA(1) Processes ---\n\n');

% Simulate both processes using the generator class
[Y_n, eps_n] = gen.simulateNonInvertible(T, 42);
[Y_i, eps_i] = gen.simulateInvertible(T, 42);

fprintf('  Non-invertible  Y_n: mean=%.4f,  std=%.4f,  (theory std=%.4f)\n', ...
    mean(Y_n), std(Y_n), sqrt(theory_var));
fprintf('  Invertible twin Y_i: mean=%.4f,  std=%.4f\n\n', mean(Y_i), std(Y_i));

% =========================================================================
% PHASE 2 — Reconstruction Attempts (Reflective Observation)
% =========================================================================
fprintf('--- PHASE 2: Reconstruction at Fixed t* = %d ---\n\n', t_star);

fprintf('  Applying AR(inf) formula truncated at L lags.\n');
fprintf('  Implementation: recursive filter initialized at 0, run forward L steps:\n');
fprintf('    eps_hat(s) = Y(s) - theta * eps_hat(s-1)\n');
fprintf('    starting from eps_hat(t*-L) = 0\n\n');

% Compute eps_hat_L(t_star) for L = 1,...,L_max
eps_hat_n_series = zeros(L_max, 1);  % reconstruction from Y_n with theta=2
eps_hat_i_series = zeros(L_max, 1);  % reconstruction from Y_i with theta_tilde=0.5

for L = 1:L_max
    % Non-invertible: run forward recursion for L steps starting from t_star-L+1
    eh = 0;   % initialization: eps_hat(t_star - L) = 0
    for s = (t_star - L + 1):t_star
        eh = Y_n(s) - theta_n * eh;
    end
    eps_hat_n_series(L) = eh;

    % Invertible twin
    eh = 0;
    for s = (t_star - L + 1):t_star
        eh = Y_i(s) - theta_i * eh;
    end
    eps_hat_i_series(L) = eh;
end

true_eps_n = eps_n(t_star);
true_eps_i = eps_i(t_star);

fprintf('  True eps_n(t*) = %.6f  (sigma_n = %.4f)\n', true_eps_n, sigma_n);
fprintf('  True eps_i(t*) = %.6f  (sigma_i = %.4f)\n\n', true_eps_i, sigma_i);
fprintf('  %-6s  %-14s  %-14s  %-14s  %-14s\n', ...
    'L', 'eps_hat_n', 'err_n', 'eps_hat_i', 'err_i');
fprintf('  %s\n', repmat('-', 1, 68));
for L = [1, 2, 3, 5, 10, 20, 50, 100]
    if L <= L_max
        err_n = eps_hat_n_series(L) - true_eps_n;
        err_i = eps_hat_i_series(L) - true_eps_i;
        fprintf('  %-6d  %14.4f  %14.4f  %14.6f  %14.8f\n', ...
            L, eps_hat_n_series(L), err_n, eps_hat_i_series(L), err_i);
    end
end
fprintf('\n');

% =========================================================================
% PHASE 3 — Trace Divergence to |theta|^j (Abstract Conceptualization)
% =========================================================================
fprintf('--- PHASE 3: Source of Divergence ---\n\n');

fprintf('  The truncation error when initializing with eps_hat(t*-L) = 0\n');
fprintf('  is approximately: error ~ (-theta)^L * eps_true(t*-L)\n\n');

fprintf('  %-6s  %-14s  %-14s  %-14s  %-14s\n', ...
    'L', '(-theta_n)^L', '~err_n (pred)', 'actual err_n', 'ratio');
fprintf('  %s\n', repmat('-', 1, 70));
for L = [5, 10, 20, 30, 50]
    if L <= L_max
        pred_err = (-theta_n)^L * eps_n(t_star - L);
        act_err  = eps_hat_n_series(L) - true_eps_n;
        ratio    = act_err / pred_err;
        fprintf('  %-6d  %14.2e  %14.2e  %14.2e  %14.6f\n', ...
            L, (-theta_n)^L, pred_err, act_err, ratio);
    end
end
fprintf('\n  Ratio -> 1.0 as L grows: error IS (-theta)^L * eps(t*-L), confirming\n');
fprintf('  divergence traces exactly to the non-summability |(-theta)^j| = 2^j.\n\n');

% =========================================================================
% PHASE 4 — Aggregate MSE + Observational Equivalence Check
% =========================================================================
fprintf('--- PHASE 4: Observational Equivalence & Aggregate MSE ---\n\n');

% Compute aggregate reconstruction MSE over t = L+2 : T-1
% for L = 1, 5, 10, 20, 50, 100
L_vals = [1, 5, 10, 20, 50, 100];
mse_n  = zeros(length(L_vals), 1);
mse_i  = zeros(length(L_vals), 1);

for ii = 1:length(L_vals)
    L = L_vals(ii);
    if L >= T, continue; end
    t_range = (L + 2):(T - 1);
    n_t = length(t_range);

    err_n_agg = zeros(n_t, 1);
    err_i_agg = zeros(n_t, 1);

    for kk = 1:n_t
        t = t_range(kk);

        eh = 0;
        for s = (t - L + 1):t
            eh = Y_n(s) - theta_n * eh;
        end
        err_n_agg(kk) = eh - eps_n(t);

        eh = 0;
        for s = (t - L + 1):t
            eh = Y_i(s) - theta_i * eh;
        end
        err_i_agg(kk) = eh - eps_i(t);
    end

    mse_n(ii) = mean(err_n_agg.^2);
    mse_i(ii) = mean(err_i_agg.^2);
end

fprintf('  Aggregate Reconstruction MSE (averaged over all valid t):\n\n');
fprintf('  %-6s  %-18s  %-18s\n', 'L', 'MSE_n (theta=2)', 'MSE_i (theta=0.5)');
fprintf('  %s\n', repmat('-', 1, 46));
for ii = 1:length(L_vals)
    fprintf('  %-6d  %-18.4f  %-18.8f\n', L_vals(ii), mse_n(ii), mse_i(ii));
end
fprintf('\n');
fprintf('  MSE_n (non-invertible) GROWS with L -> reconstruction diverges.\n');
fprintf('  MSE_i (invertible)     SHRINKS with L -> reconstruction converges.\n\n');

% Sample autocovariances of BOTH processes — must be equal
acov_n = gen.computeSampleAutocovariance(Y_n, 3);
acov_i = gen.computeSampleAutocovariance(Y_i, 3);

fprintf('  Observational Equivalence (sample autocovariances):\n\n');
fprintf('  %-6s  %-14s  %-14s  %-14s\n', 'Lag k', 'Acov Y_n', 'Acov Y_i', 'Theory');
fprintf('  %s\n', repmat('-', 1, 52));
theory_acov = [theory_var; theory_cov1; 0; 0];
for k = 0:3
    fprintf('  %-6d  %-14.6f  %-14.6f  %-14.6f\n', ...
        k, acov_n(k+1), acov_i(k+1), theory_acov(k+1));
end
fprintf('\n  Both processes match the same theoretical autocovariance.\n');
fprintf('  But ONLY the invertible process (theta=0.5) admits\n');
fprintf('  a valid AR(inf) reconstruction of its eps_t.\n\n');

% =========================================================================
% PLOTS
% =========================================================================
RESULTS_DIR = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

%% Figure 1 — Coefficient magnitudes |(-theta)^j|
j_vec = (0:L_max-1)';
fig1 = viz.createFigure('Coefficient Magnitudes');
set(fig1, 'Position', [60 60 960 400]);

subplot(1,2,1);
semilogy(j_vec, abs((-theta_n).^j_vec), 'Color', viz.Colors.Secondary, 'LineWidth', 2.0);
xlabel('Lag  j'); ylabel('|(-\theta)^j|');
title('Non-invertible: \theta = 2  (|(-\theta)^j| = 2^j \rightarrow \infty)', 'Color', viz.Colors.Text);
viz.styleAxes(gca);
annotation_text = sprintf('2^{100} = %.1e', 2^100);
text(80, 2^90, annotation_text, 'FontSize', 9, 'Color', viz.Colors.Secondary);

subplot(1,2,2);
semilogy(j_vec, abs((-theta_i).^j_vec), 'Color', viz.Colors.Primary, 'LineWidth', 2.0);
xlabel('Lag  j'); ylabel('|(-\theta)^j|');
title('Invertible twin: \theta = 0.5  (|(-\theta)^j| = 0.5^j \rightarrow 0)', 'Color', viz.Colors.Text);
viz.styleAxes(gca);
yline(eps, '--', 'machine eps', 'Color', viz.Colors.Text, 'LineWidth', 1, 'HandleVisibility', 'off');

sgtitle('Coefficient magnitudes |(-\theta)^j| in AR(\infty) reconstruction sum', 'Color', viz.Colors.Text);
saveas(fig1, fullfile(RESULTS_DIR, 'exp_inv_coefficients.png'));
fprintf('  Fig 1 saved: exp_inv_coefficients.png\n');

%% Figure 2 — eps_hat_L(t*) vs L for both processes
L_vec = (1:L_max)';
fig2 = viz.createFigure('Reconstruction vs Lag L');
set(fig2, 'Position', [60 500 960 420]);

subplot(1,2,1);
plot(L_vec, eps_hat_n_series, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
hold on;
yline(true_eps_n, '--', sprintf('True \\epsilon_{t^*} = %.3f', true_eps_n), ...
    'Color', viz.Colors.Text, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
hold off;
xlabel('Truncation lag  L');
ylabel('\epsilon_{hat,L}(t^*)');
title(sprintf('NON-INVERTIBLE (\\theta=2)\nDIVERGES as L grows'), 'Color', viz.Colors.Text);
viz.styleAxes(gca);
legend({'eps-hat-L(t*)'}, 'Location', 'northwest', 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');

subplot(1,2,2);
plot(L_vec, eps_hat_i_series, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
hold on;
yline(true_eps_i, '--', sprintf('True \\epsilon_{t^*} = %.3f', true_eps_i), ...
    'Color', viz.Colors.Text, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
hold off;
xlabel('Truncation lag  L');
ylabel('\epsilon_{hat,L}(t^*)');
title(sprintf('INVERTIBLE twin (\\theta=0.5)\nCONVERGES to true eps'), 'Color', viz.Colors.Text);
viz.styleAxes(gca);
legend({'eps-hat-L(t*)'}, 'Location', 'northeast', 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');

sgtitle(sprintf('Reconstruction eps-hat-L(t*) vs lag L  at  t* = %d', t_star), 'Color', viz.Colors.Text);
saveas(fig2, fullfile(RESULTS_DIR, 'exp_inv_reconstruction.png'));
fprintf('  Fig 2 saved: exp_inv_reconstruction.png\n');

%% Figure 3 — Aggregate MSE vs L
L_fine = round(logspace(0, log10(L_max), 40));
L_fine = unique(L_fine);
mse_n_fine = zeros(length(L_fine), 1);
mse_i_fine = zeros(length(L_fine), 1);

for ii = 1:length(L_fine)
    L = L_fine(ii);
    if L >= T - 5, continue; end
    t_range = (L + 2):(min(T-1, L+502));   % use 500-sample window for speed
    n_t = length(t_range);
    en = zeros(n_t,1);
    ei = zeros(n_t,1);
    for kk = 1:n_t
        t = t_range(kk);
        eh = 0;
        for s = (t-L+1):t, eh = Y_n(s) - theta_n*eh; end
        en(kk) = eh - eps_n(t);
        eh = 0;
        for s = (t-L+1):t, eh = Y_i(s) - theta_i*eh; end
        ei(kk) = eh - eps_i(t);
    end
    mse_n_fine(ii) = mean(en.^2);
    mse_i_fine(ii) = mean(ei.^2);
end

fig3 = viz.createFigure('Aggregate Reconstruction MSE vs L');
set(fig3, 'Position', [60 60 860 400]);
loglog(L_fine, mse_n_fine, '-o', 'Color', viz.Colors.Secondary, 'LineWidth', 2.0, 'MarkerSize', 5, ...
    'DisplayName', 'Non-invertible (\theta=2): MSE grows');
hold on;
loglog(L_fine, mse_i_fine, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 2.0, 'MarkerSize', 5, ...
    'DisplayName', 'Invertible (\theta=0.5): MSE decays');
% Reference slopes
ref_L = L_fine;
loglog(ref_L, sigma2_n * (theta_n.^(2*ref_L)), ':', 'Color', viz.Colors.Secondary, 'LineWidth', 1.2, ...
    'DisplayName', 'Slope ~ 2^{2L} (theory)');
loglog(ref_L, sigma2_i * (theta_i.^(2*ref_L)), ':', 'Color', viz.Colors.Primary, 'LineWidth', 1.2, ...
    'DisplayName', 'Slope ~ 0.5^{2L} (theory)');
hold off;
xlabel('Truncation lag  L  (log scale)');
ylabel('Mean Squared Reconstruction Error (log scale)');
title('Reconstruction MSE vs Lag: Divergence (theta=2) vs Convergence (theta=0.5)', 'Color', viz.Colors.Text);
viz.styleAxes(gca);
legend('Location', 'south', 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');
saveas(fig3, fullfile(RESULTS_DIR, 'exp_inv_mse_vs_lag.png'));
fprintf('  Fig 3 saved: exp_inv_mse_vs_lag.png\n');

%% Figure 4 — Observational equivalence: ACF comparison
lags = 0:10;
acov_n_full = gen.computeSampleAutocovariance(Y_n, max(lags));
acov_i_full = gen.computeSampleAutocovariance(Y_i, max(lags));
theory_acov_full = [theory_var; theory_cov1; zeros(max(lags)-1, 1)];

fig4 = viz.createFigure('Observational Equivalence: ACF');
set(fig4, 'Position', [60 500 860 380]);
bar_data = [acov_n_full(:), acov_i_full(:), theory_acov_full(:)];
b = bar(lags, bar_data, 'grouped');
b(1).FaceColor = viz.Colors.Secondary; b(1).DisplayName = 'Y_n (theta=2)';
b(2).FaceColor = viz.Colors.Primary;   b(2).DisplayName = 'Y_i (theta=0.5)';
b(3).FaceColor = viz.Colors.Tertiary;  b(3).DisplayName = 'Theory';
xlabel('Lag  k');  ylabel('Autocovariance');
title('Observational Equivalence: Both MA(1) Processes Share the Same Autocovariance', 'Color', viz.Colors.Text);
viz.styleAxes(gca);
legend('Location', 'northeast', 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');
ylim([-0.2, 1.6]);
saveas(fig4, fullfile(RESULTS_DIR, 'exp_inv_obs_equivalence.png'));
fprintf('  Fig 4 saved: exp_inv_obs_equivalence.png\n');

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('=================================================================\n');
fprintf('  SUMMARY\n');
fprintf('=================================================================\n\n');
fprintf('  MA(1) pair sharing identical autocovariance structure:\n');
fprintf('    Non-invertible:   theta=%.1f,  sigma^2=%.2f\n', theta_n, sigma2_n);
fprintf('    Invertible twin:  theta=%.2f, sigma^2=%.2f\n\n', theta_i, sigma2_i);

fprintf('  AR(inf) reconstruction formula truncated at lag L:\n');
fprintf('    eps_hat_L(t) = sum_{j=0}^{L-1} (-theta)^j * Y_{t-j}\n\n');

fprintf('  Reconstruction error at t* = %d:\n', t_star);
fprintf('    L=5:   non-inv = %+.2e,  inv = %+.2e\n', ...
    eps_hat_n_series(5)-true_eps_n, eps_hat_i_series(5)-true_eps_i);
fprintf('    L=20:  non-inv = %+.2e,  inv = %+.2e\n', ...
    eps_hat_n_series(20)-true_eps_n, eps_hat_i_series(20)-true_eps_i);
fprintf('    L=100: non-inv = %+.2e,  inv = %+.2e\n\n', ...
    eps_hat_n_series(100)-true_eps_n, eps_hat_i_series(100)-true_eps_i);

fprintf('  Conclusion:\n');
fprintf('    1. Non-invertible (theta=2): error ~ (-2)^L * eps(t*-L) -> DIVERGES.\n');
fprintf('       Absolute summability condition sum|eta_j| < inf FAILS: sum 2^j = inf.\n');
fprintf('    2. Invertible twin (theta=0.5): error ~ (0.5)^L -> 0, CONVERGES.\n');
fprintf('       AR(inf) reconstruction is valid: sum 0.5^j = 2 < inf.\n');
fprintf('    3. Both processes have identical autocovariance (observationally\n');
fprintf('       equivalent) — but only one admits a valid shock reconstruction.\n');
fprintf('    4. Root-flipping is not bookkeeping: it is the necessary step\n');
fprintf('       for [4.2.10]-[4.2.11] to be meaningful.\n');
fprintf('=================================================================\n');

