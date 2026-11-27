%% EXP_BACKFIT_EX94_CONVERGENCE
%
% EXERCISE 9.4 — Gauss-Seidel / Backfitting with Two Identical Smoothers
% (Hastie & Tibshirani, "Elements of Statistical Learning", Ex. 9.4)
%
% MATHEMATICAL SETUP:
%   Two-term additive model: y = f1(x) + f2(x)
%   Both terms use the SAME symmetric smoother S.
%   S has eigenvalues strictly in [0, 1) by construction.
%
% BACKFITTING (GAUSS-SEIDEL) LOOP:
%   Iterate until convergence:
%     f1 <- S * (y - f2)    [partial residual for term 1]
%     f2 <- S * (y - f1)    [partial residual for term 2, uses NEW f1]
%
% ANALYTICAL LIMIT (Ex. 9.4 result):
%   The backfitting residual  r = y - f1 - f2  converges to:
%
%       r_limit = (I + S)^{-1} (I - S) y
%
%   The limit is computed via MATLAB's \ (backslash) solver, which is
%   numerically stable:
%       r_limit = (I + S) \ ((I - S) * y)
%
%   Therefore:  RSS_limit = ||r_limit||^2
%
% THEORY CONNECTION:
%   Using eigen-decomposition S = Q * Lambda * Q':
%       Fitted values -> 2*Lambda/(1+Lambda) * Q'y  (per eigenvector)
%   This multiplier is strictly GREATER than Lambda alone.
%   Forcing two identical smoothers competes for signal, creating a more
%   aggressive filter and causing initial overfitting (high RSS drop, slow
%   final convergence).
%
% AUTHOR: Ocean Library System
% DATE:   2026-02-28

clear; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..'));

%% =========================================================================
% STEP 1: Construct the Constrained Smoother Operator S
% =========================================================================
% We MUST NOT use a random matrix directly.
% Construction via spectral decomposition guarantees:
%   (a) Symmetry                (S = S')
%   (b) Eigenvalues in [0, 1)   (convergence guaranteed)
%
% S = Q * Lambda * Q'
%   Q      — orthogonal matrix  (from QR of a random Gaussian)
%   Lambda — diagonal, entries drawn uniformly from [0, 1)

rng(42);           % Reproducibility
N      = 50;       % Dimension of the smoother (number of observations)
n_iter = 60;       % Backfitting iterations to run

% Build Q via QR decomposition of a random matrix
[Q, ~] = qr(randn(N));                       % Q is (N x N) orthogonal

% Draw eigenvalues strictly in [0, 1)
lambda_vals = rand(N, 1) * 0.95;             % max 0.95 to avoid near-1 degenerate
Lambda      = diag(lambda_vals);             % diagonal eigenvalue matrix

% Assemble the smoother: S = Q * diag(lambda) * Q'
S = Q * Lambda * Q';

% Sanity check (un-comment to verify):
fprintf('[Setup] S is symmetric: %d\n',      norm(S - S', 'fro') < 1e-10);
fprintf('[Setup] Max eigenvalue:  %.4f\n',   max(lambda_vals));

%% =========================================================================
% STEP 2: Execute the Algorithmic Loop
% =========================================================================
% Initialize target y and component functions f1, f2.
% Record RSS = ||y - f1 - f2||^2 after each HALF-STEP (each single update).

y  = randn(N, 1);  % Random target signal
f1 = zeros(N, 1);
f2 = zeros(N, 1);

% We record RSS after EVERY single f-update (2 updates per outer iteration)
% so the trajectory shows the within-iteration behaviour.
n_halfsteps = n_iter * 2;
rss_history = zeros(n_halfsteps, 1);

half = 0;
for iter = 1:n_iter

    % --- Half-step 1: Update f1 (uses current f2) ---
    f1   = S * (y - f2);
    half = half + 1;
    rss_history(half) = sum((y - f1 - f2).^2);

    % --- Half-step 2: Update f2 (uses NEW f1 — Gauss-Seidel rule) ---
    f2   = S * (y - f1);
    half = half + 1;
    rss_history(half) = sum((y - f1 - f2).^2);

end

%% =========================================================================
% STEP 3: Calculate the Analytical Limit
% =========================================================================
% The residual at convergence: r_limit = (I + S)^{-1} (I - S) y
% Computed with \ to avoid explicit inversion (numerically stable).

I        = eye(N);
r_limit  = (I + S) \ ((I - S) * y);
RSS_limit = sum(r_limit.^2);

fprintf('[Theory] Analytical RSS limit: %.6f\n', RSS_limit);
fprintf('[Sim]    Simulated final RSS:   %.6f\n', rss_history(end));
fprintf('[Error]  Abs diff:              %.2e\n', abs(rss_history(end) - RSS_limit));

%% =========================================================================
% STEP 4: Visualize the Trajectory
% =========================================================================
% Use OceanViz MultiComparison strategy for the convergence curve.
% The analytical limit appears as a second "flat" line (RSS_limit repeated).

viz = OceanViz.Core('Theme', 'Scientific');
fig = viz.createFigure('Ex. 9.4 — Backfitting Convergence');

% --- Build the layout ---
t = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
t.Title.String     = 'Ex. 9.4 — Backfitting with Two Identical Smoothers S';
t.Title.Color      = viz.Colors.Text;
t.Title.FontSize   = 13;
t.Title.FontWeight = 'bold';

% ---- Panel A: RSS Trajectory (absolute scale) ----
ax1 = nexttile(t);
viz.styleAxes(ax1);

half_iters = (1:n_halfsteps)';

data_abs.X = half_iters;
data_abs.Lines = { ...
    struct('Name', 'Backfitting RSS',      'Y', rss_history), ...
    struct('Name', sprintf('Limit = (I+S)^{-1}(I-S)y, ||r||^2 = %.3f', RSS_limit), ...
           'Y', repmat(RSS_limit, n_halfsteps, 1))  ...
};

viz.plot(ax1, 'MultiComparison', data_abs, ...
    'Title',      'RSS Trajectory (Absolute)', ...
    'XLabel',     'Half-Step (2 per outer iteration)', ...
    'YLabel',     'RSS = ||y - f_1 - f_2||^2', ...
    'ShowLegend', true);

% Style the asymptote line as dashed
ax1.Children(end).LineStyle = '--';
ax1.Children(end).LineWidth = 1.5;

% ---- Panel B: log10 scale — see convergence rate ----
ax2 = nexttile(t);
viz.styleAxes(ax2);

% Distance from analytical limit: |RSS(t) - RSS_limit|
gap = abs(rss_history - RSS_limit);
gap(gap < 1e-16) = 1e-16;   % floor to avoid log(0)

data_gap.X = half_iters;
data_gap.Lines = { ...
    struct('Name', '|RSS(t) - RSS_{limit}|', 'Y', log10(gap))   ...
};

viz.plot(ax2, 'MultiComparison', data_gap, ...
    'Title',      'Convergence to Analytical Limit (log_{10} scale)', ...
    'XLabel',     'Half-Step', ...
    'YLabel',     'log_{10}  |RSS - RSS_{limit}|', ...
    'ShowLegend', true);

% ----- Annotate eigenvalue stats -----
lambda_min = min(lambda_vals);
lambda_max = max(lambda_vals);
lambda_avg = mean(lambda_vals);

annotation_str = sprintf( ...
    'N=%d,  \\lambda \\in [%.2f, %.2f],  mean=%.2f', ...
    N, lambda_min, lambda_max, lambda_avg);
xlabel(t, annotation_str, 'Color', viz.Colors.Text, 'FontSize', 9);

fprintf('\n[OceanViz] Figure rendered.\n');
fprintf('Eigenvalue range of S: [%.4f, %.4f]  mean=%.4f\n', ...
    lambda_min, lambda_max, lambda_avg);
fprintf('Convergence rate (spectral): rho = max_j [lambda_j^2 / (1 + lambda_j)] = %.4f\n', ...
    max(lambda_vals.^2 ./ (1 + lambda_vals)));
