% EXP_ICA_SIMULATION  ESL Fig 14.42 — ICA Benchmark: FastICA vs ProDenICA.
%
%   REFERENCE:  ESL §14.7.4, Fig 14.42.
%   GOAL:       Compare average Amari distance (ESL Eq. 14.97) of FastICA and
%               ProDenICA across 18 source distributions using 30 MC repeats.
%   DATA:       Synthetic — 2D ICA, N=1024 per trial.
%   METRIC:     Amari distance d(A0, A_hat) — lower is better.
%
%   NOTE: KernelICA is currently a stub and is excluded from this comparison.
%         The plot shows 2 lines (FastICA, ProDenICA) instead of 3.
%
%   PROTOCOL (per distribution, per repeat):
%     1. Sample S (N x 2) — two independent sources from the distribution
%     2. Random mixing A0 (2x2) with cond(A0) in [1, 2]
%     3. Mix: X = S * A0'  (N x 2)
%     4. Run FastICA (5 random starts, pick best by negentropy criterion)
%     5. Run ProDenICA (5 random starts, pick best by negentropy criterion)
%     6. Compute Amari distance vs A0 for each
%     Average over 30 repeats.
%
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

N_SAMPLES  = 1024;   % samples per trial
N_REPS     = 30;     % Monte Carlo repeats per distribution
N_STARTS   = 5;      % random starts per ICA method (pick best by negentropy)
COND_MAX   = 2.0;    % max condition number of mixing matrix

% === 1. Define 18 Source Distributions ===
% Each entry: name (single char a-r), label (display), sample function
% All sample functions return (N x 1) zero-mean unit-variance samples.

std_fn = @(x) (x - mean(x)) / (std(x) + 1e-12);

D = struct('name', {}, 'label', {}, 'group', {}, 'sample', {});
k = 0;

% --- Group 1: t-distributions (super-Gaussian, heavy-tailed) ---
for nu = [1, 2, 3, 5]
    k = k + 1;
    nu_ = nu;
    D(k).name   = char('a' + k - 1);
    D(k).label  = sprintf('t(%d)', nu_);
    D(k).group  = 't';
    D(k).sample = @(N) std_fn(trnd(nu_, N, 1));
end

% --- Group 2: Uniform (sub-Gaussian) ---
for b = [1, 2]
    k = k + 1;
    b_ = b;
    D(k).name   = char('a' + k - 1);
    D(k).label  = sprintf('Unif(%.0f)', b_);
    D(k).group  = 'uniform';
    D(k).sample = @(N) std_fn((rand(N, 1) - 0.5) * 2 * b_);
end

% --- Group 3: Exponential (skewed, super-Gaussian) ---
k = k + 1;
D(k).name   = char('a' + k - 1);
D(k).label  = 'Exp(1)';
D(k).group  = 'exp';
D(k).sample = @(N) std_fn(exprnd(1, N, 1));

k = k + 1;
D(k).name   = char('a' + k - 1);
D(k).label  = 'Laplace';
D(k).group  = 'exp';
D(k).sample = @(N) std_fn(log(rand(N,1) ./ rand(N,1)));  % Laplace via log-ratio

% --- Group 4: Mixture of Exponentials ---
mix_params = [0.5, 1.0, 2.0;   % p, lam1, lam2
              0.3, 0.5, 3.0;
              0.7, 2.0, 0.5];
for ii = 1:3
    k = k + 1;
    p_ = mix_params(ii,1); l1_ = mix_params(ii,2); l2_ = mix_params(ii,3);
    D(k).name   = char('a' + k - 1);
    D(k).label  = sprintf('MixExp');
    D(k).group  = 'mixexp';
    D(k).sample = @(N) std_fn(sampleMixExp(N, p_, l1_, l2_));
end

% --- Group 5: Symmetric Gaussian Mixtures: 0.5*N(-mu,1) + 0.5*N(mu,1) ---
for mu = [1, 2, 4]
    k = k + 1;
    mu_ = mu;
    D(k).name   = char('a' + k - 1);
    D(k).label  = sprintf('SymGMM(%d)', mu_);
    D(k).group  = 'symgmm';
    D(k).sample = @(N) std_fn(sampleSymGMM(N, mu_));
end

% --- Group 6: Asymmetric Gaussian Mixtures: p*N(mu1,1) + (1-p)*N(mu2,1) ---
asym_params = [0.3, -1, 3;    % p, mu1, mu2
               0.2, -2, 1;
               0.4,  0, 4;
               0.1, -3, 0.5];
for ii = 1:4
    k = k + 1;
    p_ = asym_params(ii,1); m1_ = asym_params(ii,2); m2_ = asym_params(ii,3);
    D(k).name   = char('a' + k - 1);
    D(k).label  = sprintf('AsymGMM');
    D(k).group  = 'asymgmm';
    D(k).sample = @(N) std_fn(sampleAsymGMM(N, p_, m1_, m2_));
end

n_dists = numel(D);   % should be 18
fprintf('Defined %d source distributions.\n', n_dists);

% === 2. Monte Carlo Benchmark ===
amari_fast = zeros(n_dists, N_REPS);
amari_pden = zeros(n_dists, N_REPS);

strat_fast = OceanStats.Unsupervised.ICASolvers.FastICAStrategy('MaxIter', 200, 'Tol', 1e-5);
strat_pden = OceanStats.Unsupervised.ICASolvers.ProDenICAStrategy('MaxIter', 100, 'Tol', 1e-4);

for d = 1:n_dists
    fprintf('Distribution %s (%s)...', D(d).name, D(d).label);
    for rep = 1:N_REPS
        % 1. Generate independent sources (two columns, same distribution)
        S = [D(d).sample(N_SAMPLES), D(d).sample(N_SAMPLES)];  % (N x 2)

        % 2. Random mixing matrix, cond in [1, COND_MAX]
        A0 = randMixing(COND_MAX);

        % 3. Mix
        X = S * A0';   % (N x 2)

        % 4. Best-of-N_STARTS ICA runs (select by total negentropy)
        amari_fast(d, rep) = bestICA(X, strat_fast, N_STARTS, A0);
        amari_pden(d, rep) = bestICA(X, strat_pden, N_STARTS, A0);
    end
    fprintf(' done. FastICA=%.4f  ProDenICA=%.4f\n', ...
        mean(amari_fast(d,:)), mean(amari_pden(d,:)));
end

mean_fast = mean(amari_fast, 2);   % (18 x 1)
mean_pden = mean(amari_pden, 2);   % (18 x 1)

% === 3. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');
bg  = [0.10 0.10 0.13];
xs  = 1:n_dists;
dist_labels = {D.name};

% --- Figure 1: Distribution shapes (3 rows x 6 cols) ---
fig1 = figure('Name', 'ESL Fig 14.42 — 18 Source Distributions', ...
    'Color', bg, 'Position', [20 50 860 500]);
tl1  = tiledlayout(fig1, 3, 6, 'TileSpacing', 'tight', 'Padding', 'compact');
u_plot = linspace(-4, 4, 300)';
for d = 1:n_dists
    ax = nexttile(tl1);
    % Estimate density via KDE from a large sample
    samp = D(d).sample(5000);
    h_bw = 0.9 * std(samp) * 5000^(-0.2);
    K_plot = mean(exp(-0.5 * ((u_plot - samp') / h_bw).^2), 2) / (sqrt(2*pi) * h_bw);
    plot(ax, u_plot, K_plot, 'Color', [0.2 0.85 0.6], 'LineWidth', 1.5);
    title(ax, D(d).name, 'Color', 'w', 'FontSize', 9);
    set(ax, 'XTickLabel', [], 'YTickLabel', []);
    xlim(ax, [-4, 4]);
    viz.styleAxes(ax);
end
sgtitle(fig1, 'ESL Fig 14.42 (left) — 18 Source Distributions for ICA Benchmark', ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');

% --- Figure 2: Amari distance comparison (log scale) ---
fig2 = figure('Name', 'ESL Fig 14.42 — ICA Amari Distance Benchmark', ...
    'Color', bg, 'Position', [900 50 850 550]);
ax2  = axes(fig2, 'Color', [0.07 0.07 0.09]);

semilogy(ax2, xs, mean_fast, 'o-', 'Color', [1.0 0.75 0.2], 'LineWidth', 2, ...
    'MarkerFaceColor', [1.0 0.75 0.2], 'MarkerSize', 7);
hold(ax2, 'on');
semilogy(ax2, xs, mean_pden, 's-', 'Color', [0.2 0.85 0.85], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.85 0.85], 'MarkerSize', 7);
hold(ax2, 'off');

set(ax2, 'XTick', xs, 'XTickLabel', dist_labels, 'FontSize', 10);
xlabel(ax2, 'Distribution', 'Color', 'w', 'FontSize', 11);
ylabel(ax2, 'Amari Distance from True A', 'Color', 'w', 'FontSize', 11);
title(ax2, sprintf('Average Amari distance (%d reps, %d starts each)', N_REPS, N_STARTS), ...
    'Color', 'w', 'FontSize', 10);
legend(ax2, {'FastICA (tanh)', 'ProDenICA (KDE)'}, ...
    'TextColor', 'w', 'Color', [0.15 0.15 0.2], 'FontSize', 10, 'Location', 'best');
grid(ax2, 'on');
ylim(ax2, [0.005, 1]);
viz.styleAxes(ax2);
sgtitle(fig2, 'ESL Fig 14.42 (right) — ICA Benchmark: FastICA vs ProDenICA', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% === 4. Summary ===
fprintf('\n=== Summary ===\n');
fprintf('  FastICA wins (lower Amari):    %d / %d distributions\n', ...
    sum(mean_fast < mean_pden), n_dists);
fprintf('  ProDenICA wins (lower Amari):  %d / %d distributions\n', ...
    sum(mean_pden < mean_fast), n_dists);
fprintf('  Overall mean — FastICA: %.4f  ProDenICA: %.4f\n', ...
    mean(mean_fast), mean(mean_pden));

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function A = randMixing(cond_max)
    % Random 2x2 mixing matrix with condition number in [1, cond_max]
    c = 1 + (cond_max - 1) * rand();
    [U, ~] = qr(randn(2, 2));
    [V, ~] = qr(randn(2, 2));
    A = U * diag([1, 1/c]) * V';
end

function best_amari = bestICA(X, strategy, n_starts, A0)
    % Run ICA n_starts times, pick best result by negentropy, return Amari dist.
    % negentropy proxy: E{log cosh(S)} - 0.3745 (Gaussian baseline)
    best_negen = -Inf;
    best_amari = Inf;
    GAUSS_REF  = 0.3745;   % E{log cosh(v)} for v ~ N(0,1)

    for s = 1:n_starts
        try
            ica = OceanStats.Unsupervised.ICA();
            ica.fit(X, 2, 'Strategy', strategy);
            S     = ica.getComponents();             % (N x 2)
            A_hat = ica.getMixingMatrix();           % (2 x 2)
            negen = sum(abs(mean(log(cosh(S))) - GAUSS_REF));
            if negen > best_negen
                best_negen = negen;
                best_amari = OceanStats.Unsupervised.ICASolvers.amariDist(A0, A_hat);
            end
        catch
            % Strategy failed to converge on this start — skip
        end
    end
end

function x = sampleMixExp(N, p, lam1, lam2)
    % Sample from p*Exp(lam1) + (1-p)*Exp(lam2)
    u = rand(N, 1) < p;
    x = u .* exprnd(1/lam1, N, 1) + (1 - u) .* exprnd(1/lam2, N, 1);
end

function x = sampleSymGMM(N, mu)
    % Symmetric GMM: 0.5*N(-mu,1) + 0.5*N(mu,1)
    u = rand(N, 1) < 0.5;
    x = u .* (randn(N, 1) + mu) + (1 - u) .* (randn(N, 1) - mu);
end

function x = sampleAsymGMM(N, p, mu1, mu2)
    % Asymmetric GMM: p*N(mu1,1) + (1-p)*N(mu2,1)
    u = rand(N, 1) < p;
    x = u .* (randn(N, 1) + mu1) + (1 - u) .* (randn(N, 1) + mu2);
end
