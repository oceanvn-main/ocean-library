% EXP_KMEANS_VS_GMM  ESL Exercise 13.1 — K-Means vs Fixed-Sigma GMM.
%
%   Empirically demonstrates that K-means is the limiting case of EM for
%   a Gaussian mixture model with Sigma_r = sigma * I as sigma → 0.
%
%   DATASET:        ESLMixtureGenerator (N=300, 3 classes, 10 centers/class)
%   MODEL:          Fixed-sigma GMM (sigma varies) vs K-means (Lloyd's)
%   ASSUMPTIONS:    - All K components share the same isotropic covariance
%                   - Sigma is a frozen hyperparameter, NOT learned by EM
%                   - Both algorithms initialized from identical centroids
%   REPRODUCIBILITY: rng(42, 'twister')
%
% Phase 1: "Shrinking Sigma" — watch soft responsibilities harden
% Phase 2: "Trajectory Race" — centroid paths converge as sigma → 0
% Phase 3: Convergence metric — ||mu_kmeans - mu_gmm|| vs sigma
%
% Ref: Hastie et al. (2009), ESL Exercise 13.1
clc; clear; close all; rng(42, 'twister');

%% === 1. Data Generation ===
fprintf('=== Step 1: Data Generation ===\n');

gen = OceanData.Generators.ESLMixtureGenerator('N', 300, 'NumClasses', 3, 'Seed', 42);
[X, Y] = gen.generate();
[N, D] = size(X);
K = 3;

fprintf('  N=%d, D=%d, K=%d\n', N, D, K);

% Fix shared initial centroids for ALL runs
rng(7, 'twister');
initIdx = randperm(N, K);
C0 = X(initIdx, :);
fprintf('  Shared init centroids (rows %s)\n', mat2str(initIdx));

% Class colors
classColors = [
    0.30 0.65 0.95;   % blue
    1.00 0.50 0.15;   % orange
    0.35 0.80 0.40;   % green
];

%% === Phase 1: Shrinking Sigma Responsibility Test ===
fprintf('\n=== Phase 1: Shrinking Sigma ===\n');

sigmas_phase1 = [2.0, 1.0, 0.5, 0.1, 0.01];
nSig = numel(sigmas_phase1);

viz = OceanViz.Core('Theme', 'Cyberpunk');
fig1 = viz.createFigure('ESL Ex 13.1 — Shrinking Sigma Responsibility Test');
fig1.Position = [30, 30, 1400, 350];

for si = 1:nSig
    sigma = sigmas_phase1(si);
    
    % Create fixed-sigma GMM
    model = OceanStats.Unsupervised.GMM(K);
    recipe = OceanStats.Inference.Recipes.Recipe_FixedSigmaGMM_EM(model, sigma);
    recipe.initialize(X, C0);
    
    % Run EM
    optimizer = OceanStats.Optimization.GenericOptimizer('MaxIter', 50, 'Verbose', false);
    optimizer.run(recipe, X);
    
    % Get final responsibilities
    gammas = recipe.getResponsibilities(X);
    
    % Max responsibility (how "hard" is the assignment?)
    maxGamma = max(gammas, [], 2);
    meanMaxGamma = mean(maxGamma);
    fprintf('  sigma=%.3f: mean(max gamma)=%.4f\n', sigma, meanMaxGamma);
    
    % Plot
    ax = subplot(1, nSig, si, 'Parent', fig1);
    hold(ax, 'on'); viz.styleAxes(ax);
    
    % Color each point by blended responsibility
    pointColors = gammas * classColors;  % (N x 3) blended RGB
    scatter(ax, X(:,1), X(:,2), 12, pointColors, 'filled', ...
        'MarkerFaceAlpha', 0.8);
    
    % Show centroids
    scatter(ax, model.Mu(:,1), model.Mu(:,2), 100, classColors, 'd', ...
        'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
    
    title(ax, sprintf('\\sigma = %.2f', sigma), ...
        'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
    axis(ax, 'equal');
    set(ax, 'XTick', [], 'YTick', []);
    hold(ax, 'off');
end

% Add annotation
annotation(fig1, 'textbox', [0.0 0.92 1.0 0.08], ...
    'String', 'Soft \rightarrow Hard: Responsibilities harden as \sigma \rightarrow 0', ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');

resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig1, fullfile(resultsDir, 'fig_ex13_1_shrinking_sigma.png'));
fprintf('[Phase 1] Saved: fig_ex13_1_shrinking_sigma.png\n');

%% === Phase 2: Trajectory Race ===
fprintf('\n=== Phase 2: Trajectory Race ===\n');

sigmas_phase2 = [2.0, 0.5, 0.1, 0.01];
nSig2 = numel(sigmas_phase2);
maxIter = 30;

% Run K-means with tracking (same init)
km = OceanStats.Unsupervised.KMeans();
km.fit(X, K, 'InitCentroids', C0, 'TrackHistory', true, 'MaxIter', maxIter);
km_traj = km.History;
km_final = km.Centroids;

fprintf('  K-means converged in %d iterations\n', numel(km_traj));

fig2 = viz.createFigure('ESL Ex 13.1 — Trajectory Race: K-Means vs GMM');
fig2.Position = [50, 50, 1000, 900];

mu_diffs = zeros(nSig2, 1);

for si = 1:nSig2
    sigma = sigmas_phase2(si);
    
    % Run fixed-sigma GMM from same init
    model = OceanStats.Unsupervised.GMM(K);
    recipe = OceanStats.Inference.Recipes.Recipe_FixedSigmaGMM_EM(model, sigma);
    recipe.initialize(X, C0);
    
    optimizer = OceanStats.Optimization.GenericOptimizer( ...
        'MaxIter', maxIter, 'Verbose', false);
    optimizer.run(recipe, X);
    
    gmm_traj = recipe.CentroidHistory;
    gmm_final = model.Mu;
    
    fprintf('  sigma=%.3f: GMM iters=%d\n', sigma, numel(gmm_traj));
    
    % Sort centroids by nearest match for fair comparison
    perm = matchCentroids(km_final, gmm_final);
    gmm_final_matched = gmm_final(perm, :);
    mu_diffs(si) = max(sqrt(sum((km_final - gmm_final_matched).^2, 2)));
    fprintf('    max ||mu_km - mu_gmm|| = %.6f\n', mu_diffs(si));
    
    % Plot
    ax = subplot(2, 2, si, 'Parent', fig2);
    hold(ax, 'on'); viz.styleAxes(ax);
    
    % Data as faint background
    scatter(ax, X(:,1), X(:,2), 5, [0.4 0.4 0.4], 'filled', ...
        'MarkerFaceAlpha', 0.3, 'HandleVisibility', 'off');
    
    % K-means trajectories
    for k = 1:K
        coords_km = zeros(numel(km_traj), 2);
        for t = 1:numel(km_traj)
            if ~isempty(km_traj(t).Centroids)
                coords_km(t,:) = km_traj(t).Centroids(k,:);
            end
        end
        valid = any(coords_km ~= 0, 2);
        coords_km = coords_km(valid, :);
        
        plot(ax, coords_km(:,1), coords_km(:,2), '-', ...
            'Color', [0.30 0.65 0.95 0.9], 'LineWidth', 2.0, ...
            'HandleVisibility', 'off');
        scatter(ax, coords_km(end,1), coords_km(end,2), 80, ...
            [0.30 0.65 0.95], 's', 'filled', 'MarkerEdgeColor', 'w', ...
            'HandleVisibility', 'off');
    end
    
    % GMM trajectories
    for k = 1:K
        coords_gmm = zeros(numel(gmm_traj), 2);
        for t = 1:numel(gmm_traj)
            coords_gmm(t,:) = gmm_traj{t}(k,:);
        end
        
        plot(ax, coords_gmm(:,1), coords_gmm(:,2), '--', ...
            'Color', [1.00 0.50 0.15 0.9], 'LineWidth', 2.0, ...
            'HandleVisibility', 'off');
        scatter(ax, coords_gmm(end,1), coords_gmm(end,2), 80, ...
            [1.00 0.50 0.15], 'o', 'filled', 'MarkerEdgeColor', 'w', ...
            'HandleVisibility', 'off');
    end
    
    % Initial centroids
    scatter(ax, C0(:,1), C0(:,2), 60, [1 1 1], '+', 'LineWidth', 2, ...
        'HandleVisibility', 'off');
    
    % Legend proxies
    plot(ax, NaN, NaN, '-', 'Color', [0.30 0.65 0.95], 'LineWidth', 2, ...
        'DisplayName', 'K-Means');
    plot(ax, NaN, NaN, '--', 'Color', [1.00 0.50 0.15], 'LineWidth', 2, ...
        'DisplayName', 'GMM');
    
    title(ax, sprintf('\\sigma = %.2f  |  \\Delta\\mu = %.4f', sigma, mu_diffs(si)), ...
        'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
    axis(ax, 'equal');
    legend(ax, 'Location', 'best', 'FontSize', 8, ...
        'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, ...
        'EdgeColor', [0.3 0.3 0.3]);
    hold(ax, 'off');
end

saveas(fig2, fullfile(resultsDir, 'fig_ex13_1_trajectory_race.png'));
fprintf('[Phase 2] Saved: fig_ex13_1_trajectory_race.png\n');

%% === Phase 3: Convergence Metric ===
fprintf('\n=== Phase 3: Convergence Metric ===\n');

sigmas_sweep = logspace(log10(5), log10(0.001), 20);
mu_diffs_sweep = zeros(numel(sigmas_sweep), 1);

for si = 1:numel(sigmas_sweep)
    sigma = sigmas_sweep(si);
    
    model = OceanStats.Unsupervised.GMM(K);
    recipe = OceanStats.Inference.Recipes.Recipe_FixedSigmaGMM_EM(model, sigma);
    recipe.initialize(X, C0);
    
    optimizer = OceanStats.Optimization.GenericOptimizer( ...
        'MaxIter', 50, 'Verbose', false);
    optimizer.run(recipe, X);
    
    gmm_final = model.Mu;
    perm = matchCentroids(km_final, gmm_final);
    mu_diffs_sweep(si) = max(sqrt(sum((km_final - gmm_final(perm,:)).^2, 2)));
end

fig3 = viz.createFigure('ESL Ex 13.1 — GMM→K-Means Convergence');
fig3.Position = [80, 80, 700, 500];
ax3 = axes('Parent', fig3);
hold(ax3, 'on'); viz.styleAxes(ax3);

loglog(ax3, sigmas_sweep, mu_diffs_sweep, '-o', ...
    'Color', [0.30 0.65 0.95], 'MarkerFaceColor', [0.30 0.65 0.95], ...
    'MarkerSize', 5, 'LineWidth', 2);

xlabel(ax3, '\sigma (log scale)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax3, 'max ||\mu_{KM} - \mu_{GMM}|| (log scale)', ...
    'Color', viz.Colors.Text, 'FontSize', 12);
title(ax3, 'GMM \rightarrow K-Means as \sigma \rightarrow 0', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
set(ax3, 'XScale', 'log', 'YScale', 'log');
grid(ax3, 'on');
hold(ax3, 'off');

saveas(fig3, fullfile(resultsDir, 'fig_ex13_1_convergence.png'));
fprintf('[Phase 3] Saved: fig_ex13_1_convergence.png\n');

%% === 8. Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ESLMixture (N=%d, p=%d, K=%d)\n', N, D, K);
fprintf('  Model         : Fixed-sigma GMM (EM) vs K-means (Lloyd)\n');
fprintf('  Phase 1       : Shrinking sigma [%s]\n', num2str(sigmas_phase1, '%.2f '));
fprintf('  Phase 2       : Trajectory race [%s]\n', num2str(sigmas_phase2, '%.2f '));
fprintf('  Phase 3       : Convergence sweep (%d sigma values)\n', numel(sigmas_sweep));
fprintf('  Final ||diff||: %.6f (at sigma=%.4f)\n', mu_diffs_sweep(end), sigmas_sweep(end));
fprintf('==========================\n');

%% === Local Helper ===
function perm = matchCentroids(A, B)
    % MATCHCENTROIDS  Find permutation of B rows that best matches A rows.
    %   Uses Hungarian-style greedy nearest matching on Euclidean distance.
    K = size(A, 1);
    D2 = sum(A.^2, 2) + sum(B.^2, 2)' - 2 * (A * B');
    perm = zeros(K, 1);
    used = false(K, 1);
    for k = 1:K
        D2_row = D2(k, :);
        D2_row(used) = inf;
        [~, j] = min(D2_row);
        perm(k) = j;
        used(j) = true;
    end
end
