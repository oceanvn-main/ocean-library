% EXP_SPECTRAL_CLUSTERING2  ESL Fig 14.29 — Spectral Clustering with Fully-Connected RBF Affinity.
%
%   DATASET:        450 points, 3 concentric circles (r=1, 2.8, 5) with N(0, 0.25^2) noise
%   MODEL:          Unnormalized Spectral Clustering, fully-connected RBF graph (sigma auto-estimated)
%   ASSUMPTIONS:    Sigma for RBF kernel estimated from median pairwise distance (Gaussian bandwidth heuristic).
%                   Contrast experiment to exp_spectral_clustering.m (mutual KNN affinity).
%                   Using custom SpectralClustering and KMeans implementations.
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading ===
N_per_cluster = 150;
K = 3;
N = N_per_cluster * K;

radii = [1, 2.8, 5];
X = zeros(N, 2);
Y_true = zeros(N, 1);

noise_std = 0.25;

for k = 1:K
    idx = (1:N_per_cluster) + (k-1)*N_per_cluster;
    angles = unifrnd(0, 2*pi, N_per_cluster, 1);
    r = radii(k);

    X(idx, 1) = r * cos(angles) + normrnd(0, noise_std, N_per_cluster, 1);
    X(idx, 2) = r * sin(angles) + normrnd(0, noise_std, N_per_cluster, 1);
    Y_true(idx) = k; % Used for verification
end

% === 2. Feature Engineering ===
% None required for this geometric problem.

% === 3. Train / Test Split ===
% Unsupervised learning on full dataset.

% === 4. Model Configuration ===
% Estimate sigma via the median-distance heuristic (common bandwidth rule for RBF kernels).
% Compute pairwise squared distances on a subsample for efficiency.
D2 = sum(X.^2, 2) + sum(X.^2, 2)' - 2*(X*X');
D2(D2 < 0) = 0;
sigma_est = sqrt(median(D2(D2 > 0)) / 2);  % sigma = sqrt(median_sq_dist / 2)

num_eigenvectors = 2; % 2nd and 3rd eigenvectors (matching ESL Fig 14.29)
affinity_graph = OceanStats.Unsupervised.Affinity.FullyConnectedRBFStrategy('Sigma', sigma_est);
sc = OceanStats.Unsupervised.SpectralClustering();

% === 5. Training ===
labels = sc.fit(X, K, 'AffinityStrategy', affinity_graph, 'NumEigenvectors', num_eigenvectors);

% === 6. Evaluation ===
% Not directly applicable for unsupervised geometry, but we verify cluster sizes.

% === 7. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'ESL Fig 14.29: Spectral Clustering (RBF)', 'Position', [100, 100, 1000, 900]);

colors = viz.getColorSequence(K);

% --- Panel 1: Top Left — Data points colored by predicted cluster ---
ax1 = subplot(2,2,1);
hold(ax1, 'on');
for k = 1:K
    scatter(ax1, X(labels == k, 1), X(labels == k, 2), 25, colors(k,:), 'o', ...
        'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
end
title(ax1, 'Data (K-Means on Z) — RBF Affinity');
xlabel(ax1, 'x_1'); ylabel(ax1, 'x_2');
axis(ax1, 'equal');
viz.styleAxes(ax1);

% --- Panel 2: Top Right — 15 smallest eigenvalues ---
ax2 = subplot(2,2,2);
n_show = min(15, length(sc.Eigenvalues));
plot(ax2, 1:n_show, sc.Eigenvalues(1:n_show), 'o-', 'MarkerSize', 6, ...
    'MarkerFaceColor', colors(1,:), 'MarkerEdgeColor', 'w', 'Color', colors(1,:));
yline(ax2, 0, '--', 'Color', [0.5 0.5 0.5]);
title(ax2, sprintf('Eigenvalues (\\sigma = %.2f)', sigma_est));
xlabel(ax2, 'Number'); ylabel(ax2, 'Eigenvalue');
viz.styleAxes(ax2);

% --- Panel 3: Bottom Left — 2nd and 3rd eigenvectors sorted by cluster ---
ax3 = subplot(2,2,3);
hold(ax3, 'on');

Z2 = sc.AllEigenvectors(:, 2);
Z3 = sc.AllEigenvectors(:, 3);

% Sort by predicted cluster label to create block-wise plot (ESL style)
[~, sort_idx] = sort(labels);
Z2_sorted = Z2(sort_idx);
Z3_sorted = Z3(sort_idx);
labels_sorted = labels(sort_idx);

offset = 0.15; % Vertical separation between Z2 and Z3 strips

for k = 1:K
    idx = find(labels_sorted == k);
    % Z2 strip (top, shifted up)
    scatter(ax3, idx, Z2_sorted(idx) + offset, 15, colors(k,:), 'o', ...
        'filled', 'MarkerEdgeColor', 'none');
    % Z3 strip (bottom, shifted down)
    scatter(ax3, idx, Z3_sorted(idx) - offset, 15, colors(k,:), 'o', ...
        'filled', 'MarkerEdgeColor', 'none');
end

yline(ax3,  offset, '--', 'Color', [0.5 0.5 0.5]);
yline(ax3, -offset, '--', 'Color', [0.5 0.5 0.5]);
title(ax3, 'Eigenvectors Z_2 (top) and Z_3 (bottom)');
xlabel(ax3, 'Index (sorted by cluster)');
yticks(ax3, []);
viz.styleAxes(ax3);

% --- Panel 4: Bottom Right — Spectral embedding Z2 vs Z3 ---
ax4 = subplot(2,2,4);
hold(ax4, 'on');
for k = 1:K
    scatter(ax4, Z2(labels == k), Z3(labels == k), 30, colors(k,:), 'o', ...
        'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
end
title(ax4, 'Spectral Embedding (Z_2 vs Z_3)');
xlabel(ax4, 'Second Smallest Eigenvector Z_2');
ylabel(ax4, 'Third Smallest Eigenvector Z_3');
axis(ax4, 'equal');
viz.styleAxes(ax4);

% === 8. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : 3 Concentric Circles (N=%d, p=2)\n', N);
fprintf('  Model         : Unnormalized Spectral Clustering (RBF)\n');
fprintf('  Sigma (auto)  : %.4f  (median-distance heuristic)\n', sigma_est);
fprintf('  Num EigVecs   : %d\n', num_eigenvectors);
fprintf('  Zero EigVal   : %.6f\n', sc.Eigenvalues(1));
fprintf('  Eigval gap    : %.4f  (EigVal(4) - EigVal(3) = %.4f)\n', ...
    sc.Eigenvalues(4) - sc.Eigenvalues(3), sc.Eigenvalues(4) - sc.Eigenvalues(3));
fprintf('  Cluster Sizes :');
for k = 1:K
    fprintf('  Cluster%d=%d', k, sum(labels == k));
end
fprintf('\n==========================\n');

viz.export(fig, fullfile('c:\', 'Ocean Library', 'Matlab code practice', 'Results', 'exp_spectral_clustering2.png'));