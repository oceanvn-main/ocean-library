% EXP_SPECTRAL_CLUSTERING  Recreation of ESL Figure 14.29 Spectral Clustering.
%
%   DATASET:        450 points, 3 concentric circles (r=1, 2.8, 5) with N(0, 0.25^2) noise
%   MODEL:          Unnormalized Spectral Clustering (k=10 NN graph)
%   ASSUMPTIONS:    Unweighted adjacency matrix based on mutual K-nearest neighbors.
%                   Using custom SpectralClustering and KMeans implementation.
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
num_neighbors = 10;
num_eigenvectors = 2; % We extract 2 eigenvectors to match the book (2nd and 3rd)
affinity_graph = OceanStats.Unsupervised.Affinity.MutualKNNStrategy('NumNeighbors', num_neighbors);
sc = OceanStats.Unsupervised.SpectralClustering();

% === 5. Training ===
labels = sc.fit(X, K, 'AffinityStrategy', affinity_graph, 'NumEigenvectors', num_eigenvectors);

% === 6. Evaluation ===
% Not applicable for unsupervised geometry, but we can visualize the separation.

% === 7. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'ESL Fig 14.29: Spectral Clustering', 'Position', [100, 100, 900, 900]);

% Panel 1: Top Left - Data points
ax1 = subplot(2,2,1);
hold on;
% We color by the identified K-Means labels
colors = viz.getColorSequence(K);
for k = 1:K
    scatter(ax1, X(labels == k, 1), X(labels == k, 2), 25, colors(k,:), 'o', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
end
title(ax1, 'Data (K-Means on Z)');
xlabel(ax1, 'x_1'); ylabel(ax1, 'x_2');
axis(ax1, 'equal');
viz.styleAxes(ax1);

% Panel 2: Top Right - 15 smallest eigenvalues
ax2 = subplot(2,2,2);
plot(ax2, 1:15, sc.Eigenvalues(1:15), 'o', 'MarkerSize', 6, 'MarkerFaceColor', colors(1,:), 'MarkerEdgeColor', 'w');
yline(ax2, 0, '--', 'Color', [0.5 0.5 0.5]);
title(ax2, 'Eigenvalues');
xlabel(ax2, 'Number'); ylabel(ax2, 'Eigenvalue');
viz.styleAxes(ax2);

% Panel 3: Bottom Left - 2nd and 3rd smallest eigenvectors
ax3 = subplot(2,2,3);
hold on;
Z2 = sc.AllEigenvectors(:, 2);
Z3 = sc.AllEigenvectors(:, 3);

% Sort data by cluster to match ESL's block-wise plot
[~, sort_idx] = sort(labels); 
Z2_sorted = Z2(sort_idx);
Z3_sorted = Z3(sort_idx);
labels_sorted = labels(sort_idx);

% Plot Z2 (top half of panel, shifted by +0.15)
for k = 1:K
    idx = find(labels_sorted == k);
    scatter(ax3, idx, Z2_sorted(idx) + 0.15, 15, colors(k,:), 'o', 'filled', 'MarkerEdgeColor', 'w');
end

% Plot Z3 (bottom half of panel, shifted by -0.15)
for k = 1:K
    idx = find(labels_sorted == k);
    scatter(ax3, idx, Z3_sorted(idx) - 0.15, 15, colors(k,:), 'o', 'filled', 'MarkerEdgeColor', 'w');
end

yline(ax3, 0.15, '--', 'Color', [0.5 0.5 0.5]);
yline(ax3, -0.15, '--', 'Color', [0.5 0.5 0.5]);
title(ax3, 'Eigenvectors');
xlabel(ax3, 'Index'); 
yticks(ax3, []); % Hide artificial y-ticks
viz.styleAxes(ax3);

% Panel 4: Bottom Right - Spectral Clustering (Z2 vs Z3)
ax4 = subplot(2,2,4);
hold on;
for k = 1:K
    scatter(ax4, Z2(labels == k), Z3(labels == k), 25, colors(k,:), 'o', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
end
title(ax4, 'Spectral Clustering');
xlabel(ax4, 'Second Smallest Eigenvector'); 
ylabel(ax4, 'Third Smallest Eigenvector');
viz.styleAxes(ax4);

% === 8. Summary printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : 3 Concentric Circles (N=%d, p=2)\n', N);
fprintf('  Model         : Unnormalized Spectral Clustering\n');
fprintf('  Graph         : Mutual %d-NN Adjacency\n', num_neighbors);
fprintf('  Zero EigVal   : %.4f\n', sc.Eigenvalues(1));
fprintf('  Cluster Sizes : %d, %d, %d\n', sum(labels==1), sum(labels==2), sum(labels==3));
fprintf('==========================\n');

viz.export(fig, fullfile('c:\', 'Ocean Library', 'Matlab code practice', 'Results', 'exp_spectral_clustering.png'));
