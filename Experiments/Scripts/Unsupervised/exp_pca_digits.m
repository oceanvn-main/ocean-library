% EXP_PCA_DIGITS  Standard PCA on USPS handwritten digit '3' — ESL §14.5.1 recreation.
%
%   DATASET:        USPS zip code digits — digit '3' from zip.train
%                   Each row: 1 label + 256 pixel intensities (16x16 image, range [-1, 1])
%                   N ~ 658, p = 256
%   MODEL:          Standard PCA via economy SVD (explicit — no MATLAB pca() builtin)
%   ASSUMPTIONS:    Columns centered (zero-mean) before SVD. No scaling (pixels already
%                   normalized to [-1,1] by USPS standard preprocessing).
%                   Explained variance computed as sigma_k^2 / sum(sigma^2).
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading ===
data_path = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Data', 'zip', 'zip.train', 'zip.train');

fprintf('Loading USPS digit data from: %s\n', data_path);
raw = load(data_path);   % Numeric array: N x 257 (col 1 = label, cols 2:257 = pixels)

% Filter to digit '3' only (label == 3)
digit_label = 3;
mask = raw(:, 1) == digit_label;
X_raw = raw(mask, 2:end);  % (N3 x 256), pixel intensities in [-1, 1]

[N, p] = size(X_raw);
img_dim = sqrt(p);         % 16 (for 16x16 images)

fprintf('  Loaded digit "%d": N=%d samples, p=%d features\n', digit_label, N, p);

% === 2. Feature Engineering ===
% Center columns (subtract column means — required for PCA).
% Do NOT scale further: USPS pixels are already in [-1, 1].
mu = mean(X_raw, 1);        % (1 x p) column means
Xc = X_raw - mu;            % (N x p) centered

% === 3. Train / Test Split ===
% PCA is unsupervised — use full dataset.

% === 4. Model Configuration ===
K_components = 10;  % Number of PCs to retain for visualization / comparison with SPCA

% === 5. Training — Explicit Economy SVD ===
% PCA via SVD: Xc = U * S * V^T
%   V columns = principal directions (loadings)
%   Scores = Xc * V = U * S
%   Singular values sigma_k relate to eigenvalues: lambda_k = sigma_k^2 / (N-1)
fprintf('Running economy SVD on %d x %d centered matrix...\n', N, p);
[U, S, V] = svd(Xc, 'econ');   % U:(N x r), S:(r x r), V:(p x r),  r=min(N,p)

sigma   = diag(S);              % (r x 1) singular values
lambda  = sigma.^2 / (N - 1);  % (r x 1) PCA eigenvalues (variance per PC)
pve     = lambda / sum(lambda); % (r x 1) proportion of variance explained
cumPVE  = cumsum(pve);          % (r x 1) cumulative PVE

total_var = sum(lambda);        % Total variance in centered data

% Extract K principal directions and scores
V_K = V(:, 1:K_components);       % (p x K) loadings
scores_K = Xc * V_K;              % (N x K) scores

% === 6. Evaluation ===
% No supervised metric — evaluate via reconstruction error
K_reconstruct_list = [1, 2, 5, 10, 50, 100];
recon_mse = zeros(1, numel(K_reconstruct_list));
for i = 1:numel(K_reconstruct_list)
    K_r = min(K_reconstruct_list(i), size(V, 2));
    X_recon = scores_K(:, 1:min(K_r, K_components)) * V_K(:, 1:min(K_r, K_components))' + mu;
    if K_r > K_components
        % Need more components — recompute
        scores_full = Xc * V(:, 1:K_r);
        X_recon = scores_full * V(:, 1:K_r)' + mu;
    end
    recon_mse(i) = mean(sum((X_raw - X_recon).^2, 2));
end

% How many PCs needed for 90% and 95% variance?
K_90 = find(cumPVE >= 0.90, 1, 'first');
K_95 = find(cumPVE >= 0.95, 1, 'first');

% === 7. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.getColorSequence(6);

fig = figure('Name', 'ESL §14.5 PCA on Digit 3', 'Position', [50, 50, 1400, 900]);

% ---- Panel 1: Mean face (mean digit image) ----
ax1 = subplot(3, 5, 1);
imagesc(ax1, reshape(mu, img_dim, img_dim)');
colormap(ax1, gray);
axis(ax1, 'image', 'off');
title(ax1, 'Mean Image', 'FontSize', 9);
viz.styleAxes(ax1);

% ---- Panels 2-5: First 4 principal component directions (loadings) ----
for k = 1:4
    ax_k = subplot(3, 5, k + 1);
    pc_img = reshape(V(:, k), img_dim, img_dim)';
    imagesc(ax_k, pc_img);
    colormap(ax_k, gray);
    axis(ax_k, 'image', 'off');
    title(ax_k, sprintf('PC %d (%.1f%%)', k, 100*pve(k)), 'FontSize', 9);
    viz.styleAxes(ax_k);
end

% ---- Panels 6-10: Scree plot of first 50 eigenvalues ----
ax_scree = subplot(3, 5, [6 7 8]);
n_show = min(50, length(pve));
bar(ax_scree, 1:n_show, 100*pve(1:n_show), 'FaceColor', colors(1,:), 'EdgeColor', 'none');
hold(ax_scree, 'on');
plot(ax_scree, 1:n_show, 100*cumPVE(1:n_show), '-', 'Color', colors(2,:), 'LineWidth', 1.5);
xline(ax_scree, K_90, '--', sprintf('90%% (K=%d)', K_90), ...
    'Color', colors(3,:), 'LineWidth', 1, 'LabelVerticalAlignment', 'bottom');
xline(ax_scree, K_95, '--', sprintf('95%% (K=%d)', K_95), ...
    'Color', colors(4,:), 'LineWidth', 1, 'LabelVerticalAlignment', 'bottom');
title(ax_scree, 'Scree Plot — % Variance Explained');
xlabel(ax_scree, 'Principal Component');
ylabel(ax_scree, '% Variance');
legend(ax_scree, {'Individual', 'Cumulative'}, 'Location', 'east');
viz.styleAxes(ax_scree);

% ---- Panel 9-10: PC1 vs PC2 scatter colored by quartile of PC1 score ----
ax_scatter = subplot(3, 5, [9 10]);
hold(ax_scatter, 'on');
[~, q_order] = sort(scores_K(:, 1));
n_quartile = floor(N / 4);
q_colors = viz.getColorSequence(4);
for q = 1:4
    q_start = (q-1)*n_quartile + 1;
    q_end   = min(q*n_quartile, N);
    q_idx   = q_order(q_start:q_end);
    scatter(ax_scatter, scores_K(q_idx, 1), scores_K(q_idx, 2), 12, ...
        q_colors(q,:), 'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.7);
end
xlabel(ax_scatter, 'PC_1 Score');
ylabel(ax_scatter, 'PC_2 Score');
title(ax_scatter, 'PC_1 vs PC_2 (colored by PC_1 quartile)');
viz.styleAxes(ax_scatter);

% ---- Panels 11-15: Reconstruction quality at different K ----
K_recon_demo = [1, 5, 10, 30, 100];
demo_idx = 1;  % Always show first sample for consistency
for i = 1:5
    ax_r = subplot(3, 5, 10 + i);
    K_r = min(K_recon_demo(i), size(V, 2));
    scores_r = Xc(demo_idx, :) * V(:, 1:K_r);
    X_recon_single = scores_r * V(:, 1:K_r)' + mu;
    imagesc(ax_r, reshape(X_recon_single, img_dim, img_dim)');
    colormap(ax_r, gray);
    axis(ax_r, 'image', 'off');
    title(ax_r, sprintf('K=%d', K_r), 'FontSize', 9);
    viz.styleAxes(ax_r);
end

% === 8. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY (PCA on Digit %d) ===\n', digit_label);
fprintf('  Dataset       : USPS zip digit "%d" (N=%d, p=%d)\n', digit_label, N, p);
fprintf('  Method        : Standard PCA (economy SVD)\n');
fprintf('  Total Variance: %.4f\n', total_var);
fprintf('  90%% Variance  : K = %d components\n', K_90);
fprintf('  95%% Variance  : K = %d components\n', K_95);
fprintf('  Top-5 PVE     :');
for k = 1:5; fprintf('  PC%d=%.2f%%', k, 100*pve(k)); end
fprintf('\n');
fprintf('  Recon MSE     :');
for i = 1:numel(K_reconstruct_list)
    fprintf('  K=%d:%.3f', K_reconstruct_list(i), recon_mse(i));
end
fprintf('\n=============================================\n');

viz.export(fig, fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Results', 'exp_pca_digits.png'));
