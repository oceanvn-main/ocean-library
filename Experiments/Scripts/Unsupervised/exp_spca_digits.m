% EXP_SPCA_DIGITS  Sparse PCA on USPS digit '3' — ESL §14.5.5 (Zou et al., 2006).
%
%   DATASET:        USPS zip code digits — digit '3' from zip.train
%                   N ~ 658, p = 256 pixels (16x16 images, range [-1, 1])
%   MODEL:          Sparse PCA (Zou et al. 2006) via alternating minimization.
%                   Contrast with standard PCA in exp_pca_digits.m.
%   ASSUMPTIONS:    Columns are centered before fit (handled inside SparsePCA.fit).
%                   Lambda (L2) = 1e-4 for uniqueness when p > N.
%                   Lambda1 (L1) varied in a sweep to demonstrate sparsity control.
%                   Standard PCA recovered as baseline (Lambda1 = 0, Lambda = 0).
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading ===
data_path = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Data', 'zip', 'zip.train', 'zip.train');

fprintf('Loading USPS digit data...\n');
raw = load(data_path);

digit_label = 3;
mask = raw(:, 1) == digit_label;
X_raw = raw(mask, 2:end);       % (N x 256)

[N, p] = size(X_raw);
img_dim = sqrt(p);              % 16
fprintf('  Digit "%d": N=%d, p=%d\n', digit_label, N, p);

% === 2. Feature Engineering ===
% SparsePCA.fit() centers X internally — no pre-centering needed.
% Document assumption explicitly:
% ASSUMPTION: Raw USPS pixels in [-1,1] — no additional scaling applied.

% === 3. Train / Test Split ===
% Unsupervised — full dataset used.

% === 4. Model Configuration ===
K_components = 5;      % Number of sparse PCs (matches ESL Fig 14.31 setup)
lambda    = 1e-4;      % L2 ridge: small, for uniqueness when p > N
lambda1_sweep = [0, 0.01, 0.1, 0.3, 0.6, 1.0];  % L1 values to sweep

% Also compute standard PCA baseline for direct comparison
fprintf('Computing standard PCA baseline via SVD...\n');
mu_pca = mean(X_raw, 1);
Xc     = X_raw - mu_pca;
[~, S_pca, V_pca] = svd(Xc, 'econ');
sigma_pca  = diag(S_pca);
lambda_pca = sigma_pca.^2 / (N - 1);
pve_pca    = lambda_pca / sum(lambda_pca);

% === 5. Training ===
% ---- 5a. Main fit: Sparse PCA with a selected Lambda1 ----
lambda1_main = 0.1;  % Main experiment value (moderate sparsity, visually interesting)

fprintf('\nFitting SparsePCA (Lambda1=%.4f, Lambda=%.4e, K=%d)...\n', ...
    lambda1_main, lambda, K_components);

spca = OceanStats.Unsupervised.SparsePCA();
spca.fit(X_raw, K_components, ...
    'Lambda',  lambda, ...
    'Lambda1', lambda1_main, ...
    'MaxIter', 500, ...
    'Tol',     1e-6, ...
    'Verbose', true);
spca.printSummary();

% ---- 5b. Lambda1 sweep ----
% Fit SPCA for each Lambda1 value and record sparsity and explained variance
n_sweep   = numel(lambda1_sweep);
sparsity_sweep = zeros(n_sweep, K_components);  % % zeros per component
expvar_sweep   = zeros(n_sweep, K_components);  % explained variance per component
total_var_sweep = zeros(n_sweep, 1);

fprintf('\nRunning Lambda1 sweep (%d values)...\n', n_sweep);
for s = 1:n_sweep
    spca_s = OceanStats.Unsupervised.SparsePCA();
    spca_s.fit(X_raw, K_components, ...
        'Lambda',  lambda, ...
        'Lambda1', lambda1_sweep(s), ...
        'MaxIter', 500, ...
        'Tol',     1e-6, ...
        'Verbose', false);

    sparsity_sweep(s, :) = spca_s.sparsityRatio() * 100;   % % zeros
    expvar_sweep(s, :)   = spca_s.ExplainedVariance;
    total_var_sweep(s)   = spca_s.TotalVariance;
    fprintf('  lambda1=%.3f | Sparsity(PC1)=%.1f%% | ExpVar(PC1)=%.4f\n', ...
        lambda1_sweep(s), sparsity_sweep(s,1), expvar_sweep(s,1));
end

% === 6. Evaluation ===
% Percentage of zero loadings at main lambda1
sparsity_main = spca.sparsityRatio() * 100;   % (1 x K) vector

% Reconstruction MSE (using sparse loadings)
Xc_centered = X_raw - spca.ColumnMeans;
X_recon_spca = spca.Scores * spca.Loadings' + spca.ColumnMeans;
recon_mse_spca = mean(sum((X_raw - X_recon_spca).^2, 2));

% Standard PCA reconstruction with same K (for comparison)
scores_pca_K = Xc * V_pca(:, 1:K_components);
X_recon_pca  = scores_pca_K * V_pca(:, 1:K_components)' + mu_pca;
recon_mse_pca = mean(sum((X_raw - X_recon_pca).^2, 2));

% === 7. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.getColorSequence(K_components + 2);

fig = figure('Name', 'ESL §14.5.5 Sparse PCA on Digit 3', 'Position', [50, 50, 1400, 1000]);

% ---- Row 1: Standard PCA loadings (first K_components PCs as images) ----
for k = 1:K_components
    ax = subplot(4, K_components, k);
    pc_img = reshape(V_pca(:, k), img_dim, img_dim)';
    imagesc(ax, pc_img, [-0.2, 0.2]);
    colormap(ax, gray);
    axis(ax, 'image', 'off');
    if k == 1
        ylabel(ax, 'Standard PC', 'FontSize', 8, 'Rotation', 90);
    end
    title(ax, sprintf('PC%d (%.1f%%)', k, 100*pve_pca(k)), 'FontSize', 8);
    viz.styleAxes(ax);
end

% ---- Row 2: Sparse PCA loadings (same K_components at lambda1_main) ----
for k = 1:K_components
    ax = subplot(4, K_components, K_components + k);
    sp_img = reshape(spca.Loadings(:, k), img_dim, img_dim)';
    imagesc(ax, sp_img, [-0.2, 0.2]);
    colormap(ax, gray);
    axis(ax, 'image', 'off');
    if k == 1
        ylabel(ax, sprintf('Sparse PC\n(\\lambda_1=%.2f)', lambda1_main), ...
            'FontSize', 8, 'Rotation', 90);
    end
    title(ax, sprintf('SPC%d (%.0f%% zero)', k, sparsity_main(k)), 'FontSize', 8);
    viz.styleAxes(ax);
end

% ---- Row 3 left: Sparsity sweep (% zeros vs Lambda1, per component) ----
ax_sparsity = subplot(4, 3, 7);
hold(ax_sparsity, 'on');
for k = 1:K_components
    plot(ax_sparsity, lambda1_sweep, sparsity_sweep(:, k), 'o-', ...
        'Color', colors(k,:), 'LineWidth', 1.5, 'MarkerFaceColor', colors(k,:), ...
        'DisplayName', sprintf('PC%d', k));
end
xlabel(ax_sparsity, '\lambda_1 (Lasso penalty)');
ylabel(ax_sparsity, '% Zero Loadings');
title(ax_sparsity, 'Sparsity vs \lambda_1');
legend(ax_sparsity, 'Location', 'northwest', 'FontSize', 7);
viz.styleAxes(ax_sparsity);

% ---- Row 3 center: Explained variance sweep ----
ax_expvar = subplot(4, 3, 8);
hold(ax_expvar, 'on');
% Cumulative explained variance (sum over K components) vs lambda1
cumExpVar_sweep = sum(expvar_sweep, 2);  % (n_sweep x 1)
total_var_ref   = spca.TotalVariance;
plot(ax_expvar, lambda1_sweep, 100 * cumExpVar_sweep / total_var_ref, 'o-', ...
    'Color', colors(1,:), 'LineWidth', 2, 'MarkerFaceColor', colors(1,:));
% Reference: standard PCA baseline (K components)
cumExpVar_pca_K = 100 * sum(pve_pca(1:K_components));
yline(ax_expvar, cumExpVar_pca_K, '--', sprintf('PCA (%.1f%%)', cumExpVar_pca_K), ...
    'Color', colors(2,:), 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
xlabel(ax_expvar, '\lambda_1 (Lasso penalty)');
ylabel(ax_expvar, sprintf('%% Variance Explained (K=%d PCs)', K_components));
title(ax_expvar, sprintf('Cum. Explained Var. (K=%d)', K_components));
viz.styleAxes(ax_expvar);

% ---- Row 3 right: Loading profiles — PC1 comparison (standard vs sparse) ----
ax_loading = subplot(4, 3, 9);
hold(ax_loading, 'on');
plot(ax_loading, 1:p, V_pca(:, 1), '-', 'Color', [0.85, 0.85, 0.85], ...
    'LineWidth', 1.2, 'DisplayName', 'Standard PC1');
plot(ax_loading, 1:p, spca.Loadings(:, 1), '-', 'Color', colors(2,:), ...
    'LineWidth', 1.2, 'DisplayName', sprintf('Sparse PC1 (\\lambda_1=%.2f)', lambda1_main));
yline(ax_loading, 0, '--', 'Color', [0.4, 0.4, 0.4]);
xlabel(ax_loading, 'Feature (pixel) index');
ylabel(ax_loading, 'Loading value');
title(ax_loading, 'PC1 Loading Profile: Standard vs Sparse');
legend(ax_loading, 'Location', 'best', 'FontSize', 7);
viz.styleAxes(ax_loading);

% ---- Row 4: Demo reconstruction comparison (sample digit) ----
demo_idx = 1;

% Original
ax_orig = subplot(4, 5, 16);
imagesc(ax_orig, reshape(X_raw(demo_idx,:), img_dim, img_dim)', [-1, 1]);
colormap(ax_orig, gray); axis(ax_orig, 'image', 'off');
title(ax_orig, 'Original', 'FontSize', 8);
viz.styleAxes(ax_orig);

% Standard PCA reconstruction at K_components
ax_pca_r = subplot(4, 5, 17);
imagesc(ax_pca_r, reshape(X_recon_pca(demo_idx,:), img_dim, img_dim)', [-1, 1]);
colormap(ax_pca_r, gray); axis(ax_pca_r, 'image', 'off');
title(ax_pca_r, sprintf('PCA K=%d', K_components), 'FontSize', 8);
viz.styleAxes(ax_pca_r);

% Sparse PCA reconstruction at lambda1_main
ax_spca_r = subplot(4, 5, 18);
imagesc(ax_spca_r, reshape(X_recon_spca(demo_idx,:), img_dim, img_dim)', [-1, 1]);
colormap(ax_spca_r, gray); axis(ax_spca_r, 'image', 'off');
title(ax_spca_r, sprintf('SPCA \\lambda_1=%.2f', lambda1_main), 'FontSize', 8);
viz.styleAxes(ax_spca_r);

% Residual PCA — use tighter range to amplify residual structure
ax_res_pca = subplot(4, 5, 19);
res_pca = X_raw(demo_idx,:) - X_recon_pca(demo_idx,:);
imagesc(ax_res_pca, reshape(res_pca, img_dim, img_dim)', [-0.5, 0.5]);
colormap(ax_res_pca, gray); axis(ax_res_pca, 'image', 'off');
title(ax_res_pca, 'Residual (PCA)', 'FontSize', 8);
viz.styleAxes(ax_res_pca);

% Residual SPCA
ax_res_spca = subplot(4, 5, 20);
res_spca = X_raw(demo_idx,:) - X_recon_spca(demo_idx,:);
imagesc(ax_res_spca, reshape(res_spca, img_dim, img_dim)', [-0.5, 0.5]);
colormap(ax_res_spca, gray); axis(ax_res_spca, 'image', 'off');
title(ax_res_spca, 'Residual (SPCA)', 'FontSize', 8);
viz.styleAxes(ax_res_spca);

% === 8. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY (Sparse PCA on Digit %d) ===\n', digit_label);
fprintf('  Dataset       : USPS digit "%d" (N=%d, p=%d)\n', digit_label, N, p);
fprintf('  Model         : SparsePCA (Zou et al. 2006)\n');
fprintf('  K components  : %d\n', K_components);
fprintf('  Lambda (L2)   : %.1e\n', lambda);
fprintf('  Lambda1 (L1)  : %.4f  [main experiment]\n', lambda1_main);
fprintf('  Sparsity%%     :');
for k = 1:K_components; fprintf('  PC%d=%.1f%%', k, sparsity_main(k)); end
fprintf('\n');
fprintf('  ExpVar (SPCA) : %.4f  (vs PCA: %.4f)\n', ...
    sum(spca.ExplainedVariance), sum(lambda_pca(1:K_components)));
fprintf('  Recon MSE     : PCA=%.4f  SPCA=%.4f\n', recon_mse_pca, recon_mse_spca);
fprintf('====================================================\n');

viz.export(fig, fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Results', 'exp_spca_digits.png'));
