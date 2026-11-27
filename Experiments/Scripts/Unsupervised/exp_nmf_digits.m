% EXP_NMF_DIGITS  ESL Exercise 14.25: NMF vs PCA on digits data.
%
%   DATASET:        'zip.train' (subset: label '2')
%   MODEL:          NMF (r=25) vs PCA (r=24 + mean)
%   ASSUMPTIONS:    NMF requires non-negative inputs, so X is shifted to [0,1].
%                   For fair visual comparison, the PCA basis is rendered 
%                   with the mean image as the first component.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; close all; rng(42, 'twister');

% === 1. Data Loading & Preprocessing ===
fprintf('Loading digits database...\n');
filepath = 'C:\Ocean Library\Matlab code practice\Data\zip\zip.train\zip.train';
data = load(filepath, '-ascii');
labels = data(:, 1);
X_raw = data(:, 2:end);

% Filter for '2's
mask_twos = (labels == 2);
X_twos = X_raw(mask_twos, :);
[N, p] = size(X_twos);
fprintf('Found %d examples of digit "2". Each has %d pixels.\n', N, p);

% Preprocessing for NMF: Shift to [0, 1]
X_nn = (X_twos + 1) / 2;

% === 2. Model Configuration & Training ===

% --- NMF ---
r_nmf = 25;
fprintf('\nFitting NMF (r=%d)...\n', r_nmf);
nmf = OceanStats.Unsupervised.NMF();
nmf.fit(X_nn, r_nmf, 'MaxIter', 500, 'Verbose', true);
W_nmf = nmf.W;
H_nmf = nmf.getBasisImages(); % 25 x 256

% --- PCA ---
r_pca = 24;
fprintf('\nFitting PCA (r=%d)...\n', r_pca);
pca_transformer = OceanStats.Pipeline.PCA(r_pca);
pca_fit = pca_transformer.train(X_nn, []);
score = pca_transformer.predict(pca_fit, X_nn);

% The basis for PCA is 'Coeffs' (p x r). We transpose to get (r x p)
H_pca_components = pca_fit.Coeffs';
W_pca = score;

% As per exercise, we display the mean image + 24 components
H_pca = [pca_fit.Mu; H_pca_components]; % 25 x 256

% === 3. Visualization ===
fprintf('\nRendering visualizations...\n');
viz = OceanViz.Core('Theme', 'Light');

if ~exist('Results', 'dir')
    mkdir('Results');
end

% Figure 1: NMF Basis Images (H)
fig_nmf_H = figure('Name', 'NMF Basis Images (H)', 'Position', [100, 100, 600, 600], 'Color', viz.Colors.Background);
t_nmf = tiledlayout(5, 5, 'TileSpacing', 'none', 'Padding', 'compact');
gray_inv = 1 - gray(256); % White background, black ink

for k = 1:r_nmf
    ax = nexttile(t_nmf);
    img = reshape(H_nmf(k, :), 16, 16)';
    imagesc(ax, img);
    colormap(ax, gray_inv);
    axis(ax, 'image');
    axis(ax, 'off');
    text(ax, 1, 2, sprintf('H_{%d}', k), 'Color', 'blue', 'FontSize', 8, 'FontWeight', 'bold');
end
title(t_nmf, 'NMF Basis Images (H)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(t_nmf, 'Gallery of r=25 independent structural components (Lego Bricks)', 'FontSize', 12, 'Color', '#555555');

% Figure 2: PCA Basis Images (H)
fig_pca_H = figure('Name', 'PCA Basis Images (Mean + 24 PCs)', 'Position', [750, 100, 600, 600], 'Color', viz.Colors.Background);
t_pca = tiledlayout(5, 5, 'TileSpacing', 'none', 'Padding', 'compact');

for k = 1:25
    ax = nexttile(t_pca);
    img = reshape(H_pca(k, :), 16, 16)';
    
    imagesc(ax, img);
    colormap(ax, gray_inv);
    axis(ax, 'image');
    axis(ax, 'off');
    
    if k == 1
        % Add subtle text for mean
        text(ax, 1, 2, 'Mean', 'Color', 'red', 'FontSize', 8, 'FontWeight', 'bold');
    else
        text(ax, 1, 2, sprintf('PC_{%d}', k-1), 'Color', 'blue', 'FontSize', 8, 'FontWeight', 'bold');
    end
end
title(t_pca, 'PCA Basis Images (Mean + 24 PCs)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(t_pca, 'Gallery of Mean Image + r=24 independent principal components', 'FontSize', 12, 'Color', '#555555');

% Figure 3: Coefficients Comparison (W)
% Display the activation coefficients for the first 50 digits.
fig_W = figure('Name', 'Activation Coefficients (W)', 'Position', [400, 400, 800, 400], 'Color', viz.Colors.Background);
t_W = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Create custom diverging colormap (Cyan -> Blue -> Black -> Red -> Yellow -> White)
pos_cmap = hot(128);
neg_cmap = pos_cmap(end:-1:1, [3 2 1]); % Swap RGB and flip
div_cmap = [neg_cmap; pos_cmap];

% NMF W (strictly positive, sparse)
ax1 = nexttile(t_W);
imagesc(ax1, W_nmf(1:50, :));
colormap(ax1, hot); % 0 is black
colorbar(ax1);
title(ax1, 'NMF Coefficients (W_{1:50, :})', 'Color', 'k');
xlabel(ax1, 'Basis Component', 'Color', 'k');
ylabel(ax1, 'Sample Index', 'Color', 'k');

% PCA W (mixed signs, dense)
ax2 = nexttile(t_W);
imagesc(ax2, W_pca(1:50, :));
colormap(ax2, div_cmap); % 0 is exactly in the middle (black)
colorbar(ax2);
% Center the color limits around 0 so black perfectly matches 0
max_w = max(abs(W_pca(1:50, :)), [], 'all');
clim(ax2, [-max_w, max_w]);
title(ax2, 'PCA Coefficients (Scores_{1:50, :})', 'Color', 'k');
xlabel(ax2, 'Principal Component', 'Color', 'k');

% We will save all figures at the very end.

% Figure 4: Reconstruction Recipe (Practical Example)
% Pick the first sample to demonstrate how NMF builds the image
sample_idx = 1;
w_recipe = W_nmf(sample_idx, :);

% Sort components by how 'active' they are for this specific digit
[w_sorted, sorted_idx] = sort(w_recipe, 'descend');

% Find how many components are significantly active (e.g., > 10% of max)
active_threshold = 0.1 * w_sorted(1);
num_active = sum(w_sorted > active_threshold);
num_active = min(max(num_active, 3), 6); % Keep between 3 and 6 for nice plotting

fig_recipe = figure('Name', sprintf('NMF Reconstruction Recipe (Sample %d)', sample_idx), 'Position', [200, 200, 1200, 300], 'Color', viz.Colors.Background);
t_recipe = tiledlayout(1, num_active + 2, 'TileSpacing', 'compact', 'Padding', 'compact');
gray_inv = 1 - gray(256); % White background, black ink

% 1. Original Image
ax_orig = nexttile(t_recipe);
img_orig = reshape(X_nn(sample_idx, :), 16, 16)';
imagesc(ax_orig, img_orig);
colormap(ax_orig, gray_inv);
axis(ax_orig, 'image');
axis(ax_orig, 'off');
title(ax_orig, 'Original Image', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');

% Active Components
recon_img = zeros(16, 16);
max_intensity = max(img_orig(:)); % For consistent color scaling

for k = 1:num_active
    comp_idx = sorted_idx(k);
    comp_weight = w_recipe(comp_idx);
    comp_img = reshape(H_nmf(comp_idx, :), 16, 16)';
    
    % Add to reconstruction
    recon_img = recon_img + comp_weight * comp_img;
    
    % Plot the weighted component
    ax_comp = nexttile(t_recipe);
    imagesc(ax_comp, comp_weight * comp_img);
    colormap(ax_comp, gray_inv);
    clim(ax_comp, [0 max_intensity]); % Lock color limits so brighter = higher weight
    axis(ax_comp, 'image');
    axis(ax_comp, 'off');
    
    if k == 1
        prefix = ' \approx ';
    else
        prefix = ' + ';
    end
    title(ax_comp, sprintf('%s W_{%d} \\times H_{%d}\n(Weight: %.2f)', prefix, comp_idx, comp_idx, comp_weight), 'Color', 'k', 'FontSize', 10);
end

% Reconstructed Image
ax_recon = nexttile(t_recipe);
imagesc(ax_recon, recon_img);
colormap(ax_recon, gray_inv);
clim(ax_recon, [0 max_intensity]);
axis(ax_recon, 'image');
axis(ax_recon, 'off');
title(ax_recon, ' = Reconstructed', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');

% Suppress axes toolbars before saving to avoid export warnings
figs = [fig_nmf_H, fig_pca_H, fig_W, fig_recipe];
for f = figs
    set(findall(f, 'type', 'axes'), 'Toolbar', []);
end

% Save figures
saveas(fig_nmf_H, 'Results/nmf_basis.png');
saveas(fig_pca_H, 'Results/pca_basis.png');
saveas(fig_W, 'Results/nmf_pca_weights.png');
saveas(fig_recipe, 'Results/nmf_recipe.png');
fprintf('Done! Saved to Results/ directory.\n');
