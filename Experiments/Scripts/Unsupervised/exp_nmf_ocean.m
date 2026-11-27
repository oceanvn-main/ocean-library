% EXP_NMF_OCEAN  NMF vs PCA on ocean image patches (Texture/Dictionary Learning).
%
%   DATASET:        'ocean_img.jpg' (extracted as 32x32 non-overlapping patches)
%   MODEL:          NMF (r=25) vs PCA (r=24 + mean)
%   ASSUMPTIONS:    Image is converted to grayscale and normalized to [0,1].
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; close all; rng(42, 'twister');

% === 1. Data Loading & Preprocessing ===
fprintf('Loading ocean image...\n');
filepath = 'C:\Ocean Library\Matlab code practice\Data\images\ocean_img.jpg';
img = imread(filepath);

if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Normalize to [0, 1]
img_gray = double(img_gray) / 255.0;

% Extract 32x32 non-overlapping patches
patch_size = 32;
[H, W] = size(img_gray);
H_crop = floor(H / patch_size) * patch_size;
W_crop = floor(W / patch_size) * patch_size;
img_crop = img_gray(1:H_crop, 1:W_crop);

fprintf('Cropped image to %dx%d to extract exact %dx%d patches.\n', H_crop, W_crop, patch_size, patch_size);

rows = repmat(patch_size, 1, H_crop/patch_size);
cols = repmat(patch_size, 1, W_crop/patch_size);
patches = mat2cell(img_crop, rows, cols);

% Convert to N x p dataset (samples x pixels)
X_nn = cell2mat(cellfun(@(x) x(:)', patches(:), 'UniformOutput', false));
[N, p] = size(X_nn);
fprintf('Extracted %d patches. Each has %d pixels.\n', N, p);

% === 2. Model Configuration & Training ===

% --- NMF ---
r_nmf = 25;
fprintf('\nFitting NMF (r=%d)...\n', r_nmf);
nmf = OceanStats.Unsupervised.NMF();
nmf.fit(X_nn, r_nmf, 'MaxIter', 500, 'Verbose', true);
W_nmf = nmf.W;
H_nmf = nmf.getBasisImages(); % 25 x 1024

% --- PCA ---
r_pca = 24;
fprintf('\nFitting PCA (r=%d)...\n', r_pca);
pca_transformer = OceanStats.Pipeline.PCA(r_pca);
pca_fit = pca_transformer.train(X_nn, []);
score = pca_transformer.predict(pca_fit, X_nn);

% The basis for PCA is 'Coeffs' (p x r). We transpose to get (r x p)
H_pca_components = pca_fit.Coeffs';
W_pca = score;

% As per exercise format, we display the mean image + 24 components
H_pca = [pca_fit.Mu; H_pca_components]; % 25 x 1024

% === 3. Visualization ===
fprintf('\nRendering visualizations...\n');
viz = OceanViz.Core('Theme', 'Light');

if ~exist('Results', 'dir')
    mkdir('Results');
end

% We don't use inverted gray here because it's a natural image. We just use standard gray or bone.
color_map = gray(256);

% Figure 1: NMF Basis Patches (H)
fig_nmf_H = figure('Name', 'NMF Basis Patches (H)', 'Position', [100, 100, 600, 600], 'Color', viz.Colors.Background);
t_nmf = tiledlayout(5, 5, 'TileSpacing', 'none', 'Padding', 'compact');

for k = 1:r_nmf
    ax = nexttile(t_nmf);
    % Reshape the 1024-pixel vector back into a 32x32 patch
    % Note: x(:) goes column-wise, so we reshape(..., 32, 32)
    img_patch = reshape(H_nmf(k, :), patch_size, patch_size);
    imagesc(ax, img_patch);
    colormap(ax, color_map);
    axis(ax, 'image');
    axis(ax, 'off');
    text(ax, 2, 4, sprintf('H_{%d}', k), 'Color', 'yellow', 'FontSize', 8, 'FontWeight', 'bold');
end
title(t_nmf, 'NMF Basis Patches (H)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(t_nmf, 'Gallery of r=25 independent structural components (Lego Bricks)', 'FontSize', 12, 'Color', '#555555');

% Figure 2: PCA Basis Patches (H)
fig_pca_H = figure('Name', 'PCA Basis Patches (Mean + 24 PCs)', 'Position', [750, 100, 600, 600], 'Color', viz.Colors.Background);
t_pca = tiledlayout(5, 5, 'TileSpacing', 'none', 'Padding', 'compact');

for k = 1:25
    ax = nexttile(t_pca);
    img_patch = reshape(H_pca(k, :), patch_size, patch_size);
    
    imagesc(ax, img_patch);
    colormap(ax, color_map);
    axis(ax, 'image');
    axis(ax, 'off');
    
    if k == 1
        % Add subtle text for mean
        text(ax, 2, 4, 'Mean', 'Color', 'yellow', 'FontSize', 8, 'FontWeight', 'bold');
    else
        text(ax, 2, 4, sprintf('PC_{%d}', k-1), 'Color', 'yellow', 'FontSize', 8, 'FontWeight', 'bold');
    end
end
title(t_pca, 'PCA Basis Patches (Mean + 24 PCs)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(t_pca, 'Gallery of Mean Patch + r=24 independent principal components', 'FontSize', 12, 'Color', '#555555');

% Figure 3: Coefficients Comparison (W)
% Display the activation coefficients for a subset of patches (e.g. 50 patches)
fig_W = figure('Name', 'Activation Coefficients (W)', 'Position', [400, 400, 800, 400], 'Color', viz.Colors.Background);
t_W = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

num_display = min(50, N);

% Create custom diverging colormap (Cyan -> Blue -> Black -> Red -> Yellow -> White)
pos_cmap = hot(128);
neg_cmap = pos_cmap(end:-1:1, [3 2 1]); % Swap RGB and flip
div_cmap = [neg_cmap; pos_cmap];

% NMF W (strictly positive, sparse)
ax1 = nexttile(t_W);
imagesc(ax1, W_nmf(1:num_display, :));
colormap(ax1, hot); % 0 is black
colorbar(ax1);
title(ax1, sprintf('NMF Coefficients (W_{1:%d, :})', num_display), 'Color', 'k');
xlabel(ax1, 'Basis Component', 'Color', 'k');
ylabel(ax1, 'Patch Index', 'Color', 'k');

% PCA W (mixed signs, dense)
ax2 = nexttile(t_W);
imagesc(ax2, W_pca(1:num_display, :));
colormap(ax2, div_cmap); % 0 is exactly in the middle (black)
colorbar(ax2);
% Center the color limits around 0 so black perfectly matches 0
max_w = max(abs(W_pca(1:num_display, :)), [], 'all');
clim(ax2, [-max_w, max_w]);
title(ax2, sprintf('PCA Coefficients (Scores_{1:%d, :})', num_display), 'Color', 'k');
xlabel(ax2, 'Principal Component', 'Color', 'k');

% We will save all figures at the very end.

% === Figure 4: Full Image Reconstruction ===
fprintf('\nStitching patches to reconstruct full images...\n');
num_rows = H_crop / patch_size;
num_cols = W_crop / patch_size;

% 1. NMF Reconstruction
X_nmf_recon = W_nmf * H_nmf;
recon_nmf_patches = cell(num_rows, num_cols);

% 2. PCA Reconstruction
X_pca_recon = repmat(pca_fit.Mu, N, 1) + W_pca * H_pca_components;
recon_pca_patches = cell(num_rows, num_cols);

patch_idx = 1;
for c = 1:num_cols
    for r = 1:num_rows
        % NMF
        patch_nmf = X_nmf_recon(patch_idx, :);
        recon_nmf_patches{r, c} = reshape(patch_nmf, patch_size, patch_size);
        
        % PCA
        patch_pca = X_pca_recon(patch_idx, :);
        recon_pca_patches{r, c} = reshape(patch_pca, patch_size, patch_size);
        
        patch_idx = patch_idx + 1;
    end
end

img_nmf_recon = cell2mat(recon_nmf_patches);
img_pca_recon = cell2mat(recon_pca_patches);

fig_recon = figure('Name', 'Full Image Reconstruction', 'Position', [100, 100, 1500, 500], 'Color', viz.Colors.Background);
t_recon = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot Original
ax1 = nexttile(t_recon);
imagesc(ax1, img_crop);
colormap(ax1, gray(256));
axis(ax1, 'image'); axis(ax1, 'off');
title(ax1, 'Original Image', 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');

% Plot NMF
ax2 = nexttile(t_recon);
imagesc(ax2, img_nmf_recon);
colormap(ax2, gray(256));
clim(ax2, [0 1]);
axis(ax2, 'image'); axis(ax2, 'off');
title(ax2, sprintf('NMF Reconstruction (r=%d)', r_nmf), 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');

% Plot PCA
ax3 = nexttile(t_recon);
imagesc(ax3, img_pca_recon);
colormap(ax3, gray(256));
clim(ax3, [0 1]);
axis(ax3, 'image'); axis(ax3, 'off');
title(ax3, sprintf('PCA Reconstruction (r=%d + Mean)', r_pca), 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig_recon, 'Results/ocean_reconstruction.png');

% === Figure 5: Scree Plot (Eigen Spectrum) ===
fprintf('\nGenerating PCA Scree Plot...\n');
fig_scree = figure('Name', 'PCA Eigen Spectrum', 'Position', [100, 100, 600, 400], 'Color', viz.Colors.Background);
ax_scree = axes(fig_scree);

% The eigenvalues are sorted in descending order
eigenvalues = pca_fit.Eigenvalues(1:r_pca);
total_variance = sum(pca_fit.Eigenvalues);
explained_variance_pct = (eigenvalues / total_variance) * 100;

% Plot log of variance to clearly see the drop-off
semilogy(ax_scree, 1:r_pca, eigenvalues, '-o', 'LineWidth', 2, 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);
grid(ax_scree, 'on');

% Highlight PC1 vs the rest
hold(ax_scree, 'on');
plot(ax_scree, 1, eigenvalues(1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(ax_scree, 2, eigenvalues(1), sprintf('PC1 (DC Brightness)\n%.1f%% Variance', explained_variance_pct(1)), 'Color', 'r', 'FontWeight', 'bold');

title(ax_scree, 'PCA Scree Plot (Log Scale)', 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax_scree, 'Principal Component Index', 'Color', 'k', 'FontSize', 12);
ylabel(ax_scree, 'Eigenvalue (Variance)', 'Color', 'k', 'FontSize', 12);

% Suppress axes toolbars before saving to avoid export warnings
figs = [fig_nmf_H, fig_pca_H, fig_W, fig_recon, fig_scree];
for f = figs
    set(findall(f, 'type', 'axes'), 'Toolbar', []);
end

% Save figures
saveas(fig_nmf_H, 'Results/ocean_nmf_basis.png');
saveas(fig_pca_H, 'Results/ocean_pca_basis.png');
saveas(fig_W, 'Results/ocean_nmf_pca_weights.png');
saveas(fig_recon, 'Results/ocean_reconstruction.png');
saveas(fig_scree, 'Results/ocean_scree_plot.png');
fprintf('Done! Saved to Results/ directory.\n');
