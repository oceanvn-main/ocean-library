% EXP_NMF_OCEAN_TEXTURE  NMF vs PCA on ocean textures (DC-Offset Removed)
%
%   DATASET:        'ocean_img.jpg' (extracted as 32x32 non-overlapping patches)
%   PREPROCESSING:  PCA gets local-mean subtracted patches. NMF gets Sobel edge map.
%   MODEL:          NMF (r=25) vs PCA (r=25)
%   ASSUMPTIONS:    Image is converted to grayscale.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; close all; rng(42, 'twister');

% === 1. Data Loading & Preprocessing ===
fprintf('Loading ocean image...\n');
filepath = 'C:\Ocean Library\Matlab code practice\Data\images\ocean_img.jpg';
img = imread(filepath);
if size(img, 3) == 3
    img_gray = double(rgb2gray(img)) / 255.0;
else
    img_gray = double(img) / 255.0;
end

patch_size = 32;
[H, W] = size(img_gray);
H_crop = floor(H / patch_size) * patch_size;
W_crop = floor(W / patch_size) * patch_size;
img_crop = img_gray(1:H_crop, 1:W_crop);

% Create Edge Map for NMF
[Gmag, ~] = imgradient(img_crop, 'sobel');
% Normalize Gmag to [0, 1] across the entire image
Gmag = Gmag / max(Gmag(:));

rows = repmat(patch_size, 1, H_crop/patch_size);
cols = repmat(patch_size, 1, W_crop/patch_size);

% Extract patches
patches_raw = mat2cell(img_crop, rows, cols);
patches_edge = mat2cell(Gmag, rows, cols);

X_raw = cell2mat(cellfun(@(x) x(:)', patches_raw(:), 'UniformOutput', false));
X_edge = cell2mat(cellfun(@(x) x(:)', patches_edge(:), 'UniformOutput', false));

[N, p] = size(X_raw);
fprintf('Extracted %d patches. Each has %d pixels.\n', N, p);

% Pipeline 1: PCA with Local Mean Subtraction
local_means = mean(X_raw, 2);
X_pca_input = X_raw - local_means; % Zero-centered patches

% Pipeline 2: NMF with Edge Map
X_nmf_input = X_edge; % Already strictly non-negative

% === 2. Model Configuration & Training ===
r_comp = 25;

% --- NMF ---
fprintf('\nFitting NMF (r=%d) on Edge Map...\n', r_comp);
nmf = OceanStats.Unsupervised.NMF();
nmf.fit(X_nmf_input, r_comp, 'MaxIter', 500, 'Verbose', true);
W_nmf = nmf.W;
H_nmf = nmf.getBasisImages(); 

% --- PCA ---
fprintf('\nFitting PCA (r=%d) on Zero-Centered Patches...\n', r_comp);
pca_transformer = OceanStats.Pipeline.PCA(r_comp);
pca_fit = pca_transformer.train(X_pca_input, []);
W_pca = pca_transformer.predict(pca_fit, X_pca_input);
H_pca = pca_fit.Coeffs';

% === 3. Visualization ===
fprintf('\nRendering visualizations...\n');
viz = OceanViz.Core('Theme', 'Light');
if ~exist('Results', 'dir'), mkdir('Results'); end

color_map = gray(256);

% Figure 1: NMF Basis Patches (Edge Textures)
fig_nmf_H = figure('Name', 'NMF Basis Patches (Edge Map)', 'Position', [100, 100, 600, 600], 'Color', viz.Colors.Background);
t_nmf = tiledlayout(5, 5, 'TileSpacing', 'none', 'Padding', 'compact');
for k = 1:r_comp
    ax = nexttile(t_nmf);
    imagesc(ax, reshape(H_nmf(k, :), patch_size, patch_size));
    colormap(ax, color_map); axis(ax, 'image'); axis(ax, 'off');
    % Annotate the component index inside the top-left of the image
    text(ax, 2, 4, sprintf('H_{%d}', k), 'Color', 'yellow', 'FontSize', 8, 'FontWeight', 'bold');
end
title(t_nmf, 'NMF Basis Patches (Edge Map)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(t_nmf, 'Gallery of r=25 independent structural components (The "Lego Bricks")', 'FontSize', 12, 'Color', '#555555');

% Figure 2: PCA Basis Patches (Zero-Centered Textures)
fig_pca_H = figure('Name', 'PCA Basis Patches', 'Position', [750, 100, 600, 600], 'Color', viz.Colors.Background);
t_pca = tiledlayout(5, 5, 'TileSpacing', 'none', 'Padding', 'compact');
for k = 1:r_comp
    ax = nexttile(t_pca);
    imagesc(ax, reshape(H_pca(k, :), patch_size, patch_size));
    colormap(ax, color_map); axis(ax, 'image'); axis(ax, 'off');
    text(ax, 2, 4, sprintf('PC_{%d}', k), 'Color', 'yellow', 'FontSize', 8, 'FontWeight', 'bold');
end
title(t_pca, 'PCA Basis Patches (Zero-Centered)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(t_pca, 'Gallery of r=25 independent principal components', 'FontSize', 12, 'Color', '#555555');

% Figure 3: Coefficients Comparison (W)
fig_W = figure('Name', 'Activation Coefficients (W)', 'Position', [400, 400, 800, 400], 'Color', viz.Colors.Background);
t_W = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
num_display = min(50, N);

pos_cmap = hot(128); neg_cmap = pos_cmap(end:-1:1, [3 2 1]); div_cmap = [neg_cmap; pos_cmap];

ax1 = nexttile(t_W);
imagesc(ax1, W_nmf(1:num_display, :)); colormap(ax1, hot); colorbar(ax1);
title(ax1, sprintf('NMF Edge Coefficients (W_{1:%d, :})', num_display), 'Color', 'k');
xlabel(ax1, 'Basis Component', 'Color', 'k'); ylabel(ax1, 'Patch Index', 'Color', 'k');

ax2 = nexttile(t_W);
imagesc(ax2, W_pca(1:num_display, :)); colormap(ax2, div_cmap); colorbar(ax2);
max_w = max(abs(W_pca(1:num_display, :)), [], 'all'); clim(ax2, [-max_w, max_w]);
title(ax2, sprintf('PCA Texture Coefficients (Scores_{1:%d, :})', num_display), 'Color', 'k');
xlabel(ax2, 'Principal Component', 'Color', 'k');

% === Figure 4: Full Image Reconstruction ===
fprintf('\nStitching patches...\n');
num_rows = H_crop / patch_size;
num_cols = W_crop / patch_size;

X_nmf_recon = W_nmf * H_nmf;
% Note: PCA reconstruction requires adding the local means back AND the global mean (pca_fit.Mu)
X_pca_recon = repmat(pca_fit.Mu, N, 1) + W_pca * H_pca + local_means;

recon_nmf_patches = cell(num_rows, num_cols);
recon_pca_patches = cell(num_rows, num_cols);
patch_idx = 1;
for c = 1:num_cols
    for r = 1:num_rows
        recon_nmf_patches{r, c} = reshape(X_nmf_recon(patch_idx, :), patch_size, patch_size);
        recon_pca_patches{r, c} = reshape(X_pca_recon(patch_idx, :), patch_size, patch_size);
        patch_idx = patch_idx + 1;
    end
end
img_nmf_recon = cell2mat(recon_nmf_patches);
img_pca_recon = cell2mat(recon_pca_patches);

fig_recon = figure('Name', 'Full Image Reconstruction', 'Position', [100, 100, 1500, 500], 'Color', viz.Colors.Background);
t_recon = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(t_recon); imagesc(ax1, img_crop); colormap(ax1, gray(256)); axis(ax1, 'image'); axis(ax1, 'off');
title(ax1, 'Original Image', 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');

ax2 = nexttile(t_recon); imagesc(ax2, img_nmf_recon); colormap(ax2, gray(256)); clim(ax2, [0 max(img_nmf_recon(:))]); axis(ax2, 'image'); axis(ax2, 'off');
title(ax2, 'NMF Reconstruction (Edge Map)', 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');

ax3 = nexttile(t_recon); imagesc(ax3, img_pca_recon); colormap(ax3, gray(256)); clim(ax3, [0 1]); axis(ax3, 'image'); axis(ax3, 'off');
title(ax3, 'PCA Reconstruction (Textures + Means)', 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');

% === Figure 5: Scree Plot (Eigen Spectrum) ===
fig_scree = figure('Name', 'PCA Eigen Spectrum', 'Position', [100, 100, 600, 400], 'Color', viz.Colors.Background);
ax_scree = axes(fig_scree);
eigenvalues = pca_fit.Eigenvalues(1:r_comp);
total_variance = sum(pca_fit.Eigenvalues);
explained_variance_pct = (eigenvalues / total_variance) * 100;

semilogy(ax_scree, 1:r_comp, eigenvalues, '-o', 'LineWidth', 2, 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);
grid(ax_scree, 'on'); hold(ax_scree, 'on');
plot(ax_scree, 1, eigenvalues(1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(ax_scree, 2, eigenvalues(1), sprintf('PC1 (Texture)\n%.1f%% Variance', explained_variance_pct(1)), 'Color', 'r', 'FontWeight', 'bold');
title(ax_scree, 'PCA Scree Plot (Zero-Centered Patches)', 'Color', 'k', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax_scree, 'Principal Component Index', 'Color', 'k', 'FontSize', 12);
ylabel(ax_scree, 'Eigenvalue (Variance)', 'Color', 'k', 'FontSize', 12);

% === Figure 6: NMF Reconstruction Recipe (Practical Example) ===
fprintf('\nGenerating NMF Reconstruction Recipe...\n');
% The glasses/eye are roughly around Col=25, Row=22 -> Index = (25-1)*60 + 22 = 1462
sample_idx = 600; 
w_recipe = W_nmf(sample_idx, :);

[w_sorted, sorted_idx] = sort(w_recipe, 'descend');
active_threshold = 0.05 * w_sorted(1);
num_active = sum(w_sorted > active_threshold);
num_active = min(max(num_active, 3), 6); 

fig_recipe = figure('Name', sprintf('NMF Edge Recipe (Patch %d)', sample_idx), 'Position', [100, 200, 1500, 300], 'Color', viz.Colors.Background);
t_recipe = tiledlayout(1, num_active + 3, 'TileSpacing', 'compact', 'Padding', 'compact');
gray_inv = 1 - gray(256); 

% 0. Context Image (Show where the patch came from)
ax_context = nexttile(t_recipe);
imagesc(ax_context, img_crop); colormap(ax_context, gray(256)); axis(ax_context, 'image'); axis(ax_context, 'off');
hold(ax_context, 'on');

% Calculate patch coordinates for bounding box
num_rows = H_crop / patch_size;
p_col = ceil(sample_idx / num_rows);
p_row = mod(sample_idx - 1, num_rows) + 1;
x_rect = (p_col - 1) * patch_size + 1;
y_rect = (p_row - 1) * patch_size + 1;
rectangle(ax_context, 'Position', [x_rect, y_rect, patch_size, patch_size], 'EdgeColor', 'r', 'LineWidth', 2);
title(ax_context, sprintf('Patch Location\n(Row %d, Col %d)', p_row, p_col), 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');

% 1. Original Edge Patch
ax_orig = nexttile(t_recipe);
img_orig = reshape(X_nmf_input(sample_idx, :), patch_size, patch_size);
imagesc(ax_orig, img_orig);
colormap(ax_orig, gray_inv);
axis(ax_orig, 'image'); axis(ax_orig, 'off');
title(ax_orig, 'Original Edge Patch', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');

recon_img = zeros(patch_size, patch_size);
max_intensity = max(img_orig(:)); 
if max_intensity == 0; max_intensity = 1; end % Safety fallback

for k = 1:num_active
    comp_idx = sorted_idx(k);
    comp_weight = w_recipe(comp_idx);
    comp_img = reshape(H_nmf(comp_idx, :), patch_size, patch_size);
    
    % Add to reconstruction
    recon_img = recon_img + comp_weight * comp_img;
    
    % Plot the weighted component
    ax_comp = nexttile(t_recipe);
    imagesc(ax_comp, comp_weight * comp_img);
    colormap(ax_comp, gray_inv);
    clim(ax_comp, [0 max_intensity]); % Lock color limits so brighter = higher weight
    axis(ax_comp, 'image'); axis(ax_comp, 'off');
    
    if k == 1
        prefix = ' \approx ';
    else
        prefix = ' + ';
    end
    title(ax_comp, sprintf('%s W_{%d} \\times H_{%d}\n(Weight: %.3f)', prefix, comp_idx, comp_idx, comp_weight), 'Color', 'k', 'FontSize', 10);
end

% Reconstructed Image
ax_recon = nexttile(t_recipe);
imagesc(ax_recon, recon_img);
colormap(ax_recon, gray_inv);
clim(ax_recon, [0 max_intensity]);
axis(ax_recon, 'image'); axis(ax_recon, 'off');
title(ax_recon, ' = Reconstructed', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');

% Suppress axes toolbars before saving to avoid export warnings
figs = [fig_nmf_H, fig_pca_H, fig_W, fig_recon, fig_scree, fig_recipe];
for f = figs
    set(findall(f, 'type', 'axes'), 'Toolbar', []);
end

% Save all figures
saveas(fig_nmf_H, 'Results/ocean_texture_nmf_basis.png');
saveas(fig_pca_H, 'Results/ocean_texture_pca_basis.png');
saveas(fig_W, 'Results/ocean_texture_weights.png');
saveas(fig_recon, 'Results/ocean_texture_recon.png');
saveas(fig_scree, 'Results/ocean_texture_scree.png');
saveas(fig_recipe, 'Results/ocean_texture_recipe.png');

fprintf('Done!\n');
