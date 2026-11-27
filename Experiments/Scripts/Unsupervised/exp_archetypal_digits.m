% EXP_ARCHETYPAL_DIGITS  ESL §14.6.1 — Archetypal Analysis.
%
%   REFERENCE:  ESL §14.6.1, Cutler & Breiman (1994). Figs 14.35 & 14.36.
%   PART A:     2D synthetic (bivariate Gaussian, N=50) — ESL Fig 14.35.
%               Compares AA archetypes (convex hull) vs K-Means centroids (interior).
%   PART B:     USPS digit '3' (N~658, p=256) — ESL Fig 14.36.
%               Runs AA with r=2, 3, 4 and displays archetype images.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% =========================================================================
% PART A — Synthetic 2D data: AA vs K-Means (ESL Fig 14.35)
% =========================================================================
fprintf('=== PART A: 2D Synthetic Data (ESL Fig 14.35) ===\n');

N2d   = 50;
mu2d  = [2 2];
Sig2d = [1.5 0.8; 0.8 1.2];
X2d   = mvnrnd(mu2d, Sig2d, N2d);    % (50 x 2)

r_vals = [2, 4, 8];                   % archetype counts to try

% Fit AA and K-Means for each r
aa_results = cell(1, 3);
km_results = cell(1, 3);
for s = 1:3
    r_s = r_vals(s);
    fprintf('  Fitting AA  r=%d...\n', r_s);
    aa_s = OceanStats.Unsupervised.ArchetypalAnalysis();
    aa_s.fit(X2d, r_s, 'MaxIter', 200, 'NumRestarts', 5, 'Tol', 1e-5);
    aa_results{s} = aa_s;

    fprintf('  Fitting VQ  K=%d...\n', r_s);
    [km_idx, km_c] = kmeans(X2d, r_s, 'Replicates', 5, 'Display', 'off');
    km_results{s}  = struct('idx', km_idx, 'centroids', km_c);
end

% Plot Part A — 2x3 grid: Row 1 = AA, Row 2 = K-Means
viz = OceanViz.Core('Theme', 'Dark');
fig_a = figure('Name', 'ESL §14.35 — AA vs K-Means (2D)', ...
    'Color', [0.10 0.10 0.13], 'Position', [30 30 1100 600]);
tl_a  = tiledlayout(fig_a, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Convex hull of 2D data (for visual reference)
hull_idx = convhull(X2d(:,1), X2d(:,2));

for s = 1:3
    r_s = r_vals(s);

    % Row 1: AA
    ax = nexttile(tl_a);
    fill(ax, X2d(hull_idx,1), X2d(hull_idx,2), [0.2 0.2 0.3], ...
        'FaceAlpha', 0.2, 'EdgeColor', [0.5 0.5 0.7], 'LineWidth', 1);
    hold(ax, 'on');
    scatter(ax, X2d(:,1), X2d(:,2), 30, 'o', ...
        'MarkerEdgeColor', [0.7 0.7 0.9], 'MarkerFaceColor', 'none', 'LineWidth', 1);
    H_aa = aa_results{s}.getArchetypes() + aa_results{s}.ColumnMeans;
    scatter(ax, H_aa(:,1), H_aa(:,2), 120, 'filled', ...
        'MarkerFaceColor', [1.0 0.2 0.2], 'MarkerEdgeColor', 'w', 'LineWidth', 1.2);
    hold(ax, 'off');
    title(ax, sprintf('AA: %d Prototypes', r_s), 'Color', 'w', 'FontSize', 9);
    viz.styleAxes(ax);

    % Row 2: K-Means
    ax = nexttile(tl_a, s + 3);
    fill(ax, X2d(hull_idx,1), X2d(hull_idx,2), [0.2 0.2 0.3], ...
        'FaceAlpha', 0.2, 'EdgeColor', [0.5 0.5 0.7], 'LineWidth', 1);
    hold(ax, 'on');
    scatter(ax, X2d(:,1), X2d(:,2), 30, 'o', ...
        'MarkerEdgeColor', [0.7 0.7 0.9], 'MarkerFaceColor', 'none', 'LineWidth', 1);
    scatter(ax, km_results{s}.centroids(:,1), km_results{s}.centroids(:,2), 120, 'filled', ...
        'MarkerFaceColor', [0.2 0.5 1.0], 'MarkerEdgeColor', 'w', 'LineWidth', 1.2);
    hold(ax, 'off');
    title(ax, sprintf('K-Means: %d Centroids', r_s), 'Color', 'w', 'FontSize', 9);
    viz.styleAxes(ax);
end

sgtitle(fig_a, 'ESL §14.35 — Archetypal Analysis (red) vs K-Means (blue)', ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');

% =========================================================================
% PART B — USPS Digit '3': AA with r=2, 3, 4 (ESL Fig 14.36)
% =========================================================================
fprintf('\n=== PART B: USPS Digit 3 Archetypes (ESL Fig 14.36) ===\n');

data_path = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Data', 'zip', 'zip.train', 'zip.train');
raw       = load(data_path);
mask      = raw(:, 1) == 3;
X_raw     = raw(mask, 2:end);          % (N x 256), range [-1, 1]
[N, p]    = size(X_raw);
img_dim   = sqrt(p);                   % 16
fprintf('  Digit "3": N=%d, p=%d\n', N, p);

% ASSUMPTION: use raw pixel values in [-1,1]. AA centers internally.
r_list = [2, 3, 4];

aa_digit = cell(1, 3);
for s = 1:3
    r_s = r_list(s);
    fprintf('\n  Fitting AA r=%d on digit "3"...\n', r_s);
    aa_s = OceanStats.Unsupervised.ArchetypalAnalysis();
    aa_s.fit(X_raw, r_s, 'MaxIter', 200, 'NumRestarts', 3, ...
        'Tol', 1e-4, 'Verbose', true);
    aa_s.printSummary();
    aa_digit{s} = aa_s;
end

% Plot Part B — archetype images (matches ESL Fig 14.36 layout)
% Rows: r=2 | r=3 | r=4. Cols: one per archetype.
max_r = max(r_list);

fig_b = figure('Name', 'ESL §14.36 — Archetypal Analysis on Digit 3', ...
    'Color', [0.10 0.10 0.13], 'Position', [30 680 900 500]);
tl_b  = tiledlayout(fig_b, 3, max_r, 'TileSpacing', 'compact', 'Padding', 'compact');

for s = 1:3
    r_s    = r_list(s);
    H_arch = aa_digit{s}.getArchetypes();       % (r x p) centered archetypes
    mu_s   = aa_digit{s}.ColumnMeans;           % (1 x p) training means
    H_raw  = H_arch + mu_s;                     % shift back to pixel space

    % Sort archetypes by brightness (left=darkest, right=brightest)
    brightness = mean(H_raw, 2);
    [~, sort_idx] = sort(brightness, 'ascend');
    H_sorted = H_raw(sort_idx, :);

    for k = 1:r_s
        ax = nexttile(tl_b, (s-1)*max_r + k);
        img_k = reshape(H_sorted(k, :), img_dim, img_dim)';
        imagesc(ax, img_k, [-1, 1]);
        colormap(ax, hot); axis(ax, 'image', 'off');
        if k == 1
            ylabel(ax, sprintf('r=%d', r_s), 'Color', 'w', 'FontSize', 9);
        end
        title(ax, sprintf('A%d', k), 'Color', 'w', 'FontSize', 8);
        viz.styleAxes(ax);
    end

    % Pad remaining tiles in row with blank axes
    for k = (r_s+1):max_r
        ax = nexttile(tl_b, (s-1)*max_r + k);
        axis(ax, 'off');
        viz.styleAxes(ax);
    end
end

sgtitle(fig_b, 'ESL §14.36 — Archetypes of Digit "3" (r=2, 3, 4)', ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');

% =========================================================================
% Summary
% =========================================================================
fprintf('\n=== Summary ===\n');
fprintf('Part A: Verify AA prototypes are near edges (convex hull), K-Means in interior.\n');
for s = 1:3
    r_s = r_list(s);
    err = aa_digit{s}.ReconError(end);
    tot = aa_digit{s}.TotalEnergy;
    fprintf('Part B: r=%d | Recon error: %.4f (%.1f%% unexplained)\n', ...
        r_s, err, 100*err/tot);
end
