% EXP_ICA_DIGITS  ESL §14.7 — ICA vs PCA on USPS Digit "3".
%
%   REFERENCE:  ESL §14.7, Figs 14.39 & 14.40. Hyvärinen & Oja (2000).
%   DATA:       USPS digit 3. N~658, p=256 (16x16 images, range [-1,1]).
%   MODEL:      FastICA (q=5 components) vs PCA (q=5 components).
%
%   EXPERIMENT PARTS:
%     Part A — Fig 14.39: 5x5 scatter matrix
%       Above diagonal: ICA component pairs (red, long-tailed)
%       Below diagonal: PCA component pairs (green, approx Gaussian)
%       Blue squares: extreme samples per component
%     Part B — Fig 14.40: Extreme digit images per IC
%       For each IC: digit with highest/lowest score (extreme styles)
%
%   EXPECTED:
%     PCA pairs: ellipsoidal blobs (joint Gaussian — variance only)
%     ICA pairs: cross/star shapes (independent, non-Gaussian distributions)
%
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading ===
data_path = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Data', 'zip', 'zip.train', 'zip.train');
fprintf('Loading USPS digit data...\n');
raw = load(data_path);
digit_label = 3;
mask  = raw(:, 1) == digit_label;
X_raw = raw(mask, 2:end);   % (N x 256), range [-1, 1]
[N, p] = size(X_raw);
img_dim = sqrt(p);           % 16
fprintf('  Digit "%d": N=%d, p=%d\n', digit_label, N, p);

% === 2. FastICA (q=5) ===
q = 5;
fprintf('\n--- Fitting FastICA (q=%d) ---\n', q);
ica = OceanStats.Unsupervised.ICA();
ica.fit(X_raw, q, 'Verbose', true);
ica.printSummary();

S_ica = ica.getComponents();    % (N x q) ICA scores
X_w   = ica.getWhitenedData();  % (N x q) standardized PCA scores (for lower triangle)
S_pca = X_w;                    % PCA scores are the whitened data (unit variance each)

% === 3. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');
bg  = [0.10 0.10 0.13];
N_EXTREME = 5;   % number of extreme points to highlight with blue squares

% --- FIGURE 1: 5x5 scatter matrix (Fig 14.39) ---
fig1 = figure('Name', 'ESL Fig 14.39 — ICA vs PCA Scatter Matrix', ...
    'Color', bg, 'Position', [20 40 950 920]);

clr_ica  = [1.0  0.35 0.35];   % red  — ICA
clr_pca  = [0.30 0.85 0.30];   % green — PCA
clr_ext  = [0.25 0.50 1.00];   % blue — extreme

for i = 1:q
    for j = 1:q
        ax = subplot(q, q, (i-1)*q + j);
        set(ax, 'Color', [0.07 0.07 0.09]);

        if i == j
            % Diagonal — component label
            axis(ax, 'off');
            text(0.5, 0.5, sprintf('Component\n%d', i), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Units', 'normalized', 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');

        elseif i < j
            % Above diagonal — ICA component pairs (red)
            scatter(ax, S_ica(:,j), S_ica(:,i), 5, clr_ica, 'filled', ...
                'MarkerFaceAlpha', 0.4);
            hold(ax, 'on');
            % Highlight extreme points for component j (horizontal axis)
            [~, sorted_j] = sort(S_ica(:, j));
            ext_idx = [sorted_j(1:N_EXTREME); sorted_j(end-N_EXTREME+1:end)];
            scatter(ax, S_ica(ext_idx,j), S_ica(ext_idx,i), 22, clr_ext, ...
                's', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
            hold(ax, 'off');

        else
            % Below diagonal — PCA component pairs (green)
            scatter(ax, S_pca(:,j), S_pca(:,i), 5, clr_pca, 'filled', ...
                'MarkerFaceAlpha', 0.35);
        end

        % Minimal axis styling
        box(ax, 'on');
        set(ax, 'XTickLabel', [], 'YTickLabel', [], 'TickLength', [0 0]);
        lim = 4;
        xlim(ax, [-lim lim]); ylim(ax, [-lim lim]);
        viz.styleAxes(ax);
    end
end

% Column headers (ICA top, PCA left)
annotation(fig1, 'textbox', [0.2 0.96 0.6 0.03], ...
    'String', 'ICA Components', 'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
annotation(fig1, 'textbox', [0.01 0.2 0.03 0.6], ...
    'String', 'PCA Components', 'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Rotation', 90);

sgtitle(fig1, sprintf('ESL Fig 14.39 — ICA (above diag, red) vs PCA (below diag, green) | Digit "%d"', digit_label), ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');

% --- FIGURE 2: Extreme digit images per IC (Fig 14.40) ---
fig2 = figure('Name', 'ESL Fig 14.40 — ICA Extreme Digits', ...
    'Color', bg, 'Position', [990 40 1050 380]);
tl2  = tiledlayout(fig2, 2, q+1, 'TileSpacing', 'tight', 'Padding', 'compact');

% Column 1: Mean digit (shown in both rows)
mu_img = reshape(mean(X_raw, 1), img_dim, img_dim)';
for row = 1:2
    ax = nexttile(tl2);
    imagesc(ax, mu_img, [-1, 1]);
    colormap(ax, gray); axis(ax, 'image', 'off');
    if row == 1
        title(ax, 'Mean', 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    end
    viz.styleAxes(ax);
end

% Columns 2..q+1: IC extremes
for k = 1:q
    [~, sorted_k] = sort(S_ica(:, k));
    top_idx = sorted_k(end);    % highest ICA score
    bot_idx = sorted_k(1);      % lowest ICA score

    % Row 1: most positive extreme
    ax = nexttile(tl2, 2 + (k-1));   % row 1, col k+1
    img_top = reshape(X_raw(top_idx, :), img_dim, img_dim)';
    imagesc(ax, img_top, [-1, 1]);
    colormap(ax, gray); axis(ax, 'image', 'off');
    title(ax, sprintf('ICA %d', k), 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    viz.styleAxes(ax);

    % Row 2: most negative extreme
    ax = nexttile(tl2, (q+1) + 1 + (k-1));  % row 2, col k+1
    img_bot = reshape(X_raw(bot_idx, :), img_dim, img_dim)';
    imagesc(ax, img_bot, [-1, 1]);
    colormap(ax, gray); axis(ax, 'image', 'off');
    viz.styleAxes(ax);
end

sgtitle(fig2, sprintf('ESL Fig 14.40 — ICA Extreme Digits (top=high score, bottom=low score) | Digit "%d"', digit_label), ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');

% === 4. Summary ===
fprintf('\n=== ICA vs PCA Summary ===\n');
fprintf('  q=%d components, N=%d samples, p=%d features\n', q, N, p);
for k = 1:q
    [~, sk] = sort(S_ica(:, k));
    fprintf('  IC%d: score range [%.2f, %.2f] | kurtosis=%.2f\n', ...
        k, S_ica(sk(1),k), S_ica(sk(end),k), kurtosis(S_ica(:,k)));
end
fprintf('\nNote: ICA kurtosis >> 3 => heavy-tailed (non-Gaussian), as expected.\n');
fprintf('      PCA kurtosis ~= 3 => approximately Gaussian.\n');
for k = 1:q
    fprintf('  PC%d: kurtosis=%.2f\n', k, kurtosis(S_pca(:,k)));
end
