%% exp_nci_som_kmeans_pca.m
%  ============================================================
%  ESL Exercise 14.6 — K-Means & SOM on NCI Microarray Data
%  WITH PCA PREPROCESSING (Built-in functions)
%  ============================================================
%
%  MOTIVATION:
%    Raw data (64 x 6830) suffers from the curse of dimensionality:
%      - All pairwise distances become approximately equal
%      - SOM suffers severe dead-node collapse
%      - Neighbourhood convergence runs in the wrong direction
%    PCA reduces 6830 → n_pcs dimensions, retaining most variance
%    while making the distance geometry tractable.
%
%  PIPELINE:
%    Phase 1 : Load & standardise  (same as before)
%    Phase 1b: PCA reduction       [NEW] pca() built-in
%    Phase 2 : K-Means sweep       kmeans()
%    Phase 3 : SOM sweep           selforgmap()
%    Phase 4 : Neighbourhood convergence experiment
%    Phase 5 : Visualisation (4 figures: scree + WCSS/MQE + heatmap + convergence)
%
%  Ref: Hastie, Tibshirani, Friedman (2009), ESL Ex. 14.6, Section 14.4
%  ============================================================

clear; clc; close all;
rng(42, 'twister');

% ============================================================
%  CONFIGURATION
% ============================================================
SCRIPT_DIR = fileparts(mfilename('fullpath'));
ROOT_DIR   = fullfile(SCRIPT_DIR, '..', '..', '..');
DATA_PATH  = fullfile(ROOT_DIR, 'Data', 'NCI', 'nci.data.csv');
LABEL_PATH = fullfile(ROOT_DIR, 'Data', 'NCI', 'nci.label.txt');

K_VALUES   = [2, 5, 10, 20];
SOM_GRIDS  = {[1,2], [1,5], [2,5], [4,5]};   % Q1*Q2 = K

KM_RESTARTS   = 20;
SOM_EPOCHS    = 500;
INIT_NEIGHBOR = 3;      % default SOM sweep neighbourhood

% PCA configuration
VAR_THRESHOLD = 0.90;   % keep PCs until cumulative variance >= 90%
MAX_PCS       = 50;     % hard cap (max rank is N-1 = 63)

% Convergence experiment
CONV_K         = 10;
CONV_GRID      = [2, 5];
CONV_INIT_NBRS = [6, 4, 3, 2, 1];   % integer only (selforgmap requirement)

TOP_GENES_VIZ = 150;   % genes for heatmap (show in original space)

fprintf('=============================================================\n');
fprintf('  ESL Ex 14.6 — NCI60  [PCA + BUILTIN VERSION]\n');
fprintf('  K sweep: [%s]   PCA threshold: %.0f%%\n', num2str(K_VALUES), VAR_THRESHOLD*100);
fprintf('=============================================================\n\n');

% ============================================================
%  PHASE 1: DATA LOADING & PREPROCESSING
% ============================================================
fprintf('>>> PHASE 1: DATA LOADING & PREPROCESSING <<<\n');

T     = readtable(DATA_PATH, 'ReadRowNames', true);
X_raw = T{:,:}';             % (64 x 6830)
[N, p] = size(X_raw);

fid   = fopen(LABEL_PATH, 'r');
raw   = textscan(fid, '%s', 'Delimiter', '\n', 'WhiteSpace', '');
fclose(fid);
cancer_labels = strtrim(raw{1});
cancer_labels = cancer_labels(~cellfun(@isempty, cancer_labels));
cancer_labels = cancer_labels(1:N);

% Gene-wise standardisation
col_mean = mean(X_raw, 1);
col_std  = std(X_raw, 0, 1);
col_std(col_std == 0) = 1;
X = (X_raw - col_mean) ./ col_std;   % (64 x 6830)

[uniq_cancers, ~, cancer_idx] = unique(cancer_labels, 'stable');
n_types = numel(uniq_cancers);
fprintf('  Shape: %d cell lines x %d genes   Cancer types: %d\n\n', N, p, n_types);

% ============================================================
%  PHASE 1b: PCA REDUCTION
%
%  pca(X) — built-in (Statistics & Machine Learning Toolbox)
%    coeff    (p x p)   — loading matrix (PC directions)
%    score    (N x p)   — PC scores (projections)
%    latent   (p x 1)   — eigenvalues (variance per PC)
%    explained(p x 1)   — % variance explained per PC
%
%  X_pca = score(:, 1:n_pcs)  — (N x n_pcs) reduced data
% ============================================================
fprintf('>>> PHASE 1b: PCA REDUCTION <<<\n');

[coeff, score, latent, ~, explained] = pca(X);

% Choose n_pcs: fewest PCs reaching VAR_THRESHOLD cumulative variance
cum_var = cumsum(explained) / 100;
n_pcs   = find(cum_var >= VAR_THRESHOLD, 1, 'first');
n_pcs   = min(n_pcs, MAX_PCS);   % cap at MAX_PCS

X_pca = score(:, 1:n_pcs);       % (64 x n_pcs)

fprintf('  n_pcs = %d  (%.1f%% of variance explained, threshold = %.0f%%)\n', ...
    n_pcs, cum_var(n_pcs)*100, VAR_THRESHOLD*100);
fprintf('  Reduced: %d x %d  →  %d x %d\n\n', N, p, N, n_pcs);

% selforgmap expects (features x samples)
Xt_pca = X_pca';   % (n_pcs x 64)

% ============================================================
%  PHASE 2: K-MEANS SWEEP  (on PCA data)
% ============================================================
fprintf('>>> PHASE 2: K-MEANS SWEEP (PCA + builtin kmeans) <<<\n');

km_results = struct('K', {}, 'labels', {}, 'centroids', {}, 'wcss', {});

for ki = 1:numel(K_VALUES)
    K = K_VALUES(ki);
    fprintf('  K = %2d  ... ', K);

    [lbl, C, sumd] = kmeans(X_pca, K, ...
        'Replicates', KM_RESTARTS, ...
        'MaxIter',    500, ...
        'Display',    'off');

    wcss = sum(sumd);
    km_results(ki).K         = K;
    km_results(ki).labels    = lbl;
    km_results(ki).centroids = C;
    km_results(ki).wcss      = wcss;

    fprintf('WCSS = %.4e   Sizes: [%s]\n', wcss, num2str(histcounts(lbl, 1:K+1)));
end
fprintf('\n');

% ============================================================
%  PHASE 3: SOM SWEEP  (on PCA data)
% ============================================================
fprintf('>>> PHASE 3: SOM SWEEP (PCA + builtin selforgmap) <<<\n');

som_results = struct('K', {}, 'grid', {}, 'labels', {}, 'mqe', {}, 'weights', {});

for ki = 1:numel(K_VALUES)
    K    = K_VALUES(ki);
    grid = SOM_GRIDS{ki};
    fprintf('  K = %2d  Grid = [%dx%d]  ... ', K, grid(1), grid(2));

    net = selforgmap(grid, 100, INIT_NEIGHBOR, 'gridtop', 'dist');
    net.trainParam.epochs     = SOM_EPOCHS;
    net.trainParam.showWindow = false;
    net.trainParam.show       = inf;

    net = train(net, Xt_pca);

    W   = net.IW{1,1};                  % (K x n_pcs)
    lbl = vec2ind(net(Xt_pca))';        % (N x 1)
    dists = pdist2(X_pca, W);
    mqe   = mean(min(dists, [], 2));

    som_results(ki).K       = K;
    som_results(ki).grid    = grid;
    som_results(ki).labels  = lbl;
    som_results(ki).mqe     = mqe;
    som_results(ki).weights = W;

    fprintf('MQE = %.4f   Sizes: [%s]\n', mqe, num2str(histcounts(lbl, 1:K+1)));
end
fprintf('\n');

% ============================================================
%  PHASE 4: NEIGHBOURHOOD CONVERGENCE  (on PCA data)
% ============================================================
fprintf('>>> PHASE 4: NEIGHBOURHOOD CONVERGENCE (K=%d, PCA) <<<\n', CONV_K);

km_k10_idx = find([km_results.K] == CONV_K);
km_k10_lbl = km_results(km_k10_idx).labels;

conv_agreement   = zeros(1, numel(CONV_INIT_NBRS) + 1);
conv_labels_used = [CONV_INIT_NBRS, NaN];

for ri = 1:numel(CONV_INIT_NBRS)
    iN = CONV_INIT_NBRS(ri);

    net_conv = selforgmap(CONV_GRID, 100, iN, 'gridtop', 'dist');
    net_conv.trainParam.epochs     = SOM_EPOCHS;
    net_conv.trainParam.showWindow = false;
    net_conv.trainParam.show       = inf;

    net_conv  = train(net_conv, Xt_pca);
    lbl_conv  = vec2ind(net_conv(Xt_pca))';

    [~, agr] = deal_labels_(lbl_conv, km_k10_lbl, CONV_K);
    conv_agreement(ri) = agr;

    fprintf('  initNeighbor = %d  →  Agreement: %.1f%%\n', iN, agr*100);
end

% Degenerate case: coverSteps=0
fprintf('  coverSteps=0 (K-means limit) ... ');
net_km = selforgmap(CONV_GRID, 0, 1, 'gridtop', 'dist');
net_km.trainParam.epochs     = SOM_EPOCHS;
net_km.trainParam.showWindow = false;
net_km.trainParam.show       = inf;
net_km = train(net_km, Xt_pca);
lbl_km = vec2ind(net_km(Xt_pca))';
[~, agr_km] = deal_labels_(lbl_km, km_k10_lbl, CONV_K);
conv_agreement(end) = agr_km;
fprintf('Agreement: %.1f%%\n\n', agr_km*100);

% ============================================================
%  PHASE 5: VISUALISATION
% ============================================================
fprintf('>>> PHASE 5: VISUALISATION <<<\n');
viz = OceanViz.Core('Theme', 'Dark');

% ---- Fig 0: PCA Scree plot ----
fig0 = viz.createFigure('ESL 14.6 (PCA) — NCI60: PCA Scree Plot');

ax0a = subplot(1, 2, 1, 'Parent', fig0);
bar(ax0a, 1:min(30, numel(explained)), explained(1:min(30,end)), ...
    'FaceColor', viz.Colors.Primary, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax0a);
title(ax0a, 'Variance Explained per PC (top 30)', ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax0a, 'Principal Component', 'Color', viz.Colors.Text);
ylabel(ax0a, '% Variance Explained', 'Color', viz.Colors.Text);

ax0b = subplot(1, 2, 2, 'Parent', fig0);
plot(ax0b, 1:numel(explained), cumsum(explained)/100 * 100, '-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 2);
hold(ax0b, 'on');
xline(ax0b, n_pcs, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 1.5, ...
    'Label', sprintf('n_{pcs}=%d (%.0f%%)', n_pcs, cum_var(n_pcs)*100));
yline(ax0b, VAR_THRESHOLD*100, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.5);
hold(ax0b, 'off');
viz.styleAxes(ax0b);
title(ax0b, 'Cumulative Variance Explained', ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax0b, 'Number of PCs', 'Color', viz.Colors.Text);
ylabel(ax0b, 'Cumulative % Variance', 'Color', viz.Colors.Text);
ylim(ax0b, [0, 105]);

% ---- Fig 1: WCSS vs MQE bar charts ----
fig1 = viz.createFigure('ESL 14.6 (PCA) — NCI60: K-Means WCSS vs SOM MQE');

ax1a = subplot(1, 2, 1, 'Parent', fig1);
bar(ax1a, K_VALUES, [km_results.wcss], ...
    'FaceColor', viz.Colors.Primary, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax1a);
title(ax1a, sprintf('K-Means WCSS  (PCA, %d PCs)', n_pcs), ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1a, 'K', 'Color', viz.Colors.Text);
ylabel(ax1a, 'WCSS', 'Color', viz.Colors.Text);
xticks(ax1a, K_VALUES);

ax1b = subplot(1, 2, 2, 'Parent', fig1);
bar(ax1b, K_VALUES, [som_results.mqe], ...
    'FaceColor', viz.Colors.Secondary, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax1b);
title(ax1b, sprintf('SOM MQE  (PCA, %d PCs)', n_pcs), ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1b, 'K (grid nodes)', 'Color', viz.Colors.Text);
ylabel(ax1b, 'MQE (mean dist to BMU in PC space)', 'Color', viz.Colors.Text);
xticks(ax1b, K_VALUES);

% ---- Fig 2: Cluster heatmap at K=5 (original gene space, sorted by PCA cluster) ----
gene_vars = var(X, 0, 1);
[~, sidx] = sort(gene_vars, 'descend');
X_sub     = X(:, sidx(1:TOP_GENES_VIZ));

ki5      = find([km_results.K] == 5);
si5      = find([som_results.K] == 5);
km5_lbl  = km_results(ki5).labels;
som5_lbl = som_results(si5).labels;

fig2 = viz.createFigure('ESL 14.6 (PCA) — NCI60: Cluster Heatmap (K=5)');
for panel = 1:2
    lbl    = km5_lbl * (panel==1) + som5_lbl * (panel==2);
    ptitle = {sprintf('K-Means K=5 [PCA %dPC]', n_pcs), ...
              sprintf('SOM K=5 [PCA %dPC]', n_pcs)};

    [~, sord]     = sort(lbl);
    X_sorted      = X_sub(sord, :);
    lbl_sorted    = lbl(sord);
    cancer_sorted = cancer_idx(sord);

    ax2 = subplot(1, 2, panel, 'Parent', fig2);
    imagesc(ax2, X_sorted);
    colormap(ax2, 'turbo');
    caxis(ax2, [-2, 2]);

    hold(ax2, 'on');
    for b = find(diff(lbl_sorted))'
        plot(ax2, [0.5, TOP_GENES_VIZ+0.5], [b+0.5, b+0.5], 'w-', 'LineWidth', 1.5);
    end
    hold(ax2, 'off');

    viz.styleAxes(ax2);
    title(ax2, ptitle{panel}, 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    xlabel(ax2, sprintf('Top %d variable genes', TOP_GENES_VIZ), 'Color', viz.Colors.Text);
    ylabel(ax2, 'Cell lines (sorted by cluster)', 'Color', viz.Colors.Text);
    ax2.YTick = 1:N;
    ax2.YTickLabel = cancer_sorted;
    ax2.TickLabelInterpreter = 'none';
    ax2.FontSize = 7;
end

% ---- Fig 3: PC1 vs PC2 scatter coloured by K-means cluster (K=5) ----
fig3 = viz.createFigure('ESL 14.6 (PCA) — NCI60: PC1 vs PC2 (K-Means K=5)');
ax_sc = axes('Parent', fig3);
hold(ax_sc, 'on');
cmap_k5 = lines(5);
for k = 1:5
    mask = (km5_lbl == k);
    scatter(ax_sc, X_pca(mask,1), X_pca(mask,2), 80, cmap_k5(k,:), ...
        'filled', 'MarkerFaceAlpha', 0.85, 'DisplayName', sprintf('Cluster %d', k));
end
% Annotate cancer types
for i = 1:N
    text(ax_sc, X_pca(i,1), X_pca(i,2), cancer_labels{i}, ...
        'FontSize', 7, 'Color', [0.8, 0.8, 0.8], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'Interpreter', 'none');
end
hold(ax_sc, 'off');
viz.styleAxes(ax_sc);
title(ax_sc, sprintf('PC1 vs PC2 — K-Means (K=5, %d PCs)', n_pcs), ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax_sc, sprintf('PC1 (%.1f%% var)', explained(1)), 'Color', viz.Colors.Text);
ylabel(ax_sc, sprintf('PC2 (%.1f%% var)', explained(2)), 'Color', viz.Colors.Text);
legend(ax_sc, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', 'none', 'EdgeColor', 'none');

% ---- Fig 4: Neighbourhood convergence curve ----
fig4 = viz.createFigure('ESL 14.6 (PCA) — SOM Neighbourhood Convergence (K=10)');
ax4  = axes('Parent', fig4);

x_pos    = 1:numel(conv_agreement);
x_labels = [arrayfun(@num2str, CONV_INIT_NBRS, 'UniformOutput', false), {'cov=0 (KM)'}];

plot(ax4, x_pos, conv_agreement * 100, 'o-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 2.5, ...
    'MarkerFaceColor', viz.Colors.Primary, 'MarkerSize', 9);
hold(ax4, 'on');
yline(ax4, 100, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.5, ...
    'Label', 'K-Means (reference)', 'LabelHorizontalAlignment', 'left');
hold(ax4, 'off');

viz.styleAxes(ax4);
set(ax4, 'XTick', x_pos, 'XTickLabel', x_labels);
title(ax4, sprintf('SOM \\rightarrow K-Means as initNeighbor \\rightarrow 0  [PCA %d PCs]', n_pcs), ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax4, 'initNeighbor (decreasing \rightarrow)', 'Color', viz.Colors.Text);
ylabel(ax4, 'Agreement with K-Means (%)', 'Color', viz.Colors.Text);
ylim(ax4, [0, 105]);

for ri = 1:numel(conv_agreement)
    if isnan(conv_labels_used(ri)), lbl_str = 'cov=0';
    else, lbl_str = sprintf('%d', conv_labels_used(ri)); end
    text(ax4, ri, conv_agreement(ri)*100 + 2, lbl_str, ...
        'Color', viz.Colors.Text, 'FontSize', 8, 'HorizontalAlignment', 'center');
end

% ============================================================
%  SUMMARY
% ============================================================
fprintf('=============================================================\n');
fprintf('  PIPELINE COMPLETE  [PCA + BUILTIN]\n\n');
fprintf('  PCA: %d x %d  -->  %d x %d  (%.1f%% variance retained)\n', ...
    N, p, N, n_pcs, cum_var(n_pcs)*100);
fprintf('\n  K-Means WCSS (PCA space):\n');
for ki = 1:numel(K_VALUES)
    fprintf('    K=%2d  WCSS = %.4e   Sizes: [%s]\n', K_VALUES(ki), ...
        km_results(ki).wcss, num2str(histcounts(km_results(ki).labels, 1:K_VALUES(ki)+1)));
end
fprintf('\n  SOM MQE (PCA space):\n');
for ki = 1:numel(K_VALUES)
    fprintf('    K=%2d  [%dx%d]  MQE = %.4f   Sizes: [%s]\n', K_VALUES(ki), ...
        SOM_GRIDS{ki}(1), SOM_GRIDS{ki}(2), som_results(ki).mqe, ...
        num2str(histcounts(som_results(ki).labels, 1:K_VALUES(ki)+1)));
end
fprintf('\n  Convergence (initNeighbor → Agreement with K-Means):\n');
for ri = 1:numel(conv_agreement)
    if isnan(conv_labels_used(ri)), tag = 'cov=0';
    else, tag = sprintf('initN=%d', conv_labels_used(ri)); end
    fprintf('    %8s  →  %.1f%%\n', tag, conv_agreement(ri)*100);
end
fprintf('=============================================================\n');


% ============================================================
%  LOCAL HELPER: permutation-matched agreement
% ============================================================
function [mapped, acc] = deal_labels_(pred_raw, ref_lbl, K)
    pred_ids = unique(pred_raw);
    mapping  = zeros(1, max(pred_ids));
    for k = pred_ids'
        mask  = (pred_raw == k);
        votes = histcounts(ref_lbl(mask), 1:K+1);
        [~, best]  = max(votes);
        mapping(k) = best;
    end
    mapped = mapping(pred_raw);
    acc    = mean(mapped(:) == ref_lbl(:));
end
