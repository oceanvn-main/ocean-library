%% exp_nci_som_kmeans.m
%  ============================================================
%  ESL Exercise 14.6 — K-Means & SOM on NCI Microarray Data
%  ============================================================
%
%  PROBLEM:
%    Apply K-means and a 2D-grid SOM to the *columns* of the
%    NCI60 human tumor microarray data:
%      - 64 cell lines (columns) = observations
%      - 6830 genes (rows)       = features
%    Run both methods at K = 2, 5, 10, 20 centroids.
%    Demonstrate that as SOM neighbourhood radius shrinks → 0,
%    SOM assignments converge to K-means assignments.
%
%  DATA:
%    Data/NCI/nci.data.csv   — (6830 x 64) gene expression matrix
%    Data/NCI/nci.label.txt  — 64 cancer-type labels
%
%  LIBRARY COMPONENTS:
%    OceanStats.Unsupervised.KMeans  (Lloyd, multiple restarts)
%    OceanStats.Unsupervised.SOM     (online Kohonen, linear decay)
%    OceanViz.Core                   (Dark theme)
%
%  Ref: Hastie, Tibshirani, Friedman (2009), ESL Ex. 14.6, Section 14.4
%  ============================================================

clear; clc; close all;
rng(42, 'twister');

% ============================================================
%  CONFIGURATION
% ============================================================
% Anchor paths to the script's location (robust regardless of MATLAB cwd)
SCRIPT_DIR = fileparts(mfilename('fullpath'));          % .../Experiments/Scripts/Unsupervised
ROOT_DIR   = fullfile(SCRIPT_DIR, '..', '..', '..');   % project root
DATA_DIR   = fullfile(ROOT_DIR,   'Data', 'NCI');
DATA_PATH  = fullfile(DATA_DIR,   'nci.data.csv');
LABEL_PATH = fullfile(DATA_DIR,   'nci.label.txt');

K_VALUES   = [2, 5, 10, 20];   % K sweep

% SOM grid sizes: Q1*Q2 = K, roughly square-ish
% K=2:  [1x2]   K=5:  [1x5]   K=10: [2x5]   K=20: [4x5]
SOM_GRIDS  = {[1,2], [1,5], [2,5], [4,5]};

% SOM training settings
SOM_PASSES    = 20;      % passes per fit (64 samples, so fast)
SOM_INIT_LR   = 0.5;
SOM_FINAL_LR  = 0.01;
SOM_INIT_R    = 2.0;     % standard neighbourhood
SOM_FINAL_R   = 0.5;

KM_RESTARTS   = 20;      % K-means restarts

% Convergence experiment (K=10, GridSize=[2,5], vary FinalRadius)
CONV_K        = 10;
CONV_GRID     = [2, 5];
CONV_RADII    = [2.0, 1.0, 0.5, 0.2, 0.1, 0.02, 0.005];  % FinalRadius sweep
TOP_GENES_VIZ = 150;    % top variable genes to display in heatmap

fprintf('=============================================================\n');
fprintf('  ESL Ex 14.6 — K-Means & SOM on NCI Microarray Data\n');
fprintf('  K sweep: [%s]\n', num2str(K_VALUES));
fprintf('=============================================================\n\n');

% ============================================================
%  PHASE 1: DATA ACQUISITION & PREPROCESSING
% ============================================================
fprintf('>>> PHASE 1: DATA LOADING & PREPROCESSING <<<\n');

% Load gene expression matrix
% CSV: first column = gene index (row names), header row = cell line IDs
% After readtable: T is (6830 genes x 64 cell lines)
T = readtable(DATA_PATH, 'ReadRowNames', true);
X_raw = T{:, :};            % (6830 x 64) double

% Transpose: rows = observations (cell lines), cols = features (genes)
X_raw = X_raw';             % (64 x 6830)
[N, p] = size(X_raw);       % N=64, p=6830

% Load cancer-type labels
fid  = fopen(LABEL_PATH, 'r');
raw  = textscan(fid, '%s', 'Delimiter', '\n', 'WhiteSpace', '');
fclose(fid);
cancer_labels = strtrim(raw{1});
cancer_labels = cancer_labels(~cellfun(@isempty, cancer_labels));
cancer_labels = cancer_labels(1:N);   % ensure exactly 64

% Standardise GENES (columns): zero-mean, unit-variance across the 64 samples
% This is standard microarray preprocessing — each gene is centred/scaled
col_mean = mean(X_raw, 1);           % (1 x 6830)
col_std  = std(X_raw, 0, 1);         % (1 x 6830)
col_std(col_std == 0) = 1;           % guard against constant genes
X = (X_raw - col_mean) ./ col_std;  % (64 x 6830)

% Unique cancer types for colour coding
[uniq_cancers, ~, cancer_idx] = unique(cancer_labels, 'stable');
n_types = numel(uniq_cancers);

fprintf('  Shape (after transpose): %d cell lines x %d genes\n', N, p);
fprintf('  Cancer types (%d): %s\n', n_types, strjoin(uniq_cancers, ', '));
fprintf('  Standardised: mean~0, std~1 per gene\n\n');

% ============================================================
%  PHASE 2: K-MEANS SWEEP  (K = 2, 5, 10, 20)
% ============================================================
fprintf('>>> PHASE 2: K-MEANS SWEEP <<<\n');

km_results = struct('K', {}, 'labels', {}, 'wcss', {});

for ki = 1:numel(K_VALUES)
    K = K_VALUES(ki);
    fprintf('  K = %2d  ... ', K);

    km = OceanStats.Unsupervised.KMeans();
    lbl = km.fit(X, K, 'NumRestarts', KM_RESTARTS);

    % Compute WCSS
    wcss = 0;
    for k = 1:K
        mask = (lbl == k);
        if any(mask)
            wcss = wcss + sum(sum((X(mask,:) - km.Centroids(k,:)).^2));
        end
    end

    km_results(ki).K      = K;
    km_results(ki).labels = lbl;
    km_results(ki).wcss   = wcss;

    fprintf('WCSS = %.2f   Cluster sizes: [%s]\n', wcss, ...
        num2str(histcounts(lbl, 1:K+1)));
end
fprintf('\n');

% ============================================================
%  PHASE 3: SOM SWEEP  (K = 2, 5, 10, 20, 2D grids)
% ============================================================
fprintf('>>> PHASE 3: SOM SWEEP <<<\n');

som_results = struct('K', {}, 'grid', {}, 'labels', {}, 'mqe', {}, 'weights', {});

for ki = 1:numel(K_VALUES)
    K    = K_VALUES(ki);
    grid = SOM_GRIDS{ki};
    fprintf('  K = %2d  Grid = [%dx%d]  ... ', K, grid(1), grid(2));

    som = OceanStats.Unsupervised.SOM( ...
        'GridSize',    grid, ...
        'NumPasses',   SOM_PASSES, ...
        'InitLR',      SOM_INIT_LR, ...
        'FinalLR',     SOM_FINAL_LR, ...
        'InitRadius',  SOM_INIT_R, ...
        'FinalRadius', SOM_FINAL_R);

    lbl = som.fit(X, 'Verbose', false);
    mqe = som.meanQuantizationError(X);

    som_results(ki).K       = K;
    som_results(ki).grid    = grid;
    som_results(ki).labels  = lbl;
    som_results(ki).mqe     = mqe;
    som_results(ki).weights = som.Weights;

    fprintf('MQE = %.4f   Node sizes: [%s]\n', mqe, ...
        num2str(histcounts(lbl, 1:K+1)));
end
fprintf('\n');

% ============================================================
%  PHASE 4: NEIGHBOURHOOD CONVERGENCE EXPERIMENT
%  Fix K=10, GridSize=[2,5]; sweep FinalRadius from large to ~0
%  Hypothesis: as FinalRadius → 0, SOM → K-Means
% ============================================================
fprintf('>>> PHASE 4: NEIGHBOURHOOD CONVERGENCE (K=%d) <<<\n', CONV_K);
fprintf('  FinalRadius sweep: [%s]\n\n', num2str(CONV_RADII));

% Reference: K-means at K=10 (already computed in Phase 2)
km_k10_idx  = find([km_results.K] == CONV_K);
km_k10_lbl  = km_results(km_k10_idx).labels;   % (64 x 1)

conv_agreement = zeros(1, numel(CONV_RADII));

for ri = 1:numel(CONV_RADII)
    fr = CONV_RADII(ri);
    % InitRadius should be >= FinalRadius; clamp to at least 2x FinalRadius
    ir = max(2 * fr, 0.5);

    som_conv = OceanStats.Unsupervised.SOM( ...
        'GridSize',    CONV_GRID, ...
        'NumPasses',   SOM_PASSES, ...
        'InitLR',      SOM_INIT_LR, ...
        'FinalLR',     SOM_FINAL_LR, ...
        'InitRadius',  ir, ...
        'FinalRadius', fr);

    lbl_conv = som_conv.fit(X, 'Verbose', false);

    % Permutation-match SOM labels to K-means labels then compute agreement
    [~, agr] = deal_labels_(lbl_conv, km_k10_lbl, CONV_K);
    conv_agreement(ri) = agr;

    fprintf('  FinalRadius = %.4f  (InitR=%.4f)  Agreement with K-Means: %.1f%%\n', ...
        fr, ir, agr * 100);
end
fprintf('\n');

% ============================================================
%  PHASE 5: VISUALISATION (OceanViz Dark theme)
% ============================================================
fprintf('>>> PHASE 5: VISUALISATION <<<\n');

viz = OceanViz.Core('Theme', 'Dark');

% Build colour palette for cancer types
n_types_needed = max(n_types, 20);
cmap_types  = lines(n_types_needed);   % distinct colours per cancer type
cmap_clust  = hsv(max(K_VALUES));      % cluster colours for heatmap

% ---- Fig 1: K-sweep comparison (WCSS vs MQE) ----
fig1 = viz.createFigure('ESL 14.6 — NCI60: K-Means WCSS vs SOM MQE');

ax1a = subplot(1, 2, 1, 'Parent', fig1);
wcss_vals = [km_results.wcss];
bar(ax1a, K_VALUES, wcss_vals / 1e6, 'FaceColor', viz.Colors.Primary, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax1a);
title(ax1a, 'K-Means: WCSS', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1a, 'K (number of clusters)', 'Color', viz.Colors.Text);
ylabel(ax1a, 'WCSS (x10^6)', 'Color', viz.Colors.Text);
xticks(ax1a, K_VALUES);

ax1b = subplot(1, 2, 2, 'Parent', fig1);
mqe_vals = [som_results.mqe];
bar(ax1b, K_VALUES, mqe_vals, 'FaceColor', viz.Colors.Secondary, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax1b);
title(ax1b, 'SOM: Mean Quantization Error', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1b, 'K (grid nodes)', 'Color', viz.Colors.Text);
ylabel(ax1b, 'MQE (mean Euclidean distance to BMU)', 'Color', viz.Colors.Text);
xticks(ax1b, K_VALUES);

% ---- Fig 2: Cluster heatmap at K=5 (K-means vs SOM) ----
% Use top TOP_GENES_VIZ most variable genes for display
gene_vars  = var(X, 0, 1);                               % (1 x 6830) variance across 64 samples
[~, sidx]  = sort(gene_vars, 'descend');
X_sub      = X(:, sidx(1:TOP_GENES_VIZ));                % (64 x TOP_GENES_VIZ)

ki5 = find([km_results.K] == 5);
si5 = find([som_results.K] == 5);

km5_lbl  = km_results(ki5).labels;
som5_lbl = som_results(si5).labels;

fig2 = viz.createFigure('ESL 14.6 — NCI60: Cluster Heatmap (K=5)');

for panel = 1:2
    if panel == 1
        lbl   = km5_lbl;
        ptitle = 'K-Means (K=5)';
    else
        lbl   = som5_lbl;
        ptitle = 'SOM (K=5)';
    end

    [~, sort_order] = sort(lbl);
    X_sorted = X_sub(sort_order, :);
    lbl_sorted = lbl(sort_order);
    cancer_sorted = cancer_idx(sort_order);

    ax2 = subplot(1, 2, panel, 'Parent', fig2);
    imagesc(ax2, X_sorted);
    colormap(ax2, 'turbo');
    caxis(ax2, [-2, 2]);   % clamp z-score range

    % Draw cluster boundaries as horizontal lines
    hold(ax2, 'on');
    boundaries = find(diff(lbl_sorted));
    for b = boundaries'
        plot(ax2, [0.5, TOP_GENES_VIZ + 0.5], [b+0.5, b+0.5], ...
            'w-', 'LineWidth', 1.5);
    end
    hold(ax2, 'off');

    viz.styleAxes(ax2);
    title(ax2, ptitle, 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    xlabel(ax2, sprintf('Top %d variable genes', TOP_GENES_VIZ), 'Color', viz.Colors.Text);
    ylabel(ax2, 'Cell lines (sorted by cluster)', 'Color', viz.Colors.Text);

    % Y-tick labels: show cancer type for each row
    ax2.YTick = 1:N;
    ax2.YTickLabel = cancer_sorted;
    ax2.TickLabelInterpreter = 'none';
    ax2.FontSize = 7;
end
colorbar(subplot(1, 2, 2, 'Parent', fig2));

% ---- Fig 3: Neighbourhood convergence curve ----
fig3 = viz.createFigure('ESL 14.6 — SOM Neighbourhood Convergence (K=10)');
ax3  = axes('Parent', fig3);

semilogx(ax3, CONV_RADII, conv_agreement * 100, 'o-', ...
    'Color',           viz.Colors.Primary, ...
    'LineWidth',       2.5, ...
    'MarkerFaceColor', viz.Colors.Primary, ...
    'MarkerSize',      9);
hold(ax3, 'on');

% Reference line at 100% (K-means with itself)
yline(ax3, 100, '--', ...
    'Color',     viz.Colors.Boundary, ...
    'LineWidth', 1.5, ...
    'Label',     'K-Means (reference)', ...
    'LabelHorizontalAlignment', 'left', ...
    'FontSize',  10);
hold(ax3, 'off');

viz.styleAxes(ax3);
set(ax3, 'XDir', 'reverse');   % large radius on left → near-zero on right
title(ax3, 'SOM → K-Means as Neighbourhood Radius → 0', ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax3, 'FinalRadius (log scale, decreasing →)', 'Color', viz.Colors.Text);
ylabel(ax3, 'Agreement with K-Means (%)', 'Color', viz.Colors.Text);
ylim(ax3, [0, 105]);
xlim(ax3, [min(CONV_RADII)/2, max(CONV_RADII)*1.5]);

% Label each data point with its FinalRadius value
for ri = 1:numel(CONV_RADII)
    text(ax3, CONV_RADII(ri), conv_agreement(ri)*100 + 2, ...
        sprintf('%.3f', CONV_RADII(ri)), ...
        'Color', viz.Colors.Text, 'FontSize', 8, ...
        'HorizontalAlignment', 'center');
end

% ============================================================
%  SUMMARY
% ============================================================
fprintf('=============================================================\n');
fprintf('  PIPELINE COMPLETE\n\n');
fprintf('  K-Means WCSS:\n');
for ki = 1:numel(K_VALUES)
    fprintf('    K=%2d  WCSS = %.2e\n', K_VALUES(ki), km_results(ki).wcss);
end
fprintf('\n  SOM MQE:\n');
for ki = 1:numel(K_VALUES)
    fprintf('    K=%2d  [%dx%d]  MQE = %.4f\n', K_VALUES(ki), ...
        SOM_GRIDS{ki}(1), SOM_GRIDS{ki}(2), som_results(ki).mqe);
end
fprintf('\n  Convergence (FinalRadius → Agreement with K-Means):\n');
for ri = 1:numel(CONV_RADII)
    fprintf('    FinalR=%.4f → %.1f%%\n', CONV_RADII(ri), conv_agreement(ri)*100);
end
fprintf('=============================================================\n');


% ============================================================
%  LOCAL HELPER: match cluster labels to reference via majority vote
%  (permutation-invariant agreement)
% ============================================================
function [mapped, acc] = deal_labels_(pred_raw, ref_lbl, K)
    pred_ids = unique(pred_raw);
    mapping  = zeros(1, max(pred_ids));
    for k = pred_ids'
        mask  = (pred_raw == k);
        votes = histcounts(ref_lbl(mask), 1:K+1);
        [~, best]    = max(votes);
        mapping(k)   = best;
    end
    mapped = mapping(pred_raw);
    acc    = mean(mapped(:) == ref_lbl(:));
end
