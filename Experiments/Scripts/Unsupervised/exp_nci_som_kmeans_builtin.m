%% exp_nci_som_kmeans_builtin.m
%  ============================================================
%  ESL Exercise 14.6 — K-Means & SOM on NCI Microarray Data
%  Using MATLAB Built-in Functions
%  ============================================================
%
%  PROBLEM:
%    Apply K-means and a 2D-grid SOM to the *columns* of the
%    NCI60 human tumor microarray data:
%      - 64 cell lines (columns) = observations
%      - 6830 genes (rows)       = features
%    Run both at K = 2, 5, 10, 20.
%    Demonstrate that as SOM neighbourhood shrinks → 0, SOM → K-Means.
%
%  BUILT-IN FUNCTIONS:
%    kmeans()      — Statistics & Machine Learning Toolbox
%    selforgmap()  — Deep Learning Toolbox (Neural Network Toolbox)
%
%  KEY DIFFERENCES vs OceanStats version:
%    kmeans()  : WCSS directly from sumd output (no manual loop)
%    selforgmap: uses 'initNeighbor' to control neighbourhood width
%                (controls ordering phase radius in the Kohonen schedule)
%                net.IW{1,1} = (M x p) prototype weight matrix
%                vec2ind(net(X')) = BMU index per sample
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

K_VALUES  = [2, 5, 10, 20];

% SOM grids: Q1*Q2 = K
% K=2:[1x2]  K=5:[1x5]  K=10:[2x5]  K=20:[4x5]
SOM_GRIDS = {[1,2], [1,5], [2,5], [4,5]};

KM_RESTARTS   = 20;
SOM_EPOCHS    = 500;     % selforgmap epochs (covers ordering + tuning phases)
INIT_NEIGHBOR = 3;       % default initial neighbourhood for SOM sweep

% Convergence experiment: fix K=10 grid [2x5], sweep initNeighbor
% Large initNeighbor → strong ordering phase (SOM-like topology)
% Small initNeighbor → weak ordering, hard assignment (K-means-like)
% NOTE: selforgmap requires initNeighbor to be a positive INTEGER
% Special case: coverSteps=0 → skip ordering entirely (pure K-means update)
CONV_K          = 10;
CONV_GRID       = [2, 5];
CONV_INIT_NBRS  = [6, 4, 3, 2, 1];   % integer sweep (large → small)
% coverSteps=0 case is appended separately (labelled as 'K-means limit')

TOP_GENES_VIZ = 150;

fprintf('=============================================================\n');
fprintf('  ESL Ex 14.6 — NCI60 K-Means & SOM  [BUILTIN VERSION]\n');
fprintf('  K sweep: [%s]\n', num2str(K_VALUES));
fprintf('=============================================================\n\n');

% ============================================================
%  PHASE 1: DATA LOADING & PREPROCESSING
%  (identical to OceanStats version)
% ============================================================
fprintf('>>> PHASE 1: DATA LOADING & PREPROCESSING <<<\n');

T     = readtable(DATA_PATH, 'ReadRowNames', true);
X_raw = T{:,:}';            % (64 x 6830)
[N, p] = size(X_raw);

fid   = fopen(LABEL_PATH, 'r');
raw   = textscan(fid, '%s', 'Delimiter', '\n', 'WhiteSpace', '');
fclose(fid);
cancer_labels = strtrim(raw{1});
cancer_labels = cancer_labels(~cellfun(@isempty, cancer_labels));
cancer_labels = cancer_labels(1:N);

% Gene-wise standardisation (zero-mean, unit-variance across 64 samples)
col_mean = mean(X_raw, 1);
col_std  = std(X_raw, 0, 1);
col_std(col_std == 0) = 1;
X = (X_raw - col_mean) ./ col_std;   % (64 x 6830)

[uniq_cancers, ~, cancer_idx] = unique(cancer_labels, 'stable');
n_types = numel(uniq_cancers);

fprintf('  Shape: %d cell lines x %d genes\n', N, p);
fprintf('  Cancer types (%d): %s\n', n_types, strjoin(uniq_cancers, ', '));
fprintf('  Standardised per gene\n\n');

% ============================================================
%  PHASE 2: K-MEANS SWEEP  (built-in kmeans)
%
%  kmeans(X, K, ...)
%    'Replicates' — number of random restarts (best WCSS kept)
%    'MaxIter'    — max Lloyd iterations per restart
%    'Display'    — suppress output
%  Returns:
%    lbl  (N x 1)  — cluster assignments 1..K
%    C    (K x p)  — centroids
%    sumd (K x 1)  — within-cluster sum of squared distances
%  WCSS = sum(sumd)
% ============================================================
fprintf('>>> PHASE 2: K-MEANS SWEEP (built-in kmeans) <<<\n');

km_results = struct('K', {}, 'labels', {}, 'centroids', {}, 'wcss', {});

for ki = 1:numel(K_VALUES)
    K = K_VALUES(ki);
    fprintf('  K = %2d  ... ', K);

    [lbl, C, sumd] = kmeans(X, K, ...
        'Replicates', KM_RESTARTS, ...
        'MaxIter',    500, ...
        'Display',    'off');

    wcss = sum(sumd);   % total within-cluster sum of squared distances

    km_results(ki).K         = K;
    km_results(ki).labels    = lbl;
    km_results(ki).centroids = C;
    km_results(ki).wcss      = wcss;

    fprintf('WCSS = %.4e   Cluster sizes: [%s]\n', wcss, ...
        num2str(histcounts(lbl, 1:K+1)));
end
fprintf('\n');

% ============================================================
%  PHASE 3: SOM SWEEP  (built-in selforgmap)
%
%  selforgmap(dimensions, coverSteps, initNeighbor, topologyFcn, distanceFcn)
%    dimensions  — [Q1, Q2] grid (Q1*Q2 = K)
%    coverSteps  — steps in ordering phase (default 100)
%    initNeighbor— initial neighbourhood radius (default 3)
%    topologyFcn — 'gridtop' (rectangular) or 'hextop' (hexagonal)
%    distanceFcn — 'dist' (Euclidean) or 'linkdist' (link distance)
%
%  NOTE: selforgmap expects data as (features x samples) — transpose X
%
%  Weights:  net.IW{1,1}      (M x p)  — prototype weight matrix
%  Labels:   vec2ind(net(X')) (1 x N)  — BMU index per sample
%  MQE:      computed manually from weights
% ============================================================
fprintf('>>> PHASE 3: SOM SWEEP (built-in selforgmap) <<<\n');

som_results = struct('K', {}, 'grid', {}, 'labels', {}, 'mqe', {}, 'weights', {});

Xt = X';   % selforgmap expects (features x samples) = (6830 x 64)

for ki = 1:numel(K_VALUES)
    K    = K_VALUES(ki);
    grid = SOM_GRIDS{ki};
    fprintf('  K = %2d  Grid = [%dx%d]  ... ', K, grid(1), grid(2));

    % Create SOM: rectangular topology, Euclidean distance
    net = selforgmap(grid, 100, INIT_NEIGHBOR, 'gridtop', 'dist');
    net.trainParam.epochs     = SOM_EPOCHS;
    net.trainParam.showWindow = false;   % suppress Training GUI
    net.trainParam.show       = inf;     % suppress console progress

    net = train(net, Xt);

    % Extract results
    W   = net.IW{1,1};              % (M x p) prototype weights
    lbl = vec2ind(net(Xt))';        % (N x 1) BMU index per sample

    % MQE: mean Euclidean distance from each sample to its BMU
    dists = pdist2(X, W);           % (N x M) pairwise distances
    mqe   = mean(min(dists, [], 2));

    som_results(ki).K       = K;
    som_results(ki).grid    = grid;
    som_results(ki).labels  = lbl;
    som_results(ki).mqe     = mqe;
    som_results(ki).weights = W;

    fprintf('MQE = %.4f   Node sizes: [%s]\n', mqe, ...
        num2str(histcounts(lbl, 1:K+1)));
end
fprintf('\n');

% ============================================================
%  PHASE 4: NEIGHBOURHOOD CONVERGENCE EXPERIMENT
%  Fix K=10, GridSize=[2,5]; sweep initNeighbor from large to ~0
%
%  selforgmap neighbourhood schedule:
%    Ordering phase: radius shrinks from initNeighbor → 1  (coverSteps iters)
%    Tuning phase:   radius = 1 (only BMU updates)
%
%  As initNeighbor → 0: ordering phase has negligible effect →
%  prototypes update independently → equivalent to online K-means
% ============================================================
fprintf('>>> PHASE 4: NEIGHBOURHOOD CONVERGENCE (K=%d) <<<\n', CONV_K);
fprintf('  initNeighbor sweep: [%s]\n\n', num2str(CONV_INIT_NBRS));

km_k10_idx = find([km_results.K] == CONV_K);
km_k10_lbl = km_results(km_k10_idx).labels;   % (64 x 1) reference

% Sweep: standard integer initNeighbor values
conv_agreement   = zeros(1, numel(CONV_INIT_NBRS) + 1);
conv_labels_used = [CONV_INIT_NBRS, NaN];   % NaN = coverSteps=0 special case

for ri = 1:numel(CONV_INIT_NBRS)
    iN = CONV_INIT_NBRS(ri);   % always an integer

    net_conv = selforgmap(CONV_GRID, 100, iN, 'gridtop', 'dist');
    net_conv.trainParam.epochs     = SOM_EPOCHS;
    net_conv.trainParam.showWindow = false;
    net_conv.trainParam.show       = inf;

    net_conv = train(net_conv, Xt);
    lbl_conv = vec2ind(net_conv(Xt))';

    [~, agr] = deal_labels_(lbl_conv, km_k10_lbl, CONV_K);
    conv_agreement(ri) = agr;

    fprintf('  initNeighbor = %d  →  Agreement with K-Means: %.1f%%\n', iN, agr*100);
end

% Special degenerate case: coverSteps=0 (skip ordering phase entirely)
% → SOM only runs tuning phase (neighborhood=1 throughout) → closest to K-means
fprintf('  coverSteps=0 (K-means limit) ... ');
net_km = selforgmap(CONV_GRID, 0, 1, 'gridtop', 'dist');
net_km.trainParam.epochs     = SOM_EPOCHS;
net_km.trainParam.showWindow = false;
net_km.trainParam.show       = inf;
net_km = train(net_km, Xt);
lbl_km = vec2ind(net_km(Xt))';
[~, agr_km] = deal_labels_(lbl_km, km_k10_lbl, CONV_K);
conv_agreement(end) = agr_km;
fprintf('Agreement: %.1f%%\n', agr_km*100);
fprintf('\n');

% ============================================================
%  PHASE 5: VISUALISATION (OceanViz Dark theme)
% ============================================================
fprintf('>>> PHASE 5: VISUALISATION <<<\n');

viz = OceanViz.Core('Theme', 'Dark');

% ---- Fig 1: K-sweep WCSS vs MQE ----
fig1 = viz.createFigure('ESL 14.6 (Builtin) — NCI60: K-Means WCSS vs SOM MQE');

ax1a = subplot(1, 2, 1, 'Parent', fig1);
bar(ax1a, K_VALUES, [km_results.wcss] / 1e6, ...
    'FaceColor', viz.Colors.Primary, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax1a);
title(ax1a, 'K-Means: WCSS  [builtin kmeans]', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1a, 'K (number of clusters)', 'Color', viz.Colors.Text);
ylabel(ax1a, 'WCSS (x10^6)', 'Color', viz.Colors.Text);
xticks(ax1a, K_VALUES);

ax1b = subplot(1, 2, 2, 'Parent', fig1);
bar(ax1b, K_VALUES, [som_results.mqe], ...
    'FaceColor', viz.Colors.Secondary, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
viz.styleAxes(ax1b);
title(ax1b, 'SOM: MQE  [builtin selforgmap]', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1b, 'K (grid nodes)', 'Color', viz.Colors.Text);
ylabel(ax1b, 'MQE (mean Euclidean distance to BMU)', 'Color', viz.Colors.Text);
xticks(ax1b, K_VALUES);

% ---- Fig 2: Cluster heatmap at K=5 ----
gene_vars = var(X, 0, 1);
[~, sidx] = sort(gene_vars, 'descend');
X_sub     = X(:, sidx(1:TOP_GENES_VIZ));

ki5     = find([km_results.K] == 5);
si5     = find([som_results.K] == 5);
km5_lbl = km_results(ki5).labels;
som5_lbl= som_results(si5).labels;

fig2 = viz.createFigure('ESL 14.6 (Builtin) — NCI60: Cluster Heatmap (K=5)');
for panel = 1:2
    lbl    = km5_lbl * (panel==1) + som5_lbl * (panel==2);
    ptitle = {'K-Means (K=5) [builtin]', 'SOM (K=5) [builtin]'};

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

% ---- Fig 3: Neighbourhood convergence ----
fig3 = viz.createFigure('ESL 14.6 (Builtin) — SOM Neighbourhood Convergence (K=10)');
ax3  = axes('Parent', fig3);

% X-axis labels for convergence plot:
% [6, 4, 3, 2, 1, 'KM-limit']
x_labels = [arrayfun(@num2str, CONV_INIT_NBRS, 'UniformOutput', false), {'cov=0\n(KM)'}];
x_pos    = 1:numel(conv_agreement);

semilogx(ax3, x_pos, conv_agreement * 100, 'o-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 2.5, ...
    'MarkerFaceColor', viz.Colors.Primary, 'MarkerSize', 9);
hold(ax3, 'on');
yline(ax3, 100, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.5, ...
    'Label', 'K-Means (reference)', 'LabelHorizontalAlignment', 'left');
hold(ax3, 'off');

viz.styleAxes(ax3);
set(ax3, 'XTick', x_pos, 'XTickLabel', x_labels, ...
    'XDir', 'normal');   % left=large neighbor, right=small
title(ax3, 'SOM \rightarrow K-Means as initNeighbor \rightarrow 0', ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax3, 'initNeighbor (decreasing \rightarrow)', 'Color', viz.Colors.Text);
ylabel(ax3, 'Agreement with K-Means (%)', 'Color', viz.Colors.Text);
ylim(ax3, [0, 105]);

for ri = 1:numel(conv_agreement)
    if isnan(conv_labels_used(ri))
        lbl_str = 'cov=0';
    else
        lbl_str = sprintf('%d', conv_labels_used(ri));
    end
    text(ax3, ri, conv_agreement(ri)*100 + 2, lbl_str, ...
        'Color', viz.Colors.Text, 'FontSize', 8, 'HorizontalAlignment', 'center');
end

% ============================================================
%  SUMMARY
% ============================================================
fprintf('=============================================================\n');
fprintf('  PIPELINE COMPLETE  [BUILTIN VERSION]\n\n');
fprintf('  Function mapping (OceanStats → MATLAB Builtin):\n');
fprintf('    OceanStats.Unsupervised.KMeans → kmeans()\n');
fprintf('    OceanStats.Unsupervised.SOM    → selforgmap() + train()\n\n');
fprintf('  K-Means WCSS:\n');
for ki = 1:numel(K_VALUES)
    fprintf('    K=%2d  WCSS = %.4e\n', K_VALUES(ki), km_results(ki).wcss);
end
fprintf('\n  SOM MQE:\n');
for ki = 1:numel(K_VALUES)
    fprintf('    K=%2d  [%dx%d]  MQE = %.4f\n', K_VALUES(ki), ...
        SOM_GRIDS{ki}(1), SOM_GRIDS{ki}(2), som_results(ki).mqe);
end
fprintf('\n  Convergence (initNeighbor → Agreement with K-Means):\n');
for ri = 1:numel(CONV_INIT_NBRS)
    fprintf('    initN=%.2f → %.1f%%\n', CONV_INIT_NBRS(ri), conv_agreement(ri)*100);
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
