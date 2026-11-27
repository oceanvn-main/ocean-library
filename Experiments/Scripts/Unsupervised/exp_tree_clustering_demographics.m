%% exp_tree_clustering_demographics.m
%  ============================================================
%  ESL Exercise 14.4 -- Classification Tree Clustering
%  San Francisco Bay Area Demographic Data (Table 14.1)
%  ============================================================
%
%  METHOD:
%    1. Load the demographics dataset (N=8993, 14 features).
%    2. Generate a reference (null) sample: permute values within
%       each feature column independently.  This preserves each
%       feature's marginal distribution but destroys all cross-
%       feature correlations.
%    3. Label real data = 1, reference data = 0.  Stack them.
%    4. Fit a CART classification tree (Gini, depth-limited).
%    5. Identify terminal nodes with highest P(Class=1) — those
%       are dense real-data regions (clusters).
%    6. Compare to K-means clustering on the real data.
%
%  LIBRARY COMPONENTS USED:
%    OceanStats.Supervised.Tree.DecisionTree  (Gini CART)
%    OceanStats.Unsupervised.KMeans
%    OceanViz.Core                            (Dark theme)
%
%  DATASET:  Data/Demographics/data.txt
%            Col 1  = Annual Income (1-9, Ordinal)
%            Col 2  = Sex (1-2, Categorical)
%            Col 3  = Marital Status (1-5, Categorical)
%            Col 4  = Age (1-7, Ordinal)
%            Col 5  = Education (1-6, Ordinal)
%            Col 6  = Occupation (1-9, Categorical)
%            Col 7  = Years in Bay Area (1-5, Ordinal)
%            Col 8  = Dual Incomes (1-3, Categorical)
%            Col 9  = Number in Household (1-9, Ordinal)
%            Col 10 = Number of Children (0-9, Ordinal)
%            Col 11 = Householder Status (1-3, Categorical)
%            Col 12 = Type of Home (1-5, Categorical)
%            Col 13 = Ethnic Classification (1-8, Categorical)
%            Col 14 = Language in Home (1-3, Categorical)
%
%  Ref: Hastie, Tibshirani, Friedman (2009), ESL Ex. 14.4
%  ============================================================

clear; clc; close all;
rng(42, 'twister');

% ============================================================
%  CONFIGURATION
% ============================================================
DATA_PATH     = fullfile('c:', 'Ocean Library', 'Matlab code practice', ...
                         'Data', 'Demographics', 'data.txt');
MAX_DEPTH     = 6;    % CART tree depth limit (interpretability)
MIN_LEAF      = 50;   % Minimum leaf size (~0.5% support floor)
TOP_N         = 5;    % How many high-P(C=1) leaves to report
K_MEANS_K     = 3;    % K for comparison clustering

FEATURE_NAMES = { ...
    'Income', 'Sex', 'Marital Status', 'Age', 'Education', ...
    'Occupation', 'Years in Bay Area', 'Dual Incomes', ...
    'Number in Household', 'Number of Children', ...
    'Householder Status', 'Type of Home', ...
    'Ethnic Classification', 'Language in Home' };

fprintf('=============================================================\n');
fprintf('  ESL Ex 14.4 -- Classification Tree Clustering\n');
fprintf('  Dataset: SF Bay Area Demographics\n');
fprintf('  MaxDepth=%d  MinLeaf=%d  K(means)=%d\n', MAX_DEPTH, MIN_LEAF, K_MEANS_K);
fprintf('=============================================================\n\n');

% ============================================================
%  PHASE 1: DATA ACQUISITION & PREPROCESSING
% ============================================================
fprintf('>>> PHASE 1: DATA ACQUISITION & PREPROCESSING <<<\n');

% Read whitespace-delimited file.
% TreatAsEmpty converts 'NA' strings to NaN during parsing.
% ReadVariableNames=false because the file has no header row.
T = readtable(DATA_PATH, ...
    'FileType',       'text', ...
    'Delimiter',      ' ', ...
    'ReadVariableNames', false, ...
    'TreatAsEmpty',   {'NA', ''}, ...
    'ConsecutiveDelimitersRule', 'join', ...
    'LeadingDelimitersRule',    'ignore');
X_raw = table2array(T);

[N_raw, p] = size(X_raw);
fprintf('  Raw shape: %d x %d\n', N_raw, p);
fprintf('  Missing per feature: ');
fprintf('%d ', sum(isnan(X_raw)));
fprintf('\n');

% -- Median imputation (anti-leakage: fit medians on raw data only)
% Assumption: MCAR/MAR; median preserves ordinal scale.
X = X_raw;
for j = 1:p
    col = X(:, j);
    med = median(col, 'omitnan');
    col(isnan(col)) = med;
    X(:, j) = col;
end
X = round(X);   % all values are integer ordinal codes

[N, p] = size(X);
fprintf('  After imputation: %d x %d (0 missing)\n', N, p);

% ============================================================
%  PHASE 2: GENERATE NULL REFERENCE SAMPLE
% ============================================================
fprintf('\n>>> PHASE 2: NULL REFERENCE SAMPLE (column permutation) <<<\n');

% Permute each column independently.
% This preserves each marginal distribution but destroys all
% inter-feature associations -- the signal we want to find.
X_ref = X;
for j = 1:p
    X_ref(:, j) = X_ref(randperm(N), j);
end

fprintf('  Reference sample: %d x %d\n', size(X_ref));
fprintf('  Marginals preserved per column; cross-feature structure destroyed.\n');

% Stack: real = class 1, reference = class 0
X_combined = [X; X_ref];
Y_combined  = [ones(N, 1); zeros(N, 1)];

fprintf('  Combined: %d x %d  (Class 1=%d, Class 0=%d)\n', ...
    size(X_combined, 1), size(X_combined, 2), sum(Y_combined), sum(~Y_combined));

% ============================================================
%  PHASE 3: CLASSIFICATION TREE (OceanStats.Supervised.Tree)
% ============================================================
fprintf('\n>>> PHASE 3: CLASSIFICATION TREE FITTING <<<\n');

% Gini CART, depth-limited for interpretability.
% In a balanced problem (N class-1 == N class-0), leaves with
% P(class=1) >> 0.5 identify regions denser than expected by
% the null (independent-marginal) model.
tree = OceanStats.Supervised.Tree.DecisionTree( ...
    'Criterion',   'gini', ...
    'MaxDepth',    MAX_DEPTH, ...
    'MinLeafSize', MIN_LEAF);

tree.fit(X_combined, Y_combined);

nLeaves = tree.countTerminalNodes();
treeDepth = tree.depth();
Y_pred = tree.predict(X_combined);
acc = mean(Y_pred == Y_combined);

fprintf('  Tree depth:       %d\n', treeDepth);
fprintf('  Terminal nodes:   %d\n', nLeaves);
fprintf('  Training accuracy: %.4f\n', acc);

% ============================================================
%  PHASE 4: TERMINAL NODE ANALYSIS
% ============================================================
fprintf('\n>>> PHASE 4: TERMINAL NODE ANALYSIS <<<\n');

% -- Route each sample to its leaf
[leafIds_combined, ~] = tree.getLeafIds(X_combined);
leafIds_real = leafIds_combined(1:N);     % real data leaves
leafIds_ref  = leafIds_combined(N+1:end); % reference data leaves

% -- Per-leaf statistics
uniqueLeaves = unique(leafIds_combined);
nUnique = numel(uniqueLeaves);

leafStats = zeros(nUnique, 4);  % [leafId, n_real, n_ref, p_class1]
for k = 1:nUnique
    lid    = uniqueLeaves(k);
    n_real = sum(leafIds_real == lid);
    n_ref  = sum(leafIds_ref  == lid);
    n_tot  = n_real + n_ref;
    p1     = n_real / max(n_tot, 1);
    leafStats(k, :) = [lid, n_real, n_ref, p1];
end

% Sort by P(class=1) descending
[~, order] = sort(leafStats(:, 4), 'descend');
leafStats   = leafStats(order, :);

% -- Print top leaves
fprintf('\n  Top %d terminal nodes by P(Class=1):\n\n', TOP_N);

for r = 1:TOP_N
    lid    = leafStats(r, 1);
    n_real = leafStats(r, 2);
    n_ref  = leafStats(r, 3);
    p1     = leafStats(r, 4);
    sup    = n_real / N;

    fprintf('  [%d] Leaf %d\n', r, lid);
    fprintf('      P(Class=1) = %.4f\n', p1);
    fprintf('      Support    = %.2f%%  (%d real samples)\n', sup*100, n_real);
    fprintf('      n_total    = %d  (real=%d, ref=%d)\n\n', n_real+n_ref, n_real, n_ref);
end

% ============================================================
%  PHASE 5: K-MEANS CLUSTERING (COMPARISON)
% ============================================================
fprintf('>>> PHASE 5: K-MEANS CLUSTERING (K=%d) <<<\n', K_MEANS_K);

% K-means on real data only
km = OceanStats.Unsupervised.KMeans();
km_labels = km.fit(X, K_MEANS_K, 'NumRestarts', 20);

clusterCounts = histcounts(km_labels, 1:K_MEANS_K+1);
fprintf('  Cluster sizes: ');
fprintf('%d ', clusterCounts);
fprintf('\n');

% Compute inertia
inertia = 0;
for k = 1:K_MEANS_K
    mask = (km_labels == k);
    if any(mask)
        inertia = inertia + sum(sum((X(mask,:) - km.Centroids(k,:)).^2));
    end
end
fprintf('  Inertia (WCSS): %.2f\n', inertia);

% Cluster mean profiles
fprintf('\n  Cluster mean profiles:\n');
fprintf('  %-25s', 'Feature');
for k = 1:K_MEANS_K
    fprintf('  Cluster %d', k);
end
fprintf('\n  %s\n', repmat('-', 1, 25 + K_MEANS_K*11));
for j = 1:p
    fprintf('  %-25s', FEATURE_NAMES{j});
    for k = 1:K_MEANS_K
        fprintf('  %8.3f', mean(X(km_labels==k, j)));
    end
    fprintf('\n');
end

% ============================================================
%  PHASE 6: VISUALIZATION (OceanViz)
% ============================================================
fprintf('\n>>> PHASE 6: VISUALIZATION <<<\n');

viz = OceanViz.Core('Theme', 'Dark');

% -- Fig 1: P(class=1) vs Support for all leaves ----------------
fig1 = viz.createFigure('ESL 14.4 - Leaf Density Analysis');
ax1  = axes(fig1);
hold(ax1, 'on');

p_vals = leafStats(:, 4);
supp   = leafStats(:, 2) / N * 100;
sz     = max(leafStats(:, 2) / 30, 5);

scatter(ax1, supp, p_vals, sz*20, p_vals, 'filled', ...
    'MarkerEdgeColor', [0.8 0.8 0.8], 'LineWidth', 0.3);

% Highlight top-N leaves
top_sup = leafStats(1:TOP_N, 2) / N * 100;
top_p   = leafStats(1:TOP_N, 4);
plot(ax1, top_sup, top_p, 'o', ...
    'MarkerSize', 14, 'LineStyle', 'none', ...
    'MarkerFaceColor', 'none', ...
    'MarkerEdgeColor', viz.Colors.Boundary, ...
    'LineWidth', 2.5, 'HandleVisibility', 'off');
for r = 1:TOP_N
    text(ax1, top_sup(r), top_p(r) + 0.008, sprintf('#%d', r), ...
        'Color', viz.Colors.Boundary, 'FontSize', 10, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

yline(ax1, 0.5, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5);
text(ax1, max(supp)*0.85, 0.515, 'Chance (P=0.5)', ...
    'Color', [0.6 0.6 0.6], 'FontSize', 9);

colormap(ax1, 'parula');
c = colorbar(ax1);
c.Label.String = 'P(Class=1)';
c.Label.Color  = viz.Colors.Text;
c.Color        = viz.Colors.Text;

viz.styleAxes(ax1);
title(ax1, 'Classification Tree Leaves: P(Class=1) vs Support', ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax1, 'Support (% of real data in leaf)', 'Color', viz.Colors.Text);
ylabel(ax1, 'Estimated P(Class=1)', 'Color', viz.Colors.Text);
ylim(ax1, [0, 1.05]);
hold(ax1, 'off');


% -- Fig 2: K-means cluster profiles (heat map) -----------------
fig2 = viz.createFigure('ESL 14.4 - K-Means Cluster Profiles');
ax2  = axes(fig2);

% Build cluster-mean matrix and normalize per feature to [0,1]
C_means = zeros(K_MEANS_K, p);
for k = 1:K_MEANS_K
    C_means(k, :) = mean(X(km_labels == k, :));
end
C_min  = min(C_means, [], 1);
C_max  = max(C_means, [], 1);
C_norm = (C_means - C_min) ./ max(C_max - C_min, 1e-9);

imagesc(ax2, C_norm);
colormap(ax2, 'turbo');   % perceptually vivid (MATLAB R2020b+; use 'parula' on older)
colorbar(ax2);

ax2.XTick = 1:p;
ax2.XTickLabel = FEATURE_NAMES;
ax2.XTickLabelRotation = 40;
ax2.YTick = 1:K_MEANS_K;
ax2.YTickLabel = arrayfun(@(k) sprintf('Cluster %d', k), 1:K_MEANS_K, ...
    'UniformOutput', false);

% Annotate raw means
for k = 1:K_MEANS_K
    for j = 1:p
        text(ax2, j, k, sprintf('%.1f', C_means(k,j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', 'white', 'FontSize', 7);
    end
end

viz.styleAxes(ax2);
title(ax2, sprintf('K-Means (K=%d) Cluster Feature Profiles', K_MEANS_K), ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');


% -- Fig 3: K-means cluster membership inside top leaves ---------
fig3 = viz.createFigure('ESL 14.4 - K-Means vs Tree Leaf Membership');
ax3  = axes(fig3);
hold(ax3, 'on');

palette = viz.getColorSequence(K_MEANS_K);
bar_data = zeros(TOP_N, K_MEANS_K);

for r = 1:TOP_N
    lid  = leafStats(r, 1);
    mask = (leafIds_real == lid);
    if any(mask)
        km_in_leaf = km_labels(mask);
        for k = 1:K_MEANS_K
            bar_data(r, k) = sum(km_in_leaf == k) / sum(mask);
        end
    end
end

b = bar(ax3, bar_data, 'stacked');
for k = 1:K_MEANS_K
    b(k).FaceColor = palette(k, :);
    b(k).DisplayName = sprintf('K-means Cluster %d', k);
end

ax3.XTickLabel = arrayfun(@(r) sprintf('Leaf #%d', r), 1:TOP_N, ...
    'UniformOutput', false);
viz.styleAxes(ax3);
legend(ax3, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', 'none', 'EdgeColor', 'none');
title(ax3, 'K-Means Cluster Composition in Top Tree Leaves', ...
    'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax3, 'Terminal Node', 'Color', viz.Colors.Text);
ylabel(ax3, 'Proportion', 'Color', viz.Colors.Text);
ylim(ax3, [0 1.05]);
hold(ax3, 'off');


% ============================================================
%  PHASE 7: SUMMARY
% ============================================================
fprintf('\n=============================================================\n');
fprintf('  PHASE 7: SUMMARY\n');
fprintf('=============================================================\n\n');

fprintf('COMPARISON FRAMEWORK\n');
fprintf('  Method               | Key characteristic\n');
fprintf('  ---------------------|-----------------------------------------------\n');
fprintf('  Classification Tree  | Finds hyper-rectangle rules vs null reference\n');
fprintf('  (Ex 14.4)            | P(C=1) >> 0.5 => dense real-data region\n');
fprintf('  ---------------------|-----------------------------------------------\n');
fprintf('  K-Means (K=%d)       | Partitions all data, no null reference needed\n', K_MEANS_K);
fprintf('                       | Spherical clusters; misses interaction effects\n');
fprintf('\n');

fprintf('TOP TERMINAL NODES (Classification Tree):\n');
fprintf('%s\n', repmat('-', 1, 65));
for r = 1:TOP_N
    lid  = leafStats(r, 1);
    n_r  = leafStats(r, 2);
    p1   = leafStats(r, 4);
    sup  = n_r / N;
    fprintf('  [%d] Leaf %d  |  P(C=1)=%.4f  |  Support=%.2f%%  |  n=%d\n', ...
        r, lid, p1, sup*100, n_r);
end

fprintf('\n=============================================================\n');
fprintf('  PIPELINE COMPLETE\n');
fprintf('=============================================================\n');
