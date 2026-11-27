% EXP_KMEANS_NCI
% Reproduces ESL Section 14.3.8 "Example: Human Tumor Microarray Data"
%
% Runs 4 comparative experiments manipulating PCA reduction and cluster depth (K)
% mapping tables and visual distribution automatically.
%
% Ref: Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
%      Section 14.3.8, Figure 14.8, Table 14.2

clc; clear; rng(42, 'twister');
fprintf('=== K-Means on Human Tumor Microarray Data ===\n');

%% PHASE 1: Data Ingestion (Anti-Leakage)
dataPath = fullfile('c:', 'Ocean Library', 'Matlab code practice', 'Data', 'NCI');
dataFile = fullfile(dataPath, 'nci.data.csv');
lblFile  = fullfile(dataPath, 'nci.label.txt');

% 1. Read labels and clean up reproducible cell lines to match textbook exactly
trueLabels = strtrim(string(readcell(lblFile)));
trueLabels = strrep(trueLabels, 'K562A-repro', 'K562');
trueLabels = strrep(trueLabels, 'K562B-repro', 'K562');
trueLabels = strrep(trueLabels, 'MCF7A-repro', 'MCF7');
trueLabels = strrep(trueLabels, 'MCF7D-repro', 'MCF7');
N = numel(trueLabels);
uniqueLabels = unique(trueLabels);
L = numel(uniqueLabels);

% 2. Read Expression feature matrix
T = readtable(dataFile, 'ReadVariableNames', true, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');

% X_raw is 6830 genes x 64 samples. Transpose to canonical N x p (64 x 6830)
X = table2array(T)'; 
fprintf('[DataOps] Ingested %d samples with %d genes.\n', size(X, 1), size(X, 2));

%% PHASE 2: Dimensionality Reduction (PCA Spectrum)
fprintf('\n[Phase 2: PCA] Analyzing Principal Component Spectrum...\n');

% Calculate native mathematical spectrum without mutating library
[~, ~, latent] = pca(X - mean(X));
explained_var = (latent / sum(latent)) * 100;
cumulative_var = cumsum(explained_var);

% Dynamically pick optimal K (e.g. 90% variance target) instead of heuristics
best_k = find(cumulative_var >= 90, 1);
fprintf('  Selected K=%d principal components to capture 90%% of variance.\n', best_k);

viz = OceanViz.Core('Theme', 'Dark');

% Render the Scree Plot
f_scree = viz.createFigure('PCA Scree Plot');
ax_scree = axes(f_scree);
hold(ax_scree, 'on');

bar(ax_scree, 1:numel(explained_var), explained_var, 'FaceColor', viz.Colors.Secondary, 'EdgeColor', 'none', 'FaceAlpha', 0.8, 'DisplayName', 'Individual Variance');
plot(ax_scree, 1:numel(cumulative_var), cumulative_var, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 2, 'MarkerFaceColor', viz.Colors.Primary, 'MarkerSize', 5, 'DisplayName', 'Cumulative Variance');
yline(ax_scree, 90, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'DisplayName', '90% Variance Threshold');
xline(ax_scree, best_k, ':', 'Color', viz.Colors.Boundary, 'LineWidth', 2.0, 'DisplayName', sprintf('Optimal Cutoff (K=%d)', best_k));

viz.styleAxes(ax_scree);
title(ax_scree, 'PCA Scree Plot (NCI-60 Spectrum)', 'FontWeight', 'bold', 'Color', viz.Colors.Text);
xlabel(ax_scree, 'Principal Component Rank', 'Color', viz.Colors.Text);
ylabel(ax_scree, 'Percentage of Variance (%)', 'Color', viz.Colors.Text);
legend(ax_scree, 'Location', 'east', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);
hold(ax_scree, 'off');

% Project the actual basis space dynamically based on the discovered selection
pcaPipe = OceanStats.Pipeline.PCA(best_k); 
pcaModel = pcaPipe.train(X, []);
X_pca = pcaPipe.predict(pcaModel, X);

%% PHASE 3: Elbow Curves (K=1..10)
fprintf('\n[Phase 3: Tuning] Running K-Means (K=1...10) for Elbow plotting...\n');
K_max = 10;
wcss_raw = zeros(K_max, 1);
wcss_pca = zeros(K_max, 1);
km = OceanStats.Unsupervised.KMeans();

for K = 1:K_max
    % Raw Data WCSS
    km.fit(X, K, 'NumRestarts', 10);
    ss_raw = 0;
    for k = 1:K
        mask = (km.Labels == k);
        if any(mask), ss_raw = ss_raw + sum(sum((X(mask, :) - km.Centroids(k, :)).^2)); end
    end
    wcss_raw(K) = ss_raw;

    % PCA Data WCSS
    km.fit(X_pca, K, 'NumRestarts', 10);
    ss_pca = 0;
    for k = 1:K
        mask = (km.Labels == k);
        if any(mask), ss_pca = ss_pca + sum(sum((X_pca(mask, :) - km.Centroids(k, :)).^2)); end
    end
    wcss_pca(K) = ss_pca;
end

% Plot Elbows
f_elbow = viz.createFigure('K-Means WCSS Tuning');
ax1 = subplot(1, 2, 1, 'Parent', f_elbow);
viz.plot(ax1, 'LearningCurve', struct('X', (1:K_max)', 'Lines', {{struct('Y', wcss_raw, 'Name', 'Raw WCSS (6830 dims)')}}), ...
    'Title', 'Raw NCI-60 WCSS', 'XLabel', 'K', 'YLabel', 'Sum of Squares');

ax2 = subplot(1, 2, 2, 'Parent', f_elbow);
viz.plot(ax2, 'LearningCurve', struct('X', (1:K_max)', 'Lines', {{struct('Y', wcss_pca, 'Name', 'PCA WCSS (2 dims)')}}), ...
    'Title', 'PCA-Reduced WCSS', 'XLabel', 'K', 'YLabel', 'Sum of Squares');


%% PHASE 4: The Four Diagnostic Experiments
experiments = {
    struct('usePCA', false, 'K', 3,  'Desc', '1. Textbook Reference (Raw Data, K=3)'),
    struct('usePCA', false, 'K', 11, 'Desc', '2. Extended Clusters (Raw Data, K=11)'),
    struct('usePCA', true,  'K', 3,  'Desc', sprintf('3. PCA-Reduced Space (%d Dims, K=3)', best_k)),
    struct('usePCA', true,  'K', 11, 'Desc', sprintf('4. PCA-Reduced Space (%d Dims, K=11)', best_k))
};

f_scatter = viz.createFigure('K-Means Comparative Clustering');

for i = 1:4
    exp = experiments{i};
    fprintf('\n=================================================================\n');
    fprintf(' EXPERIMENT %s\n', exp.Desc);
    fprintf('=================================================================\n');
    
    % Execute Clustering
    if exp.usePCA
        km.fit(X_pca, exp.K, 'NumRestarts', 50);
    else
        km.fit(X, exp.K, 'NumRestarts', 50);
    end
    
    % Print Table 14.2 Equivalents
    counts = zeros(exp.K, L);
    for c = 1:exp.K
        for l = 1:L
            counts(c, l) = sum((km.Labels == c) & (trueLabels == uniqueLabels(l)));
        end
    end

    fprintf('\nCross-Tabulation Matrix:\n');
    fprintf('Cluster |');
    for l = 1:L, fprintf(' %9s', uniqueLabels(l)); end
    fprintf('\n%s\n', repmat('-', 1, 9 + 10*L));
    for c = 1:exp.K
        fprintf('   %-2d   |', c);
        for l = 1:L, fprintf(' %9d', counts(c, l)); end
        fprintf('\n');
    end
    fprintf('\n');

    % Render Scatter
    ax = subplot(2, 2, i, 'Parent', f_scatter);
    hold(ax, 'on');
    palette = viz.getColorSequence(exp.K);
    
    for k = 1:exp.K
        mask = (km.Labels == k);
        if any(mask)
            % Points are ALWAYS plotted on the 2D PCA basis, regardless of what space
            % the K-means algorithm used. We're just visualizing the resulting cluster assignments!
            scatter(ax, X_pca(mask, 1), X_pca(mask, 2), 40, palette(k, :), 'filled', ...
                'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.8, 'DisplayName', sprintf('C %d', k));
        end
    end
    
    viz.styleAxes(ax);
    title(ax, exp.Desc, 'FontWeight', 'bold', 'Color', viz.Colors.Text, 'Interpreter', 'none');
    xlabel(ax, 'PC1', 'Color', viz.Colors.Text);
    ylabel(ax, 'PC2', 'Color', viz.Colors.Text);
    
    % Condense legend to avoid cramping
    if exp.K == 3
        legend(ax, 'Location', 'northeast', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none', 'FontSize', 7);
    end
    hold(ax, 'off');
end

fprintf('\n[Done] All 4 experiments analyzed and plotted.\n');
