% EXP_DEMOGRAPHICS_GBM  Replicates ESL §10.14.3 — Demographics Data.
%
%   PURPOSE:    Use MART (Multiple Additive Regression Trees) for K=9
%               multiclass classification on shopping-mall survey data.
%               Replicates Figures 10.22–10.25 from Hastie, Tibshirani &
%               Friedman, "Elements of Statistical Learning" (2nd ed.).
%
%   DATASET:    Demographics Data (Impact Resources, Inc., Columbus OH)
%               Source: Data/Demographics/Demographis_raw.txt
%               N ~8900 (space-separated, header row, NAs imputed)
%               p = 13 predictors, K = 9 occupation classes
%
%   MODEL:      MART — MulticlassDevianceLoss (softmax), one regression
%               tree per class per boosting round (Algorithm 10.4, ESL).
%
%   ASSUMPTIONS:
%     A1. NAs in predictors are filled with column-wise mode (most frequent
%         integer value). Rows with NA in the response (occup) are dropped.
%         This matches more closely the textbook's N~9243 sample count.
%     A2. 80% train / 20% test, stratified by class, rng(42).
%     A3. J=6 terminal nodes → MaxDepth=ceil(log2(J))=3; nu=0.1; M=800.
%     A4. Variable importance = sum of RSS reduction at each split,
%         averaged across all trees and classes, normalized to 100.
%     A5. Partial dependence uses 500-sample subsample from training set.
%
%   REPRODUCIBILITY: rng(42, 'twister')

clc; clear; close all;
rng(42, 'twister');

%% === [1] Data Loading ===

scriptDir = fileparts(mfilename('fullpath'));
lvl1      = fileparts(scriptDir);   % Scripts/
lvl2      = fileparts(lvl1);        % Experiments/
rootDir   = fileparts(lvl2);        % repo root

dataPath = fullfile(rootDir, 'Data', 'Demographics', 'Demographis_raw.txt');

assert(isfile(dataPath), 'EXP:DataNotFound', ...
    'Data file not found:\n  %s', dataPath);

% Read space-separated table with header; 'NA' → NaN automatically
opts           = detectImportOptions(dataPath, 'FileType', 'text');
opts.Delimiter = ' ';
opts           = setvaropts(opts, opts.VariableNames, 'TreatAsMissing', 'NA');
T              = readtable(dataPath, opts);

fprintf('[1] Loaded %d rows, %d columns\n', height(T), width(T));

%% === [2] Feature Extraction & NA Handling (Mode Imputation) ===

% Column layout (from header):
%   income sex mar-stat age edu occup yrs-BA mar-dlinc num-hsld
%   children hsld-stat typ-home ethnic lang
%
% Response: col 6 (occup), classes 1..9
% Predictors: all other 13 columns

colNames = T.Properties.VariableNames;     % 14 names
respCol  = 6;                              % occup

% Build feature matrix and response vector (all as doubles)
X_raw = table2array(T);                    % N_raw × 14
Y_raw = X_raw(:, respCol);               % N_raw × 1
X_raw(:, respCol) = [];                   % remove response → N_raw × 13

featNames = colNames;
featNames(respCol) = [];                  % 13 feature names

% Drop rows where response (occup) is invalid — keep all others
respValid = ~isnan(Y_raw) & (Y_raw >= 1) & (Y_raw <= 9) & (Y_raw == floor(Y_raw));
X_raw = X_raw(respValid, :);
Y_raw = Y_raw(respValid);

% Mode-impute each predictor column (fills NAs with most frequent integer value)
% This matches textbook more closely (N~9243 vs our 8993→6876 with row-drop).
for j = 1:size(X_raw, 2)
    col    = X_raw(:, j);
    nanIdx = isnan(col);
    if any(nanIdx)
        vals      = col(~nanIdx);
        colMode   = mode(vals);            % most frequent value
        col(nanIdx) = colMode;
        X_raw(:, j) = col;
    end
end

X = X_raw;
Y = Y_raw;
N = size(X, 1);
K = numel(unique(Y));   % should be 9

fprintf('[2] N=%d, p=%d, K=%d classes (mode-imputed NAs, dropped %d invalid resp rows)\n', ...
    N, size(X,2), K, sum(~respValid));
assert(K == 9, 'EXP:WrongClasses', 'Expected K=9 occupation classes, got %d.', K);

% Print class distribution
classCounts = histcounts(Y, 0.5:(K+0.5))';
% Class labels indexed by their numeric code (1..9), from data.info.txt:
%   1=Prof/Man, 2=Sales, 3=Labor, 4=Clerical, 5=Homemaker,
%   6=Student, 7=Military, 8=Retired, 9=Unemployed
classLabels = {'Prof/Man','Sales','Labor','Clerical','Homemaker', ...
               'Student','Military','Retired','Unemployed'};
fprintf('[2] Class distribution:\n');
for k = 1:K
    fprintf('    %d (%s): %d samples (%.1f%%)\n', k, classLabels{k}, ...
        classCounts(k), 100*classCounts(k)/N);
end

%% === [3] Stratified 80/20 Train / Test Split ===

% Split within each class so each class proportion is preserved
trainMask = false(N, 1);
rng(42, 'twister');
for k = 1:K
    classIdx  = find(Y == k);
    nTr       = round(0.80 * numel(classIdx));
    selPerm   = randperm(numel(classIdx), nTr);
    trainMask(classIdx(selPerm)) = true;
end

X_train = X(trainMask, :);   Y_train = Y(trainMask);
X_test  = X(~trainMask, :);  Y_test  = Y(~trainMask);

N_tr = sum(trainMask);
N_te = N - N_tr;
fprintf('[3] N_train=%d, N_test=%d\n', N_tr, N_te);

% Null error rate: predict most-frequent class for all test samples
[~, modeClass] = max(histcounts(Y_train, 0.5:(K+0.5)));
nullErr = mean(Y_test ~= modeClass);
fprintf('[3] Null error rate (most-frequent class): %.1f%%\n', nullErr*100);

%% === [4] Model Configuration ===

import OceanStats.Loss.Classification.MulticlassDevianceLoss
import OceanStats.Boosting.Algorithms.GradientBoostingMachine

M   = 800;    % boosting rounds
nu  = 0.10;   % learning rate (shrinkage)
J   = 6;      % target terminal nodes
% MaxDepth: for J terminal nodes in a balanced binary tree, depth = ceil(log2(J))
% J=6 → MaxDepth=3 (gives trees with up to 8 leaves, typically ~6 in practice)
maxD = ceil(log2(J));   % = 3
mls  = 10;              % min leaf size

loss = MulticlassDevianceLoss();

gbm = GradientBoostingMachine( ...
    'Loss',         loss, ...
    'NumTrees',     M, ...
    'MaxDepth',     maxD, ...
    'LearningRate', nu, ...
    'MinLeafSize',  mls);

fprintf('[4] MART: K=%d, M=%d, J=%d (MaxDepth=%d), nu=%.2f, MinLeafSize=%d\n', ...
    K, M, J, maxD, nu, mls);

%% === [5] Training (timed) ===

fprintf('\n[5] Training MART (%d classes × %d rounds = %d trees)...\n', K, M, K*M);
tStart = tic;

gbm.fit(X_train, Y_train);

elapsed = toc(tStart);
fprintf('[5] Training complete in %.1f seconds (%.1f min)\n', elapsed, elapsed/60);

%% === [6] Evaluation — Per-Class Error Rates (ESL Figure 10.22) ===

fprintf('[6] Evaluating on test set...\n');

% Predict: returns N_te × K probability matrix
P_test   = gbm.predict(X_test);          % N_te × K
[~, Y_hat] = max(P_test, [], 2);          % N_te × 1 predicted class

% Overall error rate
overallErr = mean(Y_hat ~= Y_test);
fprintf('[6] Overall test error rate: %.1f%% (textbook target: ~42.5%%)\n', overallErr*100);

% Per-class error rates
classErr = zeros(K, 1);
for k = 1:K
    mask = (Y_test == k);
    if sum(mask) == 0, continue; end
    classErr(k) = mean(Y_hat(mask) ~= k);
end

% Sort by error rate (ascending) so best class is at top of bar chart
[sortedErr, sIdx] = sort(classErr, 'ascend');
sortedNames = classLabels(sIdx);

fprintf('[6] Per-class error rates:\n');
for k = 1:K
    fprintf('    Class %d (%s): %.1f%%\n', sIdx(k), classLabels{sIdx(k)}, sortedErr(k)*100);
end

%% === [7] Variable Importance (ESL Figures 10.23 & 10.24) ===
% Average relative influence over all K classes and all M trees.
% For each tree, each internal split contributes its RSS reduction (gain).
% Importance_j = mean over all trees and classes of gain at splits using feature j.

fprintf('[7] Computing variable importance...\n');

p = size(X_train, 2);
[M_trees, K_trees] = size(gbm.Trees_);

% Overall importance via built-in (averages across all trees and classes)
% featureImportance() returns p×1 normalized to sum=1; scale to max=100.
impAll_frac = gbm.featureImportance(p);
impAll = 100 * impAll_frac / max(impAll_frac + eps);

% Per-class importance: accumulate gain for each class column separately
impPerClass = zeros(p, K);
for k = 1:K_trees
    for m = 1:M_trees
        t = gbm.Trees_{m, k};
        fi_mk = gatherImportance(t.RootNode, p);   % p×1
        impPerClass(:, k) = impPerClass(:, k) + fi_mk;
    end
end
for k = 1:K
    impPerClass(:, k) = 100 * impPerClass(:, k) / max(impPerClass(:, k) + eps);
end

% Sort ascending for bar chart (most important at top)
[sortedImpAll, iIdx] = sort(impAll, 'ascend');
sortedFeatAll = featNames(iIdx);

fprintf('[7] Top 3 predictors (averaged): %s, %s, %s\n', ...
    featNames{iIdx(end)}, featNames{iIdx(end-1)}, featNames{iIdx(end-2)});

%% === [8] Partial Dependence on AGE (ESL Figure 10.25) ===
% Show log-odds of 3 classes (Retired, Student, Prof/Man) vs `age`.
% Partial dependence: PD_k(x_j) = (1/N_ss) sum_i F_k(x_i | x_j fixed)
% where F_k is the raw additive score for class k (before softmax).

fprintf('[8] Computing partial dependence on age...\n');

% Find the `age` column index
ageCol = find(strcmpi(featNames, 'age'));
assert(~isempty(ageCol), 'EXP:NoAge', 'Could not find "age" column.');

nPDGrid  = 50;
nSubsamp = min(500, N_tr);
rng(42, 'twister');
ssIdx = randperm(N_tr, nSubsamp);
X_ss  = X_train(ssIdx, :);

ageGrid = (1:7)';    % age is coded 1..7 in this dataset (equally spaced intervals)

% Classes to show (index = numeric occup code, matching classLabels above):
%   Retired=8, Student=6, Prof/Man=1
pdClasses = [8, 6, 1];
pdClassNames = {'Retired', 'Student', 'Prof/Man'};
nPDClasses = numel(pdClasses);

pdLogOdds = zeros(numel(ageGrid), nPDClasses);

for ci = 1:nPDClasses
    classK = pdClasses(ci);
    pd_k   = zeros(numel(ageGrid), 1);

    for gi = 1:numel(ageGrid)
        X_fixed = X_ss;
        X_fixed(:, ageCol) = ageGrid(gi);

        % Get raw scores for all K classes (N_ss × K)
        P_fixed = gbm.predict(X_fixed);   % N_ss × K (probabilities)

        % Convert to log-odds for this class: log(p_k / (1 - p_k))
        p_k = max(P_fixed(:, classK), 1e-8);
        logOdds_k = log(p_k ./ max(1 - p_k, 1e-8));
        pd_k(gi)  = mean(logOdds_k);
    end

    % Center at zero (ESL convention)
    pdLogOdds(:, ci) = pd_k - mean(pd_k);
end

fprintf('[8] Partial dependence computed.\n');

%% === [9] Visualization ===

fprintf('[9] Plotting figures...\n');
viz = OceanViz.Core('Theme', 'Dark');
C   = viz.Colors;

% -------------------------------------------------------------------------
% Figure 10.22 — Per-class error rates
% -------------------------------------------------------------------------
fig1 = viz.createFigure('ESL Fig 10.22 — Error Rate per Occupation Class');
fig1.Position = [50 80 680 450];
ax1  = axes('Parent', fig1);
viz.styleAxes(ax1);

barh(ax1, 1:K, sortedErr, 'FaceColor', [0.2 0.8 0.2], 'EdgeColor', 'none');
ax1.YTick      = 1:K;
ax1.YTickLabel = sortedNames;
xlim(ax1, [0, 1.05]);
xlabel(ax1, 'Error Rate', 'Color', C.Text, 'FontSize', 10);
title(ax1, 'Error Rate per Occupation (ESL Fig 10.22)', ...
    'Color', C.Text, 'FontSize', 11, 'FontWeight', 'bold');
xline(ax1, overallErr, 'Color', [1 1 0], 'LineStyle', '--', 'LineWidth', 1.5, ...
    'Label', sprintf('Overall=%.1f%%', overallErr*100), 'LabelColor', [1 1 0]);
ax1.XGrid = 'on';

% -------------------------------------------------------------------------
% Figure 10.23 — Relative predictor importance (averaged over all classes)
% -------------------------------------------------------------------------
fig2 = viz.createFigure('ESL Fig 10.23 — Relative Predictor Importance (Averaged)');
fig2.Position = [80 80 700 480];
ax2  = axes('Parent', fig2);
viz.styleAxes(ax2);

barh(ax2, 1:p, sortedImpAll, 'FaceColor', [0.9 0.2 0.2], 'EdgeColor', 'none');
ax2.YTick      = 1:p;
ax2.YTickLabel = sortedFeatAll;
xlim(ax2, [0, 115]);
xlabel(ax2, 'Relative Importance', 'Color', C.Text, 'FontSize', 10);
title(ax2, 'Relative Importance of Predictors — Averaged (ESL Fig 10.23)', ...
    'Color', C.Text, 'FontSize', 11, 'FontWeight', 'bold');
ax2.XGrid = 'on';

% -------------------------------------------------------------------------
% Figure 10.24 — Per-class variable importance for 4 best-predicted classes
% -------------------------------------------------------------------------
% Best 4 = lowest error classes
[~, bestClassIdx] = sort(classErr, 'ascend');
best4 = bestClassIdx(1:4);

fig3 = viz.createFigure('ESL Fig 10.24 — Predictor Importance per Class');
fig3.Position = [100 50 1050 700];

for bi = 1:4
    k = best4(bi);

    % Sort importance for this class
    [sortedImpK, iIdxK] = sort(impPerClass(:, k), 'ascend');
    sortedFeatK = featNames(iIdxK);

    ax = subplot(2, 2, bi, 'Parent', fig3);
    viz.styleAxes(ax);
    barh(ax, 1:p, sortedImpK, 'FaceColor', [0.9 0.2 0.2], 'EdgeColor', 'none');
    ax.YTick      = 1:p;
    ax.YTickLabel = sortedFeatK;
    xlim(ax, [0, 115]);
    xlabel(ax, 'Relative Importance', 'FontSize', 8);
    title(ax, sprintf('Class = %s', classLabels{k}), 'FontSize', 9, 'FontWeight', 'bold');
    ax.XGrid = 'on';
end
sgtitle(fig3, 'Predictor Importance Distributions by Class (ESL Fig 10.24)', ...
    'Color', C.Text, 'FontSize', 11);

% -------------------------------------------------------------------------
% Figure 10.25 — Partial dependence on age for 3 classes
% -------------------------------------------------------------------------
fig4 = viz.createFigure('ESL Fig 10.25 — Partial Dependence on Age');
fig4.Position = [120 80 900 380];

% Use 3 distinct colors — Accent not available in all themes
lineColors = {C.Primary, C.Secondary, [0.2 0.95 0.5]};

for ci = 1:nPDClasses
    ax = subplot(1, 3, ci, 'Parent', fig4);
    viz.styleAxes(ax);
    hold(ax, 'on');

    plot(ax, ageGrid, pdLogOdds(:, ci), ...
        'Color', lineColors{ci}, 'LineWidth', 2.0);

    % Red decile tick marks at bottom (ESL convention)
    yLims = ylim(ax);
    tickH = 0.04 * (yLims(2) - yLims(1) + 0.1);
    deciles = prctile(X_train(:, ageCol), 10:10:90);
    for dk = 1:numel(deciles)
        line(ax, [deciles(dk) deciles(dk)], [yLims(1), yLims(1)+tickH], ...
            'Color', [0.9 0.2 0.2], 'LineWidth', 1.5);
    end

    yline(ax, 0, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 0.8);
    xlabel(ax, 'age',              'FontSize', 9);
    ylabel(ax, 'Partial Dependence', 'FontSize', 9);
    title(ax, pdClassNames{ci},   'FontSize', 10, 'FontWeight', 'bold');
    xticks(ax, 1:7);
    hold(ax, 'off');
end
sgtitle(fig4, 'Partial Dependence on Age (ESL Fig 10.25)', ...
    'Color', C.Text, 'FontSize', 11);

fprintf('[9] All figures plotted.\n');

%% === [10] Summary Printout ===

fprintf('\n=== EXPERIMENT SUMMARY (ESL §10.14.3 — Demographics GBM) ===\n');
fprintf('  Dataset       : Demographics (N=%d, p=%d, K=%d classes)\n', N, p, K);
fprintf('  Model         : MART — MulticlassDevianceLoss\n');
fprintf('  Rounds M      : %d  |  Trees total : %d (M×K)\n', M, M*K);
fprintf('  Learning Rate : %.2f |  MaxDepth: %d  |  MinLeafSize: %d\n', nu, J-1, mls);
fprintf('  Training Time : %.1f seconds (%.1f min)\n', elapsed, elapsed/60);
fprintf('  ---\n');
fprintf('  Overall Error : %.1f%%  (textbook target: ~42.5%%)\n', overallErr*100);
fprintf('  Null Error    : %.1f%%  (textbook target: ~69%%)\n',    nullErr*100);
fprintf('  ---\n');
fprintf('  Top Predictor : %s (avg. importance = 100)\n', featNames{iIdx(end)});
fprintf('================================================================\n');

%% === LOCAL HELPER FUNCTION ===
% Recursively sums RSS reduction per feature across a tree.

function fi = gatherImportance(node, p)
    fi = zeros(p, 1);
    if node.IsLeaf, return; end
    % node.Split is a SplitResult object with .FeatureIndex property
    featJ = node.Split.FeatureIndex;
    if ~isnan(featJ) && featJ >= 1 && featJ <= p
        fi(featJ) = fi(featJ) + node.Risk;
    end
    for c = 1:length(node.Children)
        fi = fi + gatherImportance(node.Children{c}, p);
    end
end
