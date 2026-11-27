% experiments/exp_spam_roc.m
% Replicates ESL Figure 9.6: ROC Curves for Spam Data
% 
% DATA-DRIVEN REPLICATION:
% 1. Standard Tree: Size selected via 10-fold CV (1-SE Rule).
% 2. Weighted Tree: Loss L01=5, Size selected via 10-fold CV (1-SE Rule).
% 3. GAM: Internal OceanStats.GAM baseline.
% 4. Metrics: Internal OceanStats.Metrics.ROC.

import OceanStats.Supervised.Tree.*
import OceanStats.Selection.CrossValidation.*
import OceanStats.Selection.Strategies.*
import OceanStats.Metrics.*
import OceanViz.*

% --- 1. Load Data ---
fprintf('Loading Spambase dataset...\n');
currentDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(currentDir, 'Data', 'spambase');
for i = 1:3
    if isfolder(dataDir), break; end
    currentDir = fileparts(currentDir);
    dataDir = fullfile(currentDir, 'Data', 'spambase');
end
dataFile = fullfile(dataDir, 'spambase.data');
if ~isfile(dataFile), error('Spambase data not found.'); end

raw = load(dataFile);
X_raw = raw(:, 1:end-1);
Y = raw(:, end); 

% Transform features as per GAM experiment (log x + 0.1)
% This helps GAM performance significantly.
X = log(X_raw + 0.1);

% Split (ESL: 1536 test samples)
rng(42); 
cvObj = cvpartition(Y, 'HoldOut', 1536/length(Y));
X_train = X(training(cvObj), :);
Y_train = Y(training(cvObj));
X_test  = X(test(cvObj), :);
Y_test  = Y(test(cvObj));

fprintf('Train Size: %d, Test Size: %d\n', length(Y_train), length(Y_test));

cv = CrossValidator('PartitionStrategy', 'KFold', 'NumFolds', 10, 'UseParallel', true);

% --- Step 1: Standard Tree ---
fprintf('\n--- Step 1: Standard Tree (CV Selection) ---\n');
treeStd = DecisionTree('Criterion', 'entropy', 'MinLeafSize', 10, 'MaxDepth', 30);
treeStd.fit(X_train, Y_train);
alphas = treeStd.weakestLinkSequence();

% Find best size via CV
cv_errors = zeros(size(alphas));
cv_se = zeros(size(alphas));
tree_sizes = zeros(size(alphas));

fprintf('Running CV for Standard Tree...\n');
for i = 1:length(alphas)
    strategy = OceanStats.Selection.Strategies.DecisionTree(10, 30, 'entropy');
    strategy.PruneAlpha = alphas(i);
    results = cv.evaluate(strategy, X_train, Y_train);
    cv_errors(i) = results.MeanError;
    cv_se(i)     = results.StdError;
    
    pT = treeStd.prune(alphas(i));
    tree_sizes(i) = pT.countTerminalNodes(pT.RootNode);
end

[minErr, minIdx] = min(cv_errors);
threshold = minErr + cv_se(minIdx);
best_idx = find(cv_errors <= threshold, 1, 'last'); % Smallest tree within 1 SE
bestAlpha = alphas(best_idx);

treeStdFinal = treeStd.prune(bestAlpha);
szStd = treeStdFinal.countTerminalNodes(treeStdFinal.RootNode);
fprintf('Standard Tree Selection: Size %d (Alpha %.4f)\n', szStd, bestAlpha);

scoresStd = treeStdFinal.predictProb(X_test);
[fprS, tprS, ~, aucS] = ROC.compute(Y_test, scoresStd(:, 2), 1);
fprintf('Standard Tree AUC: %.4f\n', aucS);


% --- Step 2: Weighted Tree ---
fprintf('\n--- Step 2: Weighted Tree (CV Selection) ---\n');
% L(0,1)=1 (FN), L(1,0)=5 (FP)
L = [0, 1; 5, 0]; 
treeW = DecisionTree('Criterion', 'entropy', 'MinLeafSize', 10, 'MaxDepth', 30, 'LossMatrix', L);
treeW.fit(X_train, Y_train);
alphasW = treeW.weakestLinkSequence();

cv_errorsW = zeros(size(alphasW));
cv_seW = zeros(size(alphasW));

fprintf('Running CV for Weighted Tree...\n');
for i = 1:length(alphasW)
    % IMPORTANT: Pass LossMatrix to the strategy so CV uses cost-sensitive pruning
    strategy = OceanStats.Selection.Strategies.DecisionTree(10, 30, 'entropy');
    strategy.PruneAlpha = alphasW(i);
    strategy.LossMatrix = L; 
    
    results = cv.evaluate(strategy, X_train, Y_train);
    cv_errorsW(i) = results.MeanError;
    cv_seW(i)     = results.StdError;
end

[minErrW, minIdxW] = min(cv_errorsW);
threshW = minErrW + cv_seW(minIdxW);
best_idxW = find(cv_errorsW <= threshW, 1, 'last');
bestAlphaW = alphasW(best_idxW);

treeWFinal = treeW.prune(bestAlphaW);
szW = treeWFinal.countTerminalNodes(treeWFinal.RootNode);
fprintf('Weighted Tree Selection: Size %d\n', szW);

scoresW = treeWFinal.predictProb(X_test);
[fprW, tprW, ~, aucW] = ROC.compute(Y_test, scoresW(:, 2), 1);
fprintf('Weighted Tree AUC: %.4f\n', aucW);


% --- Step 3: Internal GAM Baseline ---
fprintf('\n--- Step 3: GAM Baseline (Internal) ---\n');
import OceanStats.GAM.*
P = size(X_train, 2);
strategies = cell(1, P);
for j = 1:P
    x_col = X_train(:, j);
    knots = linspace(min(x_col), max(x_col), 20); % Fewer knots for speed in ROC test
    strategies{j} = SplineSmoother(knots, 0.1, [min(x_col), max(x_col)]);
    strategies{j}.TargetDF = 4;
end

gamObj = AdditiveModel(strategies);
optimizer = LocalScoring('Binomial');
optimizer.MaxIter = 5; % Reduced for faster ROC verification
optimizer.Tol = 1e-3;

fprintf('Fitting Internal GAM (Fast Mode: MaxIter=5, Tol=1e-3)...\n');
gam_model = optimizer.fit(gamObj, X_train, Y_train);

logit_pred = gam_model.predict(X_test);
scoresGAM = 1 ./ (1 + exp(-logit_pred));

[fprG, tprG, ~, aucG] = ROC.compute(Y_test, scoresGAM, 1);
fprintf('Internal GAM AUC: %.4f\n', aucG);


% --- Visualization ---
fprintf('\nPlotting ROC Curves (Internal Metrics)...\n');
viz = OceanViz.Core(Theme="Dark");
hFig = figure('Name', 'Spam ROC Analysis', 'Color', viz.Colors.Background, 'InvertHardcopy', 'off');
ax = axes('Parent', hFig);
viz.styleAxes(ax);
hold(ax, 'on');

colorStd = [0.2, 0.8, 0.4]; % Green
colorW   = [0.9, 0.6, 0.0]; % Orange
colorGAM = [0.2, 0.6, 1.0]; % Blue

plot(ax, fprS, tprS, '.-', 'Color', colorStd, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Tree (AUC %.2f)', aucS));

plot(ax, fprW, tprW, '.-', 'Color', colorW, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Weighted Tree (AUC %.2f)', aucW));

plot(ax, fprG, tprG, '-', 'Color', colorGAM, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('GAM (AUC %.2f)', aucG));

% Diagonal reference
plot(ax, [0, 1], [0, 1], '--', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');

xlabel(ax, 'False Positive Rate');
ylabel(ax, 'True Positive Rate (Sensitivity)');
title(ax, 'ROC Curves: Automated Selection & Internal Metrics');
legend(ax, 'Location', 'southeast', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none');
grid(ax, 'on');

% Save
outFile = fullfile(pwd, 'spam_roc_internal.png');
saveas(hFig, outFile);
fprintf('Plot saved to %s\n', outFile);
