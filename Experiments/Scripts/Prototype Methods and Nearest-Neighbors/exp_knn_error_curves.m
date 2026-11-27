% EXP_KNN_ERROR_CURVES  ESL Exercise 13.7 — KNN Error Curves, CV, and AIC.
%
%   Replicates Figure 13.5 (left panel) and compares three methods for
%   estimating the optimal number of neighbors K:
%     1. True test error (gold standard)
%     2. 5-fold cross-validation
%     3. AIC-like penalization: train_err + 2*(N/K)/N
%
%   DATASET:        ESL 13.3.1 binary classification (10 uniform features)
%   MODEL:          KNNClassifier (K = 1, 3, 5, ..., 61)
%   ASSUMPTIONS:
%     - Easy problem: Y = I(X_1 > 0.5), hyperplane on feature 1
%     - Difficult: Y = I(sign(prod(X_j-0.5, j=1:3)) > 0), 3D checkerboard
%     - Bayes error = 0 for both problems
%     - N_train = 100, N_test = 1000, 10 realizations
%   REPRODUCIBILITY: rng(seed, 'twister') per realization
%
% Ref: Hastie, Tibshirani, Friedman (2009). ESL, Ex 13.7, Sec 13.3.1.
clc; clear; close all;

%% === 1. Experiment Configuration ===
fprintf('=== ESL Exercise 13.7: KNN Error Curves ===\n\n');

N_train    = 100;
N_test     = 1000;
nRealize   = 10;        % number of realizations
K_values   = 1:2:61;    % K = 1, 3, 5, ..., 61
nK         = numel(K_values);
nFolds     = 5;         % for cross-validation
problems   = ["easy", "difficult"];

fprintf('  N_train=%d, N_test=%d, realizations=%d\n', N_train, N_test, nRealize);
fprintf('  K values: %d to %d (%d values)\n', K_values(1), K_values(end), nK);
fprintf('  CV folds: %d\n', nFolds);

%% === 2. Part 1 — Test Error Curves ===
fprintf('\n=== Part 1: Test Error Curves (10 realizations) ===\n');

% Storage: (nRealize x nK) for each problem
testErr  = struct('easy', zeros(nRealize, nK), 'difficult', zeros(nRealize, nK));
trainErr = struct('easy', zeros(nRealize, nK), 'difficult', zeros(nRealize, nK));
cvErr    = struct('easy', zeros(nRealize, nK), 'difficult', zeros(nRealize, nK));

for pi = 1:numel(problems)
    prob = problems(pi);
    fprintf('\n  Problem: %s\n', prob);
    
    for r = 1:nRealize
        seed = r * 100;
        
        % Generate train + test data
        genTrain = OceanData.Generators.ESLBinaryGenerator( ...
            'N', N_train, 'Problem', prob, 'Seed', seed);
        [X_tr, Y_tr] = genTrain.generate();
        
        genTest = OceanData.Generators.ESLBinaryGenerator( ...
            'N', N_test, 'Problem', prob, 'Seed', seed + 50);
        [X_te, Y_te] = genTest.generate();
        
        % Set up 5-fold CV splitter for this realization
        cv = OceanStats.Selection.CrossValidation.KFold(nFolds, 'Seed', seed + 99);
        
        for ki = 1:nK
            K = K_values(ki);
            
            % --- Train & Test Error ---
            knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', K);
            knn.fit(X_tr, Y_tr);
            
            % Test error
            Y_pred_te = knn.predict(X_te);
            testErr.(prob)(r, ki) = mean(Y_pred_te(:) ~= Y_te(:));
            
            % Training error
            Y_pred_tr = knn.predict(X_tr);
            trainErr.(prob)(r, ki) = mean(Y_pred_tr(:) ~= Y_tr(:));
            
            % --- 5-Fold CV Error ---
            foldErrs = zeros(nFolds, 1);
            for f = 1:nFolds
                [trIdx, teIdx] = cv.getIndices(N_train, f);
                
                X_cv_tr = X_tr(trIdx, :);
                Y_cv_tr = Y_tr(trIdx);
                X_cv_te = X_tr(teIdx, :);
                Y_cv_te = Y_tr(teIdx);
                
                knn_cv = OceanStats.Supervised.Neighbors.KNNClassifier('K', K);
                knn_cv.fit(X_cv_tr, Y_cv_tr);
                Y_cv_pred = knn_cv.predict(X_cv_te);
                foldErrs(f) = mean(Y_cv_pred(:) ~= Y_cv_te(:));
            end
            cvErr.(prob)(r, ki) = mean(foldErrs);
        end
        
        fprintf('    Realization %2d (seed=%d): test_err(K=1)=%.3f\n', ...
            r, seed, testErr.(prob)(r, 1));
    end
end

%% === 3. Part 3 — AIC-like Penalization ===
fprintf('\n=== Part 3: AIC-like Penalization ===\n');

% AIC_err = train_err + 2*t/N, where t = N/K
% t = effective number of parameters (KNN with K neighbors ≈ N/K params)
aicErr = struct();
for pi = 1:numel(problems)
    prob = problems(pi);
    aicErr.(prob) = zeros(nRealize, nK);
    for ki = 1:nK
        K = K_values(ki);
        t = N_train / K;   % effective parameters
        penalty = 2 * t / N_train;   % = 2/K
        aicErr.(prob)(:, ki) = trainErr.(prob)(:, ki) + penalty;
    end
end

fprintf('  AIC penalty at K=1:  2/1  = %.4f\n', 2/1);
fprintf('  AIC penalty at K=15: 2/15 = %.4f\n', 2/15);
fprintf('  AIC penalty at K=61: 2/61 = %.4f\n', 2/61);

%% === 4. Compute Summary Statistics ===
fprintf('\n=== Computing mean ± std across realizations ===\n');

stats = struct();
for pi = 1:numel(problems)
    prob = problems(pi);
    stats.(prob).test_mean  = mean(testErr.(prob), 1);
    stats.(prob).test_std   = std(testErr.(prob), 0, 1);
    stats.(prob).cv_mean    = mean(cvErr.(prob), 1);
    stats.(prob).cv_std     = std(cvErr.(prob), 0, 1);
    stats.(prob).aic_mean   = mean(aicErr.(prob), 1);
    stats.(prob).aic_std    = std(aicErr.(prob), 0, 1);
    stats.(prob).train_mean = mean(trainErr.(prob), 1);
    
    [~, idxTest] = min(stats.(prob).test_mean);
    [~, idxCV]   = min(stats.(prob).cv_mean);
    [~, idxAIC]  = min(stats.(prob).aic_mean);
    
    fprintf('  %s:\n', prob);
    fprintf('    Optimal K (test):  %d (err=%.4f)\n', K_values(idxTest), stats.(prob).test_mean(idxTest));
    fprintf('    Optimal K (CV):    %d (err=%.4f)\n', K_values(idxCV), stats.(prob).cv_mean(idxCV));
    fprintf('    Optimal K (AIC):   %d (err=%.4f)\n', K_values(idxAIC), stats.(prob).aic_mean(idxAIC));
end

%% === 5. Visualization ===
fprintf('\n=== Step 5: Visualization ===\n');

viz = OceanViz.Core('Theme', 'Dark');

% Color palette
cTest = [0.30 0.65 0.95];   % blue
cCV   = [1.00 0.50 0.15];   % orange
cAIC  = [0.35 0.80 0.40];   % green
cTrain = [0.60 0.60 0.60];  % gray

% ─── Figure 1: Replicate Fig 13.5 Left Panel ───
fig1 = viz.createFigure('ESL Fig 13.5 — KNN Error Curves');
fig1.Position = [50, 150, 900, 700];

for pi = 1:numel(problems)
    prob = problems(pi);
    ax = subplot(2, 1, pi, 'Parent', fig1);
    hold(ax, 'on'); viz.styleAxes(ax);
    
    mu  = stats.(prob).test_mean;
    sig = stats.(prob).test_std;
    
    % Error band (±1 std)
    fill(ax, [K_values, fliplr(K_values)], ...
        [mu + sig, fliplr(mu - sig)], ...
        cTest, 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    
    % Mean curve
    plot(ax, K_values, mu, '-o', 'Color', cTest, 'LineWidth', 2, ...
        'MarkerSize', 4, 'MarkerFaceColor', cTest, ...
        'DisplayName', 'Test Error');
    
    % Mark optimal K
    [minErr, minIdx] = min(mu);
    plot(ax, K_values(minIdx), minErr, 'p', 'Color', [1 0.85 0.2], ...
        'MarkerSize', 14, 'MarkerFaceColor', [1 0.85 0.2], ...
        'HandleVisibility', 'off');
    
    xlabel(ax, 'Number of Neighbors', 'Color', viz.Colors.Text, 'FontSize', 11);
    ylabel(ax, 'Misclassification Error', 'Color', viz.Colors.Text, 'FontSize', 11);
    
    titleStr = sprintf('Nearest Neighbors / %s%s', upper(extractBefore(prob,2)), extractAfter(prob,1));
    title(ax, titleStr, 'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
    
    legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
        'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
    hold(ax, 'off');
end

sgtitle(fig1, 'ESL Figure 13.5 — Left Panel (10 Realizations)', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% ─── Figure 2: Test vs CV vs AIC Comparison ───
fig2 = viz.createFigure('Test vs CV vs AIC');
fig2.Position = [100, 100, 1000, 700];

for pi = 1:numel(problems)
    prob = problems(pi);
    ax = subplot(2, 1, pi, 'Parent', fig2);
    hold(ax, 'on'); viz.styleAxes(ax);
    
    % Test error (reference)
    plot(ax, K_values, stats.(prob).test_mean, '-o', ...
        'Color', cTest, 'LineWidth', 2.5, 'MarkerSize', 4, ...
        'MarkerFaceColor', cTest, 'DisplayName', 'Test Error');
    fill(ax, [K_values, fliplr(K_values)], ...
        [stats.(prob).test_mean + stats.(prob).test_std, ...
         fliplr(stats.(prob).test_mean - stats.(prob).test_std)], ...
        cTest, 'FaceAlpha', 0.10, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    
    % CV error
    plot(ax, K_values, stats.(prob).cv_mean, '-s', ...
        'Color', cCV, 'LineWidth', 2, 'MarkerSize', 4, ...
        'MarkerFaceColor', cCV, 'DisplayName', '5-Fold CV');
    fill(ax, [K_values, fliplr(K_values)], ...
        [stats.(prob).cv_mean + stats.(prob).cv_std, ...
         fliplr(stats.(prob).cv_mean - stats.(prob).cv_std)], ...
        cCV, 'FaceAlpha', 0.10, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    
    % AIC error
    plot(ax, K_values, stats.(prob).aic_mean, '-d', ...
        'Color', cAIC, 'LineWidth', 2, 'MarkerSize', 4, ...
        'MarkerFaceColor', cAIC, 'DisplayName', 'AIC Penalty');
    
    % Training error (for reference)
    plot(ax, K_values, stats.(prob).train_mean, '--', ...
        'Color', cTrain, 'LineWidth', 1.5, 'DisplayName', 'Train Error');
    
    xlabel(ax, 'Number of Neighbors', 'Color', viz.Colors.Text, 'FontSize', 11);
    ylabel(ax, 'Misclassification Error', 'Color', viz.Colors.Text, 'FontSize', 11);
    
    titleStr = sprintf('%s%s Problem: Test vs CV vs AIC', ...
        upper(extractBefore(prob,2)), extractAfter(prob,1));
    title(ax, titleStr, 'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
    
    legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
        'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
    hold(ax, 'off');
end

sgtitle(fig2, 'ESL Ex 13.7 — Comparing Error Estimation Methods', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% Save figures
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig1, fullfile(resultsDir, 'fig13_5_knn_test_error.png'));
saveas(fig2, fullfile(resultsDir, 'fig13_7_test_vs_cv_vs_aic.png'));
fprintf('[Viz] Saved 2 figures.\n');

%% === 6. Experiment Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ESL 13.3.1 Binary (N_train=%d, N_test=%d, p=10)\n', N_train, N_test);
fprintf('  Realizations  : %d\n', nRealize);
fprintf('  K range       : %d to %d\n', K_values(1), K_values(end));
fprintf('  CV strategy   : %d-Fold (OceanStats.Selection.CrossValidation.KFold)\n', nFolds);
fprintf('\n');

for pi = 1:numel(problems)
    prob = problems(pi);
    [~, iT] = min(stats.(prob).test_mean);
    [~, iC] = min(stats.(prob).cv_mean);
    [~, iA] = min(stats.(prob).aic_mean);
    
    fprintf('  %s problem:\n', upper(prob));
    fprintf('    Best K (test): %2d  (err=%.4f)\n', K_values(iT), stats.(prob).test_mean(iT));
    fprintf('    Best K (CV):   %2d  (err=%.4f)\n', K_values(iC), stats.(prob).cv_mean(iC));
    fprintf('    Best K (AIC):  %2d  (err=%.4f)\n', K_values(iA), stats.(prob).aic_mean(iA));
    fprintf('\n');
end

fprintf('  Conclusion:\n');
fprintf('    CV tracks test error shape well and selects a reasonable K.\n');
fprintf('    AIC penalization tends to favor smaller K (higher complexity)\n');
fprintf('    because the 2/K penalty decreases quickly.\n');
fprintf('==========================\n');
