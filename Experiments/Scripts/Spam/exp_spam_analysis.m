% experiments/exp_spam_analysis.m
% Replicates the Spam Classification Tree analysis (Figure 9.4 from ESL).

import OceanStats.Supervised.Tree.*
import OceanViz.*

% --- Caching Logic ---
cacheFile = 'spam_results_cache.mat';
forceRetrain = false;

if isfile(cacheFile) && ~forceRetrain
    fprintf('Loading results from cache: %s\n', cacheFile);
    load(cacheFile);
else
    % --- 1. Load Data ---
    fprintf('Loading Spambase dataset...\n');
    % Locate Data relative to this script (robust search)
    currentDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(currentDir, 'Data', 'spambase');

    % Search up the directory tree
    for i = 1:3
        if isfolder(dataDir), break; end
        currentDir = fileparts(currentDir);
        dataDir = fullfile(currentDir, 'Data', 'spambase');
    end

    dataFile = fullfile(dataDir, 'spambase.data');
    if ~isfile(dataFile)
        error('Spambase data not found.');
    end

    raw = load(dataFile);
    X = raw(:, 1:end-1);
    Y = raw(:, end); 

    % --- Data Splitting Strategy ---
    % REASON: Replicating the exact ESL setup.
    % The text specifies a test set of size 1536.
    rng(42); 
    cvObj = cvpartition(Y, 'HoldOut', 1536/length(Y));
    X_train = X(training(cvObj), :);
    Y_train = Y(training(cvObj));
    X_test  = X(test(cvObj), :);
    Y_test  = Y(test(cvObj));

    fprintf('Train Size: %d, Test Size: %d\n', length(Y_train), length(Y_test));

    % --- 2. Train Full Tree ---
    % METHOD: Deviance (Entropy) for Growth
    % REASON: The text states, "We used the deviance measure to grow the tree".
    % Deviance is smoother and differentiable, making it superior for finding split points,
    % unlike Misclassification Rate which is too "blocky" for optimization.
    fprintf('Growing full tree (Criterion=Entropy)...\n');
    tree = DecisionTree('Criterion', 'entropy', 'MinLeafSize', 10, 'MaxDepth', 30); 
    tree.fit(X_train, Y_train);

    % --- 3. Pruning Sequence (Cost-Complexity) ---
    fprintf('Calculating pruning sequence...\n');
    
    % METHOD: Weakest Link Pruning
    % REASON: We need to generate a sequence of nested trees indexed by alpha ($alpha$).
    % This allows us to trade off complexity (|T|) against fit.
    alphas = tree.weakestLinkSequence(); 
    nAlphas = length(alphas);
    fprintf('Found %d pruning candidates.\n', nAlphas);

    % Pre-allocate arrays
    tree_sizes = zeros(1, nAlphas);
    test_errors = zeros(1, nAlphas);

    for i = 1:nAlphas
        % Prune the GLOBAL tree to this alpha
        pTree = tree.prune(alphas(i));
        tree_sizes(i) = pTree.countTerminalNodes(pTree.RootNode);
        
        % Calculate Test Error (The Orange Line in Fig 9.4)
        % REASON: To verify that our CV estimates track the true test error.
        y_pred_test = pTree.predict(X_test);
        test_errors(i) = mean(y_pred_test ~= Y_test);
    end

    % --- 4. Cross-Validation (Model Selection) ---
    K = 10;
    
    % METHOD: 10-Fold Cross-Validation
    % REASON: To estimate the generalization error without touching the Test Set.
    % The text explicitly uses 10-fold CV to generate the error bars.
    fprintf('Running 10-fold CV...\n');
    import OceanStats.Selection.CrossValidation.CrossValidator
    
    cv = CrossValidator('PartitionStrategy', 'KFold', 'NumFolds', K, 'UseParallel', true);
    cv_errors = zeros(1, nAlphas);
    cv_se = zeros(1, nAlphas);

    % Loop through Alphas (Grid Search)
    % CRITICAL STATISTICAL NUANCE: 
    % We index by Alpha, NOT by Tree Size.
    % REASON: "Note that the cross-validation error rates are indexed by a sequence of 
    % values of alpha and not tree size".
    % A fixed alpha might produce a 17-node tree in Fold 1 but a 19-node tree in Fold 2.
    tic;
    for i = 1:nAlphas
        val_alpha = alphas(i);
        
        % Strategy: Train using Entropy, but Prune using Misclassification
        % REASON: "We used... misclassification rate to prune it".
        strategy = OceanStats.Selection.Strategies.DecisionTree(10, 30, 'entropy');
        strategy.PruneAlpha = val_alpha;
        
        % Evaluate misclassification rate across 10 folds for this specific alpha
        results = cv.evaluate(strategy, X_train, Y_train);
        
        cv_errors(i) = results.MeanError; % The Blue Line
        cv_se(i)     = results.StdError;  % The Error Bars
        
        if mod(i, 5) == 0, fprintf('  Processed Alpha %d/%d...\n', i, nAlphas); end
    end
    toc;
    
    fprintf('Saving results to cache: %s\n', cacheFile);
    save(cacheFile, 'alphas', 'nAlphas', 'tree_sizes', 'test_errors', 'cv_errors', 'cv_se', ...
        'X_test', 'Y_test', 'tree', 'X_train', 'Y_train');
end

% --- 5. Analysis & Plotting ---
fprintf('Plotting results...\n');

% Sort data by Tree Size for the X-axis of the plot
[sorted_sizes, idx] = sort(tree_sizes);
sorted_cv = cv_errors(idx);
sorted_se = cv_se(idx);
sorted_test = test_errors(idx);

% --- METHOD: The "One-Standard-Error" Rule ---
% REASON: To avoid overfitting to the random noise in the CV folds.
% We choose the simplest model (smallest size) whose error is within 1 SE 
% of the absolute minimum.

% 1. Find absolute minimum
[min_cv, min_idx] = min(sorted_cv);
min_se = sorted_se(min_idx);

% 2. Define the ceiling (Threshold)
upper_bound = min_cv + min_se;

% 3. Find the smallest tree under that ceiling
candidates = find(sorted_cv <= upper_bound);
best_idx_1se = candidates(1); % Index 1 is the smallest size because we sorted by size
best_size_1se = sorted_sizes(best_idx_1se);
best_alpha_1se = alphas(idx(best_idx_1se)); % Map back to the alpha used

fprintf('1-SE Rule suggests tree size: %d (Alpha=%.4f)\n', best_size_1se, best_alpha_1se);

% --- Final Evaluation ---
% Prune the GLOBAL tree using the chosen alpha
finalTree = tree.prune(best_alpha_1se);
y_pred_final = finalTree.predict(X_test);

% Calculate Sensitivity & Specificity
% REASON: "In medical classification problems, the terms sensitivity and specificity 
% are used to characterize a rule".
cm = confusionmat(Y_test, y_pred_final);
TN = cm(1,1); FP = cm(1,2);
FN = cm(2,1); TP = cm(2,2);

sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);
overall_err = (FP + FN) / sum(cm, 'all');

fprintf('\n=== Step 5: Test Set Evaluation ===\n');
fprintf('Selected Tree Size: %d Terminal Nodes\n', best_size_1se);
fprintf('Overall Test Error: %.2f%%\n', overall_err * 100);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity * 100);
fprintf('Specificity:          %.2f%%\n', specificity * 100);

% --- Plotting (Dark Mode) ---
viz = OceanViz.Core(Theme="Dark");
hFig = figure('Name', 'Spam Analysis', 'Color', viz.Colors.Background);
ax = axes('Parent', hFig);
viz.styleAxes(ax);
hold(ax, 'on');

% Plot CV Error (Blue Line with Bars)
errorbar(ax, sorted_sizes, sorted_cv, sorted_se, '.-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'DisplayName', 'CV Error (10-fold)');

% Plot Test Error (Orange Line)
plot(ax, sorted_sizes, sorted_test, '.-', ...
    'Color', viz.Colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Test Error');

% Mark the 1-SE Choice
xline(ax, best_size_1se, ':', 'Color', viz.Colors.Text, ...
    'Label', sprintf('1-SE Rule: Size %d', best_size_1se));

xlabel(ax, 'Tree Size |T|');
ylabel(ax, 'Misclassification Rate');
title(ax, 'Spam Dataset: Error vs Tree Size');
legend(ax, 'Location', 'northeast');

% Save
saveas(hFig, 'spam_analysis_results.png');
fprintf('Experiment Complete.\n');