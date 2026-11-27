% experiments/exp_spam_mars.m
% Replicates the Spam Classification MARS analysis (Figure 9.12 from ESL).

import OceanStats.Supervised.MARS.*
import OceanStats.Selection.FeatureSelection.*
import OceanViz.*

% --- Caching Logic ---
cacheFile = 'spam_mars_cache.mat';
forceRetrain = false;

if isfile(cacheFile) && ~forceRetrain
    fprintf('Loading results from cache: %s\n', cacheFile);
    load(cacheFile);
else
    % --- 1. Load Data ---
    fprintf('Loading Spambase dataset...\n');
    currentDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(currentDir, '..', '..', 'Data', 'spambase');
    
    dataFile = fullfile(dataDir, 'spambase.data');
    if ~isfile(dataFile)
        error('Spambase data not found.');
    end

    raw = load(dataFile);
    X = raw(:, 1:end-1);
    Y = raw(:, end);

    % Also load feature names
    namesFile = fullfile(dataDir, 'spambase.names');
    if isfile(namesFile)
        fid = fopen(namesFile, 'rt');
        lines = textscan(fid, '%s', 'Delimiter', '\n');
        lines = lines{1};
        fclose(fid);
        featNames = cell(57, 1);
        idx = 1;
        for i=1:length(lines)
            line = strtrim(lines{i});
            if startsWith(line, 'word_freq_') || startsWith(line, 'char_freq_') || startsWith(line, 'capital_')
                parts = split(line, ':');
                featNames{idx} = strtrim(parts{1});
                idx = idx + 1;
            end
        end
    else
        featNames = cellstr(strcat('X', string(1:size(X,2))));
    end

    % --- Data Splitting Strategy ---
    rng(42);
    cvObj = cvpartition(Y, 'HoldOut', 1536/length(Y));
    X_train = X(training(cvObj), :);
    Y_train = Y(training(cvObj));
    X_test  = X(test(cvObj), :);
    Y_test  = Y(test(cvObj));

    fprintf('Train Size: %d, Test Size: %d\n', length(Y_train), length(Y_test));

    % --- 2. Train MARS Model ---
    fprintf('Growing MARS model (MaxDegree=2)...\n');
    
    opts = struct();
    opts.MaxTerms = 65; % ESL goes up to ~100 rank, cut to 65 for speed
    opts.MaxInteractionDegree = 2;
    opts.PenaltyCost = 3.0;

    forward = ForwardBasisSelector();
    backward = GCVBackwardSelector();

    tic;
    grownBasis = forward.build(X_train, Y_train, opts);
    [bestBasis, bestCoefs, history] = backward.prune(X_train, Y_train, grownBasis, opts);
    toc;

    % --- 3. Evaluate Test Error over Pruning Sequence ---
    fprintf('Evaluating Test Error...\n');
    num_models = length(history);
    test_errors = zeros(1, num_models);
    ranks = zeros(1, num_models);
    gcv_scores = zeros(1, num_models);

    for m = 1:num_models
        if isempty(history(m).BasisIndices)
            continue;
        end
        
        idx = history(m).BasisIndices;
        current_basis = grownBasis(idx);
        coefs = history(m).Coefs;
        
        num_terms = length(idx);
        B_test = zeros(size(X_test, 1), num_terms);
        for j = 1:num_terms
            B_test(:, j) = current_basis{j}.evaluate(X_test);
        end
        
        y_pred_score = B_test * coefs;
        y_pred = double(y_pred_score > 0.5);
        
        test_errors(m) = mean(y_pred ~= Y_test);
        ranks(m) = num_terms;
        gcv_scores(m) = history(m).GCV;
    end
    
    % Find Optimal model globally
    valid_mask = ranks > 0;
    valid_ranks = ranks(valid_mask);
    valid_gcv = gcv_scores(valid_mask);
    
    [min_gcv, min_idx] = min(valid_gcv);
    optimal_rank = valid_ranks(min_idx);
    
    % Find corresponding error
    optimal_err_idx = find(ranks == optimal_rank, 1);
    optimal_err = test_errors(optimal_err_idx);
    
    fprintf('Saving results to cache: %s\n', cacheFile);
    save(cacheFile, 'opts', 'history', 'grownBasis', 'bestBasis', 'bestCoefs', ...
        'test_errors', 'ranks', 'gcv_scores', 'optimal_rank', 'optimal_err', ...
        'X_train', 'Y_train', 'X_test', 'Y_test', 'featNames');
end

% --- 4. Print Leading Interactions ---
fprintf('\nOptimal Model Rank (GCV chosen): %d\n', optimal_rank);
fprintf('Test Misclassification Error: %.2f%%\n', optimal_err * 100);

fprintf('\nInteraction Terms in Optimal Model:\n');
opt_hist = history(optimal_rank);
opt_basis = grownBasis(opt_hist.BasisIndices);
opt_coefs = opt_hist.Coefs;

for m = 1:length(opt_basis)
    bStr = opt_basis{m}.toString(featNames);
    if contains(bStr, '*') 
        fprintf('Coef=%8.4f: %s\n', opt_coefs(m), bStr);
    end
end

% --- 5. Plotting ---
fprintf('Plotting results...\n');

viz = OceanViz.Core(Theme="Dark");
hFig = figure('Name', 'MARS Spam Analysis', 'Color', viz.Colors.Background);
ax = axes('Parent', hFig);
viz.styleAxes(ax);
hold(ax, 'on');

valid_idx = ranks > 0;
plot_ranks = ranks(valid_idx);
plot_errors = test_errors(valid_idx);

plot(ax, plot_ranks, plot_errors, '.-', ...
    'Color', viz.Colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Test Error');

% Mark the GCV Choice
xline(ax, optimal_rank, ':', 'Color', viz.Colors.Text, ...
    'Label', sprintf('GCV Choice: %d', optimal_rank));

xlabel(ax, 'Rank (Number of basis functions)');
ylabel(ax, 'Test Misclassification Rate');
title(ax, 'MARS on Spam Data (Interactions=2)');
legend(ax, 'Location', 'northeast');

% Save
saveas(hFig, 'spam_mars_results.png');
fprintf('Experiment Complete.\n');
