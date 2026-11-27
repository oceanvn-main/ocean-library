% EXP_SPAM_MCNEMAR_ESL  Replication of ESL Example 10.6 (McNemar's Test)
%
%   Goal: Compare GAM vs GBM on the Spam dataset using McNemar's Test.
%   Reference: ESL §10.6 / Ex 10.6 (Agresti, 1996)
%
%   Task (a): Show SE of estimates is ~0.6%.
%   Task (b): Show GAM makes significantly more errors (p=0.036).
%

% 1. Setup
rng(42, 'twister');
addpath(genpath('c:\Ocean Library\Matlab code practice'));

% Paths
dataPath = 'c:\Ocean Library\Matlab code practice\Data\spambase\spambase.data';
N_TEST   = 1536;

% 2. Load Data
data = readmatrix(dataPath, 'FileType', 'text');
X    = data(:, 1:end-1);
Y    = data(:, end);
N    = size(X, 1);

% Shuffle and Split (ESL uses 1536 for test)
idx      = randperm(N);
testIdx  = idx(1:N_TEST);
trainIdx = idx(N_TEST+1:end);

Xtrain = X(trainIdx, :); Ytrain = Y(trainIdx);
Xtest  = X(testIdx, :);  Ytest  = Y(testIdx);

% 3. Fit Models (using built-in as requested)
fprintf('Fitting GAM (this may take a moment)...\n');
mdl_gam = fitcgam(Xtrain, Ytrain);

fprintf('Fitting GBM (LogitBoost ensemble with 100 trees)...\n');
% LogitBoost is often used for binary classification in this context
mdl_gbm = fitcensemble(Xtrain, Ytrain, 'Method', 'LogitBoost', 'NumLearningCycles', 100);

% 4. Predict
yhat_gam = predict(mdl_gam, Xtest);
yhat_gbm = predict(mdl_gbm, Xtest);

% 5. Standard Error Analysis
err_gam = mean(yhat_gam ~= Ytest);
err_gbm = mean(yhat_gbm ~= Ytest);

se_gam  = sqrt(err_gam * (1-err_gam) / N_TEST);
se_gbm  = sqrt(err_gbm * (1-err_gbm) / N_TEST);

fprintf('\n(a) Error Estimates & SE:\n');
fprintf('    GAM Error: %.2f%% (SE: %.4f)\n', err_gam*100, se_gam);
fprintf('    GBM Error: %.2f%% (SE: %.4f)\n', err_gbm*100, se_gbm);

% 6. Construct Contingency Table
correct_gam = (yhat_gam == Ytest);
correct_gbm = (yhat_gbm == Ytest);

% n00: both correct
% n01: gam correct, gbm wrong
% n10: gam wrong, gbm correct
% n11: both wrong
n00 = sum( correct_gam &  correct_gbm);
n01 = sum( correct_gam & ~correct_gbm);
n10 = sum(~correct_gam &  correct_gbm);
n11 = sum(~correct_gam & ~correct_gbm);

fprintf('\nContingency Table:\n');
fprintf('               GBM Correct    GBM Error\n');
fprintf('GAM Correct:      %4d           %4d\n', n00, n01);
fprintf('GAM Error:        %4d           %4d\n', n10, n11);

% 7. McNemar's Test (Using Ocean Library)
t = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();

% The book (ESL) usually reports the asymptotic p-value WITHOUT correction 
% to get 0.036. Let's compare both.
data_mcnemar = struct('n01', n01, 'n10', n10, 'Correction', false);
res_nocorr   = t.test(data_mcnemar);

data_mcnemar.Correction = true;
res_corr     = t.test(data_mcnemar);

fprintf('\n(b) McNemar Test Results (Ocean Library):\n');
fprintf('    Asymptotic (No Correction): p = %.4f (Rejects H0: %d)\n', ...
    res_nocorr.PValue, res_nocorr.Reject);
fprintf('    Asymptotic (With Correction): p = %.4f (Rejects H0: %d)\n', ...
    res_corr.PValue, res_corr.Reject);

% 8. Built-in t-test baseline
% Perform a paired t-test on the individual error indicators (0 or 1 per sample)
err_vec_gam = double(yhat_gam ~= Ytest);
err_vec_gbm = double(yhat_gbm ~= Ytest);
[h_p, p_p] = ttest(err_vec_gam, err_vec_gbm);

fprintf('\nBuilt-in Paired t-test:\n');
fprintf('    p = %.4f (Rejects H0: %d)\n', p_p, h_p);

% 9. Visualization (OceanViz)
try
    V = OceanViz.Core();
    % Use a summary plot or table
    resultsTable = table([err_gam; err_gbm], [se_gam; se_gbm], ...
        'RowNames', {'GAM', 'GBM'}, 'VariableNames', {'ErrorRate', 'StdError'});
    
    fprintf('\nResults stored in table:\n');
    disp(resultsTable);
    
    % Simple plot of error rates with error bars
    figure('Color', [0.1 0.1 0.1]); % Dark mode
    hold on;
    bar(1, err_gam, 'FaceColor', [0.2 0.4 0.8]);
    bar(2, err_gbm, 'FaceColor', [0.8 0.4 0.2]);
    errorbar([1 2], [err_gam err_gbm], [se_gam se_gbm], 'w.', 'LineWidth', 2);
    set(gca, 'Color', 'none', 'XColor', 'w', 'YColor', 'w');
    xticks([1 2]); xticklabels({'GAM', 'GBM'});
    ylabel('Test Error Rate');
    title('ESL Ex 10.6: Model Comparison', 'Color', 'w');
    grid on; grid minor;
    
    % Overlay McNemar Result
    text(1.5, max(err_gam, err_gbm)*1.1, sprintf('McNemar p = %.3f', res_nocorr.PValue), ...
        'Color', 'w', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
catch e
    fprintf('OceanViz skipped: %s\n', e.message);
end
