% EXP_SVM_MIXTURE  Linear SVM with C=10000 and C=0.01 on ESL mixture data
%
%   DATASET:        ESL Mixture Data (10 centroids per class, N=200 train)
%                   Generated from scratch to perfectly mimic Figure 12.2.
%   MODEL:          1. SVMClassifier (LinearKernel, C=10000)
%                   2. SVMClassifier (LinearKernel, C=0.01)
%   ASSUMPTIONS:    - Data generated according to ESL pg. 17 description.
%                   - Training set N=200, used for decision boundary.
%                   - Test set N=10000 used for Bayes and Test Error.
%                   - SMO Tolerance=1e-3, MaxPasses=500.
%   REPRODUCIBILITY: rng(42, 'twister') sets train/test generation fixed.
%
%   Ref: ESL Chapter 12.2.2, Figure 12.2

clc; clear; rng(42, 'twister');

ROOT = 'c:\Ocean Library\Matlab code practice';
addpath(ROOT);

fprintf('=== EXP_SVM_MIXTURE: ESL Figure 12.2 Reproduction ===\n\n');

% === 1. Data Generation (ESL Mixture) ===
fprintf('Generating ESL Mixture Data (N=200 Train, N=10000 Test)...\n');

function [X, Y, m1, m2] = generateESLMixture(N, m1, m2)
    % Generates N samples (N/2 per class). If centroids m1, m2 not provided,
    % they are spawned.
    if nargin < 2
        % 10 centroids per class
        m1 = randn(10, 2) + [1, 0];  % Class +1
        m2 = randn(10, 2) + [0, 1];  % Class -1
    end
    
    X = zeros(N, 2);
    Y = zeros(N, 1);
    
    % Class 1
    ids1 = randi(10, N/2, 1);
    X(1:N/2, :) = m1(ids1, :) + randn(N/2, 2) * sqrt(1/5);
    Y(1:N/2) = 1;
    
    % Class -1
    ids2 = randi(10, N/2, 1);
    X(N/2+1:N, :) = m2(ids2, :) + randn(N/2, 2) * sqrt(1/5);
    Y(N/2+1:N) = -1;
end

% Generate Train and Test sets
[X_tr, Y_tr, m1, m2] = generateESLMixture(200);
[X_te, Y_te]         = generateESLMixture(10000, m1, m2);

% === 2. Feature Engineering ===
% Original features used directly (2D).

% === 3. Train / Test Split ===
% Already explicit above.

% === 4. Model Configuration & Training (Using MATLAB fitcsvm) ===
fprintf('Training SVM C=10000 built-in fitcsvm...\n');
% BoxConstraint is the equivalent of C
svmHigh = fitcsvm(X_tr, Y_tr, 'KernelFunction', 'linear', 'BoxConstraint', 10000);

fprintf('Training SVM C=0.01 built-in fitcsvm...\n');
svmLow = fitcsvm(X_tr, Y_tr, 'KernelFunction', 'linear', 'BoxConstraint', 0.01);

% === 6. Evaluation ===
% Train Error
errTrHigh = mean(Y_tr ~= predict(svmHigh, X_tr));
errTrLow  = mean(Y_tr ~= predict(svmLow, X_tr));

% Test Error
errTeHigh = mean(Y_te ~= predict(svmHigh, X_te));
errTeLow  = mean(Y_te ~= predict(svmLow, X_te));

% Bayes Error (using mixture probabilities)
% P(x | Y=1) = 1/10 sum_k N(x; m1_k, I/5)
function p = evalMix(X, m)
    p = zeros(size(X,1), 1);
    for k = 1:10
        d2 = sum((X - m(k,:)).^2, 2);
        p = p + exp(-d2 * 2.5);  % precision is 5, factor is 1/2*5 = 2.5
    end
    p = p / 10;
end

p1 = evalMix(X_te, m1);
p2 = evalMix(X_te, m2);
bayesPred = ones(size(Y_te));
bayesPred(p2 > p1) = -1;
bayesErr = mean(Y_te ~= bayesPred);

% === 7. Visualization ===
fprintf('Rendering OceanViz Figure...\n');
viz = OceanViz.Core('Theme', 'Dark');  % Using Scientific for textbook look
fig = figure('Name', 'Linear SVM - Mixture Data (ESL Fig 12.2)', 'Units','normalized','Position',[0.1 0.1 0.6 0.8]);
fig.Color = viz.Colors.Background;

% Build a dense grid for contours
[xx, yy] = meshgrid(linspace(min(X_tr(:,1))-1, max(X_tr(:,1))+1, 150), ...
                    linspace(min(X_tr(:,2))-1, max(X_tr(:,2))+1, 150));
Xgrid = [xx(:), yy(:)];

% built-in predict returns [label, score]. The score(:,2) is the distance from boundary 
[~, scoreHigh] = predict(svmHigh, Xgrid);
fHigh = scoreHigh(:, 2);  % For fitcsvm, column 2 is the score for the positive class (+1)
zzHigh = reshape(fHigh, size(xx));

% Subplot 1: C = 10000
ax1 = subplot(2, 1, 1);
viz.styleAxes(ax1); hold(ax1, 'on');

% Plot Data points
scatter(ax1, X_tr(Y_tr==1, 1), X_tr(Y_tr==1, 2), 30, [0.9 0.6 0.1], 'o', 'LineWidth', 1.5);    % Orange
scatter(ax1, X_tr(Y_tr==-1, 1), X_tr(Y_tr==-1, 2), 30, [0.4 0.7 0.9], 'o', 'LineWidth', 1.5);   % Light blue

% Mark Support Vectors
svMask = svmHigh.IsSupportVector;
scatter(ax1, X_tr(svMask, 1), X_tr(svMask, 2), 15, 'k', 'filled');

% Boundaries
contour(ax1, xx, yy, zzHigh, [0 0], 'Color', 'k', 'LineWidth', 1.5);             % Decision
contour(ax1, xx, yy, zzHigh, [1 1], '--', 'Color', 'k', 'LineWidth', 1.0);       % +1 Margin
contour(ax1, xx, yy, zzHigh, [-1 -1], '--', 'Color', 'k', 'LineWidth', 1.0);     % -1 Margin

% Bayes boundary
p1g = evalMix(Xgrid, m1);
p2g = evalMix(Xgrid, m2);
% bayes decision is where p1g == p2g, so p1g - p2g == 0
contour(ax1, xx, yy, reshape(p1g - p2g, size(xx)), [0 0], '--', 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);

title(ax1, 'C = 10000', 'Color', viz.Colors.Text);
xlabel(ax1, 'Feature 1 (X_1)', 'Color', viz.Colors.Text);
ylabel(ax1, 'Feature 2 (X_2)', 'Color', viz.Colors.Text);

% Create a legend for the first subplot
legend(ax1, {'Class +1 (Orange)', 'Class -1 (Blue)', 'Support Vectors', 'Decision Boundary', 'Margin (+1)', 'Margin (-1)', 'Bayes Optimal'}, ...
       'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);

text(ax1, 0.05, 0.15, sprintf('Training Error: %.3f\nTest Error:     %.3f\nBayes Error:    %.3f', errTrHigh, errTeHigh, bayesErr), ...
     'Units', 'normalized', 'Color', viz.Colors.Text, 'FontName', 'Courier');

% Subplot 2: C = 0.01
ax2 = subplot(2, 1, 2);
viz.styleAxes(ax2); hold(ax2, 'on');

[~, scoreLow] = predict(svmLow, Xgrid);
fLow = scoreLow(:, 2);
zzLow = reshape(fLow, size(xx));

% Plot Data points
scatter(ax2, X_tr(Y_tr==1, 1), X_tr(Y_tr==1, 2), 30, [0.9 0.6 0.1], 'o', 'LineWidth', 1.5);
scatter(ax2, X_tr(Y_tr==-1, 1), X_tr(Y_tr==-1, 2), 30, [0.4 0.7 0.9], 'o', 'LineWidth', 1.5);

% Mark Support Vectors
svMask2 = svmLow.IsSupportVector;
scatter(ax2, X_tr(svMask2, 1), X_tr(svMask2, 2), 15, 'k', 'filled');

% Boundaries
contour(ax2, xx, yy, zzLow, [0 0], 'Color', 'k', 'LineWidth', 1.5);             % Decision
contour(ax2, xx, yy, zzLow, [1 1], '--', 'Color', 'k', 'LineWidth', 1.0);       % +1 Margin
contour(ax2, xx, yy, zzLow, [-1 -1], '--', 'Color', 'k', 'LineWidth', 1.0);     % -1 Margin

% Bayes boundary
contour(ax2, xx, yy, reshape(p1g - p2g, size(xx)), [0 0], '--', 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);

title(ax2, 'C = 0.01', 'Color', viz.Colors.Text);
xlabel(ax2, 'Feature 1 (X_1)', 'Color', viz.Colors.Text);
ylabel(ax2, 'Feature 2 (X_2)', 'Color', viz.Colors.Text);

text(ax2, 0.05, 0.15, sprintf('Training Error: %.3f\nTest Error:     %.3f\nBayes Error:    %.3f', errTrLow, errTeLow, bayesErr), ...
     'Units', 'normalized', 'Color', viz.Colors.Text, 'FontName', 'Courier');

% Save figure
if ~exist('Results', 'dir'); mkdir('Results'); end
saveas(fig, 'Results/exp_svm_mixture.png');

% === 8. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ESL Mixture Data (N=%d, d=%d)\n', size(X_tr,1), size(X_tr,2));
fprintf('  Bayes Error   : %.3f\n\n', bayesErr);
fprintf('  Model (C=10000)  %d Support Vectors\n', sum(svmHigh.IsSupportVector));
fprintf('  Train Err     : %.3f\n', errTrHigh);
fprintf('  Test  Err     : %.3f\n\n', errTeHigh);
fprintf('  Model (C=0.01)   %d Support Vectors\n', sum(svmLow.IsSupportVector));
fprintf('  Train Err     : %.3f\n', errTrLow);
fprintf('  Test  Err     : %.3f\n', errTeLow);
fprintf('==========================\n');
