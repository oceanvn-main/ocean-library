%% exp_adaboost_classification.m
% AdaBoost.M1 — Classification Experiment
%
%   Compares a single Decision Stump vs AdaBoost.M1 on a 2D classification
%   dataset using 5-fold cross-validation. Plots:
%     1. Decision boundary of the final AdaBoost model
%     2. Training error vs boosting round (convergence curve)
%
%   ARCHITECTURE:
%     Uses OceanStats.Boosting.BoostingEnsemble + AdaBoostM1 strategy.
%     Uses OceanViz.Core for all plots (Dark theme).
%     Uses OceanStats.Selection.CrossValidation.CrossValidator for CV.
%
%   COMMAND:
%     cd('c:\Ocean Library\Matlab code practice\')
%     run Experiments/exp_adaboost_classification.m
% =========================================================================

clearvars; clc;

%% ---- 1. Generate Dataset ------------------------------------------------
rng(42);
import OceanStats.Data.*;

N = 200;
% Two interleaved moons-like blobs
theta1 = linspace(0, pi, N/2)';
theta2 = linspace(pi, 2*pi, N/2)';
X = [ cos(theta1) + 0.3*randn(N/2,1),  sin(theta1) + 0.3*randn(N/2,1) ;
      cos(theta2) + 0.3*randn(N/2,1), -sin(theta2) + 0.3*randn(N/2,1) + 0.5 ];
Y = [ones(N/2, 1); 2*ones(N/2, 1)];

fprintf('Dataset: %d samples, 2 features, 2 classes\n', N);

%% ---- 2. Build Models ----------------------------------------------------
% Weak learner: Decision Stump (max depth = 1)
stump = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');

% AdaBoost.M1 with T=50 rounds
strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 50);
adaboost = OceanStats.Boosting.BoostingEnsemble(strategy, stump);

% Single stump baseline (IModelStrategy wrapper — same DecisionTree)
baseline = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');

%% ---- 3. 5-Fold Cross-Validation -----------------------------------------
fprintf('\n--- 5-Fold Cross-Validation ---\n');

kfold = OceanStats.Selection.CrossValidation.KFold(5);
cv    = OceanStats.Selection.CrossValidation.CrossValidator(kfold);

cv_err_stump    = cv.run(baseline,  X, Y);
cv_err_adaboost = cv.run(adaboost,  X, Y);

fprintf('Decision Stump CV Error : %.4f (%.1f%%)\n', cv_err_stump,    cv_err_stump    * 100);
fprintf('AdaBoost.M1   CV Error  : %.4f (%.1f%%)\n', cv_err_adaboost, cv_err_adaboost * 100);

%% ---- 4. Train Full Model (for plotting) ---------------------------------
fit = adaboost.train(X, Y);
fprintf('\nAdaBoost ran %d / 50 rounds.\n', fit.NumRounds);

%% ---- 5. Decision Boundary Plot ------------------------------------------
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'AdaBoost.M1 Results', 'NumberTitle', 'off', ...
             'Position', [100 100 1400 560]);

% Grid for boundary
x1r = linspace(min(X(:,1)) - 0.3, max(X(:,1)) + 0.3, 200);
x2r = linspace(min(X(:,2)) - 0.3, max(X(:,2)) + 0.3, 200);
[G1, G2] = meshgrid(x1r, x2r);
Xgrid    = [G1(:), G2(:)];
Zgrid    = adaboost.predict(fit, Xgrid);
Zgrid    = reshape(Zgrid, size(G1));

ax1 = subplot(1, 2, 1);
contourf(ax1, G1, G2, Zgrid, 'LineStyle', 'none', 'FaceAlpha', 0.3);
colormap(ax1, [0.2 0.4 0.8; 0.8 0.3 0.2]);
hold(ax1, 'on');
scatter(ax1, X(Y==1,1), X(Y==1,2), 30, 'b', 'filled', 'DisplayName', 'Class 1');
scatter(ax1, X(Y==2,1), X(Y==2,2), 30, 'r', 'filled', 'DisplayName', 'Class 2');
hold(ax1, 'off');
title(ax1, sprintf('AdaBoost.M1 Decision Boundary\nCV Error: %.1f%%', cv_err_adaboost*100));
xlabel(ax1, 'x_1'); ylabel(ax1, 'x_2');
legend(ax1, 'Location', 'best');
grid(ax1, 'on');

%% ---- 6. Convergence Curve (Training Error vs Round) ---------------------
ax2 = subplot(1, 2, 2);
errors = adaboost.getTrainingErrors();
rounds = 1:numel(errors);

plot(ax2, rounds, errors * 100, 'Color', [0.3 0.8 0.5], 'LineWidth', 2);
yline(ax2, cv_err_stump * 100, '--', 'Color', [0.9 0.5 0.2], ...
    'LineWidth', 1.5, 'Label', 'Single Stump CV Error');
xlabel(ax2, 'Boosting Round (t)');
ylabel(ax2, 'Weighted Training Error (%)');
title(ax2, 'AdaBoost.M1 — Convergence');
grid(ax2, 'on');
ylim(ax2, [0, max(errors * 100) * 1.2 + 1]);

fprintf('\nFigure ready. Training error dropped from %.1f%% → %.1f%% over %d rounds.\n', ...
    errors(1)*100, errors(end)*100, fit.NumRounds);
