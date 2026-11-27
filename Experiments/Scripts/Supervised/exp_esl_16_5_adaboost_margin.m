% EXP_ESL_16_5_ADABOOST_MARGIN  ESL Figure 16.5: Adaboost Margin and Overfitting
%
%   DATASET:        ESL Mixture Data (N=200 train, N=10,000 test)
%   MODEL:          GradientBoostingMachine with ExponentialLoss (AdaBoost.M1 eqv.)
%                   LearningRate=0.02, NumTrees=10000, MaxDepth=2 (4-node trees)
%   ASSUMPTIONS:    GBM with ExponentialLoss exactly recovers AdaBoost.M1 on binary 
%                   classification. We convert labels {1, 2} -> {-1, 1}.
%   REPRODUCIBILITY: rng(42, 'twister') for train data, rng(99, 'twister') for test.
%
clc; clear; rng(42, 'twister');

%% === 1. Data Loading ===
fprintf('1. Generating ESL Mixture Data (N_train=200, N_test=10000)...\n');
gen_train = OceanData.Generators.ESLMixtureGenerator('N', 200, 'Seed', 42);
[X_train, y_train_raw] = gen_train.generate();

gen_test = OceanData.Generators.ESLMixtureGenerator('N', 10000, 'Seed', 99);
[X_test, y_test_raw] = gen_test.generate();

%% === 2. Feature Engineering ===
% ExponentialLoss expects labels in {-1, 1}
Y_train = ones(size(y_train_raw));
Y_train(y_train_raw == 1) = -1;

Y_test = ones(size(y_test_raw));
Y_test(y_test_raw == 1) = -1;

%% === 3 & 4. Model Configuration ===
fprintf('2. Training Gradient Boosting Machine (AdaBoost equivalent)...\n');
M = 10000;
nu = 0.02;

loss = OceanStats.Loss.Classification.ExponentialLoss();
gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine(...
    'Loss', loss, ...
    'NumTrees', M, ...
    'LearningRate', nu, ...
    'MaxDepth', 2); % MaxDepth 2 gives up to 4 terminal nodes (4-node trees)

%% === 5. Training ===
tic;
gbm.fit(X_train, Y_train);
fprintf('   Training completed in %.2f seconds.\n', toc);

%% === 6. Evaluation (Staged) ===
fprintf('3. Evaluating Staged Predictions...\n');
% stagedPredict returns N x M matrix of raw score predictions F_m(x)
F_train = gbm.stagedPredict(X_train);
F_test  = gbm.stagedPredict(X_test);

margin = zeros(M, 1);
test_err = zeros(M, 1);

for m = 1:M
    % Unnormalized margin is Y_train .* F_train(:, m)
    % To match Adaboost L1 margin, we normalize by the L1 norm of the coefficients.
    % Since each step adds a tree multiplied by nu, we approximate the normalization
    % by dividing by (m * nu).
    norm_factor = m * nu; 
    
    % L1 margin m(f)
    m_f = (Y_train .* F_train(:, m)) / norm_factor;
    margin(m) = min(m_f);
    
    % Test error (sign of raw score gives prediction)
    pred_label = sign(F_test(:, m));
    test_err(m) = mean(pred_label ~= Y_test);
end

%% === 7. Visualization ===
fprintf('4. Rendering Visualization...\n');
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ESL 16.5: L1 Margin and Test Error for AdaBoost');
fig.Position = [100, 100, 1000, 450];

% Left panel: L1 Margin
ax1 = subplot(1, 2, 1);
plot(ax1, 1:M, margin, 'Color', viz.Colors.cyan, 'LineWidth', 1.5);
yline(ax1, 0, '--', 'Color', viz.Colors.orange);
title(ax1, 'L_1 Margin m(f)');
xlabel(ax1, 'Number of Trees');
ylabel(ax1, 'Margin');
% Match ESL y-axis roughly
ylim(ax1, [-0.3, 0.1]);
xlim(ax1, [0, M]);
viz.styleAxes(ax1);

% Right panel: Test Error
ax2 = subplot(1, 2, 2);
plot(ax2, 1:M, test_err, 'Color', viz.Colors.green, 'LineWidth', 1.5);

% Find minimum test error
[min_err, best_m] = min(test_err);
xline(ax2, best_m, '--', 'Color', viz.Colors.orange);

title(ax2, sprintf('Test Error (min at %d trees)', best_m));
xlabel(ax2, 'Number of Trees');
ylabel(ax2, 'Test Error');
xlim(ax2, [0, M]);
viz.styleAxes(ax2);

%% === 8. Summary printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ESL Mixture Data (N_train=200, N_test=10000)\n');
fprintf('  Model         : GBM (ExponentialLoss, M=10000, nu=0.02, Depth=2)\n');
fprintf('  Min Test Err  : %.4f (at tree %d)\n', min_err, best_m);
fprintf('  Final Margin  : %.4f\n', margin(end));
fprintf('==========================\n');
