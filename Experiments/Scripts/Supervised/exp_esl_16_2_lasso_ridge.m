% EXP_ESL_16_2_LASSO_RIDGE  ESL Figure 16.2: Lasso vs Ridge 
%
%   DATASET:        Simulation (N_train=50, N_test=10000, p=300)
%                   Independent Gaussian Predictors.
%   SCENARIOS:      Dense (300 nonzero), Sparse (10 nonzero), Sparse (30 nonzero)
%   MODELS:         Lasso (L1) and Ridge (L2) regression on continuous & binary data.
%   ASSUMPTIONS:    Classification uses squared-error loss on {0, 1} responses.
%                   Optimal lambda chosen from test set directly to show "best case".
%                   NSR scaling for classification involves numerical root finding.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; close all; rng(42, 'twister');

%% === 1. Setup Simulation Parameters ===
N_train = 50;
N_test = 10000;
p = 300;

scenarios = [300, 10, 30]; % Number of non-zero coefficients
nsr_vals = [0.1, 0.2, 0.3, 0.4, 0.5];
num_runs = 20;

% Storage: (scenario, nsr, run)
res_reg_lasso = zeros(3, 5, num_runs);
res_reg_ridge = zeros(3, 5, num_runs);
res_cls_lasso = zeros(3, 5, num_runs);
res_cls_ridge = zeros(3, 5, num_runs);

% Lambda/k sequence for Ridge (Lasso computes its own path automatically)
k_vals = logspace(-3, 4, 50);

fprintf('Running Monte Carlo Simulation (600 configurations)...\n');

tic;
%% === 2. Main Simulation Loop ===
for s = 1:length(scenarios)
    num_nonzero = scenarios(s);
    
    for nsr_idx = 1:length(nsr_vals)
        target_nsr = nsr_vals(nsr_idx);
        
        for run = 1:num_runs
            % --- Generate Base Data ---
            X_train = randn(N_train, p);
            X_test  = randn(N_test, p);
            
            % True Beta
            beta = zeros(p, 1);
            beta(1:num_nonzero) = randn(num_nonzero, 1);
            
            % Linear Predictor
            f_train = X_train * beta;
            f_test  = X_test * beta;
            var_f   = var(f_test); % Use test set for population variance
            
            % ==========================================
            % REGRESSION SCENARIO
            % ==========================================
            % Scale noise to hit NSR exactly
            sigma = sqrt(target_nsr * var_f);
            Y_train_reg = f_train + sigma * randn(N_train, 1);
            Y_test_reg  = f_test  + sigma * randn(N_test, 1);
            
            % Constant Baseline (Mean)
            y_mean_reg = mean(Y_train_reg);
            mse_const = mean((Y_test_reg - y_mean_reg).^2);
            
            % Fit Lasso
            [B_lasso, info_lasso] = lasso(X_train, Y_train_reg, 'NumLambda', 50, 'Standardize', false);
            Y_pred_lasso = X_test * B_lasso + info_lasso.Intercept;
            mse_lasso = mean((Y_test_reg - Y_pred_lasso).^2, 1);
            res_reg_lasso(s, nsr_idx, run) = max(1 - mse_lasso / mse_const);
            
            % Fit Ridge
            B_ridge = ridge(Y_train_reg, X_train, k_vals, 0);
            Y_pred_ridge = X_test * B_ridge(2:end, :) + B_ridge(1, :);
            mse_ridge = mean((Y_test_reg - Y_pred_ridge).^2, 1);
            res_reg_ridge(s, nsr_idx, run) = max(1 - mse_ridge / mse_const);
            
            % ==========================================
            % CLASSIFICATION SCENARIO
            % ==========================================
            % Find scalar c so that Var(Y|c*f) / Var(c*f) = target_nsr
            % Y|c*f ~ Bernoulli(p) where p = 1 / (1 + exp(-c*f))
            % Var(Y|c*f) = E[p(1-p)]
            % Var(c*f) = c^2 * var_f
            obj_fun = @(c) (mean( (1./(1+exp(-c*f_test))) .* (1 - 1./(1+exp(-c*f_test))) ) / (c^2 * var_f)) - target_nsr;
            
            % We know NSR -> inf as c->0, NSR -> 0 as c->inf.
            % fzero can be sensitive, so we bound it
            try
                c_opt = fzero(obj_fun, [1e-3, 50]);
            catch
                % Fallback if bounds don't bracket
                c_opt = fzero(obj_fun, 1.0);
            end
            
            p_train = 1 ./ (1 + exp(-c_opt * f_train));
            p_test  = 1 ./ (1 + exp(-c_opt * f_test));
            
            Y_train_cls = double(rand(N_train, 1) < p_train);
            Y_test_cls  = double(rand(N_test, 1) < p_test);
            
            % Fit Lasso (using Squared-Error on binary targets, per textbook)
            [B_lasso_cls, info_lasso_cls] = lasso(X_train, Y_train_cls, 'NumLambda', 50, 'Standardize', false);
            Y_pred_lasso_cls = X_test * B_lasso_cls + info_lasso_cls.Intercept;
            err_lasso = mean((Y_pred_lasso_cls > 0.5) ~= Y_test_cls, 1);
            res_cls_lasso(s, nsr_idx, run) = max((0.5 - err_lasso) / 0.5);
            
            % Fit Ridge (using Squared-Error on binary targets)
            B_ridge_cls = ridge(Y_train_cls, X_train, k_vals, 0);
            Y_pred_ridge_cls = X_test * B_ridge_cls(2:end, :) + B_ridge_cls(1, :);
            err_ridge = mean((Y_pred_ridge_cls > 0.5) ~= Y_test_cls, 1);
            res_cls_ridge(s, nsr_idx, run) = max((0.5 - err_ridge) / 0.5);
        end
    end
    fprintf('  Completed Scenario %d/3\n', s);
end
fprintf('Simulation finished in %.2f seconds.\n', toc);

save('exp_esl_16_2_results.mat', 'res_reg_lasso', 'res_reg_ridge', 'res_cls_lasso', 'res_cls_ridge');

%% === 3. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');

scenario_names = {'Gaussian', 'Subset 10', 'Subset 30'};

% ----- Plot 1: Regression -----
fig_reg = viz.createFigure('ESL 16.2: Regression');
fig_reg.Position = [100, 100, 800, 1000];

for s = 1:3
    % Lasso Subplot
    ax1 = subplot(3, 2, 2*s - 1);
    data_lasso = squeeze(res_reg_lasso(s, :, :))';
    boxplot(ax1, data_lasso, 'Labels', nsr_vals, 'Colors', viz.Colors.Primary, 'Symbol', 'o');
    title(ax1, sprintf('Lasso/%s', scenario_names{s}));
    if s == 3, xlabel(ax1, 'Noise-to-Signal Ratio'); end
    if s == 2, ylabel(ax1, 'Percentage Squared Prediction Error Explained'); end
    % In MATLAB boxplot changes line colors, let's fix axes color manually
    set(ax1, 'XColor', viz.Colors.Grid, 'YColor', viz.Colors.Grid);
    ylim(ax1, [0.0, 1.0]);
    
    % Ridge Subplot
    ax2 = subplot(3, 2, 2*s);
    data_ridge = squeeze(res_reg_ridge(s, :, :))';
    boxplot(ax2, data_ridge, 'Labels', nsr_vals, 'Colors', viz.Colors.Secondary, 'Symbol', 'o');
    title(ax2, sprintf('Ridge/%s', scenario_names{s}));
    if s == 3, xlabel(ax2, 'Noise-to-Signal Ratio'); end
    viz.styleAxes(ax2);
    set(ax2, 'XColor', viz.Colors.Grid, 'YColor', viz.Colors.Grid);
    ylim(ax2, [0.0, 1.0]);
end

% ----- Plot 2: Classification -----
fig_cls = viz.createFigure('ESL 16.2: Classification');
fig_cls.Position = [950, 100, 800, 1000];

for s = 1:3
    % Lasso Subplot
    ax1 = subplot(3, 2, 2*s - 1);
    data_lasso = squeeze(res_cls_lasso(s, :, :))';
    boxplot(ax1, data_lasso, 'Labels', nsr_vals, 'Colors', viz.Colors.Primary, 'Symbol', 'o');
    title(ax1, sprintf('Lasso/%s', scenario_names{s}));
    if s == 3, xlabel(ax1, 'Noise-to-Signal Ratio'); end
    if s == 2, ylabel(ax1, 'Percentage Misclassification Error Explained'); end
    viz.styleAxes(ax1);
    set(ax1, 'XColor', viz.Colors.Grid, 'YColor', viz.Colors.Grid);
    ylim(ax1, [0.0, 1.0]);
    
    % Ridge Subplot
    ax2 = subplot(3, 2, 2*s);
    data_ridge = squeeze(res_cls_ridge(s, :, :))';
    boxplot(ax2, data_ridge, 'Labels', nsr_vals, 'Colors', viz.Colors.Secondary, 'Symbol', 'o');
    title(ax2, sprintf('Ridge/%s', scenario_names{s}));
    if s == 3, xlabel(ax2, 'Noise-to-Signal Ratio'); end
    viz.styleAxes(ax2);
    set(ax2, 'XColor', viz.Colors.Grid, 'YColor', viz.Colors.Grid);
    ylim(ax2, [0.0, 1.0]);
end
