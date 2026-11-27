% experiments/exp_mars_simulated.m
% Replicates the MARS Simulated Data Experiment (ESL 9.4.2).

import OceanStats.Supervised.MARS.*

rng(42); % For reproducibility

N_train = 100;
N_test = 1000;
N_sims = 5;

% Preallocate arrays for R^2 results
R2_S1 = zeros(N_sims, 1);
R2_S2 = zeros(N_sims, 1);
R2_S3 = zeros(N_sims, 1);

for sim = 1:N_sims
    fprintf('Running simulation %d / %d...\n', sim, N_sims);
    
    % Initialize DataFactory with different seed for each sim
    factory = OceanStats.Data.DataFactory('Seed', 42 + sim);
    
    % --- Scenario 1: Tensor Product, p=2 ---
    [X_train1, Y_train1, mu_train1] = factory.create("ComplexFunctions", "N", N_train, "Function", "TensorProduct", "Features", 2, "Sigma", 0.12, "Distribution", "Uniform");
    [X_test1,  ~,        mu_test1]  = factory.create("ComplexFunctions", "N", N_test,  "Function", "TensorProduct", "Features", 2, "Sigma", 0.12, "Distribution", "Uniform");
    
    mars1 = MARS('MaxInteractionDegree', 2, 'MaxTerms', 50, 'PenaltyCost', 3.0);
    mars1.fit(X_train1, Y_train1);
    
    y_bar1 = mean(Y_train1);
    MSE0_1 = mean((y_bar1 - mu_test1).^2);
    MSE_1  = mean((mars1.predict(X_test1) - mu_test1).^2);
    R2_S1(sim) = (MSE0_1 - MSE_1) / MSE0_1;
    
    % --- Scenario 2: Tensor Product, p=20 ---
    [X_train2, Y_train2, mu_train2] = factory.create("ComplexFunctions", "N", N_train, "Function", "TensorProduct", "Features", 20, "Sigma", 0.12, "Distribution", "Uniform");
    [X_test2,  ~,        mu_test2]  = factory.create("ComplexFunctions", "N", N_test,  "Function", "TensorProduct", "Features", 20, "Sigma", 0.12, "Distribution", "Uniform");
    
    mars2 = MARS('MaxInteractionDegree', 2, 'MaxTerms', 50, 'PenaltyCost', 3.0);
    mars2.fit(X_train2, Y_train2);
    
    y_bar2 = mean(Y_train2);
    MSE0_2 = mean((y_bar2 - mu_test2).^2);
    MSE_2  = mean((mars2.predict(X_test2) - mu_test2).^2);
    R2_S2(sim) = (MSE0_2 - MSE_2) / MSE0_2;
    
    % --- Scenario 3: Neural Network, p=10 ---
    [X_train3, Y_train3, mu_train3] = factory.create("ComplexFunctions", "N", N_train, "Function", "NeuralNetwork", "Features", 10, "Sigma", 0.12);
    [X_test3,  ~,        mu_test3]  = factory.create("ComplexFunctions", "N", N_test,  "Function", "NeuralNetwork", "Features", 10, "Sigma", 0.12);
    
    mars3 = MARS('MaxInteractionDegree', 2, 'MaxTerms', 50, 'PenaltyCost', 3.0);
    mars3.fit(X_train3, Y_train3);
    
    y_bar3 = mean(Y_train3);
    MSE0_3 = mean((y_bar3 - mu_test3).^2);
    MSE_3  = mean((mars3.predict(X_test3) - mu_test3).^2);
    R2_S3(sim) = (MSE0_3 - MSE_3) / MSE0_3;
end

fprintf('\n=== Results (R^2) over %d simulations ===\n', N_sims);
fprintf('Scenario 1 (p=2) : Mean = %.4f (S.E. = %.4f)\n', mean(R2_S1), std(R2_S1)/sqrt(N_sims));
fprintf('Scenario 2 (p=20): Mean = %.4f (S.E. = %.4f)\n', mean(R2_S2), std(R2_S2)/sqrt(N_sims));
fprintf('Scenario 3 (NN)  : Mean = %.4f (S.E. = %.4f)\n', mean(R2_S3), std(R2_S3)/sqrt(N_sims));

%% --- Plotting Scenario 1 using OceanViz Engine ---
fprintf('\nGenerating 3D Surface Plots for Scenario 1 (p=2)...\n');

% Create meshgrid for X1 and X2
x_range = linspace(-2.5, 2.5, 40);
[x1_grid, x2_grid] = meshgrid(x_range, x_range);
X_grid = [x1_grid(:), x2_grid(:)];

% Compute True Surface
f_true_grid = max(X_grid(:,1)-1, 0) + max(X_grid(:,1)-1, 0) .* max(X_grid(:,2)-0.8, 0);
z_true_grid = reshape(f_true_grid, size(x1_grid));

% Compute Predicted Surface (using the last trained mars1 model)
f_pred_grid = mars1.predict(X_grid);
z_pred_grid = reshape(f_pred_grid, size(x1_grid));

% Initialize OceanViz
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('MARS Model - ESL 9.4.2 Scenario 1');
t = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');

% Plot True Function
ax1 = nexttile(t);
data_true.X = x1_grid; data_true.Y = x2_grid; data_true.Z = z_true_grid;
data_true.RawX = X_train1(:,1); data_true.RawY = X_train1(:,2); data_true.RawZ = Y_train1;
viz.plot(ax1, "Surface", data_true, 'Title', "True Function (Scenario 1)", ...
    'XLabel', "$X_1$", 'YLabel', "$X_2$", 'ZLabel', "$Y$", 'ShowScatter', true);

% Plot MARS Prediction
ax2 = nexttile(t);
data_pred.X = x1_grid; data_pred.Y = x2_grid; data_pred.Z = z_pred_grid;
viz.plot(ax2, "Surface", data_pred, 'Title', "MARS Prediction (Scenario 1)", ...
    'XLabel', "$X_1$", 'YLabel', "$X_2$", 'ZLabel', "$\hat{Y}$", 'ShowScatter', false);
