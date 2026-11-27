%% HME Adapter Demo: Reusing Supervised Library Models
% This script verifies:
% 1. GenericExpert adapter for LinearModel.
% 2. Solution A: Input-dependent gating for specialists.
% 3. Solution C: Sparsity priors preventing expert collapse.

clear; clc; close all;
addpath(genpath(pwd));

import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*
import OceanViz.*

%% 1. Generate Non-Linear Data (W-Shape)
N = 300;
x = linspace(-2, 2, N)';
y = abs(x) + 0.1*randn(N, 1);
data = [x, y];

%% 2. Setup HME using the Adapter Pattern
% Creating specialists from the Supervised Library
specialist1 = GenericExpert(LinearModel());
specialist2 = GenericExpert(LinearModel());

% Solution A: Enable Local Gating
% Solution C: Alpha < 1 for Sparsity
pool = CompetitiveExpert({specialist1, specialist2}, ...
    'UseLocalGating', true, ...
    'Alpha', 0.1); 

hme = HierarchicalMixture(pool);

%% 3. Train
fprintf('Training HME with Library Adapter and Advanced Symmetry Breaking...\n');
trainer = EMTrainer('MaxIter', 50, 'Tol', 1e-4);
[~, ~] = trainer.train(hme, data);
% Alternatively: hme.fit(data); 

%% 4. Visualize
X_test = linspace(-2, 2, 500)';
[Y_hat, ~] = hme.predict(X_test);

% Get local hiring weights (Solution A proof)
hiring_weights = pool.getWorkerWeights(X_test);

viz = Core.getInstance();
figure('Color', 'w', 'Name', 'HME Adapter Audit');
subplot(2,1,1);
scatter(x, y, 15, 'k', 'filled', 'MarkerEdgeAlpha', 0.2); hold on;
plot(X_test, Y_hat, 'r', 'LineWidth', 2);
viz.styleAxes(gca);
title('HME Prediction using Library Specialists');
grid on;

subplot(2,1,2);
plot(X_test, hiring_weights(:, 1), 'b', 'LineWidth', 2, 'DisplayName', 'Expert 1 Weight'); hold on;
plot(X_test, hiring_weights(:, 2), 'g', 'LineWidth', 2, 'DisplayName', 'Expert 2 Weight');
viz.styleAxes(gca);
title('Solution A: Input-Dependent Hiring Weights (Mini-Gating)');
legend('Location', 'best');
grid on;

saveas(gcf, 'hme_adapter_audit.png');
fprintf('Audit saved to hme_adapter_audit.png\n');
