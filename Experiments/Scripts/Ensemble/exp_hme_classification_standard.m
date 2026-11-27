%% HME Classification: Standard MATLAB Visualization
% Objective: Visualize HME decision boundaries using native MATLAB plotting.
clear; clc; close all;
addpath(genpath(pwd));

import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*

%% 1. Generate "Double Moons" Dataset
fprintf('Generating Moons Dataset...\n');
N = 800;
t = linspace(0, pi, N/2)';

% Moon 1 (Upper)
X1 = [cos(t), sin(t)] + 0.15 * randn(N/2, 2);
Y1 = ones(N/2, 1);

% Moon 2 (Lower)
X2 = [1 - cos(t), 1 - sin(t) - 0.5] + 0.15 * randn(N/2, 2);
Y2 = zeros(N/2, 1);

X = [X1; X2];
Y = [Y1; Y2];
X_intercept = [ones(N, 1), X];

%% 2. Configure HME
fprintf('Configuring HME...\n');
def_pool = @() CompetitiveExpert({...
    GenericExpert(PolynomialLogisticModel(2), 'Mass'), ...
    GenericExpert(PolynomialLogisticModel(1), 'Mass') ...
}, 'UseLocalGating', true, 'Alpha', 1.0);

root = InternalNode({LeafNode(def_pool()), LeafNode(def_pool())});
model = HierarchicalMixture(root, 'Type', 'Classification');

%% 3. Training
fprintf('Training HME...\n');
model.fit([X_intercept, Y]);

%% 4. Standard MATLAB Plotting
fprintf('Generating Standard Visualization...\n');
figure('Color', 'w', 'Position', [100, 100, 1000, 450]);

% Create Visualization Grid
margin = 0.5;
x = linspace(min(X(:,1))-margin, max(X(:,1))+margin, 100);
y = linspace(min(X(:,2))-margin, max(X(:,2))+margin, 100);
[GX, GY] = meshgrid(x, y);
X_grid = [GX(:), GY(:)];
X_grid_intercept = [ones(size(X_grid, 1), 1), X_grid];

% Predict
prob_map = model.predict(X_grid_intercept);
prob_grid = reshape(prob_map, 100, 100);

% --- PLOT 1: Probability Heatmap ---
subplot(1, 2, 1);
contourf(GX, GY, prob_grid, 20, 'LineColor', 'none');
hold on;
colormap(parula);
cb = colorbar;
ylabel(cb, 'P(Class 1 | X)');

% Overlay Decision Boundary (P=0.5)
contour(GX, GY, prob_grid, [0.5 0.5], 'w', 'LineWidth', 2);

% Plot Data Points
scatter(X(Y==0,1), X(Y==0,2), 30, 'b', 'filled', 'MarkerEdgeColor', 'k');
scatter(X(Y==1,1), X(Y==1,2), 30, 'r', 'filled', 'MarkerEdgeColor', 'k');

title('HME Probability Surface (Standard MATLAB)');
xlabel('X1'); ylabel('X2');
axis equal; axis tight; grid on;

% --- PLOT 2: Hard Decision Regions ---
subplot(1, 2, 2);
imagesc(x, y, prob_grid > 0.5);
set(gca, 'YDir', 'normal');
hold on;

% Plot Data Points
scatter(X(Y==0,1), X(Y==0,2), 25, 'c', 'filled', 'MarkerEdgeColor', 'k');
scatter(X(Y==1,1), X(Y==1,2), 25, 'm', 'filled', 'MarkerEdgeColor', 'k');

title('Hard Decision Regions (P > 0.5)');
xlabel('X1'); ylabel('X2');
axis equal; axis tight;

saveas(gcf, 'hme_classification_standard.png');
fprintf('Visualization saved to hme_classification_standard.png\n');
