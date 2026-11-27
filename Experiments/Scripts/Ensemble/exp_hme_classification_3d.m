%% HME 3D Classification: Moons Challenge
% Objective: Visualize decision "Ramps" and "Cliffs" by lifting probabilities to 3D.
clear; clc; close all;
addpath(genpath(pwd));

import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*
import OceanViz.*

%% 1. Generate "Double Moons" Dataset (with OVERLAP)
fprintf('Generating Moons Dataset (High Noise/Overlap)...\n');
N = 1000;
t = linspace(0, pi, N/2)';

% Moon 1 (Upper) - Increased noise (0.1 -> 0.25)
X1 = [cos(t), sin(t)] + 0.25 * randn(N/2, 2);
Y1 = ones(N/2, 1);

% Moon 2 (Lower) - Increased noise (0.1 -> 0.25)
X2 = [1 - cos(t), 1 - sin(t) - 0.5] + 0.25 * randn(N/2, 2);
Y2 = zeros(N/2, 1);

X = [X1; X2];
Y = [Y1; Y2];
X_intercept = [ones(N, 1), X];

%% 2. Configure HME for Soft Decisions
% We use Alpha = 1.0 (No sparsity) to allow "Soft" overlapping experts.
fprintf('Configuring HME (Soft Classification Mode)...\n');

% Specialist Pool: Polynomial Logistic (Degree 2)
% Solution A (Mini-Gating) is ON, but Solution C (Sparsity) is OFF (Alpha=1.0)
def_pool = @() CompetitiveExpert({...
    GenericExpert(PolynomialLogisticModel(2), 'Mass'), ...
    GenericExpert(PolynomialLogisticModel(1), 'Mass') ...
}, 'UseLocalGating', true, 'Alpha', 1.0);

root = InternalNode({LeafNode(def_pool()), LeafNode(def_pool())});
model = HierarchicalMixture(root, 'Type', 'Classification');

%% 3. Training
fprintf('Training HME (EM Algorithm)...\n');
model.fit([X_intercept, Y]);

%% 4. 3D Lifting Visualization with OceanViz
fprintf('Generating 3D Manifold Visualization...\n');
viz = OceanViz.Core('Theme', 'Dark');

% Setup Grid for Surface (TIGHTER MARGIN to avoid wild extrapolation)
res = 80;
margin = 0.4; % Reduced from 0.8 to focus on data
x1_range = linspace(min(X(:,1))-margin, max(X(:,1))+margin, res);
x2_range = linspace(min(X(:,2))-margin, max(X(:,2))+margin, res);
[GX1, GX2] = meshgrid(x1_range, x2_range);
X_grid = [GX1(:), GX2(:)];
X_grid_intercept = [ones(size(X_grid, 1), 1), X_grid];

% Predict Probability Map
prob_map = model.predict(X_grid_intercept); 
prob_map_grid = reshape(prob_map, res, res); 

% Data structure for OceanViz
viz_data = struct();
viz_data.X = GX1;
viz_data.Y = GX2;
viz_data.Z = prob_map_grid; 
viz_data.RawX = X(:,1);
viz_data.RawY = X(:,2);
viz_data.RawZ = Y;     
viz_data.RawC = Y + 1; 

fig = viz.createFigure('HME 2D & 3D Classification Audit');
tbox = tiledlayout(fig, 1, 2, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% --- PANEL 1: 3D PROBABILITY MANIFOLD ---
ax1 = nexttile(tbox);
viz.plot(ax1, 'Surface', viz_data, ...
    'Title', '3D Probability Manifold', ...
    'XLabel', 'X1', 'YLabel', 'X2', 'ZLabel', 'P(C1|X)', ...
    'Projection', '3D', 'FaceAlpha', 0.6, 'ShowContour', false, ...
    'ViewAngle', [-45, 40]);

hold(ax1, 'on');
% Uncertainty Boundary (P = 0.5) - REFINED
[C, ~] = contour(GX1, GX2, prob_map_grid, [0.5 0.5]);
if ~isempty(C)
    idx = 1;
    while idx < size(C, 2)
        nPoints = C(2, idx);
        cx = C(1, idx+1:idx+nPoints); cy = C(2, idx+1:idx+nPoints);
        plot3(ax1, cx, cy, 0.5 * ones(size(cx)), 'Color', [1 0.8 0], 'LineWidth', 2.5);
        idx = idx + nPoints + 1;
    end
end
camlight(ax1, 'headlight'); lighting(ax1, 'gouraud');
axis(ax1, 'tight');

% --- PANEL 2: 2D PROBABILITY HEATMAP (HYBRID) ---
ax2 = nexttile(tbox);
viz.plot(ax2, 'Surface', viz_data, ...
    'Title', '2D Confidence Heatmap', ...
    'XLabel', 'X1', 'YLabel', 'X2', ...
    'Projection', '2D', ...
    'ZMode', 'Probability', ... 
    'ShowContour', true); % Natively draws the 0.5 dashed boundary

axis(ax2, 'equal'); 
axis(ax2, 'tight');
grid(ax2, 'on');
legend(ax2, 'off');

% Save final audit
saveas(fig, 'hme_classification_2d_3d.png');
fprintf('Combined Audit saved to hme_classification_2d_3d.png\n');
