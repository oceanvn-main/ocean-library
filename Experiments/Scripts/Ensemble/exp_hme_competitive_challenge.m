%% HME Competitive Challenge: The Tri-Spiral Labyrinth (Final Audit)
clear; clc; close all;
addpath(genpath(pwd));

import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*
import OceanViz.*

%% 1. Generate "Tri-Spiral" Dataset (3 Classes)
fprintf('Generating Tri-Spiral Dataset (3 Classes)...\n');
N = 300; 
theta = linspace(0, 3*pi, N/3)';
r1 = theta / (3*pi);
X1 = [r1 .* cos(theta), r1 .* sin(theta)] + 0.04 * randn(N/3, 2); Y1 = ones(N/3, 1);
theta2 = theta + 2*pi/3;
X2 = [r1 .* cos(theta2), r1 .* sin(theta2)] + 0.04 * randn(N/3, 2); Y2 = 2 * ones(N/3, 1);
theta3 = theta + 4*pi/3;
X3 = [r1 .* cos(theta3), r1 .* sin(theta3)] + 0.04 * randn(N/3, 2); Y3 = 3 * ones(N/3, 1);
X = [X1; X2; X3]; Y = [Y1; Y2; Y3];
X_intercept = [ones(N, 1), X];

%% 2. Configure HME with Competitive Pool
fprintf('Configuring HME (Depth 2 + Competitive Pools)...\n');
def_pool = @() CompetitiveExpert({...
    GenericExpert(LogisticModel(), 'Mass'), ...
    GenericExpert(PolynomialLogisticModel(2), 'Mass')
}, 'UseLocalGating', true, 'Alpha', 1.0);

root = InternalNode({LeafNode(def_pool()), LeafNode(def_pool())});
model = HierarchicalMixture(root, 'Type', 'Classification');
model.setTrainer(EMTrainer('MaxIter', 10)); % Fast Audit

%% 3. Training
fprintf('Training HME (EM Algorithm)...\n');
model.fit([X_intercept, Y]);

%% 4. Multi-Dimensional OceanViz Audit
fprintf('Generating 3-Panel Audit...\n');
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('HME Competitive Pool Challenge');
fig.Position = [50 100 1500 500];
tbox = tiledlayout(fig, 1, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

res = 50; margin = 0.3;
gx = linspace(min(X(:,1))-margin, max(X(:,1))+margin, res);
gy = linspace(min(X(:,2))-margin, max(X(:,2))+margin, res);
[GX, GY] = meshgrid(gx, gy);
X_grid = [ones(numel(GX), 1), GX(:), GY(:)];

prob_tensor = model.predict(X_grid);
viz_data = struct('X', GX, 'Y', GY, 'Z', prob_tensor, 'RawX', X(:,1), 'RawY', X(:,2), 'RawC', Y);

% PANEL 1: Hybrid 2D Confidence Heatmap
ax1 = nexttile(tbox);
viz.plot(ax1, 'Surface', viz_data, 'Title', 'Hybrid Decision Regions', ...
    'Projection', '2D', 'ZMode', 'Probability', 'ShowContour', true);

% PANEL 2: 3D Probability Manifold
ax2 = nexttile(tbox);
viz.plot(ax2, 'Surface', viz_data, 'Title', '3D Topological Surface', ...
    'Projection', '3D', 'FaceAlpha', 0.7, 'ViewAngle', [-35, 35]);

% PANEL 3: Specialist Resource Map
ax3 = nexttile(tbox);
gate_probs = root.getGate().predict(X_grid); % [N_grid x 2]
w1 = root.getChild(1).getExpert().getWorkerWeights(X_grid); % [N_grid x 2]
w2 = root.getChild(2).getExpert().getWorkerWeights(X_grid); % [N_grid x 2]
joint = [gate_probs(:,1) .* w1(:,1), gate_probs(:,1) .* w1(:,2), ...
         gate_probs(:,2) .* w2(:,1), gate_probs(:,2) .* w2(:,2)];
[~, active_id] = max(joint, [], 2);
imagesc(ax3, gx, gy, reshape(active_id, res, res));
set(ax3, 'YDir', 'normal'); colormap(ax3, lines(4));
cb3 = colorbar(ax3); set(cb3, 'Ticks', 1:4, 'TickLabels', {'L1:Lin', 'L1:Poly', 'L2:Lin', 'L2:Poly'}, 'Color', [0.9 0.9 0.9]);
axis(ax3, 'equal'); axis(ax3, 'tight'); grid(ax3, 'on');
title(ax3, 'Specialist Allocation'); xlabel(ax3, 'X1');

saveas(fig, 'hme_competitive_challenge.png');
fprintf('Competitive Audit saved to hme_competitive_challenge.png\n');
