%% Experiment: Hybrid HME "Specialist Selection" (Robust Version)
clear; clc; close all;
addpath(genpath(pwd));

% 1. Create a "V-Shape" Linear Dataset
% Challenge: Model must select the best specialist for two intersecting lines.
N = 600;
x_val = linspace(-1, 1, N)';
X = [ones(N, 1), x_val];
Y = zeros(N, 1);

% Left Territory (x < 0): y = 2x + 4
idx_L = x_val < 0;
Y(idx_L) = 2 * x_val(idx_L) + 4 + 0.2 * randn(sum(idx_L), 1);

% Right Territory (x >= 0): y = -3x + 4
idx_R = x_val >= 0;
Y(idx_R) = -3 * x_val(idx_R) + 4 + 0.2 * randn(sum(idx_R), 1);

% 2. Construct HME with Competitive Experts
% Every leaf gets its own FRESH pool of specialists.
import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*

def_pool = @() CompetitiveExpert({...
    GenericExpert(LinearModel(), 'Density'), ...
    GenericExpert(LinearModel(), 'Density') ...
});

e1 = LeafNode(def_pool());
e2 = LeafNode(def_pool());
root = InternalNode({e1, e2});
model = HierarchicalMixture(root);

% 3. Fit (More iterations to ensure consensus across the V-kink)
fprintf('Training Hybrid HME (Homogeneous Competitive Pool)...\n');
model.fit([X, Y]);

% 4. Detailed Visualization
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Hybrid HME: Detailed Selection Audit');
tiledlayout(fig, 2, 1, 'TileSpacing', 'Compact');

% --- Tile 1: Consensus Curve ---
ax1 = nexttile; viz.styleAxes(ax1); hold(ax1, 'on');
scatter(ax1, x_val, Y, 15, 'filled', 'MarkerFaceColor', [0.4 0.4 0.4], 'MarkerFaceAlpha', 0.4, 'DisplayName', 'Challenge Data');

x_grid = linspace(-1, 1, 400)';
X_grid = [ones(400, 1), x_grid];
Y_hat = model.predict(X_grid);
plot(ax1, x_grid, Y_hat, 'Color', viz.Colors.Secondary, 'LineWidth', 4, 'DisplayName', 'Hybrid Consensus');

% Boundary Labels
line(ax1, [0 0], [0 6], 'Color', [0.8 0.2 0.2], 'LineStyle', '--', 'LineWidth', 1.5);
text(ax1, -0.5, 5.5, 'Region A: y=2x+4', 'Color', viz.Colors.Text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(ax1, 0.5, 5.5, 'Region B: y=-3x+4', 'Color', viz.Colors.Text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

ylim(ax1, [0, 6]); ylabel(ax1, 'Output (Y)'); legend(ax1, 'Location', 'northeastoutside', 'TextColor', viz.Colors.Text);
title(ax1, 'Hybrid HME: Global Approximator');

% --- Tile 2: Hiring Audit (The Specialist Distribution) ---
ax2 = nexttile; viz.styleAxes(ax2); hold(ax2, 'on');

% Categorical Labels for clarity
cats = categorical({'Leaf 1 (Worker)', 'Leaf 2 (Worker)'});
cats = reordercats(cats, {'Leaf 1 (Worker)', 'Leaf 2 (Worker)'});

% Get Priors
p1 = e1.getExpert().getWorkerWeights(); 
p2 = e2.getExpert().getWorkerWeights();

b = bar(ax2, cats, [p1; p2]);
b(1).FaceColor = viz.Colors.Primary;
b(1).DisplayName = 'Candidate 1';
b(2).FaceColor = [1 0.5 0];
b(2).DisplayName = 'Candidate 2';

ylabel(ax2, 'Selection Priority (0-1)');
ylim(ax2, [0 1.1]); grid(ax2, 'on');
legend(ax2, 'Location', 'northeastoutside', 'TextColor', viz.Colors.Text);
title(ax2, 'Internal Market Audit: "Survival of the Fittest" Weights');

saveas(gcf, 'hme_hybrid_audit.png');
fprintf('\nFinal Verified Diagnostic saved to hme_hybrid_audit.png\n');
