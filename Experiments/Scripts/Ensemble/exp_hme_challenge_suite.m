%% HME Challenge Suite: "The Selection Duel" (Structural Diversity)
clear; clc; close all;
addpath(genpath(pwd));

% 1. Create a "Linear to Quadratic" Dataset
% Challenge: HME must select the Degree-1 worker for the left and Degree-2 for the right.
N = 1000;
x_val = linspace(-1, 1, N)';
X = [ones(N, 1), x_val];
Y = zeros(N, 1);

% Left Territory (-1, 0): Pure Linear
idx_L = x_val < 0;
Y(idx_L) = 4 * x_val(idx_L) + 2 + 0.1 * randn(sum(idx_L), 1);

% Right Territory (0, 1): Pure Quadratic
idx_R = x_val >= 0;
Y(idx_R) = 10 * x_val(idx_R).^2 + 2 + 0.05 * randn(sum(idx_R), 1);

% 2. Construct HME with Structural Diversity Pools
% Each leaf has a [Linear vs Quadratic] pool.
import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*

def_pool = @() CompetitiveExpert({...
    GenericExpert(LinearModel(), 'Density'), ...      % Candidate 1: Linear
    GenericExpert(PolynomialModel(2), 'Density') ... % Candidate 2: Quadratic (Deg-2)
});

e1 = LeafNode(def_pool());
e2 = LeafNode(def_pool());
root = InternalNode({e1, e2});
model = HierarchicalMixture(root);

% 3. Fit
fprintf('Training HME Selection Duel (Linear vs Polynomial Degree Competition)...\n');
model.fit([X, Y]);

% 4. Visualization (Decisive Slam Audit)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('HME Challenge Suite: The Selection Duel');
tiledlayout(fig, 2, 1, 'TileSpacing', 'Compact');

% --- Tile 1: Consensus ---
ax1 = nexttile; viz.styleAxes(ax1); hold(ax1, 'on');
scatter(ax1, x_val, Y, 15, 'filled', 'MarkerFaceColor', [0.4 0.4 0.4], 'MarkerFaceAlpha', 0.4, 'DisplayName', 'Challenge Data');

x_grid = linspace(-1, 1, 400)';
X_grid = [ones(400, 1), x_grid];
Y_hat = model.predict(X_grid);
plot(ax1, x_grid, Y_hat, 'Color', viz.Colors.Secondary, 'LineWidth', 4, 'DisplayName', 'HME Duo-Complexity Fit');

% Markers
line(ax1, [0 0], [-2 13], 'Color', [0.8 0.2 0.2], 'LineStyle', '--', 'LineWidth', 1.5);
text(ax1, -0.5, 12, 'Target: Linear (y=4x+2)', 'Color', viz.Colors.Text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(ax1, 0.5, 12, 'Target: Quadratic (y=10x^2+2)', 'Color', viz.Colors.Text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

ylim(ax1, [-2, 13]); ylabel(ax1, 'Response'); legend(ax1, 'Location', 'northeastoutside', 'TextColor', viz.Colors.Text);
title(ax1, 'Structural HME: Multi-Complexity Consensus Curve');

% --- Tile 2: Hiring Weights (The Decision Slam) ---
ax2 = nexttile; viz.styleAxes(ax2); hold(ax2, 'on');

cats = categorical({'Leaf 1 (Linear Domain)', 'Leaf 2 (Quadratic Domain)'});
cats = reordercats(cats, {'Leaf 1 (Linear Domain)', 'Leaf 2 (Quadratic Domain)'});

p1 = e1.getExpert().getWorkerWeights(); 
p2 = e2.getExpert().getWorkerWeights();

b = bar(ax2, cats, [p1; p2]);
b(1).FaceColor = viz.Colors.Primary;
b(1).DisplayName = 'Linear Specialist (Deg 1)';
b(2).FaceColor = [1 0.5 0];
b(2).DisplayName = 'Quadratic Specialist (Deg 2)';

ylabel(ax2, 'Hiring Priority');
ylim(ax2, [0 1.1]); grid(ax2, 'on');
legend(ax2, 'Location', 'northeastoutside', 'TextColor', viz.Colors.Text);
title(ax2, 'Autonomous Model Selection: Degree Competition Result');

saveas(gcf, 'hme_selection_duel.png');
fprintf('\nSelection Duel Results saved to hme_selection_duel.png\n');
fprintf('\nFinal Hiring Weights Audit:\n');
fprintf('  Leaf 1: Linear=%.4f, Quadratic=%.4f\n', p1(1), p1(2));
fprintf('  Leaf 2: Linear=%.4f, Quadratic=%.4f\n', p2(1), p2(2));
