%% Experiment: HME Diagnostic on Complex Non-Linear Curves
clear; clc; close all;
addpath(genpath(pwd));

% 1. Generate Synthetic "Non-Linear Curve" Data
% Challenge: A steep sine wave transitioning into a quadratic curve
N = 500;
x_val = linspace(-1, 1, N)';
X = [ones(N, 1), x_val]; 
Y = zeros(N, 1);

% Region 1 (x < 0): High-frequency Sine Wave
idx_left = x_val < 0;
Y(idx_left) = 2 * sin(1.5 * pi * x_val(idx_left)) + 0.1 * randn(sum(idx_left), 1);

% Region 2 (x >= 0): Quadratic Parabola (Inverted)
idx_right = x_val >= 0;
Y(idx_right) = -4 * (x_val(idx_right) - 0.5).^2 + 1 + 0.1 * randn(sum(idx_right), 1);

% 2. Construct 2-Level HME (Tree Structure)
import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*

% Left Department (Expected to handle the Sine wave)
e1 = LeafNode(GenericExpert(LinearModel(), 'Density'));
e2 = LeafNode(GenericExpert(LinearModel(), 'Density'));
left_gate = InternalNode({e1, e2});

% Right Department (Expected to handle the Parabola)
e3 = LeafNode(GenericExpert(LinearModel(), 'Density'));
e4 = LeafNode(GenericExpert(LinearModel(), 'Density'));
right_gate = InternalNode({e3, e4});

root_node = InternalNode({left_gate, right_gate});
model = HierarchicalMixture(root_node);

% 3. Fit Model
fprintf('Starting 2-Level HME training on Non-Linear Curves...\n');
model.fit([X, Y]);

% 4. Visualization (Optimized for Dark Mode)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('HME Non-Linear Curve Approximation');
ax = axes(fig); viz.styleAxes(ax);
hold(ax, 'on');

% Plot Data
scatter(ax, x_val, Y, 25, 'filled', 'MarkerFaceColor', [0.4 0.4 0.4], 'MarkerFaceAlpha', 0.6, 'DisplayName', 'Complex Data');

% Plot Global Prediction
xGrid = linspace(-1, 1, 300)';
XGrid = [ones(300, 1), xGrid];
Y_hat = model.predict(XGrid);
plot(ax, xGrid, Y_hat, 'Color', viz.Colors.Secondary, 'LineWidth', 4, 'DisplayName', 'HME Global Curve');

% Plot Individual Linear Experts (The "Tangents")
experts = {e1, e2, e3, e4};
colors = {viz.Colors.Primary, viz.Colors.Boundary, [0 1 0], [1 0.5 0]};
for i = 1:4
    plot(ax, xGrid, experts{i}.predict(XGrid), 'Color', colors{i}, 'LineStyle', '--', ...
        'LineWidth', 1.5, 'DisplayName', sprintf('Linear Expert %d', i));
end

% Set Y-limits to keep the visualization clean from runaway expert lines
ylim(ax, [-3.5, 3.5]);
legend(ax, 'Location', 'northeastoutside', 'TextColor', viz.Colors.Text);
title(ax, 'Hierarchical "Divide and Conquer" on Non-Linear Regimes');
saveas(gcf, 'hme_viz_nonlinear_audit.png');
fprintf('Visualization saved to hme_viz_nonlinear_audit.png\n');