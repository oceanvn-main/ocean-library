% exp_prim_test.m
% Generates synthetic overlapping data to verify the custom PRIM implementation.
% Specifically hunts for a "bump" where the target class has high density.
clear; clc; close all;

% Ensure path is correct for packages
addpath(fullfile(pwd, '..', '..'));

%% 1. Generate Synthetic Bump Data
rng(42); % For reproducibility
% Background noise (class 0) spread across [0, 10] x [0, 10]
N_bg = 1000;
X_bg = rand(N_bg, 2) * 10;
Y_bg = zeros(N_bg, 1);

% Target bump (class 1) concentrated around [7, 7] with some variance
N_target = 200;
X_target = randn(N_target, 2) * 0.5 + 7;
Y_target = ones(N_target, 1);

X = [X_bg; X_target];
Y = [Y_bg; Y_target];

%% 2. Configure and Train PRIM
% We use TargetClass objective to hunt for regions dense with Class 1
import OceanStats.Supervised.PRIM.PRIM;
import OceanStats.Supervised.PRIM.Criterion.TargetClass;
import OceanStats.Supervised.PRIM.Selection.CVSelector;
import OceanStats.Selection.CrossValidation.KFold;

primModel = PRIM('Alpha', 0.05, 'MinBoxSize', 20, 'MaxBoxes', 2);
primModel.ObjectiveFunction = TargetClass(1);
% Use robust K-Fold CV to select the optimal sub-box from peeling sequence
primModel.BoxSelector = CVSelector(KFold(5, 'Seed', 42));

fprintf('Fitting PRIM to Bump Data...\n');
tic;
primModel.fit(X, Y);
t_fit = toc;
fprintf('PRIM fitted in %.4f seconds. Found %d boxes.\n', t_fit, length(primModel.Boxes));

%% 3. Output box metrics
for i = 1:length(primModel.Boxes)
    box = primModel.Boxes{i};
    bounds = box.Bounds;
    fprintf('Box %d: Target Class Proportion Score = %.4f, NSamples = %d\n', i, box.Score, box.NSamples);
    fprintf('  Bounds Feature 1: [%.2f, %.2f]\n', bounds(1, 1), bounds(1, 2));
    fprintf('  Bounds Feature 2: [%.2f, %.2f]\n', bounds(2, 1), bounds(2, 2));
end

%% 4. Graphical Plot using OceanViz
import OceanViz.Core;
viz = Core('Theme', 'Dark'); % Use the professional Dark theme
fig = viz.createFigure('PRIM Bump Hunting');
ax = axes(fig);
viz.styleAxes(ax);
hold(ax, 'on');

% Extract Theme Colors
c_bg = viz.ensureRow(viz.Colors.Scatter);     % Background points
c_target = viz.ensureRow(viz.Colors.Primary); % Target points
c_boxes = viz.ensureRow(viz.Colors.Boundary); % Box edges

% Plot Background
scatter(ax, X(Y==0, 1), X(Y==0, 2), 15, c_bg, 'filled', 'MarkerFaceAlpha', 0.2);
% Plot Target
scatter(ax, X(Y==1, 1), X(Y==1, 2), 25, c_target, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);

% Draw Boxes using a generated sequence of colors if multiple boxes
boxColors = viz.getColorSequence(length(primModel.Boxes));

for i = 1:length(primModel.Boxes)
    box = primModel.Boxes{i};
    bounds = box.Bounds;
    w = bounds(1, 2) - bounds(1, 1);
    h = bounds(2, 2) - bounds(2, 1);
    rectangle(ax, 'Position', [bounds(1, 1), bounds(2, 1), w, h], ...
              'EdgeColor', boxColors(i,:), 'LineWidth', 2.5, 'LineStyle', '--');
end

title(ax, 'PRIM Sequence (Dashed Boxes) Targeting Class 1', 'Color', viz.Colors.Text, 'FontSize', 12);
xlabel(ax, 'Feature 1', 'Color', viz.Colors.Text);
ylabel(ax, 'Feature 2', 'Color', viz.Colors.Text);

leg = legend(ax, 'Class 0 (Background)', 'Class 1 (Bump)', 'Location', 'best');
set(leg, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);

viz.export(fig, 'prim_bump_hunting.png');
fprintf('Saved OceanViz styled figure to prim_bump_hunting.png\n');
