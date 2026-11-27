% EXP_BEZIER_CONSTRUCTION  Cubic Bezier construction visualization.
%
%   Replicates the classic Bezier curve demo showing:
%     - The cubic Bezier curve colored by dominant basis function
%     - Control polygon with labeled control points
%     - De Casteljau construction at a given t
%     - Bernstein basis functions with vertical t-marker
%     - Weight bars showing each basis function's contribution
%     - The Bernstein polynomial formula
%
%   ASSUMPTIONS:
%     - Cubic Bezier (4 control points, degree 3)
%     - Uses OceanViz Dark theme
%
%   REPRODUCIBILITY: Deterministic (no randomness)
%
% Author: Ocean Library
% Date:   2026-03-20

clc; clear; close all;

%% 0. Setup
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;
bg = colors.Background;

% 4 distinct colors for P0, P1, P2, P3
cP = [1.00, 0.20, 0.60;   % P0 — hot pink
      0.30, 0.75, 1.00;   % P1 — sky blue
      0.20, 1.00, 0.60;   % P2 — mint green
      1.00, 0.85, 0.20];  % P3 — gold

%% 1. Define control points (cubic)
P = [1 1; 3 7; 8 7; 9 2]; %P0, P1, P2, P3
n = size(P, 1) - 1;  % degree = 3

% Current parameter value for construction display
t_val = 0.35; %35% of the way along the curve

%% 2. Build curve
curve = OceanMath.Geometry.BezierCurve(P);
t_dense = linspace(0, 1, 500)';
C = curve.evaluate(t_dense);

%% 3. Compute Bernstein basis at all t and at t_val
Bmat = OceanMath.Geometry.BernsteinBasis.matrix(n, t_dense);
w = OceanMath.Geometry.BernsteinBasis.matrix(n, t_val);

%% =======================================================================
%  CREATE FIGURE
%  =======================================================================
fig = viz.createFigure('Bezier Curve Construction');
set(fig, 'Position', [100, 100, 1400, 700]);

%% =======================================================================
%  PANEL 1 (Left-Bottom): Bernstein Basis Functions
%  =======================================================================
ax1 = axes('Parent', fig, 'Position', [0.04, 0.08, 0.28, 0.42]);
viz.styleAxes(ax1);
hold(ax1, 'on');

for i = 0:n
    area(ax1, t_dense, Bmat(:, i+1), ...
        'FaceColor', cP(i+1,:), 'FaceAlpha', 0.15, ...
        'EdgeColor', cP(i+1,:), 'LineWidth', 2.2);
end

% Vertical line at t_val
xline(ax1, t_val, '-', 'Color', colors.Text, 'LineWidth', 1.5, 'Alpha', 0.8);

% Mark the weight values on the vertical line
for i = 0:n
    plot(ax1, t_val, w(i+1), 'o', ...
        'MarkerSize', 10, 'MarkerFaceColor', cP(i+1,:), ...
        'MarkerEdgeColor', 'w', 'LineWidth', 1.2);
end

xlabel(ax1, 't', 'Color', colors.Text, 'FontSize', 14);
ylabel(ax1, 'w', 'Color', colors.Text, 'FontSize', 14);
xlim(ax1, [0, 1]); ylim(ax1, [0, 1.05]);
title(ax1, '', 'Color', colors.Text);
hold(ax1, 'off');

%% =======================================================================
%  PANEL 2 (Left-Top): Bernstein Formula
%  =======================================================================
ax_formula = axes('Parent', fig, 'Position', [0.04, 0.55, 0.28, 0.40]);
set(ax_formula, 'Color', [0.08, 0.08, 0.12], 'XColor', 'none', 'YColor', 'none');
axis(ax_formula, 'off'); hold(ax_formula, 'on');

% Compute the polynomial coefficients for display
% B_{0,3} = -t^3+3t^2-3t+1,  B_{1,3} = 3t^3-6t^2+3t
% B_{2,3} = -3t^3+3t^2,      B_{3,3} = t^3
labels = {'P_0', 'P_1', 'P_2', 'P_3'};
polys  = {' -t^3+3t^2-3t+1', ...
           '  3t^3-6t^2+3t ', ...
           ' -3t^3+3t^2    ', ...
           '     t^3       '};
suffixes = {' ) +', ' ) +', ' ) +', ' )  '};

% Title
text(ax_formula, 0.05, 0.92, 'P(t) =', ...
    'Color', colors.Text, 'FontSize', 16, 'FontWeight', 'bold', ...
    'FontName', 'Consolas', 'Interpreter', 'none');

for i = 0:n
    ypos = 0.78 - i * 0.18;
    % Colored label
    text(ax_formula, 0.05, ypos, labels{i+1}, ...
        'Color', cP(i+1,:), 'FontSize', 14, 'FontWeight', 'bold', ...
        'FontName', 'Consolas', 'Interpreter', 'none');
    % Polynomial
    text(ax_formula, 0.18, ypos, ['(' polys{i+1} suffixes{i+1}], ...
        'Color', colors.Text, 'FontSize', 13, ...
        'FontName', 'Consolas', 'Interpreter', 'none');
end

hold(ax_formula, 'off');

%% =======================================================================
%  PANEL 3 (Right): Bezier Curve + Control Polygon + De Casteljau
%  =======================================================================
ax2 = axes('Parent', fig, 'Position', [0.38, 0.08, 0.38, 0.85]);
viz.styleAxes(ax2);
hold(ax2, 'on');

% --- Control polygon (thin lines between control points) ---
plot(ax2, P(:,1), P(:,2), '-', ...
    'Color', [colors.Text, 0.3], 'LineWidth', 1.2);

% --- The curve, colored by dominant basis function ---
% Find which basis function is largest at each t
[~, dominant] = max(Bmat, [], 2);
% Draw segments colored by dominance
for seg = 1:n+1
    mask = (dominant == seg);
    % Find contiguous runs
    starts = find(diff([0; mask]) == 1);
    ends   = find(diff([mask; 0]) == -1);
    for r = 1:numel(starts)
        idx = starts(r):ends(r);
        if numel(idx) < 2, continue; end
        plot(ax2, C(idx,1), C(idx,2), '-', ...
            'Color', cP(seg,:), 'LineWidth', 3.5);
    end
end

% --- De Casteljau construction at t_val ---
% Level 1: linear interpolation between consecutive points
L1 = zeros(n, 2);
for i = 1:n
    L1(i,:) = (1 - t_val) * P(i,:) + t_val * P(i+1,:);
end
% Draw level-1 construction lines
for i = 1:n
    plot(ax2, [P(i,1), P(i+1,1)], [P(i,2), P(i+1,2)], '--', ...
        'Color', [colors.Text, 0.15], 'LineWidth', 0.8);
end
plot(ax2, L1(:,1), L1(:,2), '-', ...
    'Color', [colors.Text, 0.4], 'LineWidth', 1);
plot(ax2, L1(:,1), L1(:,2), 'o', ...
    'MarkerSize', 7, 'MarkerFaceColor', [0.5 0.5 0.5], ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1);

% Level 2
L2 = zeros(n-1, 2);
for i = 1:n-1
    L2(i,:) = (1 - t_val) * L1(i,:) + t_val * L1(i+1,:);
end
plot(ax2, L2(:,1), L2(:,2), '-', ...
    'Color', [colors.Text, 0.5], 'LineWidth', 1);
plot(ax2, L2(:,1), L2(:,2), 'o', ...
    'MarkerSize', 7, 'MarkerFaceColor', [0.6 0.6 0.6], ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1);

% Level 3 (the point on the curve)
pt_on_curve = (1 - t_val) * L2(1,:) + t_val * L2(2,:);
plot(ax2, pt_on_curve(1), pt_on_curve(2), 'o', ...
    'MarkerSize', 14, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', colors.Text, 'LineWidth', 2);

% --- Control points with labels ---
for i = 1:n+1
    plot(ax2, P(i,1), P(i,2), 'o', ...
        'MarkerSize', 13, 'MarkerFaceColor', bg, ...
        'MarkerEdgeColor', cP(i,:), 'LineWidth', 2.5);
    text(ax2, P(i,1), P(i,2) + 0.45, sprintf('P_{%d}', i-1), ...
        'Color', cP(i,:), 'FontSize', 16, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Interpreter', 'tex');
end

axis(ax2, 'equal');
xlim(ax2, [min(P(:,1))-1, max(P(:,1))+1]);
ylim(ax2, [min(P(:,2))-1, max(P(:,2))+1]);
hold(ax2, 'off');

%% =======================================================================
%  PANEL 4 (Far-Right): Weight Bars
%  =======================================================================
ax3 = axes('Parent', fig, 'Position', [0.79, 0.08, 0.19, 0.45]);
set(ax3, 'Color', bg, 'XColor', 'none', 'YColor', 'none');
axis(ax3, 'off'); hold(ax3, 'on');

barH = 0.08;
maxBarW = 0.60;

for i = 0:n
    yb = 0.85 - i * 0.22;

    % Label
    text(ax3, 0.02, yb + barH/2, sprintf('W_%d', i), ...
        'Color', cP(i+1,:), 'FontSize', 14, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'middle', 'Interpreter', 'tex');

    % Background bar (dark)
    rectangle(ax3, 'Position', [0.15, yb, maxBarW, barH], ...
        'FaceColor', [0.15 0.15 0.18], 'EdgeColor', 'none', ...
        'Curvature', 0.3);

    % Filled bar proportional to weight
    fillW = max(w(i+1) * maxBarW, 0.01);
    rectangle(ax3, 'Position', [0.15, yb, fillW, barH], ...
        'FaceColor', cP(i+1,:), 'EdgeColor', 'none', ...
        'Curvature', 0.3);

    % Value text
    text(ax3, 0.78, yb + barH/2, sprintf('%.2f', w(i+1)), ...
        'Color', cP(i+1,:), 'FontSize', 13, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'middle', 'FontName', 'Consolas');
end

% t-value label
text(ax3, 0.15, 0.98, sprintf('t = %.2f', t_val), ...
    'Color', colors.Text, 'FontSize', 16, 'FontWeight', 'bold');

hold(ax3, 'off');

%% =======================================================================
%  Summary
%  =======================================================================
fprintf('\n=== BEZIER CONSTRUCTION ===\n');
fprintf('  Degree:       %d (cubic)\n', n);
fprintf('  t =           %.2f\n', t_val);
fprintf('  Weights:      W0=%.3f  W1=%.3f  W2=%.3f  W3=%.3f\n', w(1), w(2), w(3), w(4));
fprintf('  Point C(t):   [%.3f, %.3f]\n', pt_on_curve(1), pt_on_curve(2));
fprintf('  Sum of W:     %.6f (should be 1)\n', sum(w));
fprintf('===========================\n');
