% EXP_BEZIER_EXPLORER  Visual exploration of Bezier curve properties.
%
%   Demonstrates the BezierCurve class from the parametric curve system.
%   Shows how control points, degree, tangent vectors, and the convex hull
%   property work together.
%
%   COMPONENTS:
%     Panel 1 — Bezier curve with control polygon and tangent vectors
%     Panel 2 — Bernstein basis functions for the curve's degree
%     Panel 3 — Curvature profile along the curve
%     Panel 4 — Degree comparison (linear, quadratic, cubic, quartic)
%
%   ASSUMPTIONS:
%     - All control points are 2D
%     - Uses OceanViz Dark theme for all plots
%
%   REPRODUCIBILITY: Deterministic (no randomness)
%
% Author: Ocean Library
% Date:   2026-03-20

clc; clear; close all;

%% 0. Setup
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;
palette = viz.getColorSequence(6);

%% 1. Define control points (a nice S-shaped curve)
P = [0 0; 1 4; 3 5; 5 3; 7 5; 9 1];

%% =======================================================================
%  PANEL 1: Bezier Curve + Control Polygon + Tangent Vectors
%  =======================================================================
fig = viz.createFigure('Bezier Curve Explorer');

ax1 = subplot(2, 2, 1, 'Parent', fig);
viz.styleAxes(ax1);
hold(ax1, 'on');

% Build the curve
curve = OceanMath.Geometry.BezierCurve(P);
t_dense = linspace(0, 1, 300)';
C = curve.evaluate(t_dense);

% Control polygon (dashed)
plot(ax1, P(:,1), P(:,2), '--o', ...
    'Color', colors.Scatter, ...
    'MarkerSize', 10, 'MarkerFaceColor', colors.Scatter, ...
    'LineWidth', 1.2, 'DisplayName', 'Control Polygon');

% The curve itself
plot(ax1, C(:,1), C(:,2), '-', ...
    'Color', colors.Primary, 'LineWidth', 2.5, ...
    'DisplayName', 'Bezier Curve');

% Tangent arrows at several t values
t_arrows = [0, 0.2, 0.4, 0.6, 0.8, 1.0]';
P_arr = curve.evaluate(t_arrows);
T_arr = curve.tangent(t_arrows);
% Normalize and scale for display
for i = 1:numel(t_arrows)
    T_norm = T_arr(i,:) / norm(T_arr(i,:)) * 0.6;
    quiver(ax1, P_arr(i,1), P_arr(i,2), T_norm(1), T_norm(2), 0, ...
        'Color', colors.Secondary, 'LineWidth', 1.5, ...
        'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
end

% Mark start/end points
plot(ax1, P(1,1), P(1,2), 's', 'MarkerSize', 14, ...
    'MarkerFaceColor', palette(1,:), 'MarkerEdgeColor', 'none', ...
    'DisplayName', 'Start (t=0)');
plot(ax1, P(end,1), P(end,2), 'd', 'MarkerSize', 14, ...
    'MarkerFaceColor', palette(3,:), 'MarkerEdgeColor', 'none', ...
    'DisplayName', 'End (t=1)');

title(ax1, sprintf('Degree-%d Bezier Curve', curve.Degree_), 'Color', colors.Text);
xlabel(ax1, 'x'); ylabel(ax1, 'y');
legend(ax1, 'Location', 'best', 'TextColor', colors.Text);
axis(ax1, 'equal');
hold(ax1, 'off');

%% =======================================================================
%  PANEL 2: Bernstein Basis Functions
%  =======================================================================
ax2 = subplot(2, 2, 2, 'Parent', fig);
viz.styleAxes(ax2);
hold(ax2, 'on');

n = curve.Degree_;
t_basis = linspace(0, 1, 200)';
Bmat = OceanMath.Geometry.BernsteinBasis.matrix(n, t_basis);
basisPalette = viz.getColorSequence(n + 1);

for i = 0:n
    plot(ax2, t_basis, Bmat(:, i+1), '-', ...
        'Color', basisPalette(i+1,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('B_{%d,%d}', i, n));
end

% Show partition of unity as dashed sum line
plot(ax2, t_basis, sum(Bmat, 2), '--', ...
    'Color', colors.Text, 'LineWidth', 1, ...
    'DisplayName', 'Sum (=1)');

title(ax2, sprintf('Bernstein Basis (degree %d)', n), 'Color', colors.Text);
xlabel(ax2, 't'); ylabel(ax2, 'B_{i,n}(t)');
legend(ax2, 'Location', 'best', 'TextColor', colors.Text, 'NumColumns', 2);
hold(ax2, 'off');

%% =======================================================================
%  PANEL 3: Curvature Profile
%  =======================================================================
ax3 = subplot(2, 2, 3, 'Parent', fig);
viz.styleAxes(ax3);
hold(ax3, 'on');

frenet = OceanMath.Geometry.FrenetFrame(curve);
t_curv = linspace(0.01, 0.99, 200)';
kappa = frenet.curvature(t_curv);

% Color the curvature line by value using a filled area
fill(ax3, [t_curv; flipud(t_curv)], [kappa; zeros(200,1)], ...
    colors.Primary, 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
plot(ax3, t_curv, kappa, '-', ...
    'Color', colors.Primary, 'LineWidth', 2, ...
    'DisplayName', 'Curvature');

title(ax3, 'Curvature Profile kappa(t)', 'Color', colors.Text);
xlabel(ax3, 't'); ylabel(ax3, 'kappa');
legend(ax3, 'Location', 'best', 'TextColor', colors.Text);
hold(ax3, 'off');

%% =======================================================================
%  PANEL 4: Degree Comparison (same endpoints, different behavior)
%  =======================================================================
ax4 = subplot(2, 2, 4, 'Parent', fig);
viz.styleAxes(ax4);
hold(ax4, 'on');

degPalette = viz.getColorSequence(4);
degreeLabels = {'Linear (deg 1)', 'Quadratic (deg 2)', 'Cubic (deg 3)', 'Quartic (deg 4)'};

% Use subsets of P for different degrees
degSets = {P([1, end], :), ...         % degree 1: 2 points
           P([1, 3, end], :), ...      % degree 2: 3 points
           P([1, 2, 5, end], :), ...   % degree 3: 4 points
           P([1, 2, 4, 5, end], :)};   % degree 4: 5 points

for d = 1:4
    cDeg = OceanMath.Geometry.BezierCurve(degSets{d});
    Cd = cDeg.evaluate(t_dense);
    plot(ax4, Cd(:,1), Cd(:,2), '-', ...
        'Color', degPalette(d,:), 'LineWidth', 2, ...
        'DisplayName', degreeLabels{d});
    % Show control points
    pts_d = degSets{d};
    plot(ax4, pts_d(:,1), pts_d(:,2), 'o', ...
        'Color', degPalette(d,:), 'MarkerSize', 6, ...
        'MarkerFaceColor', degPalette(d,:), ...
        'HandleVisibility', 'off');
end

title(ax4, 'Degree Comparison', 'Color', colors.Text);
xlabel(ax4, 'x'); ylabel(ax4, 'y');
legend(ax4, 'Location', 'best', 'TextColor', colors.Text);
axis(ax4, 'equal');
hold(ax4, 'off');

%% =======================================================================
%  Summary
%  =======================================================================
L = curve.totalLength();
fprintf('\n=== BEZIER CURVE EXPLORER ===\n');
fprintf('  Control Points: %d\n', size(P, 1));
fprintf('  Degree:         %d\n', curve.Degree_);
fprintf('  Total Length:    %.4f\n', L);
fprintf('  Start:          [%.1f, %.1f]\n', P(1,:));
fprintf('  End:            [%.1f, %.1f]\n', P(end,:));
fprintf('  Max Curvature:  %.4f (at t=%.3f)\n', max(kappa), t_curv(kappa == max(kappa)));
fprintf('============================\n');
