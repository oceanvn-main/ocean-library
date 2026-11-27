% EXP_LOCAL_WARPING  Local image warping: Ex 3.23 Parts 1, 2, 3, & 4.
%
%   DATASET:        Synthetic Grid
%   TOPICS COVERED:
%     1. Piecewise Affine barycentric warp over Delaunay triangles (Part 1).
%     2. Thin Plate Spline (TPS) smooth global variational warp (Part 1).
%     3. Beier-Neely feature morphing using directed line segments (Part 2).
%     4. Cubic B-Spline grid Free-Form Deformation (Part 3).
%     5. Paintbrush flow field simulation via vortex swirl splatting (Part 4).
%
%   Author: Antigravity Code Assistant
%   Date: 2026-07-15

clc; clear; rng(42, 'twister');
close all;

% === 1. Setup Paths and Directories ===
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
results_dir  = fullfile(script_dir, 'Results');
addpath(genpath(project_root));

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% === 2. Load and Prepare Image ===
gridSize = 256;
[x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
I_grid = 0.5 + 0.5 * sign(sin(16 * pi * x) .* sign(sin(16 * pi * y)));
I_grid = repmat(I_grid, 1, 1, 3); % Convert to 3-channel for color visualizer

fprintf('=== EXERCISE 3.23: LOCAL IMAGE WARPING COMPREHENSIVE EXPERIMENT ===\n\n');

% === 3. Define Control Structures ===
% Grid points for Affine, RBF, and B-Spline grid (Part 1 & 3)
res = 4;
x_coords = linspace(1, gridSize, res);
y_coords = linspace(1, gridSize, res);
[X, Y] = meshgrid(x_coords, y_coords);
pts_src = [X(:), Y(:)];

% Deform the center grid points outward to create a local bulge
pts_tgt = pts_src;
center_mask = (pts_src(:, 1) > 1 & pts_src(:, 1) < gridSize) & ...
              (pts_src(:, 2) > 1 & pts_src(:, 2) < gridSize);
center = [gridSize/2 + 0.5, gridSize/2 + 0.5];
for i = 1:size(pts_src, 1)
    if center_mask(i)
        dir_vec = pts_src(i, :) - center;
        pts_tgt(i, :) = center + dir_vec * 1.35; % Push outward by 35%
    end
end

% Beier-Neely line segment coordinates (Part 2)
% We define 2 directed line segments: one horizontal, one vertical
pts_line_src_P = [0.25*gridSize, 0.45*gridSize; 0.55*gridSize, 0.25*gridSize];
pts_line_src_Q = [0.75*gridSize, 0.45*gridSize; 0.55*gridSize, 0.75*gridSize];

% Deform target lines (shift right/down and rotate slightly)
pts_line_tgt_P = pts_line_src_P + [12, 10; -15, 8];
pts_line_tgt_Q = pts_line_src_Q + [8, -12; 15, -10];

% =========================================================================
% WARP 1: Piecewise Affine Triangulation
% =========================================================================
fprintf('--- Running Piecewise Affine Warping ---\n');
tri = delaunay(pts_src(:, 1), pts_src(:, 2));
tic;
I_affine = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(I_grid, pts_src, pts_tgt, tri);
t_affine = toc;
fprintf('  Piecewise Affine completed in %.4f seconds.\n', t_affine);

% =========================================================================
% WARP 2: Thin Plate Spline (TPS RBF)
% =========================================================================
fprintf('--- Running Thin Plate Spline (TPS) ---\n');
tic;
tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, 'thinplate');
I_tps = tps.warpImage(I_grid);
t_tps = toc;
fprintf('  Thin Plate Spline completed in %.4f seconds.\n', t_tps);

% =========================================================================
% WARP 3: Cubic B-Spline Free-Form Deformation (FFD)
% =========================================================================
fprintf('--- Running Cubic B-Spline FFD Grid Warp ---\n');
tic;
I_bspline = OceanMath.Image.Resampling.BSplineWarp.warp(I_grid, pts_src, pts_tgt, [res, res]);
t_bspline = toc;
fprintf('  Cubic B-Spline FFD completed in %.4f seconds.\n', t_bspline);

% =========================================================================
% WARP 4: Beier-Neely Line Correspondence Morph
% =========================================================================
fprintf('--- Running Beier-Neely Line Morphing ---\n');
tic;
I_beier = OceanMath.Image.Resampling.BeierNeelyWarp.warp(I_grid, ...
    pts_line_tgt_P, pts_line_tgt_Q, pts_line_src_P, pts_line_src_Q);
t_beier = toc;
fprintf('  Beier-Neely Warp completed in %.4f seconds.\n', t_beier);

% =========================================================================
% WARP 5: Paintbrush Swirl Displacement Field
% =========================================================================
fprintf('--- Running Paintbrush Swirl splatting ---\n');
tic;
R_swirl = gridSize / 3;
[X_grid, Y_grid] = meshgrid(1:gridSize, 1:gridSize);
dx = X_grid - gridSize/2;
dy = Y_grid - gridSize/2;
r_dist = sqrt(dx.^2 + dy.^2);
theta = atan2(dy, dx);
% Apply swirl vortex rotation field
angle_shift = 0.65 * exp(-r_dist.^2 / (2 * (R_swirl/2.5)^2));
X_tgt = gridSize/2 + r_dist .* cos(theta + angle_shift);
Y_tgt = gridSize/2 + r_dist .* sin(theta + angle_shift);
Dx_paint = X_tgt - X_grid;
Dy_paint = Y_tgt - Y_grid;

I_paint = OceanMath.Image.Resampling.DisplacementFieldWarp.warp(I_grid, Dx_paint, Dy_paint);
t_paint = toc;
fprintf('  Paintbrush Swirl completed in %.4f seconds.\n', t_paint);

% =========================================================================
% VISUALIZATION & COMPREHENSIVE DASHBOARD
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;
white = colors.Text;

fig = figure('Name', 'Ex 3.23: Local Image Warping & Mesh Deformation', ...
    'Color', colors.Background, ...
    'Position', [50, 50, 1300, 800]);

images = {I_grid, I_affine, I_tps, I_bspline, I_beier, I_paint};
titles = {
    'Original Grid (Reference Mesh & Lines)', ...
    sprintf('Piecewise Affine (C0) (%.3fs)', t_affine), ...
    sprintf('Thin Plate Spline (TPS) (%.3fs)', t_tps), ...
    sprintf('Cubic B-Spline FFD (C2) (%.3fs)', t_bspline), ...
    sprintf('Beier-Neely Line Morph (%.3fs)', t_beier), ...
    sprintf('Paintbrush Swirl (Flow) (%.3fs)', t_paint)
};

for i = 1:6
    ax = subplot(2, 3, i);
    imshow(images{i}, 'Parent', ax);
    title(ax, titles{i}, 'Color', white, 'FontSize', 10, 'FontWeight', 'bold');
    viz.styleAxes(ax);
    hold(ax, 'on');
    
    % Overlay matching geometric control elements
    if i == 1
        % Show Delaunay mesh
        triplot(tri, pts_src(:, 1), pts_src(:, 2), 'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.0, 'Parent', ax);
        plot(ax, pts_src(:, 1), pts_src(:, 2), 'o', 'MarkerSize', 5, 'MarkerFaceColor', [1.0, 0.2, 0.2], 'MarkerEdgeColor', [1 1 1]);
        
        % Show Beier-Neely source directed lines (cyan crosses and dashed lines)
        for l = 1:size(pts_line_src_P, 1)
            plot(ax, [pts_line_src_P(l, 1), pts_line_src_Q(l, 1)], [pts_line_src_P(l, 2), pts_line_src_Q(l, 2)], ...
                '--', 'Color', [0.0, 0.8, 1.0], 'LineWidth', 1.5);
            plot(ax, pts_line_src_P(l, 1), pts_line_src_P(l, 2), 'x', 'Color', [0.0, 0.8, 1.0], 'MarkerSize', 6, 'LineWidth', 1.5);
            plot(ax, pts_line_src_Q(l, 1), pts_line_src_Q(l, 2), 'x', 'Color', [0.0, 0.8, 1.0], 'MarkerSize', 6, 'LineWidth', 1.5);
        end
    elseif i == 2
        % Piecewise Affine target mesh
        triplot(tri, pts_tgt(:, 1), pts_tgt(:, 2), 'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.0, 'Parent', ax);
        plot(ax, pts_tgt(:, 1), pts_tgt(:, 2), 'o', 'MarkerSize', 5, 'MarkerFaceColor', [1.0, 0.2, 0.2], 'MarkerEdgeColor', [1 1 1]);
    elseif i == 3 || i == 4
        % TPS and B-Spline grid overlays (regular quad grid)
        for r = 1:res
            idx_row = (r-1)*res + (1:res);
            plot(ax, pts_tgt(idx_row, 1), pts_tgt(idx_row, 2), '-', 'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.0);
        end
        for c = 1:res
            idx_col = c:res:((res-1)*res + c);
            plot(ax, pts_tgt(idx_col, 1), pts_tgt(idx_col, 2), '-', 'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.0);
        end
        plot(ax, pts_tgt(:, 1), pts_tgt(:, 2), 'o', 'MarkerSize', 5, 'MarkerFaceColor', [1.0, 0.2, 0.2], 'MarkerEdgeColor', [1 1 1]);
    elseif i == 5
        % Beier-Neely target directed lines
        for l = 1:size(pts_line_tgt_P, 1)
            plot(ax, [pts_line_tgt_P(l, 1), pts_line_tgt_Q(l, 1)], [pts_line_tgt_P(l, 2), pts_line_tgt_Q(l, 2)], ...
                '-', 'Color', [1.0, 0.4, 0.0], 'LineWidth', 2.0);
            plot(ax, pts_line_tgt_P(l, 1), pts_line_tgt_P(l, 2), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [1.0, 0.2, 0.2], 'MarkerEdgeColor', [1 1 1]);
            plot(ax, pts_line_tgt_Q(l, 1), pts_line_tgt_Q(l, 2), 's', 'MarkerSize', 6, 'MarkerFaceColor', [1.0, 0.7, 0.0], 'MarkerEdgeColor', [1 1 1]);
        end
    end
end

sgtitle(fig, 'Ex 3.23: Local Image Warping & Mesh/Feature Deformation Methods', ...
    'Color', white, 'FontSize', 14, 'FontWeight', 'bold');

output_path = fullfile(results_dir, 'local_warping_results.png');
saveas(fig, output_path);

fprintf('\nDashboard figure successfully saved to: %s\n', output_path);
fprintf('========================================\n');
