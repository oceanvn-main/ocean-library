% EXP_MESH_WARPING  Demonstrating Section 3.6.2 Mesh-based warping.
%
%   DATASET:        Synthetic 512 x 512 checkered board with grid lines.
%   MESH SETUP:     5x5 control point grid, with boundary locked and interior perturbed.
%   MODELS:         (1) BilinearQuadWarp (Piecewise Affine / Triangulated)
%                   (2) ThinPlateSpline (C^infinity smooth RBF interpolation)
%   ASSUMPTIONS:    Target domain coordinates correspond to mesh deformations.
%                   Backward mapping maps target coordinates to source coordinates.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-30

clear; close all; clc;
rng(42, 'twister');

disp('Initializing Experiment: Mesh-Based Warping...');
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Create Results directory if not exists
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

out_fig1 = fullfile(results_dir, 'mesh_warp_comparison.png');
out_fig2 = fullfile(results_dir, 'mesh_displacement_field.png');

%% 1. Generate Structured Source Image (Checkerboard + Grid Lines)
N = 512;
[X, Y] = meshgrid(1:N, 1:N);

% Checkerboard pattern
I_checker = mod(floor(X / 32) + floor(Y / 32), 2);

% Grid lines
I_grid = zeros(N, N);
for i = 1:32:N
    I_grid(max(1, i-1):min(N, i+1), :) = 1;
    I_grid(:, max(1, i-1):min(N, i+1)) = 1;
end

I_src = 0.7 * I_checker + 0.3 * I_grid;

%% 2. Define 5x5 Control Point Grid
disp('Setting up control point mesh grid...');
grid_size = 5;
xs_line = linspace(1, N, grid_size);
ys_line = linspace(1, N, grid_size);
[grid_src_x, grid_src_y] = meshgrid(xs_line, ys_line);

% Copy for target grid and perturb interior points
grid_tgt_x = grid_src_x;
grid_tgt_y = grid_src_y;

% Shift center point and some adjacent interior points
grid_tgt_x(3, 3) = grid_src_x(3, 3) + 70;
grid_tgt_y(3, 3) = grid_src_y(3, 3) + 70;

grid_tgt_x(2, 3) = grid_src_x(2, 3) - 30;
grid_tgt_y(3, 2) = grid_src_y(3, 2) - 30;

grid_tgt_x(4, 3) = grid_src_x(4, 3) + 25;
grid_tgt_y(3, 4) = grid_src_y(3, 4) + 25;

%% 3. Perform Bilinear Quadrilateral Warp (Piecewise Affine)
disp('Executing Bilinear Quad Warp (Triangulated)...');
I_warped_quad = OceanMath.Image.Resampling.BilinearQuadWarp.warp(...
    I_src, grid_src_x, grid_src_y, grid_tgt_x, grid_tgt_y);

%% 4. Perform Thin Plate Spline Warp (Smooth C^inf)
disp('Executing Thin Plate Spline Warp...');
% Setup target-to-source mapping for backward image warping
pts_src = [grid_src_x(:), grid_src_y(:)];
pts_tgt = [grid_tgt_x(:), grid_tgt_y(:)];

tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src);
I_warped_tps = tps.warpImage(I_src);

%% 5. Compute Error / Smoothness Analysis
% Evaluate bending energy: trace(W' * K * W)
dx = pts_tgt(:, 1) - pts_tgt(:, 1)';
dy = pts_tgt(:, 2) - pts_tgt(:, 2)';
r = sqrt(dx.^2 + dy.^2);
r_clamped = max(r, 1e-12);
K = (r.^2) .* log(r_clamped);
bending_energy_x = tps.W(:, 1)' * K * tps.W(:, 1);
bending_energy_y = tps.W(:, 2)' * K * tps.W(:, 2);
total_bending_energy = bending_energy_x + bending_energy_y;

% Verify exact landmark mapping (landmark reconstruction error)
pts_mapped_landmarks = tps.evaluate(pts_tgt);
landmark_error = mean(sqrt(sum((pts_mapped_landmarks - pts_src).^2, 2)));

%% 6. Visualization
viz = OceanViz.Core('Theme', 'Dark');

% --- Figure 1: Warp Comparison ---
f1 = figure('Name', 'Mesh-Based Warping Comparison', ...
            'Color', viz.Colors.Background, 'Position', [50, 50, 1600, 550]);

% Subplot 1: Original Image with Source Mesh Overlay
ax1_1 = subplot(1, 3, 1);
imagesc(ax1_1, I_src); colormap(ax1_1, 'gray'); axis(ax1_1, 'image');
hold(ax1_1, 'on');
% Plot mesh lines
plot(ax1_1, grid_src_x, grid_src_y, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
plot(ax1_1, grid_src_x', grid_src_y', 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
plot(ax1_1, grid_src_x(:), grid_src_y(:), 'o', 'Color', viz.Colors.Primary, 'MarkerSize', 6, 'MarkerFaceColor', viz.Colors.Primary);
hold(ax1_1, 'off');
title(ax1_1, 'Original (Source Mesh)', 'Color', viz.Colors.Text);
viz.styleAxes(ax1_1);

% Subplot 2: Bilinear Quad Warp (Piecewise Affine, Creased)
ax1_2 = subplot(1, 3, 2);
imagesc(ax1_2, I_warped_quad); colormap(ax1_2, 'gray'); axis(ax1_2, 'image');
hold(ax1_2, 'on');
plot(ax1_2, grid_tgt_x, grid_tgt_y, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
plot(ax1_2, grid_tgt_x', grid_tgt_y', 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
plot(ax1_2, grid_tgt_x(:), grid_tgt_y(:), 'o', 'Color', viz.Colors.Secondary, 'MarkerSize', 6, 'MarkerFaceColor', viz.Colors.Secondary);
hold(ax1_2, 'off');
title(ax1_2, 'Bilinear Quad Warp (Piecewise C^0)', 'Color', viz.Colors.Text);
viz.styleAxes(ax1_2);

% Subplot 3: Thin Plate Spline Warp (Smooth, C^inf)
ax1_3 = subplot(1, 3, 3);
imagesc(ax1_3, I_warped_tps); colormap(ax1_3, 'gray'); axis(ax1_3, 'image');
hold(ax1_3, 'on');
% Evaluate a dense mesh to show smooth coordinates deformation
[dense_src_x, dense_src_y] = meshgrid(linspace(1, N, 21), linspace(1, N, 21));
pts_dense_src = [dense_src_x(:), dense_src_y(:)];
% Map backward: from dense target coordinates grid to source space
tps_forward = OceanMath.Image.Resampling.ThinPlateSpline(pts_src, pts_tgt);
pts_dense_tgt = tps_forward.evaluate(pts_dense_src);
dense_tgt_x = reshape(pts_dense_tgt(:, 1), 21, 21);
dense_tgt_y = reshape(pts_dense_tgt(:, 2), 21, 21);
plot(ax1_3, dense_tgt_x, dense_tgt_y, 'Color', viz.Colors.Boundary, 'LineWidth', 1.0);
plot(ax1_3, dense_tgt_x', dense_tgt_y', 'Color', viz.Colors.Boundary, 'LineWidth', 1.0);
plot(ax1_3, grid_tgt_x(:), grid_tgt_y(:), 'o', 'Color', viz.Colors.Boundary, 'MarkerSize', 6, 'MarkerFaceColor', viz.Colors.Boundary);
hold(ax1_3, 'off');
title(ax1_3, 'Thin Plate Spline Warp (Smooth C^\infty)', 'Color', viz.Colors.Text);
viz.styleAxes(ax1_3);

drawnow;
print(f1, out_fig1, '-dpng', '-r110');

% --- Figure 2: Vector Displacement Field ---
f2 = figure('Name', 'Mesh Displacement Vector Field', ...
            'Color', viz.Colors.Background, 'Position', [100, 100, 800, 600]);
ax2 = axes(f2);
quiver(ax2, pts_src(:, 1), pts_src(:, 2), pts_tgt(:, 1) - pts_src(:, 1), pts_tgt(:, 2) - pts_src(:, 2), ...
       0, 'Color', viz.Colors.Secondary, 'LineWidth', 2.0, 'MaxHeadSize', 0.5);
hold(ax2, 'on');
plot(ax2, pts_src(:, 1), pts_src(:, 2), 'o', 'Color', viz.Colors.Primary, 'MarkerSize', 6, 'MarkerFaceColor', viz.Colors.Primary);
plot(ax2, pts_tgt(:, 1), pts_tgt(:, 2), 'x', 'Color', viz.Colors.Boundary, 'MarkerSize', 8, 'LineWidth', 2.0);
hold(ax2, 'off');
axis(ax2, 'image');
set(ax2, 'YDir', 'reverse', 'XLim', [0, N+1], 'YLim', [0, N+1]);
title(ax2, 'Mesh Control Points Displacement Vector Field', 'Color', viz.Colors.Text);
xlabel(ax2, 'X Coordinates');
ylabel(ax2, 'Y Coordinates');
grid(ax2, 'on');
viz.styleAxes(ax2);

drawnow;
print(f2, out_fig2, '-dpng', '-r110');

%% 7. Output Experiment Summary
fprintf('\n=== MESH-BASED WARPING EXPERIMENT SUMMARY ===\n');
fprintf('  Mesh Grid size           : %d x %d (%d control points)\n', grid_size, grid_size, grid_size^2);
fprintf('  Max interior perturbation: (dx, dy) = (70, 70) pixels\n');
fprintf('  TPS Landmark Error       : %.6e pixels\n', landmark_error);
fprintf('  TPS Bending Energy       : %.6f\n', total_bending_energy);
fprintf('  Bilinear Quad Warp       : Piecewise C0 continuity (creases visible)\n');
fprintf('  Thin Plate Spline Warp   : C-inf smooth continuity (no creases)\n');
fprintf('==============================================\n');

if landmark_error < 1e-9
    disp('SUCCESS: Landmark reconstruction check passed.');
else
    disp('WARNING: High landmark reconstruction error.');
end
