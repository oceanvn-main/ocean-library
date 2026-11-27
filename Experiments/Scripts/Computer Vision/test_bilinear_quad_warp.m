% TEST_BILINEAR_QUAD_WARP  Simple script to test BilinearQuadWarp showing each stage.
%
%   This script generates a 256x256 checkerboard, sets up a 4x4 coordinate
%   mesh grid, deforms the interior grid vertices, and visualizes each stage
%   of the algorithm:
%     Stage 1: Original quadrilateral grid overlay.
%     Stage 2: Division of each quad into two triangles (triangulation step).
%     Stage 3: Warped output with the deformed target triangular mesh.
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-30

clc; clear; close all;

% Set up paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

fprintf('Generating test image (256x256 checkerboard)...\n');
N = 256;
[X, Y] = meshgrid(1:N, 1:N);
I_src = mod(floor(X / 16) + floor(Y / 16), 2);
I_src(1, :) = 0.5; I_src(end, :) = 0.5;
I_src(:, 1) = 0.5; I_src(:, end) = 0.5;

% Define a 4x4 coordinate grid
grid_size = 4;
xs_line = linspace(1, N, grid_size);
ys_line = linspace(1, N, grid_size);
[grid_src_x, grid_src_y] = meshgrid(xs_line, ys_line);

% Copy and perturb interior grid points to create localized shear/distortion
grid_tgt_x = grid_src_x;
grid_tgt_y = grid_src_y;

grid_tgt_x(2, 2) = grid_src_x(2, 2) + 25;
grid_tgt_y(2, 2) = grid_src_y(2, 2) + 25;
grid_tgt_x(3, 3) = grid_src_x(3, 3) - 20;
grid_tgt_y(3, 3) = grid_src_y(3, 3) - 20;

% --- Replicate Algorithm Internal Steps for Visualization ---
% Flatten coordinates to vertices (N_pts x 2)
pts_src = [grid_src_x(:), grid_src_y(:)];
pts_tgt = [grid_tgt_x(:), grid_tgt_y(:)];

% Triangulate quadrilateral elements (split each quad into 2 triangles)
rows = grid_size;
cols = grid_size;
num_quads = (rows - 1) * (cols - 1);
tri = zeros(2 * num_quads, 3);
t_idx = 1;

for c = 1:(cols - 1)
    for r = 1:(rows - 1)
        idx_TL = r + (c - 1) * rows;
        idx_TR = r + c * rows;
        idx_BL = r + 1 + (c - 1) * rows;
        idx_BR = r + 1 + c * rows;

        % Triangle 1: TL, TR, BL
        tri(t_idx, :) = [idx_TL, idx_TR, idx_BL];
        t_idx = t_idx + 1;

        % Triangle 2: TR, BR, BL
        tri(t_idx, :) = [idx_TR, idx_BR, idx_BL];
        t_idx = t_idx + 1;
    end
end

fprintf('Warping image using BilinearQuadWarp...\n');
tic;
I_warped = OceanMath.Image.Resampling.BilinearQuadWarp.warp(...
    I_src, grid_src_x, grid_src_y, grid_tgt_x, grid_tgt_y);
time_taken = toc;
fprintf('Warp completed in %.4f seconds.\n', time_taken);

% --- Stage-by-Stage Visualization ---
fprintf('Displaying stage-by-stage results...\n');
figure('Name', 'Bilinear Quad Warp Algorithmic Stages', 'Position', [100, 100, 1500, 500]);

% Stage 1: Original Quad Mesh
subplot(1, 3, 1);
imshow(I_src);
hold on;
plot(grid_src_x, grid_src_y, 'r-', 'LineWidth', 1.5);
plot(grid_src_x', grid_src_y', 'r-', 'LineWidth', 1.5);
plot(grid_src_x(:), grid_src_y(:), 'ro', 'MarkerFaceColor', 'r');
hold off;
title('Stage 1: Quadrilateral Mesh');
axis on;

% Stage 2: Triangulation Step (Show diagonal splits)
subplot(1, 3, 2);
imshow(I_src);
hold on;
triplot(tri, pts_src(:, 1), pts_src(:, 2), 'm-', 'LineWidth', 1.5);
plot(pts_src(:, 1), pts_src(:, 2), 'mo', 'MarkerFaceColor', 'm');
hold off;
title('Stage 2: Split Quads to Triangles');
axis on;

% Stage 3: Piecewise Affine Warped Output + Target Mesh
subplot(1, 3, 3);
imshow(I_warped);
hold on;
triplot(tri, pts_tgt(:, 1), pts_tgt(:, 2), 'y-', 'LineWidth', 1.5);
plot(pts_tgt(:, 1), pts_tgt(:, 2), 'yo', 'MarkerFaceColor', 'y');
hold off;
title('Stage 3: Warped Image & Target Mesh');
axis on;

fprintf('Success: BilinearQuadWarp stage-by-stage test completed.\n');
