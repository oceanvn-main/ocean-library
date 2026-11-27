% TEST_TPS_WARP  Simple script to test and understand ThinPlateSpline warping.
%
%   This script sets up a landmark warp, fits the ThinPlateSpline model, and
%   visualizes each stage of the process:
%     Stage 1: Sparse landmark correspondences with displacement vectors (quiver).
%     Stage 2: Deformed continuous grid showing the coordinate transformation field.
%     Stage 3: The final warped output image.
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

% Define 6 landmark points in the source image
% Format: [X, Y]
pts_src = [32,  32;
           224, 32;
           224, 224;
           32,  224;
           128, 80;
           128, 176];

% Perturb the landmarks to define the target shape
pts_tgt = pts_src;
pts_tgt(5, :) = pts_src(5, :) + [35, -25];  % Center-top point shifted
pts_tgt(6, :) = pts_src(6, :) + [-35, 25];  % Center-bottom point shifted

fprintf('Fitting Thin Plate Spline model...\n');
% To warp the image using backward mapping, we fit: Target -> Source
tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, 'thinplate');

% Generate a regular dense grid to visualize the continuous displacement
grid_spacing = 16;
[grid_x, grid_y] = meshgrid(1:grid_spacing:N, 1:grid_spacing:N);
pts_grid = [grid_x(:), grid_y(:)];

% Evaluate the mapping at each grid point
pts_grid_warped = tps.evaluate(pts_grid);
grid_tgt_x = reshape(pts_grid_warped(:, 1), size(grid_x));
grid_tgt_y = reshape(pts_grid_warped(:, 2), size(grid_y));

fprintf('Warping image using ThinPlateSpline...\n');
tic;
I_warped = tps.warpImage(I_src);
time_taken = toc;
fprintf('Warp completed in %.4f seconds.\n', time_taken);

% --- Stage-by-Stage Visualization ---
fprintf('Displaying stage-by-stage results...\n');
figure('Name', 'Thin Plate Spline Algorithmic Stages', 'Position', [100, 100, 1500, 550]);

% Stage 1: Sparse Landmark Correspondences
subplot(1, 3, 1);
imshow(I_src);
hold on;
% Draw displacement arrows
dx = pts_tgt(:, 1) - pts_src(:, 1);
dy = pts_tgt(:, 2) - pts_src(:, 2);
h_quiver = quiver(pts_src(:, 1), pts_src(:, 2), dx, dy, 0, 'Color', 'r', 'LineWidth', 2.0, 'MaxHeadSize', 0.5);
h_src = plot(pts_src(:, 1), pts_src(:, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
h_tgt = plot(pts_tgt(:, 1), pts_tgt(:, 2), 'yx', 'LineWidth', 2.0, 'MarkerSize', 8);
hold off;
legend([h_src, h_tgt, h_quiver], {'Source Landmarks', 'Target Landmarks', 'Displacement Vectors'}, ...
    'Location', 'southoutside');
title('Stage 1: Landmark Vectors (Src -> Tgt)');
axis on;

% Stage 2: Continuous Grid Deformation
subplot(1, 3, 2);
% Draw a blank canvas to show grid lines clearly
imshow(ones(N, N));
hold on;
h_grid = plot(grid_tgt_x, grid_tgt_y, 'b-', 'LineWidth', 1.0);
plot(grid_tgt_x', grid_tgt_y', 'b-', 'LineWidth', 1.0);
h_tgt_pt = plot(pts_tgt(:, 1), pts_tgt(:, 2), 'rx', 'LineWidth', 2.0, 'MarkerSize', 8);
hold off;
legend([h_grid(1), h_tgt_pt], {'Deformed Grid', 'Target Landmarks'}, ...
    'Location', 'southoutside');
title('Stage 2: Deformed Coordinate Grid');
axis on;

% Stage 3: Warped Output Image
subplot(1, 3, 3);
imshow(I_warped);
hold on;
h_tgt_final = plot(pts_tgt(:, 1), pts_tgt(:, 2), 'yo', 'MarkerFaceColor', 'y', 'MarkerSize', 6);
hold off;
legend(h_tgt_final, {'Target Landmarks'}, ...
    'Location', 'southoutside');
title('Stage 3: Warped Image');
axis on;

fprintf('Success: ThinPlateSpline stage-by-stage test completed.\n');
