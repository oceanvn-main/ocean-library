% TEST_PIECEWISE_AFFINE  Simple script to test PiecewiseAffineWarp.
%
%   This script loads a small image, sets up a 5-point triangular mesh,
%   perturbs the center vertex, warps the image using PiecewiseAffineWarp,
%   and visualizes the result side-by-side with the original mesh overlay.
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-30

clc; clear; close all;

% Set up paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

fprintf('Loading test image...\n');
% Use a checkerboard image for clear distortion tracking
N = 256;
[X, Y] = meshgrid(1:N, 1:N);
I_src = mod(floor(X / 16) + floor(Y / 16), 2);
% Add borders for visibility
I_src(1, :) = 0.5; I_src(end, :) = 0.5;
I_src(:, 1) = 0.5; I_src(:, end) = 0.5;

% Define 5-point mesh: 4 corners + 1 center point
% Format: [X, Y]
pts_src = [1,   1;     % Top-Left
           N,   1;     % Top-Right
           N,   N;     % Bottom-Right
           1,   N;     % Bottom-Left
           N/2, N/2];  % Center

% Perturb target coordinates (shift the center point to create deformation)
pts_tgt = pts_src;
pts_tgt(5, :) = pts_src(5, :) + [40, -30]; % shift center by dx=40, dy=-30

% Connectivity: 4 triangles sharing the center point
% tri(i, :) are indices into the pts matrix (1-based)
tri = [1, 2, 5;   % Top triangle
       2, 3, 5;   % Right triangle
       3, 4, 5;   % Bottom triangle
       4, 1, 5];  % Left triangle

fprintf('Warping image using PiecewiseAffineWarp...\n');
tic;
I_warped = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(I_src, pts_src, pts_tgt, tri);
time_taken = toc;
fprintf('Warp completed in %.4f seconds.\n', time_taken);

% --- Visualization ---
fprintf('Displaying results...\n');
figure('Name', 'Piecewise Affine Warp Test', 'Position', [100, 100, 1000, 500]);

% Left subplot: Original Image + Source Mesh
subplot(1, 2, 1);
imshow(I_src);
hold on;
triplot(tri, pts_src(:, 1), pts_src(:, 2), 'r-', 'LineWidth', 1.5);
plot(pts_src(:, 1), pts_src(:, 2), 'ro', 'MarkerFaceColor', 'r');
hold off;
title('Original Image (Source Mesh)');
axis on;

% Right subplot: Warped Image + Target Mesh
subplot(1, 2, 2);
imshow(I_warped);
hold on;
triplot(tri, pts_tgt(:, 1), pts_tgt(:, 2), 'y-', 'LineWidth', 1.5);
plot(pts_tgt(:, 1), pts_tgt(:, 2), 'yo', 'MarkerFaceColor', 'y');
hold off;
title('Warped Image (Target Mesh)');
axis on;

fprintf('Success: PiecewiseAffineWarp test run completed.\n');
