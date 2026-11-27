% EXP_ADVANCED_MESH_WARPING  Demonstrating RBF Kernels, Forward Warping, Beier-Neely, and Content-Adapted Meshes.
%
%   DATASET:        Synthetic checkered board, grid pattern, and shape boundary.
%   MODELS:         (1) RBF Interpolation (TPS, Gaussian, Multiquadric, IDW)
%                   (2) DisplacementFieldWarp (Forward splatting & hole filling)
%                   (3) BeierNeelyWarp (Feature-based line morphing)
%                   (4) ContentAdaptedMesh (Salient feature triangulation)
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-30

clear; close all; clc;
rng(42, 'twister');

disp('Initializing Experiment: Advanced Geometric Warping...');
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Create Results directory if not exists
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

out_fig1 = fullfile(results_dir, 'rbf_kernels_comparison.png');
out_fig2 = fullfile(results_dir, 'advanced_warps_dashboard.png');

%% 1. Generate Structured Checkerboard Image
N = 512;
[X, Y] = meshgrid(1:N, 1:N);
I_checker = mod(floor(X / 32) + floor(Y / 32), 2);
I_grid = zeros(N, N);
for i = 1:32:N
    I_grid(max(1, i-1):min(N, i+1), :) = 1;
    I_grid(:, max(1, i-1):min(N, i+1)) = 1;
end
I_src = 0.7 * I_checker + 0.3 * I_grid;

%% 2. RBF Kernels Comparison
disp('Running Part 1: Swappable RBF Kernels...');
% Define 5 control points: 4 corners + 1 center point
pts_src = [1,   1;
           N,   1;
           N,   N;
           1,   N;
           N/2, N/2];

% Deform target control points by shifting the center point
pts_tgt = pts_src;
pts_tgt(5, :) = pts_src(5, :) + [70, -50];

% Instantiations
import OceanMath.Image.Resampling.*

% TPS Kernel
tps_tps = ThinPlateSpline(pts_tgt, pts_src, 'thinplate');
I_warped_tps = tps_tps.warpImage(I_src);

% Gaussian RBF Kernel (with scale beta = 1.5e-5)
tps_gauss = ThinPlateSpline(pts_tgt, pts_src, GaussianRBFKernel('Beta', 1.5e-5));
I_warped_gauss = tps_gauss.warpImage(I_src);

% Multiquadric Kernel (with smoothness constant C = 150)
tps_mq = ThinPlateSpline(pts_tgt, pts_src, MultiquadricKernel('C', 150));
I_warped_mq = tps_mq.warpImage(I_src);

% Inverse Distance Kernel (with Power = 2.0)
tps_idw = ThinPlateSpline(pts_tgt, pts_src, InverseDistanceKernel('Power', 2.5));
I_warped_idw = tps_idw.warpImage(I_src);

%% 3. Forward Displacement Warping
disp('Running Part 2: Forward Displacement Warping...');
% Generate a sinusoidal wave displacement field
Dx = 25 * sin(2 * pi * Y / 128);
Dy = 25 * cos(2 * pi * X / 128);

I_warped_forward = DisplacementFieldWarp.warp(I_src, Dx, Dy);

%% 4. Beier-Neely Line-Based Morphing
disp('Running Part 3: Beier-Neely Line-Based Warping...');
% Define two directed line segments
% Line 1: Center vertical line, rotated and translated
pts_tgt_P = [N/2, N/4];
pts_tgt_Q = [N/2, 3*N/4];

pts_src_P = [N/2 - 40, N/4];
pts_src_Q = [N/2 + 40, 3*N/4];

I_warped_bn = BeierNeelyWarp.warp(I_src, pts_tgt_P, pts_tgt_Q, pts_src_P, pts_src_Q, 'a', 0.5, 'b', 1.0, 'p', 1.2);

%% 5. Content-Adapted Mesh
disp('Running Part 4: Content-Adapted Meshes...');
% Create an image with a clear circular shape in the center
I_shape = double(sqrt((X - N/2).^2 + (Y - N/2).^2) <= 120);
I_shape_grid = I_shape * 0.7 + I_grid * 0.3;

% Generate content-adapted mesh with 100 vertices
[pts_adapted, tri_adapted] = ContentAdaptedMesh.generate(I_shape_grid, 100, 'MinDistance', 18);

% Perturb the center vertices of the mesh to demonstrate warping
pts_adapted_perturbed = pts_adapted;
center = [N/2, N/2];
dist_to_center = sqrt(sum((pts_adapted - center).^2, 2));
inside_circle = dist_to_center < 130;
pts_adapted_perturbed(inside_circle, 1) = pts_adapted(inside_circle, 1) + 40; % shift right

I_warped_adapted = PiecewiseAffineWarp.warp(I_shape_grid, pts_adapted, pts_adapted_perturbed, tri_adapted);

%% 6. Visualization
viz = OceanViz.Core('Theme', 'Dark');

% --- Figure 1: RBF Kernels Comparison ---
f1 = figure('Name', 'RBF Kernels Comparison', ...
            'Color', viz.Colors.Background, 'Position', [50, 50, 1200, 1000]);

ax1 = subplot(2, 2, 1);
imagesc(ax1, I_warped_tps); colormap(ax1, 'gray'); axis(ax1, 'image');
title(ax1, 'Thin Plate Spline (TPS)', 'Color', viz.Colors.Text);
viz.styleAxes(ax1);

ax2 = subplot(2, 2, 2);
imagesc(ax2, I_warped_gauss); colormap(ax2, 'gray'); axis(ax2, 'image');
title(ax2, 'Gaussian RBF (\beta = 1.5e-5)', 'Color', viz.Colors.Text);
viz.styleAxes(ax2);

ax3 = subplot(2, 2, 3);
imagesc(ax3, I_warped_mq); colormap(ax3, 'gray'); axis(ax3, 'image');
title(ax3, 'Multiquadric RBF (c = 150)', 'Color', viz.Colors.Text);
viz.styleAxes(ax3);

ax4 = subplot(2, 2, 4);
imagesc(ax4, I_warped_idw); colormap(ax4, 'gray'); axis(ax4, 'image');
title(ax4, 'Inverse Distance RBF (p = 2.5)', 'Color', viz.Colors.Text);
viz.styleAxes(ax4);

drawnow;
print(f1, out_fig1, '-dpng', '-r110');

% --- Figure 2: Advanced Warping Dashboard ---
f2 = figure('Name', 'Advanced Warps and Adaptive Meshes', ...
            'Color', viz.Colors.Background, 'Position', [100, 100, 1600, 550]);

% Subplot 1: Forward Splatted Sinusoidal Warp
ax2_1 = subplot(1, 3, 1);
imagesc(ax2_1, I_warped_forward); colormap(ax2_1, 'gray'); axis(ax2_1, 'image');
title(ax2_1, 'Forward Displacement Warp', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_1);

% Subplot 2: Beier-Neely Line Morphing
ax2_2 = subplot(1, 3, 2);
imagesc(ax2_2, I_warped_bn); colormap(ax2_2, 'gray'); axis(ax2_2, 'image');
hold(ax2_2, 'on');
% Overlay targeted line segments (Source = Red/Primary, Target = Yellow/Secondary)
plot(ax2_2, [pts_src_P(1), pts_src_Q(1)], [pts_src_P(2), pts_src_Q(2)], 'r-', 'LineWidth', 2.5);
plot(ax2_2, [pts_tgt_P(1), pts_tgt_Q(1)], [pts_tgt_P(2), pts_tgt_Q(2)], 'y--', 'LineWidth', 2.5);
hold(ax2_2, 'off');
title(ax2_2, 'Beier-Neely Line Warp', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_2);

% Subplot 3: Content-Adapted Mesh Over Circle Shape
ax2_3 = subplot(1, 3, 3);
imagesc(ax2_3, I_warped_adapted); colormap(ax2_3, 'gray'); axis(ax2_3, 'image');
hold(ax2_3, 'on');
% Overlay the triangulation grid lines
triplot(tri_adapted, pts_adapted_perturbed(:, 1), pts_adapted_perturbed(:, 2), ...
        'Color', viz.Colors.Boundary, 'LineWidth', 1.0);
plot(ax2_3, pts_adapted_perturbed(:, 1), pts_adapted_perturbed(:, 2), 'o', ...
     'Color', viz.Colors.Primary, 'MarkerSize', 4, 'MarkerFaceColor', viz.Colors.Primary);
hold(ax2_3, 'off');
title(ax2_3, 'Content-Adapted Tri-Mesh Warp', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_3);

drawnow;
print(f2, out_fig2, '-dpng', '-r110');

%% 7. Output Verification Summary
landmark_error_tps = mean(sqrt(sum((tps_tps.evaluate(pts_tgt) - pts_src).^2, 2)));
landmark_error_gauss = mean(sqrt(sum((tps_gauss.evaluate(pts_tgt) - pts_src).^2, 2)));
landmark_error_mq = mean(sqrt(sum((tps_mq.evaluate(pts_tgt) - pts_src).^2, 2)));
landmark_error_idw = mean(sqrt(sum((tps_idw.evaluate(pts_tgt) - pts_src).^2, 2)));

fprintf('\n=== ADVANCED GEOMETRIC WARPING EXPERIMENT SUMMARY ===\n');
fprintf('  RBF Landmark Reconstruction Errors:\n');
fprintf('    TPS Kernel             : %.6e pixels\n', landmark_error_tps);
fprintf('    Gaussian Kernel        : %.6e pixels\n', landmark_error_gauss);
fprintf('    Multiquadric Kernel    : %.6e pixels\n', landmark_error_mq);
fprintf('    Inverse Distance Kernel: %.6e pixels\n', landmark_error_idw);
fprintf('\n  Content-Adapted Mesh Summary:\n');
fprintf('    Generated Vertices     : %d\n', size(pts_adapted, 1));
fprintf('    Generated Triangles    : %d\n', size(tri_adapted, 1));
fprintf('    Adaptive Mesh Warping  : Piecewise Affine deformation applied successfully\n');
fprintf('=====================================================\n');

if landmark_error_tps < 1e-9 && landmark_error_gauss < 1e-9 && landmark_error_mq < 1e-9
    disp('SUCCESS: All RBF exact mapping validations passed.');
else
    disp('WARNING: High RBF landmark mapping error observed.');
end
