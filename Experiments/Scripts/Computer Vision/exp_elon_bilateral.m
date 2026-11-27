% exp_elon_bilateral.m
%
% Scientific Experiment: Edge-Preserving Denoising on Elon Musk Image
%
% Description:
%   This script evaluates and compares standard Gaussian blur against our
%   Bilateral filter implementations (Direct vs. Bilateral Grid) on the image
%   'elon_musk.jpg'.
%
%   Visualizations are rendered in a premium dark-themed 2x2 dashboard and saved
%   to the Results/ directory.
%
% Author: Antigravity Code Assistant
% Date: 2026-06-17

clear;
close all;
clc;

% Import library classes
import OceanMath.Image.Filters.BilateralFilter
import OceanMath.Image.Pixels.SpatialFilter
import OceanViz.Core

%% 1. Set Up Directories and File Paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
data_dir = fullfile(project_root, 'Data', 'images');
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

image_path = fullfile(data_dir, 'elon_musk.jpg');
output_path = fullfile(results_dir, 'elon_bilateral_comparison.png');

if ~isfile(image_path)
    error('exp_elon_bilateral:ImageNotFound', 'Could not locate Elon Musk image at: %s', image_path);
end

%% 2. Load and Prepare Image
I_raw = imread(image_path);
I_dbl = double(I_raw) / 255.0;
% Resize high-resolution image to a manageable size to prevent out-of-memory errors
% in the direct bilateral filter's patch matrix allocation.
I_dbl = imresize(I_dbl, [512, NaN]);
[H, W, C] = size(I_dbl);

% Seed RNG for reproducibility
rng(42);

% Add moderate Gaussian noise (0.05 standard deviation)
I_noisy = max(0.0, min(1.0, I_dbl + 0.05 * randn(H, W, C)));

fprintf('Loaded image size: %dx%dx%d\n', H, W, C);
fprintf('Noisy test input generated successfully.\n');

%% 3. Apply Filters
fprintf('Applying spatial filters...\n');

% A. Standard Gaussian Blur
spatial_sig = 5.0;
range_sig = 0.15;

% Generate a standard 2D Gaussian kernel of size 31x31
k_size = 31;
limit = floor(k_size / 2);
[X, Y] = meshgrid(-limit:limit, -limit:limit);
K_gauss = exp(-(X.^2 + Y.^2) / (2 * spatial_sig^2));
K_gauss = K_gauss / sum(K_gauss(:));

gauss_filter = SpatialFilter('Kernel', K_gauss, 'Boundary', 'replicate');
tic;
I_gaussian = gauss_filter.apply(I_noisy);
t_gauss = toc;
fprintf('  Gaussian Blur applied in %.4f seconds.\n', t_gauss);

% B. Direct Bilateral Filter (O(K^2) Neighborhood Search)
bilat_direct = BilateralFilter('SpatialSigma', spatial_sig, 'RangeSigma', range_sig, ...
                               'Method', 'direct', 'Boundary', 'replicate');
tic;
I_bilat_dir = bilat_direct.apply(I_noisy);
t_bilat_dir = toc;
fprintf('  Direct Bilateral Filter applied in %.4f seconds.\n', t_bilat_dir);

% C. Bilateral Grid Filter (O(1) Dimensional Lifting)
bilat_grid = BilateralFilter('SpatialSigma', spatial_sig, 'RangeSigma', range_sig, ...
                             'Method', 'grid', 'SpatialSampling', 4.0, 'RangeSampling', 0.05, ...
                             'Boundary', 'replicate');
tic;
I_bilat_grid = bilat_grid.apply(I_noisy);
t_bilat_grid = toc;
fprintf('  Bilateral Grid Filter applied in %.4f seconds (Speedup: %.1fx).\n', ...
    t_bilat_grid, t_bilat_dir / t_bilat_grid);

%% 4. Render Dashboard using OceanViz
fprintf('Rendering OceanViz dashboard...\n');
viz = Core('Theme', 'Dark');
fig = viz.createFigure('Bilateral Filtering Sandbox: Elon Musk');
set(fig, 'Position', [100, 100, 1200, 900]);

% Create subplot axes
ax_orig  = subplot(2, 2, 1);
ax_noisy = subplot(2, 2, 2);
ax_gauss = subplot(2, 2, 3);
ax_grid  = subplot(2, 2, 4);

% Style all axes using visualizer
axes_list = [ax_orig, ax_noisy, ax_gauss, ax_grid];
for idx = 1:length(axes_list)
    viz.styleAxes(axes_list(idx));
end

% Plot original clean image
imshow(I_dbl, 'Parent', ax_orig);
title(ax_orig, 'Original Clean Image', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Plot noisy input image
imshow(I_noisy, 'Parent', ax_noisy);
title(ax_noisy, 'Noisy Input (Gaussian \sigma=0.05)', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Plot standard Gaussian blur (blurs fine structures)
imshow(I_gaussian, 'Parent', ax_gauss);
title(ax_gauss, sprintf('Gaussian Blur (\\sigma_s=%.1f)', spatial_sig), 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Plot Bilateral Grid filter (retains details and edges)
imshow(I_bilat_grid, 'Parent', ax_grid);
title(ax_grid, sprintf('Bilateral Grid (\\sigma_s=%.1f, \\sigma_r=%.2f)', spatial_sig, range_sig), 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Add main window header
annotation(fig, 'textbox', [0.05, 0.94, 0.9, 0.05], ...
    'String', 'BILATERAL FILTER EDGE-PRESERVING STUDY (ELON MUSK)', ...
    'Color', 'w', 'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'LineStyle', 'none');

drawnow;

% Save Figure
saveas(fig, output_path);

fprintf('Figure successfully saved to: %s\n', output_path);
fprintf('Execution completed successfully.\n');
