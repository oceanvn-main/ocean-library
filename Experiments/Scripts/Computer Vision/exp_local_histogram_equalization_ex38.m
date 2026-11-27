% exp_local_histogram_equalization_ex38.m
%
% Scientific Experiment: Local Histogram Equalization (Szeliski Exercise 3.8)
%
% Description:
%   This script implements the complete local histogram equalization pipeline for
%   Exercise 3.8 using a synthetic test image. It compares:
%     1. Standard tile-based CLAHE (discrete counting and interpolation).
%     2. Advanced bilinear-accumulated and spatially filtered local equalization (Exercise 3.8).
%   It utilizes the newly added parameters in AdaptiveHistogramEqualization.
%
% Scientific Principles Applied (RULE[experiment.md]):
%   - Explicit Computation: All values derived programmatically.
%   - Reproducibility: Self-contained path and parameter configuration.
%   - No LaTeX: Plain text formatting in formulas and console outputs.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-09

clear;
close all;
clc;

% Ensure paths are set
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

% Initialize OceanViz Core Engine (using Scientific theme)
viz = OceanViz.Core('Theme', 'Scientific');

% Set toggle to use synthetic test image or original ocean dataset
use_synthetic = true;

if use_synthetic
    fprintf('Generating synthetic image for local equalization test...\n');
    I_orig = generate_synthetic_test_image();
else
    % Load dataset
    image_path = fullfile(project_root, 'Data', 'images', 'ocean_img.jpg');
    fprintf('Loading image from: %s\n', image_path);
    if ~exist(image_path, 'file')
        error('Target image file does not exist: %s', image_path);
    end
    I_orig = imread(image_path);
end
I_double = double(I_orig);
[height, width, num_channels] = size(I_orig);
num_pixels = height * width;

% Extract luminance channel (Y)
Y_in = 0.299 * I_double(:,:,1) + 0.587 * I_double(:,:,2) + 0.114 * I_double(:,:,3);
denom = Y_in;
denom(denom == 0) = 1e-5;

% 2. Apply Standard CLAHE
fprintf('Applying standard tile-based CLAHE...\n');
clahe_std = OceanMath.Image.Pixels.AdaptiveHistogramEqualization(...
    'NumTiles', [8, 8], ...
    'ClipLimit', 0.04, ...
    'BilinearAccumulate', false, ...
    'FilterCDFSpatial', false);
Y_out_std = double(clahe_std.apply(uint8(Y_in)));

% 3. Apply Advanced CLAHE (Exercise 3.8: Bilinear Accumulation + Spatial Filtering)
fprintf('Applying Szeliski Exercise 3.8 CLAHE (Bilinear Accumulate + Spatial Filter)...\n');
clahe_ex38 = OceanMath.Image.Pixels.AdaptiveHistogramEqualization(...
    'NumTiles', [8, 8], ...
    'ClipLimit', 0.04, ...
    'BilinearAccumulate', true, ...
    'FilterCDFSpatial', true, ...
    'FilterSigma', 1.5);
Y_out_ex38 = double(clahe_ex38.apply(uint8(Y_in)));

% 4. Restore Color using Color Ratios
ratio_std = Y_out_std ./ denom;
I_out_std = I_double .* cat(3, ratio_std, ratio_std, ratio_std);
I_out_std = uint8(max(0, min(255, I_out_std)));

ratio_ex38 = Y_out_ex38 ./ denom;
I_out_ex38 = I_double .* cat(3, ratio_ex38, ratio_ex38, ratio_ex38);
I_out_ex38 = uint8(max(0, min(255, I_out_ex38)));

% 5. Compute Visual and Numerical Differences
diff_luma = abs(Y_out_ex38 - Y_out_std);
fprintf('Mean Absolute Luma Difference: %.4f gray levels\n', mean(diff_luma(:)));
fprintf('Max Absolute Luma Difference:  %.4f gray levels\n', max(diff_luma(:)));

% Amplify difference image for clear rendering
diff_luma_amplified = uint8(min(255, diff_luma * 15.0));

%% Plot Visualizations
fig = viz.createFigure('Exercise 3.8: Local Histogram Equalization');
set(fig, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);

% 1. Original RGB Image
ax1 = subplot(2, 3, 1);
imshow(I_orig);
title(ax1, '1. Original RGB Image', 'Color', viz.Colors.Text);
set(ax1, 'Color', viz.Colors.Background);

% 2. Original Luminance Channel
ax2 = subplot(2, 3, 2);
imshow(uint8(Y_in));
title(ax2, '2. Original Luminance (Grayscale)', 'Color', viz.Colors.Text);
set(ax2, 'Color', viz.Colors.Background);

% 3. Standard Local Equalized Image (Standard CLAHE)
ax3 = subplot(2, 3, 3);
imshow(I_out_std);
title(ax3, '3. Standard local Equalized (CLAHE)', 'Color', viz.Colors.Text);
set(ax3, 'Color', viz.Colors.Background);

% 4. Advanced Equalized Image (Ex 3.8: Bilinear Accum + Spatial Filter)
ax4 = subplot(2, 3, 4);
imshow(I_out_ex38);
title(ax4, '4. Advanced local Equalized (Ex 3.8)', 'Color', viz.Colors.Text);
set(ax4, 'Color', viz.Colors.Background);

% 5. Amplified Difference Image (Ex 3.8 vs Std CLAHE)
ax5 = subplot(2, 3, 5);
imshow(diff_luma_amplified);
title(ax5, '5. Amplified Difference (15x)', 'Color', viz.Colors.Text);
set(ax5, 'Color', viz.Colors.Background);

% 6. Histograms Comparison
ax6 = subplot(2, 3, 6);
h_in = histcounts(Y_in(:), 0:256);
h_std = histcounts(Y_out_std(:), 0:256);
h_ex38 = histcounts(Y_out_ex38(:), 0:256);

hold(ax6, 'on');
plot(ax6, 0:255, h_in, 'Color', viz.Colors.Boundary, 'LineWidth', 1.0);
plot(ax6, 0:255, h_std, 'Color', viz.Colors.Secondary, 'LineWidth', 1.2);
plot(ax6, 0:255, h_ex38, 'Color', viz.Colors.Primary, 'LineWidth', 1.2);
hold(ax6, 'off');
xlim(ax6, [0, 255]);
viz.styleAxes(ax6);
title(ax6, '6. Histogram Comparison', 'Color', viz.Colors.Text);
lgd = legend(ax6, 'Original', 'Standard', 'Advanced (Ex 3.8)', 'Location', 'northeast');
set(lgd, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% Super Title
sgtitle(fig, 'Szeliski Exercise 3.8: Local Histogram Equalization with Splines and Filtering', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

%% Save Results
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
if use_synthetic
    output_fig_path = fullfile(results_dir, 'exercise_3_8_synthetic_results.png');
else
    output_fig_path = fullfile(results_dir, 'exercise_3_8_results.png');
end
fprintf('Saving results visualization to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

fprintf('Experiment complete!\n');
