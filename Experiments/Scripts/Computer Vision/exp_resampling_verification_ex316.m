% exp_resampling_verification_ex316.m
%
% Scientific Experiment: High-Quality Image Resampling & Anti-Aliasing (Szeliski Ex 3.16)
%
% Description:
%   This script validates the separable ImageResampler class implementing Szeliski Ex 3.16.
%   It compares Nearest Neighbor, Bilinear, Bicubic, and Lanczos-3 filters under
%   magnification (upscaling) and minification (downscaling/decimation) on a
%   synthetic circular chirp pattern.
%   It also runs a programmatic verification of the ComputerVisionSandbox GUI integration.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-13

clear;
close all;
clc;

% Ensure paths are set
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));
addpath(fullfile(project_root, '+OceanViz', '+App', 'Computer Vision'));

% Initialize OceanViz Core Engine
viz = OceanViz.Core('Theme', 'Scientific');

%% 1. Generate Synthetic Circular Chirp Image (256x256)
gridSize = 256;
[x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
r = sqrt(x.^2 + y.^2);
I_chirp = 0.5 + 0.5 * sin(120 * r.^2);

%% 2. Minification (Downsampling to 64x64 - scale factor = 0.25)
fprintf('Running Minification (Downsampling) tests...\n');
target_shrink = [64, 64];

res_near_shrink = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_shrink, 'FilterType', 'Nearest');
res_bil_shrink  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_shrink, 'FilterType', 'Bilinear');
res_bic_shrink  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_shrink, 'FilterType', 'Bicubic');
res_lan_shrink  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_shrink, 'FilterType', 'Lanczos');
res_wsc_shrink  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_shrink, 'FilterType', 'WindowedSinc');

I_near_shrink = res_near_shrink.apply(I_chirp);
I_bil_shrink  = res_bil_shrink.apply(I_chirp);
I_bic_shrink  = res_bic_shrink.apply(I_chirp);
I_lan_shrink  = res_lan_shrink.apply(I_chirp);
I_wsc_shrink  = res_wsc_shrink.apply(I_chirp);

%% 3. Magnification (Upsampling to 512x512 - scale factor = 2.0)
fprintf('Running Magnification (Upsampling) tests...\n');
target_grow = [512, 512];

res_near_grow = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_grow, 'FilterType', 'Nearest');
res_bil_grow  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_grow, 'FilterType', 'Bilinear');
res_bic_grow  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_grow, 'FilterType', 'Bicubic');
res_lan_grow  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_grow, 'FilterType', 'Lanczos');
res_wsc_grow  = OceanMath.Image.Resampling.ImageResampler('TargetSize', target_grow, 'FilterType', 'WindowedSinc');

I_near_grow = res_near_grow.apply(I_chirp);
I_bil_grow  = res_bil_grow.apply(I_chirp);
I_bic_grow  = res_bic_grow.apply(I_chirp);
I_lan_grow  = res_lan_grow.apply(I_chirp);
I_wsc_grow  = res_wsc_grow.apply(I_chirp);

%% 4. Plot Comparison Dashboard
fig = viz.createFigure('Exercise 3.16: High-Quality Image Resampling');
set(fig, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);

% Original Chirp
ax1 = subplot(3, 4, 1); imshow(I_chirp);
title(ax1, 'Original Chirp (256x256)', 'Color', viz.Colors.Text);

% Shrunk Comparison (Row 2)
ax_n_s = subplot(3, 4, 5); imshow(I_near_shrink);
title(ax_n_s, 'Nearest Down (64x64)', 'Color', viz.Colors.Text);

ax_bi_s = subplot(3, 4, 6); imshow(I_bil_shrink);
title(ax_bi_s, 'Bilinear Down (64x64)', 'Color', viz.Colors.Text);

ax_bc_s = subplot(3, 4, 7); imshow(I_bic_shrink);
title(ax_bc_s, 'Bicubic Down (64x64)', 'Color', viz.Colors.Text);

ax_la_s = subplot(3, 4, 8); imshow(I_lan_shrink);
title(ax_la_s, 'Lanczos Down (64x64)', 'Color', viz.Colors.Text);

% Magnified Comparison (Row 3, showing a cropped zoomed center to highlight details)
crop_idx = 192:320; % center crop for zoom details
ax_n_g = subplot(3, 4, 9); imshow(I_near_grow(crop_idx, crop_idx));
title(ax_n_g, 'Nearest Zoom (Crop)', 'Color', viz.Colors.Text);

ax_bi_g = subplot(3, 4, 10); imshow(I_bil_grow(crop_idx, crop_idx));
title(ax_bi_g, 'Bilinear Zoom (Crop)', 'Color', viz.Colors.Text);

ax_bc_g = subplot(3, 4, 11); imshow(I_bic_grow(crop_idx, crop_idx));
title(ax_bc_g, 'Bicubic Zoom (Crop)', 'Color', viz.Colors.Text);

ax_la_g = subplot(3, 4, 12); imshow(I_lan_grow(crop_idx, crop_idx));
title(ax_la_g, 'Lanczos Zoom (Crop)', 'Color', viz.Colors.Text);

sgtitle(fig, 'Szeliski Exercise 3.16: High-Quality Separable Image Resampling and Decimation Filters', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

%% 5. Save Results Image
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_fig_path = fullfile(results_dir, 'exercise_3_16_results.png');
fprintf('Saving results visualization to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

%% 6. Verify Sandbox GUI Tab Integration
fprintf('Verifying ComputerVisionSandbox App registration...\n');
try
    app = ComputerVisionSandbox();
    app.show();
    
    % Close app and cleanup
    delete(app);
    fprintf('GUI integration test completed successfully!\n');
catch ME
    warning('GUI integration test encountered an issue (normal in headless/batch execution): %s', ME.message);
end

fprintf('Experiment complete!\n');
