% exp_filtering_lifting.m
%
% Scientific Experiment: Advanced Linear & Non-Linear Filtering and Lifting
%
% Description:
%   This script evaluates and compares the visual output and computational
%   performance of:
%     1. Box Filter: Standard O(K^2) convolution vs. O(1) Integral Image.
%     2. Median Filter: Vectorized non-linear impulse noise rejection.
%     3. Bilateral Filter: Non-linear edge preservation via Direct search
%        vs. fast 3D Bilateral Grid (Dimensional Lifting).
%     4. Wavelet Lifting Scheme: Lossless CDF 5/3 multi-resolution denoising.
%
%   Visualizations are rendered in a premium dark-themed 2x4 dashboard and saved
%   to the Results/ directory.
%
% Scientific Principles Applied (RULE[experiment.md]):
%   - Explicit Computation: No hardcoded results; all outputs are computed.
%   - Reproducibility: Relies on local paths and deterministic noise.
%   - Avoid LaTeX: Console logs use plain-text representation.
%
% Author: Antigravity Code Assistant
% Date: 2026-06-17

clear;
close all;
clc;

% Import packages
import OceanMath.Image.Filters.BoxFilter
import OceanMath.Image.Filters.MedianFilter
import OceanMath.Image.Filters.BilateralFilter
import OceanMath.Signal.Lifting.LiftingScheme
import OceanViz.Core

%% 1. Set Up Directories and File Paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
data_dir = fullfile(project_root, 'Data', 'images');
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

image_path = fullfile(data_dir, 'Squid-game-symbols.jpg');
output_path = fullfile(results_dir, 'filtering_lifting_comparison.png');

if ~isfile(image_path)
    error('exp_filtering_lifting:ImageNotFound', 'Could not locate Squid Game symbols image at: %s', image_path);
end

%% 2. Load and Prepare Image
I_raw = imread(image_path);
I_dbl = double(I_raw) / 255.0;
[H, W, C] = size(I_dbl);

% Seed RNG for reproducibility
rng(42);

% Generate Noisy Inputs
% A. Additive Gaussian Noise (for smoothing test)
I_gaussian = max(0.0, min(1.0, I_dbl + 0.08 * randn(H, W, C)));

% B. Salt-and-Pepper Noise (for Median filter test)
I_sp = I_dbl;
noise_density = 0.06;
noise_mask_salt = rand(H, W) < (noise_density / 2);
noise_mask_pepper = rand(H, W) < (noise_density / 2);
for ch = 1:C
    ch_img = I_sp(:, :, ch);
    ch_img(noise_mask_salt) = 1.0;
    ch_img(noise_mask_pepper) = 0.0;
    I_sp(:, :, ch) = ch_img;
end

fprintf('Loaded image size: %dx%dx%d\n', H, W, C);
fprintf('Noisy test inputs generated successfully.\n');

%% 3. Apply Filters
fprintf('Applying spatial filters...\n');

% A. Box Filter (using Integral Image O(1) path)
box_size = 7;
box_filter = BoxFilter('Size', box_size, 'Boundary', 'replicate');
tic;
I_box = box_filter.apply(I_gaussian);
t_box_filt = toc;
fprintf('  Box Filter O(1) applied in %.4f seconds.\n', t_box_filt);

% B. Median Filter (vectorized O(K^2))
med_size = 5;
med_filter = MedianFilter('Size', med_size, 'Boundary', 'replicate');
tic;
I_med = med_filter.apply(I_sp);
t_med_filt = toc;
fprintf('  Median Filter applied in %.4f seconds.\n', t_med_filt);

% C. Bilateral Filter - Direct Spatial Search
spatial_sig = 4.0;
range_sig = 0.12;
bilat_direct = BilateralFilter('SpatialSigma', spatial_sig, 'RangeSigma', range_sig, ...
                               'Method', 'direct', 'Boundary', 'replicate');
tic;
I_bilat_dir = bilat_direct.apply(I_gaussian);
t_bilat_dir = toc;
fprintf('  Direct Bilateral Filter applied in %.4f seconds.\n', t_bilat_dir);

% D. Bilateral Filter - Bilateral Grid Lifting Trick
% We use spatial and range sampling values matching the sigmas
bilat_grid = BilateralFilter('SpatialSigma', spatial_sig, 'RangeSigma', range_sig, ...
                             'Method', 'grid', 'SpatialSampling', 3.0, 'RangeSampling', 0.04, ...
                             'Boundary', 'replicate');
tic;
I_bilat_grid = bilat_grid.apply(I_gaussian);
t_bilat_grid = toc;
fprintf('  Bilateral Grid Filter applied in %.4f seconds (Speedup: %.1fx).\n', ...
    t_bilat_grid, t_bilat_dir / t_bilat_grid);

%% 4. Apply Wavelet Lifting Scheme Denoising
fprintf('Applying Wavelet Lifting Scheme...\n');
levels = 2;
wavelet_transform = LiftingScheme('WaveletType', 'CDF53', 'Levels', levels);

% Forward DWT
tic;
coeffs = wavelet_transform.forward(I_gaussian);
t_dwt_fwd = toc;

% Soft Thresholding on detail subbands
thresh = 0.08;
for lvl = 1:levels
    % Keep LL as-is. Threshold LH, HL, HH
    coeffs.Levels{lvl}.LH = sign(coeffs.Levels{lvl}.LH) .* max(0, abs(coeffs.Levels{lvl}.LH) - thresh);
    coeffs.Levels{lvl}.HL = sign(coeffs.Levels{lvl}.HL) .* max(0, abs(coeffs.Levels{lvl}.HL) - thresh);
    coeffs.Levels{lvl}.HH = sign(coeffs.Levels{lvl}.HH) .* max(0, abs(coeffs.Levels{lvl}.HH) - thresh);
end

% Inverse DWT
tic;
I_wavelet_denoised = wavelet_transform.inverse(coeffs);
t_dwt_inv = toc;

I_wavelet_denoised = max(0.0, min(1.0, I_wavelet_denoised));
fprintf('  Wavelet Denoising applied (DWT Fwd: %.4fs, Inv: %.4fs).\n', t_dwt_fwd, t_dwt_inv);

%% 5. Computational Profiling (Runtime Scaling)
fprintf('Profiling computational performance...\n');

% A. Bilateral Grid O(1) vs Direct O(K^2)
sigmas = [2.0, 4.0, 6.0, 8.0, 10.0];
profile_t_dir = zeros(length(sigmas), 1);
profile_t_grid = zeros(length(sigmas), 1);

for idx = 1:length(sigmas)
    s_sig = sigmas(idx);
    
    f_dir = BilateralFilter('SpatialSigma', s_sig, 'RangeSigma', 0.1, 'Method', 'direct');
    f_grid = BilateralFilter('SpatialSigma', s_sig, 'RangeSigma', 0.1, 'Method', 'grid', ...
                             'SpatialSampling', 2.0, 'RangeSampling', 0.04);
    
    tic; f_dir.apply(I_gaussian);  profile_t_dir(idx) = toc;
    tic; f_grid.apply(I_gaussian); profile_t_grid(idx) = toc;
end

% B. Box Filter O(1) vs Standard Sliding Window Convolution
% We implement a basic sliding window 2D convolution as a baseline comparison
box_sizes = [3, 7, 11, 15, 19, 23, 27, 31];
profile_t_std = zeros(length(box_sizes), 1);
profile_t_integral = zeros(length(box_sizes), 1);

for idx = 1:length(box_sizes)
    bsz = box_sizes(idx);
    f_int = BoxFilter('Size', bsz);
    
    % Measure O(1) Integral Image
    tic; f_int.apply(I_gaussian); profile_t_integral(idx) = toc;
    
    % Measure standard spatial convolving via MATLAB's default imfilter 
    % convolving an asymmetric non-separable block of ones
    K_std = ones(bsz, bsz) / (bsz * bsz);
    tic;
    for ch = 1:C
        imfilter(I_gaussian(:, :, ch), K_std, 'replicate');
    end
    profile_t_std(idx) = toc;
end

fprintf('Profiling complete.\n');

%% 6. Render Premium Dark-Themed Dashboard using OceanViz
fprintf('Rendering OceanViz dashboard...\n');
viz = Core('Theme', 'Dark');
fig = viz.createFigure('Advanced Image Filtering and Lifting Sandbox');
set(fig, 'Position', [50, 50, 1600, 900]);

% Create subplot axes
ax_orig     = subplot(2, 4, 1);
ax_noisy    = subplot(2, 4, 2);
ax_box      = subplot(2, 4, 3);
ax_med      = subplot(2, 4, 4);
ax_bil_dir  = subplot(2, 4, 5);
ax_bil_grd  = subplot(2, 4, 6);
ax_wavelet  = subplot(2, 4, 7);
ax_profile  = subplot(2, 4, 8);

% Style all axes
axes_list = [ax_orig, ax_noisy, ax_box, ax_med, ax_bil_dir, ax_bil_grd, ax_wavelet, ax_profile];
for idx = 1:length(axes_list)
    viz.styleAxes(axes_list(idx));
end

% Plot Images
imshow(I_dbl, 'Parent', ax_orig);
title(ax_orig, 'Original Image', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Display composite noisy input: Left half Gaussian, Right half Salt-and-Pepper
I_noisy_display = I_gaussian;
I_noisy_display(:, floor(W/2):end, :) = I_sp(:, floor(W/2):end, :);
imshow(I_noisy_display, 'Parent', ax_noisy);
title(ax_noisy, 'Noisy Input (Gauss L / S&P R)', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

imshow(I_box, 'Parent', ax_box);
title(ax_box, sprintf('Box Filter (%dx%d, O(1))', box_size, box_size), 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

imshow(I_med, 'Parent', ax_med);
title(ax_med, sprintf('Median Filter (%dx%d)', med_size, med_size), 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

imshow(I_bilat_dir, 'Parent', ax_bil_dir);
title(ax_bil_dir, 'Direct Bilateral (O(K^2))', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

imshow(I_bilat_grid, 'Parent', ax_bil_grd);
title(ax_bil_grd, 'Bilateral Grid (Lifting O(1))', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

imshow(I_wavelet_denoised, 'Parent', ax_wavelet);
title(ax_wavelet, 'Wavelet Lifting CDF 5/3', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Plot Profiling Curves on ax_profile
hold(ax_profile, 'on');

% Set up twin axes for plotting both benchmarks
yyaxis(ax_profile, 'left');
plot(ax_profile, sigmas, profile_t_dir, '-o', 'Color', [1.0, 0.4, 0.4], 'LineWidth', 2.0, 'MarkerFaceColor', [1.0, 0.4, 0.4]);
plot(ax_profile, sigmas, profile_t_grid, '-s', 'Color', [0.2, 0.8, 0.4], 'LineWidth', 2.0, 'MarkerFaceColor', [0.2, 0.8, 0.4]);
ylabel(ax_profile, 'Bilateral Runtime (s)', 'Color', 'w');
set(ax_profile, 'YColor', 'w');

yyaxis(ax_profile, 'right');
plot(ax_profile, box_sizes, profile_t_std, '--d', 'Color', [0.3, 0.6, 1.0], 'LineWidth', 1.8);
plot(ax_profile, box_sizes, profile_t_integral, '-^', 'Color', [1.0, 0.8, 0.2], 'LineWidth', 2.0, 'MarkerFaceColor', [1.0, 0.8, 0.2]);
ylabel(ax_profile, 'Box Filter Runtime (s)', 'Color', 'w');
set(ax_profile, 'YColor', 'w');

xlabel(ax_profile, 'Scale (Sigma / Box Size)', 'Color', 'w');
title(ax_profile, 'Profiling Performance scaling', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');
grid(ax_profile, 'on');

legend(ax_profile, ...
    {'Bilateral Direct (Left)', 'Bilateral Grid (Left)', 'Box Standard (Right)', 'Box Integral O(1) (Right)'}, ...
    'TextColor', 'w', 'Color', 'none', 'EdgeColor', 'none', 'Location', 'northwest');

% Main visual annotations
annotation(fig, 'textbox', [0.05, 0.94, 0.9, 0.05], ...
    'String', 'IMAGE FILTERING & COMPUTATIONAL LIFTING SANDBOX (SQUID GAME SYMBOLS)', ...
    'Color', 'w', 'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'LineStyle', 'none');

drawnow;

% Save Figure
saveas(fig, output_path);
close(fig);

fprintf('Figure successfully saved to: %s\n', output_path);
fprintf('Execution completed successfully.\n');
