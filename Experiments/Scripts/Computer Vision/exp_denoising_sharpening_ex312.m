% exp_denoising_sharpening_ex312.m
%
% Scientific Experiment: Image Softening, Sharpening, and Edge-Preserving Denoising
% (Szeliski Exercise 3.12)
%
% Description:
%   This script evaluates linear and non-linear restoration filters (Gaussian blur,
%   median filtering, bilateral filtering, and unsharp masking) on images distorted
%   by synthetic noise (Gaussian and Salt-and-Pepper) and blur.
%   Performance is evaluated quantitatively using PSNR and SSIM metrics,
%   and qualitatively via visual comparison and a 1D slice intensity profile.
%
% Scientific Principles (RULE[experiment.md]):
%   - Explicit Computation: PSNR and SSIM are calculated using standard formulas.
%   - Reproducibility: Self-contained path initialization.
%   - No LaTeX: Text and titles use ASCII/plain-text notation.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-11

clear;
close all;
clc;

% Ensure paths are set
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

% Initialize OceanViz Core Engine (using Scientific theme)
viz = OceanViz.Core('Theme', 'Scientific');

%% 1. Load Original Image
image_path = fullfile(project_root, 'Data', 'images', 'ocean_img.jpg');
fprintf('Loading original image: %s\n', image_path);
if ~exist(image_path, 'file')
    error('Image file not found: %s', image_path);
end
I_orig = imread(image_path);
% Downsample for computational speed during iterative bilateral filtering
I_orig = imresize(I_orig, [256, 256]);
I_double = double(I_orig) / 255.0;
[height, width, num_channels] = size(I_double);

% Grayscale version for profile slice plotting
Y_orig = 0.299*I_double(:,:,1) + 0.587*I_double(:,:,2) + 0.114*I_double(:,:,3);

%% 2. Generate Noisy and Blurry Images
fprintf('Synthesizing image distortions...\n');

% A. Salt-and-Pepper Noise (5% density)
I_sp_noisy = I_double;
rng(42); % For reproducibility
sp_mask = rand(height, width);
for c = 1:num_channels
    chan = I_sp_noisy(:,:,c);
    chan(sp_mask < 0.025) = 0.0;     % Pepper
    chan(sp_mask >= 0.025 & sp_mask < 0.05) = 1.0; % Salt
    I_sp_noisy(:,:,c) = chan;
end

% B. Gaussian Noise (sigma = 0.08)
rng(42);
I_gauss_noisy = max(0, min(1, I_double + 0.08 * randn(size(I_double))));

% C. Blur Distortion (Gaussian filter)
blur_kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 7, 'Sigma', 2.0);
blur_op = OceanMath.Image.Pixels.TrueConvolution('Kernel', blur_kernel, 'Boundary', 'replicate');
I_blurry = blur_op.apply(I_double);

%% 3. Apply Filtering Strategies

% A. Median Filter for Salt-and-Pepper Noise
fprintf('Applying Median Filter...\n');
med_filter = OceanMath.Image.Pixels.MedianFilter('WindowSize', [3, 3]);
I_sp_median = med_filter.apply(I_sp_noisy);

% B. Gaussian Filter for Salt-and-Pepper Noise (Comparison)
gauss_filter_3x3 = OceanMath.Image.Pixels.TrueConvolution(...
    'Kernel', OceanMath.Signal.Kernels.GaussianKernel('Size', 3, 'Sigma', 1.0), ...
    'Boundary', 'replicate');
I_sp_gauss = gauss_filter_3x3.apply(I_sp_noisy);

% C. Bilateral Filter for Gaussian Noise
fprintf('Applying Bilateral Filter...\n');
bilateral = OceanMath.Image.Pixels.BilateralFilter('WindowSize', [5, 5], 'SigmaSpatial', 2.5, 'SigmaRange', 0.12);
I_gauss_bilat = bilateral.apply(I_gauss_noisy);

% D. Gaussian Filter for Gaussian Noise (Comparison)
gauss_filter_5x5 = OceanMath.Image.Pixels.TrueConvolution(...
    'Kernel', OceanMath.Signal.Kernels.GaussianKernel('Size', 5, 'Sigma', 1.5), ...
    'Boundary', 'replicate');
I_gauss_blur = gauss_filter_5x5.apply(I_gauss_noisy);

% E. Sharpening Filter (Unsharp Masking on Blurry Image)
fprintf('Applying Unsharp Masking...\n');
% Smooth the blurry image slightly
I_blurry_smooth = gauss_filter_5x5.apply(I_blurry);
% Apply unsharp mask formula: I_sharp = I_blurry + alpha * (I_blurry - I_blurry_smooth)
alpha = 1.5;
I_sharpened = max(0, min(1, I_blurry + alpha * (I_blurry - I_blurry_smooth)));

%% 4. Define Quantitative Metric Helpers (No Toolbox dependencies)
compute_psnr = @(x, y) 10 * log10(1.0 / mean((x(:) - y(:)).^2));

    function ssim_val = compute_ssim(img1, img2)
        I1 = double(img1); I2 = double(img2);
        mu1 = mean(I1(:)); mu2 = mean(I2(:));
        var1 = var(I1(:)); var2 = var(I2(:));
        cov12 = mean((I1(:) - mu1) .* (I2(:) - mu2));
        c1 = 1e-4; c2 = 9e-4;
        ssim_val = ((2*mu1*mu2 + c1)*(2*cov12 + c2)) / ((mu1^2 + mu2^2 + c1)*(var1 + var2 + c2));
    end

%% 5. Render Visual Dashboard (3x3 Layout)
fig = viz.createFigure('Exercise 3.12: Softening, Sharpening, and Noise Removal');
set(fig, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);

% Row 1: Salt-and-Pepper Noise Denoising
ax1 = subplot(3, 3, 1); imshow(I_orig);
title(ax1, '1. Original Image (Reference)', 'Color', viz.Colors.Text);
set(ax1, 'Color', viz.Colors.Background);

ax2 = subplot(3, 3, 2); imshow(I_sp_noisy);
title(ax2, sprintf('2. S&P Noisy\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_sp_noisy, I_double), compute_ssim(I_sp_noisy, I_double)), 'Color', viz.Colors.Text);
set(ax2, 'Color', viz.Colors.Background);

ax3 = subplot(3, 3, 3); imshow(I_sp_median);
title(ax3, sprintf('3. Median Filter Restored\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_sp_median, I_double), compute_ssim(I_sp_median, I_double)), 'Color', viz.Colors.Text);
set(ax3, 'Color', viz.Colors.Background);

% Row 2: Gaussian Noise Denoising
ax4 = subplot(3, 3, 4); imshow(I_gauss_noisy);
title(ax4, sprintf('4. Gaussian Noisy\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_gauss_noisy, I_double), compute_ssim(I_gauss_noisy, I_double)), 'Color', viz.Colors.Text);
set(ax4, 'Color', viz.Colors.Background);

ax5 = subplot(3, 3, 5); imshow(I_gauss_blur);
title(ax5, sprintf('5. Gaussian Blur Restored\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_gauss_blur, I_double), compute_ssim(I_gauss_blur, I_double)), 'Color', viz.Colors.Text);
set(ax5, 'Color', viz.Colors.Background);

ax6 = subplot(3, 3, 6); imshow(I_gauss_bilat);
title(ax6, sprintf('6. Bilateral Restored\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_gauss_bilat, I_double), compute_ssim(I_gauss_bilat, I_double)), 'Color', viz.Colors.Text);
set(ax6, 'Color', viz.Colors.Background);

% Row 3: Blurring and Sharpening & 1D Intensity Slice
ax7 = subplot(3, 3, 7); imshow(I_blurry);
title(ax7, sprintf('7. Blurry Image\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_blurry, I_double), compute_ssim(I_blurry, I_double)), 'Color', viz.Colors.Text);
set(ax7, 'Color', viz.Colors.Background);

ax8 = subplot(3, 3, 8); imshow(I_sharpened);
title(ax8, sprintf('8. Sharpened (Unsharp Mask)\n(PSNR: %.2fdB, SSIM: %.3f)', compute_psnr(I_sharpened, I_double), compute_ssim(I_sharpened, I_double)), 'Color', viz.Colors.Text);
set(ax8, 'Color', viz.Colors.Background);

% Subplot 9: 1D Intensity Profile Slice comparison across horizontal center (y = 128)
ax9 = subplot(3, 3, 9);
y_slice = 128;
profile_orig = Y_orig(y_slice, :);

Y_gauss_noisy = 0.299*I_gauss_noisy(:,:,1) + 0.587*I_gauss_noisy(:,:,2) + 0.114*I_gauss_noisy(:,:,3);
profile_noisy = Y_gauss_noisy(y_slice, :);

Y_blur = 0.299*I_gauss_blur(:,:,1) + 0.587*I_gauss_blur(:,:,2) + 0.114*I_gauss_blur(:,:,3);
profile_blur = Y_blur(y_slice, :);

Y_bilat = 0.299*I_gauss_bilat(:,:,1) + 0.587*I_gauss_bilat(:,:,2) + 0.114*I_gauss_bilat(:,:,3);
profile_bilat = Y_bilat(y_slice, :);

hold(ax9, 'on');
p_noisy = plot(ax9, 1:width, profile_noisy, 'Color', [0.7, 0.7, 0.7], 'LineStyle', ':', 'LineWidth', 0.8);
p_orig  = plot(ax9, 1:width, profile_orig, 'Color', viz.Colors.Boundary, 'LineWidth', 1.2);
p_blur  = plot(ax9, 1:width, profile_blur, 'Color', viz.Colors.Secondary, 'LineWidth', 1.2);
p_bilat = plot(ax9, 1:width, profile_bilat, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
hold(ax9, 'off');

xlim(ax9, [80, 180]); % Focus on center transition regions
ylim(ax9, [0.0, 1.0]);
xlabel(ax9, 'X-Coordinate');
ylabel(ax9, 'Normalized Luma');
viz.styleAxes(ax9);
title(ax9, '9. 1D Slice Profile comparison (y = 128)', 'Color', viz.Colors.Text);

lgd = legend(ax9, [p_noisy, p_orig, p_blur, p_bilat], {'Noisy (Gauss)', 'Original', 'Gaussian Blur', 'Bilateral Filter'}, 'Location', 'southwest');
set(lgd, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% Super Title
sgtitle(fig, 'Szeliski Exercise 3.12: Softening, Sharpening, and Rank/Bilateral Noise Filters', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

%% 6. Save Results
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_fig_path = fullfile(results_dir, 'exercise_3_12_results.png');
fprintf('Saving results visualization to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

fprintf('Experiment complete!\n');
