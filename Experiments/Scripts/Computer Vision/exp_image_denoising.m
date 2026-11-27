% exp_image_denoising.m
% =========================================================================
% Experiment: Image Denoising Benchmark & Sensitivity Analysis (Ex 3.28)
% =========================================================================
%
% PURPOSE:
%   Implement and compare four image denoising techniques:
%     1. Gaussian Filtering (Linear smoothing)
%     2. Median Filtering (Non-linear rank smoothing)
%     3. Bilateral Filtering (Edge-preserving domain/range smoothing)
%     4. Non-Local Means (NL-Means) Denoising (Patch-based similarity)
%
%   Evaluate them under two distinct noise models:
%     - Additive White Gaussian Noise (AWGN): representative of sensor/low-light noise.
%     - Salt & Pepper (Impulsive) Noise: representative of transmission dropouts.
%
%   Investigate:
%     - How does performance (PSNR & SSIM) vary with the choice of filtering parameter?
%     - Does optimal performance depend on correct noise level estimation?
%     - Which techniques perform best under which noise models?
%
% REPRODUCIBILITY: All random seeds are fixed. All outputs are deterministic.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-16
% =========================================================================

clear; close all; clc;
addpath(genpath('c:\Ocean Library\Matlab code practice'));
cd('c:\Ocean Library\Matlab code practice');

fprintf('=== Ex 3.28: Image Denoising and Noise Level Estimation Sensitivity ===\n\n');

% Set random seed for reproducible noise generation
rng(42);

%% =========================================================================
% STEP 1: LOAD REFERENCE IMAGE
% =========================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
img_path     = fullfile(project_root, 'Data', 'images', 'ocean_img.jpg');

I_rgb = im2double(imread(img_path));
I_ref = rgb2gray(I_rgb); % Perform experiment on grayscale luma channel
[H, W] = size(I_ref);
% Crop central 256x256 patch for fast computational execution
cy = round(H/2);
cx = round(W/2);
I_ref = I_ref(cy-127:cy+128, cx-127:cx+128);
[H, W] = size(I_ref);
fprintf('Loaded reference image and cropped to: %s (%d x %d)\n', img_path, H, W);

%% =========================================================================
% STEP 2: GENERATE SYNTHETIC NOISE
% =========================================================================
fprintf('\nGenerating noisy images...\n');

% 1. Additive White Gaussian Noise (AWGN) (sigma_n = 0.10)
sigma_n = 0.10;
I_noisy_gaussian = I_ref + sigma_n * randn(H, W);
I_noisy_gaussian = max(0, min(1, I_noisy_gaussian)); % clamp

% 2. Salt & Pepper (Impulsive) Noise (density = 0.08)
noise_density = 0.08;
I_noisy_sp = I_ref;
rand_mask = rand(H, W);
I_noisy_sp(rand_mask < noise_density / 2) = 0.0; % salt
I_noisy_sp(rand_mask >= noise_density / 2 & rand_mask < noise_density) = 1.0; % pepper

%% =========================================================================
% STEP 3: PERFORM PARAMETER SENSITIVITY SWEEPS
% =========================================================================
fprintf('\nRunning parameter sweeps on Gaussian noise...\n');

% Sweep ranges
gaussian_sigmas = linspace(0.2, 4.0, 10);
median_sizes    = 3:2:11;
bilateral_srs   = linspace(0.02, 0.4, 10);
nlm_hs          = linspace(0.02, 0.4, 10);

% Initialize metric stores for Gaussian noise
psnr_g_gaussian  = zeros(size(gaussian_sigmas));
ssim_g_gaussian  = zeros(size(gaussian_sigmas));
psnr_g_median    = zeros(size(median_sizes));
ssim_g_median    = zeros(size(median_sizes));
psnr_g_bilateral = zeros(size(bilateral_srs));
ssim_g_bilateral = zeros(size(bilateral_srs));
psnr_g_nlm       = zeros(size(nlm_hs));
ssim_g_nlm       = zeros(size(nlm_hs));

% --- Gaussian Filter Sweep ---
for i = 1:numel(gaussian_sigmas)
    s = gaussian_sigmas(i);
    flt = OceanMath.Image.Filters.GaussianFilter('Sigma', s);
    I_denoised = flt.apply(I_noisy_gaussian);
    psnr_g_gaussian(i) = compute_psnr(I_ref, I_denoised);
    ssim_g_gaussian(i) = compute_ssim(I_ref, I_denoised);
end

% --- Median Filter Sweep ---
for i = 1:numel(median_sizes)
    sz = median_sizes(i);
    flt = OceanMath.Image.Filters.MedianFilter('Size', sz);
    I_denoised = flt.apply(I_noisy_gaussian);
    psnr_g_median(i) = compute_psnr(I_ref, I_denoised);
    ssim_g_median(i) = compute_ssim(I_ref, I_denoised);
end

% --- Bilateral Filter Sweep ---
for i = 1:numel(bilateral_srs)
    sr = bilateral_srs(i);
    flt = OceanMath.Image.Filters.BilateralFilter('SpatialSigma', 3.0, 'RangeSigma', sr, 'Method', 'grid');
    I_denoised = flt.apply(I_noisy_gaussian);
    psnr_g_bilateral(i) = compute_psnr(I_ref, I_denoised);
    ssim_g_bilateral(i) = compute_ssim(I_ref, I_denoised);
end

% --- NL-Means Sweep ---
for i = 1:numel(nlm_hs)
    h = nlm_hs(i);
    flt = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 5, 'PatchRadius', 1, 'FilteringParameter', h);
    I_denoised = flt.apply(I_noisy_gaussian);
    psnr_g_nlm(i) = compute_psnr(I_ref, I_denoised);
    ssim_g_nlm(i) = compute_ssim(I_ref, I_denoised);
end


fprintf('Running parameter sweeps on Salt & Pepper noise...\n');

% Initialize metric stores for S&P noise
psnr_sp_gaussian  = zeros(size(gaussian_sigmas));
ssim_sp_gaussian  = zeros(size(gaussian_sigmas));
psnr_sp_median    = zeros(size(median_sizes));
ssim_sp_median    = zeros(size(median_sizes));
psnr_sp_bilateral = zeros(size(bilateral_srs));
ssim_sp_bilateral = zeros(size(bilateral_srs));
psnr_sp_nlm       = zeros(size(nlm_hs));
ssim_sp_nlm       = zeros(size(nlm_hs));

% --- Gaussian Filter Sweep ---
for i = 1:numel(gaussian_sigmas)
    s = gaussian_sigmas(i);
    flt = OceanMath.Image.Filters.GaussianFilter('Sigma', s);
    I_denoised = flt.apply(I_noisy_sp);
    psnr_sp_gaussian(i) = compute_psnr(I_ref, I_denoised);
    ssim_sp_gaussian(i) = compute_ssim(I_ref, I_denoised);
end

% --- Median Filter Sweep ---
for i = 1:numel(median_sizes)
    sz = median_sizes(i);
    flt = OceanMath.Image.Filters.MedianFilter('Size', sz);
    I_denoised = flt.apply(I_noisy_sp);
    psnr_sp_median(i) = compute_psnr(I_ref, I_denoised);
    ssim_sp_median(i) = compute_ssim(I_ref, I_denoised);
end

% --- Bilateral Filter Sweep ---
for i = 1:numel(bilateral_srs)
    sr = bilateral_srs(i);
    flt = OceanMath.Image.Filters.BilateralFilter('SpatialSigma', 3.0, 'RangeSigma', sr, 'Method', 'grid');
    I_denoised = flt.apply(I_noisy_sp);
    psnr_sp_bilateral(i) = compute_psnr(I_ref, I_denoised);
    ssim_sp_bilateral(i) = compute_ssim(I_ref, I_denoised);
end

% --- NL-Means Sweep ---
for i = 1:numel(nlm_hs)
    h = nlm_hs(i);
    flt = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 5, 'PatchRadius', 1, 'FilteringParameter', h);
    I_denoised = flt.apply(I_noisy_sp);
    psnr_sp_nlm(i) = compute_psnr(I_ref, I_denoised);
    ssim_sp_nlm(i) = compute_ssim(I_ref, I_denoised);
end

%% =========================================================================
% STEP 4: FIND OPTIMAL DENOISED OUTPUTS
% =========================================================================
[~, idx_opt_g_gauss] = max(psnr_g_gaussian);
[~, idx_opt_g_med]   = max(psnr_g_median);
[~, idx_opt_g_bilat] = max(psnr_g_bilateral);
[~, idx_opt_g_nlm]   = max(psnr_g_nlm);

opt_g_gaussian_sigma = gaussian_sigmas(idx_opt_g_gauss);
opt_g_median_size    = median_sizes(idx_opt_g_med);
opt_g_bilateral_sr   = bilateral_srs(idx_opt_g_bilat);
opt_g_nlm_h          = nlm_hs(idx_opt_g_nlm);

I_opt_g_gauss = OceanMath.Image.Filters.GaussianFilter('Sigma', opt_g_gaussian_sigma).apply(I_noisy_gaussian);
I_opt_g_med   = OceanMath.Image.Filters.MedianFilter('Size', opt_g_median_size).apply(I_noisy_gaussian);
I_opt_g_bilat = OceanMath.Image.Filters.BilateralFilter('SpatialSigma', 3.0, 'RangeSigma', opt_g_bilateral_sr, 'Method', 'grid').apply(I_noisy_gaussian);
I_opt_g_nlm   = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 5, 'PatchRadius', 1, 'FilteringParameter', opt_g_nlm_h).apply(I_noisy_gaussian);

[~, idx_opt_sp_gauss] = max(psnr_sp_gaussian);
[~, idx_opt_sp_med]   = max(psnr_sp_median);
[~, idx_opt_sp_bilat] = max(psnr_sp_bilateral);
[~, idx_opt_sp_nlm]   = max(psnr_sp_nlm);

opt_sp_gaussian_sigma = gaussian_sigmas(idx_opt_sp_gauss);
opt_sp_median_size    = median_sizes(idx_opt_sp_med);
opt_sp_bilateral_sr   = bilateral_srs(idx_opt_sp_bilat);
opt_sp_nlm_h          = nlm_hs(idx_opt_sp_nlm);

I_opt_sp_gauss = OceanMath.Image.Filters.GaussianFilter('Sigma', opt_sp_gaussian_sigma).apply(I_noisy_sp);
I_opt_sp_med   = OceanMath.Image.Filters.MedianFilter('Size', opt_sp_median_size).apply(I_noisy_sp);
I_opt_sp_bilat = OceanMath.Image.Filters.BilateralFilter('SpatialSigma', 3.0, 'RangeSigma', opt_sp_bilateral_sr, 'Method', 'grid').apply(I_noisy_sp);
I_opt_sp_nlm   = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 5, 'PatchRadius', 1, 'FilteringParameter', opt_sp_nlm_h).apply(I_noisy_sp);

%% =========================================================================
% STEP 5: VISUALIZE RESULTS & CURVES (OceanViz Theme)
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');
theme = viz.Colors;

results_dir = fullfile(project_root, 'Results', 'Computer Vision');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

% --- Figure 1: AWGN Denoising Comparison ---
fig1 = figure('Name', 'AWGN Denoising Comparison', 'Position', [100, 100, 1000, 650], 'Color', theme.Background);
tiledlayout(fig1, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

show_img(fig1, 1, I_ref, 'Original Reference', theme);
show_img(fig1, 2, I_noisy_gaussian, sprintf('Noisy Gaussian (\\sigma_n = %.2f)\nPSNR: %.2fdB', sigma_n, compute_psnr(I_ref, I_noisy_gaussian)), theme);
show_img(fig1, 3, I_opt_g_gauss, sprintf('Gaussian (\\sigma = %.2f)\nPSNR: %.2fdB, SSIM: %.3f', opt_g_gaussian_sigma, psnr_g_gaussian(idx_opt_g_gauss), ssim_g_gaussian(idx_opt_g_gauss)), theme);
show_img(fig1, 4, I_opt_g_med, sprintf('Median (Size = %d)\nPSNR: %.2fdB, SSIM: %.3f', opt_g_median_size, psnr_g_median(idx_opt_g_med), ssim_g_median(idx_opt_g_med)), theme);
show_img(fig1, 5, I_opt_g_bilat, sprintf('Bilateral (\\sigma_r = %.2f)\nPSNR: %.2fdB, SSIM: %.3f', opt_g_bilateral_sr, psnr_g_bilateral(idx_opt_g_bilat), ssim_g_bilateral(idx_opt_g_bilat)), theme);
show_img(fig1, 6, I_opt_g_nlm, sprintf('NL-Means (h = %.2f)\nPSNR: %.2fdB, SSIM: %.3f', opt_g_nlm_h, psnr_g_nlm(idx_opt_g_nlm), ssim_g_nlm(idx_opt_g_nlm)), theme);

exportgraphics(fig1, fullfile(results_dir, 'denoising_awgn_comparison.png'), 'Resolution', 150);

% --- Figure 2: Salt & Pepper Denoising Comparison ---
fig2 = figure('Name', 'Salt & Pepper Denoising Comparison', 'Position', [120, 120, 1000, 650], 'Color', theme.Background);
tiledlayout(fig2, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

show_img(fig2, 1, I_ref, 'Original Reference', theme);
show_img(fig2, 2, I_noisy_sp, sprintf('Noisy S&P (Density = %.2f)\nPSNR: %.2fdB', noise_density, compute_psnr(I_ref, I_noisy_sp)), theme);
show_img(fig2, 3, I_opt_sp_gauss, sprintf('Gaussian (\\sigma = %.2f)\nPSNR: %.2fdB, SSIM: %.3f', opt_sp_gaussian_sigma, psnr_sp_gaussian(idx_opt_sp_gauss), ssim_sp_gaussian(idx_opt_sp_gauss)), theme);
show_img(fig2, 4, I_opt_sp_med, sprintf('Median (Size = %d)\nPSNR: %.2fdB, SSIM: %.3f', opt_sp_median_size, psnr_sp_median(idx_opt_sp_med), ssim_sp_median(idx_opt_sp_med)), theme);
show_img(fig2, 5, I_opt_sp_bilat, sprintf('Bilateral (\\sigma_r = %.2f)\nPSNR: %.2fdB, SSIM: %.3f', opt_sp_bilateral_sr, psnr_sp_bilateral(idx_opt_sp_bilat), ssim_sp_bilateral(idx_opt_sp_bilat)), theme);
show_img(fig2, 6, I_opt_sp_nlm, sprintf('NL-Means (h = %.2f)\nPSNR: %.2fdB, SSIM: %.3f', opt_sp_nlm_h, psnr_sp_nlm(idx_opt_sp_nlm), ssim_sp_nlm(idx_opt_sp_nlm)), theme);

exportgraphics(fig2, fullfile(results_dir, 'denoising_sp_comparison.png'), 'Resolution', 150);

% --- Figure 3: Sensitivity Curves (AWGN) ---
fig3 = figure('Name', 'Noise Level Sensitivity (AWGN)', 'Position', [140, 140, 950, 420], 'Color', theme.Background);
t = tiledlayout(fig3, 1, 2, 'Padding', 'compact', 'TileSpacing', 'loose');

% PSNR Plot
ax_psnr = nexttile(t);
viz.styleAxes(ax_psnr);
hold(ax_psnr, 'on');
plot(ax_psnr, gaussian_sigmas, psnr_g_gaussian, '-o', 'Color', theme.Primary, 'LineWidth', 2, 'DisplayName', 'Gaussian (\sigma)');
plot(ax_psnr, bilateral_srs, psnr_g_bilateral, '-s', 'Color', theme.Secondary, 'LineWidth', 2, 'DisplayName', 'Bilateral (\sigma_r)');
plot(ax_psnr, nlm_hs, psnr_g_nlm, '-^', 'Color', theme.Boundary, 'LineWidth', 2, 'DisplayName', 'NL-Means (h)');
xlabel(ax_psnr, 'Filtering Parameter Value', 'Color', theme.Text);
ylabel(ax_psnr, 'PSNR (dB)', 'Color', theme.Text);
title(ax_psnr, 'PSNR vs Filtering Scale (AWGN)', 'Color', theme.Text);
lgd1 = legend(ax_psnr, 'Location', 'best');
set(lgd1, 'Color', theme.Background, 'TextColor', theme.Text, 'EdgeColor', theme.Grid);

% SSIM Plot
ax_ssim = nexttile(t);
viz.styleAxes(ax_ssim);
hold(ax_ssim, 'on');
plot(ax_ssim, gaussian_sigmas, ssim_g_gaussian, '-o', 'Color', theme.Primary, 'LineWidth', 2, 'DisplayName', 'Gaussian (\sigma)');
plot(ax_ssim, bilateral_srs, ssim_g_bilateral, '-s', 'Color', theme.Secondary, 'LineWidth', 2, 'DisplayName', 'Bilateral (\sigma_r)');
plot(ax_ssim, nlm_hs, ssim_g_nlm, '-^', 'Color', theme.Boundary, 'LineWidth', 2, 'DisplayName', 'NL-Means (h)');
xlabel(ax_ssim, 'Filtering Parameter Value', 'Color', theme.Text);
ylabel(ax_ssim, 'SSIM', 'Color', theme.Text);
title(ax_ssim, 'SSIM vs Filtering Scale (AWGN)', 'Color', theme.Text);
lgd2 = legend(ax_ssim, 'Location', 'best');
set(lgd2, 'Color', theme.Background, 'TextColor', theme.Text, 'EdgeColor', theme.Grid);

exportgraphics(fig3, fullfile(results_dir, 'denoising_awgn_sensitivity.png'), 'Resolution', 150);

% --- Figure 4: Sensitivity Curves (S&P) ---
fig4 = figure('Name', 'Noise Level Sensitivity (S&P)', 'Position', [160, 160, 950, 420], 'Color', theme.Background);
t_sp = tiledlayout(fig4, 1, 2, 'Padding', 'compact', 'TileSpacing', 'loose');

% PSNR Plot
ax_psnr_sp = nexttile(t_sp);
viz.styleAxes(ax_psnr_sp);
hold(ax_psnr_sp, 'on');
plot(ax_psnr_sp, gaussian_sigmas, psnr_sp_gaussian, '-o', 'Color', theme.Primary, 'LineWidth', 2, 'DisplayName', 'Gaussian (\sigma)');
plot(ax_psnr_sp, bilateral_srs, psnr_sp_bilateral, '-s', 'Color', theme.Secondary, 'LineWidth', 2, 'DisplayName', 'Bilateral (\sigma_r)');
plot(ax_psnr_sp, nlm_hs, psnr_sp_nlm, '-^', 'Color', theme.Boundary, 'LineWidth', 2, 'DisplayName', 'NL-Means (h)');
xlabel(ax_psnr_sp, 'Filtering Parameter Value', 'Color', theme.Text);
ylabel(ax_psnr_sp, 'PSNR (dB)', 'Color', theme.Text);
title(ax_psnr_sp, 'PSNR vs Filtering Scale (S&P)', 'Color', theme.Text);
lgd3 = legend(ax_psnr_sp, 'Location', 'best');
set(lgd3, 'Color', theme.Background, 'TextColor', theme.Text, 'EdgeColor', theme.Grid);

% SSIM Plot
ax_ssim_sp = nexttile(t_sp);
viz.styleAxes(ax_ssim_sp);
hold(ax_ssim_sp, 'on');
plot(ax_ssim_sp, gaussian_sigmas, ssim_sp_gaussian, '-o', 'Color', theme.Primary, 'LineWidth', 2, 'DisplayName', 'Gaussian (\sigma)');
plot(ax_ssim_sp, bilateral_srs, ssim_sp_bilateral, '-s', 'Color', theme.Secondary, 'LineWidth', 2, 'DisplayName', 'Bilateral (\sigma_r)');
plot(ax_ssim_sp, nlm_hs, ssim_sp_nlm, '-^', 'Color', theme.Boundary, 'LineWidth', 2, 'DisplayName', 'NL-Means (h)');
xlabel(ax_ssim_sp, 'Filtering Parameter Value', 'Color', theme.Text);
ylabel(ax_ssim_sp, 'SSIM', 'Color', theme.Text);
title(ax_ssim_sp, 'SSIM vs Filtering Scale (S&P)', 'Color', theme.Text);
lgd4 = legend(ax_ssim_sp, 'Location', 'best');
set(lgd4, 'Color', theme.Background, 'TextColor', theme.Text, 'EdgeColor', theme.Grid);

exportgraphics(fig4, fullfile(results_dir, 'denoising_sp_sensitivity.png'), 'Resolution', 150);

fprintf('\nEvaluation complete. Output images and plots saved to: %s\n', results_dir);

%% =========================================================================
% LOCAL UTILITY FUNCTIONS (Implemented from Scratch)
% =========================================================================

function psnr_val = compute_psnr(ref, target)
    % COMPUTE_PSNR Computes PSNR from scratch for [0, 1] double images.
    mse_val = mean((ref(:) - target(:)).^2);
    if mse_val < 1e-15
        psnr_val = 99.0;
    else
        psnr_val = 10 * log10(1.0 / mse_val);
    end
end

function ssim_val = compute_ssim(img1, img2)
    % COMPUTE_SSIM Computes structural similarity index from scratch.
    % Standard parameters matching Wang et al. 2004.
    K1 = 0.01; K2 = 0.03; L = 1.0;
    C1 = (K1*L)^2; C2 = (K2*L)^2;
    
    % Gaussian window sizing
    w_size = 11;
    [X, Y] = meshgrid(-5:5, -5:5);
    window = exp(-(X.^2 + Y.^2) / (2 * 1.5^2));
    window = window / sum(window(:));
    
    % Local averages
    mu1 = filter2(window, img1, 'valid');
    mu2 = filter2(window, img2, 'valid');
    
    mu1_sq = mu1.^2;
    mu2_sq = mu2.^2;
    mu1_mu2 = mu1 .* mu2;
    
    % Local variances and covariance
    sigma1_sq = filter2(window, img1.^2, 'valid') - mu1_sq;
    sigma2_sq = filter2(window, img2.^2, 'valid') - mu2_sq;
    sigma12   = filter2(window, img1 .* img2, 'valid') - mu1_mu2;
    
    % SSIM map
    num = (2 * mu1_mu2 + C1) .* (2 * sigma12 + C2);
    den = (mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2);
    ssim_map = num ./ den;
    
    ssim_val = mean(ssim_map(:));
end

function show_img(fig, tileIdx, img, titleStr, theme)
    % Helper to plot images cleanly on tile
    ax = nexttile(tileIdx);
    imshow(img, 'Parent', ax);
    title(ax, titleStr, 'Color', theme.Text, 'FontSize', 10);
    axis(ax, 'image'); axis(ax, 'off');
end
