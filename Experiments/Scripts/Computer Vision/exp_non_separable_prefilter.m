% EXP_NON_SEPARABLE_PREFILTER Demonstrating Non-Separability of the Ideal Anti-Alias Filter
%
%   DATASET:        Synthetic 2D Zone Plate (chirp signal) on a 512 x 512 grid.
%   TRANSFORM:      Rotated anisotropic scale A = R(45) * diag(s1, s2) * R(-45)
%                   with s1 = 2.0 (magnify X-45), s2 = 0.4 (minify Y-135).
%   ASSUMPTIONS:    Target domain filter has variance sigma_p^2 = 0.8^2.
%                   SVD of the rotated kernel shows multiple non-zero singular values.
%                   Rank-1 approximation (separable pass) fails to align the blur correctly.
%                   No anti-aliasing prefilter leads to massive aliasing.
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   Author: Antigravity Code Assistant
%   Date: 2026-06-25

clear;
close all;
clc;

% Set seed for reproducibility
rng(42, 'twister');

disp('Initializing Experiment H: Prefilter Non-Separability...');
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Create Results directory if not exists
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Outputs
fig1_path = fullfile(results_dir, 'prefilter_non_separability_svd.png');
fig2_path = fullfile(results_dir, 'prefilter_non_separability_spatial.png');
fig3_path = fullfile(results_dir, 'prefilter_non_separability_spectral.png');

%% 1. Setup Transform and Pre-filter Covariance
N_src = 512;
N_tgt = 512;

theta = 45; % Rotation angle in degrees
rad = deg2rad(theta);
R = [cos(rad), -sin(rad); sin(rad), cos(rad)];

s1 = 2.0; % Magnify along 45 degrees (stretch)
s2 = 0.4; % Minify along 135 degrees (squash)

A = R * [s1, 0; 0, s2] * R';

% Ideal target-domain standard deviation
sigma_p = 0.8;

% Source-domain prefilter standard deviations along primary axes
sigma_1 = sigma_p / s1; % 0.8 / 2.0 = 0.4 (magnified axis, small blur)
sigma_2 = sigma_p / s2; % 0.8 / 0.4 = 2.0 (minified axis, large blur)

% Rotated source-domain covariance matrix
Sigma_src = R * [sigma_1^2, 0; 0, sigma_2^2] * R';

%% 2. Generate 2D Zone Plate (Chirp) Source Image
disp('Generating 2D Zone Plate (chirp) source signal...');
[X_src, Y_src] = meshgrid(-N_src/2 : N_src/2 - 1, -N_src/2 : N_src/2 - 1);
% Radial frequency coefficient to get dense rings towards the borders
beta = (N_src/2)^2 / (pi * 20); 
I_src = 0.5 * cos(pi * (X_src.^2 + Y_src.^2) / beta) + 0.5;

%% 3. Construct 2D Rotated Elliptical Gaussian Kernel
disp('Constructing rotated elliptical Gaussian kernel...');
K_size = 15;
w = (K_size - 1) / 2;
[X_k, Y_k] = meshgrid(-w:w, -w:w);
coords = [X_k(:), Y_k(:)]'; % 2 x K_size^2

% Evaluate Gaussian density: exp(-0.5 * pt' * Sigma^-1 * pt)
Sigma_inv = inv(Sigma_src);
G_flat = exp(-0.5 * sum(coords .* (Sigma_inv * coords), 1));
G = reshape(G_flat, K_size, K_size);
G = G / sum(G(:)); % Normalize

% Axis-aligned comparison (0 degrees rotation)
Sigma_aligned = [sigma_1^2, 0; 0, sigma_2^2];
G_aligned_flat = exp(-0.5 * sum(coords .* (inv(Sigma_aligned) * coords), 1));
G_aligned = reshape(G_aligned_flat, K_size, K_size);
G_aligned = G_aligned / sum(G_aligned(:));

%% 4. SVD Analysis (Proving Non-Separability)
disp('Analyzing Singular Value Decomposition (SVD) of kernels...');
[U, S, V] = svd(G);
s_vals = diag(S);

[U_al, S_al, V_al] = svd(G_aligned);
s_vals_al = diag(S_al);

% Separable rank-1 approximation of G
g_v = U(:, 1) * sqrt(s_vals(1));
g_h = V(:, 1)' * sqrt(s_vals(1));
G_separable_approx = g_v * g_h;

%% 5. Apply Convolutions
disp('Applying convolution passes...');

% Path A: No prefilter (No convolution)
I_filt_A = I_src;

% Path B: Separable rank-1 SVD approximation
I_filt_B = run_separable_conv(I_src, g_h, g_v, 'replicate');

% Path C: Full 2D rotated elliptical filter
I_filt_C = run_2d_conv(I_src, G, 'replicate');

%% 6. Warp via Gather-Resampling (Algorithm 3.2)
disp('Warping outputs via Gather-resampling...');
A_inv = inv(A);

% Coordinate grids centered
[X_tgt, Y_tgt] = meshgrid(-N_tgt/2 : N_tgt/2 - 1, -N_tgt/2 : N_tgt/2 - 1);
X_src_mapped = A_inv(1,1) * X_tgt + A_inv(1,2) * Y_tgt;
Y_src_mapped = A_inv(2,1) * X_tgt + A_inv(2,2) * Y_tgt;

% 1-based indexing for source lookup
xs = X_src_mapped + N_src/2 + 1;
ys = Y_src_mapped + N_src/2 + 1;

% Perform warp for all three paths
I_warped_A = run_gather_warp(I_filt_A, xs, ys);
I_warped_B = run_gather_warp(I_filt_B, xs, ys);
I_warped_C = run_gather_warp(I_filt_C, xs, ys);

%% 7. 2D FFT Analysis
disp('Computing 2D FFT spectra of outputs...');
% Subtract mean and apply 2D cosine window to reduce edge leakage in 2D FFT
win_2d = hann(N_tgt) * hann(N_tgt)';

F_A = abs(fftshift(fft2((I_warped_A - mean(I_warped_A(:))) .* win_2d)));
F_B = abs(fftshift(fft2((I_warped_B - mean(I_warped_B(:))) .* win_2d)));
F_C = abs(fftshift(fft2((I_warped_C - mean(I_warped_C(:))) .* win_2d)));

% Log-scale for clear visualization of weak aliasing peaks
F_A_log = log(F_A + 1.0);
F_B_log = log(F_B + 1.0);
F_C_log = log(F_C + 1.0);

%% 8. Benchmark Computational Cost vs. Kernel Size
disp('Benchmarking execution times vs. kernel sizes...');
K_sizes = 3:2:21;
times_2d = zeros(size(K_sizes));
times_1d = zeros(size(K_sizes));
ops_2d = zeros(size(K_sizes));
ops_1d = zeros(size(K_sizes));

for idx = 1:length(K_sizes)
    K = K_sizes(idx);
    ops_2d(idx) = K^2;
    ops_1d(idx) = 2 * K;
    
    % Dummy kernels of size K
    K_2D = ones(K, K) / (K^2);
    h_1d = ones(1, K) / K;
    v_1d = ones(K, 1) / K;
    
    % Benchmark 2D
    t_start = tic;
    run_2d_conv(I_src, K_2D, 'replicate');
    times_2d(idx) = toc(t_start);
    
    % Benchmark 1D
    t_start = tic;
    run_separable_conv(I_src, h_1d, v_1d, 'replicate');
    times_1d(idx) = toc(t_start);
end

%% 9. Visualization Dashboards
viz = OceanViz.Core('Theme', 'Dark');

% --- Figure 1: SVD Analysis & Computational Complexity ---
disp('Rendering Figure 1 (SVD & Complexity)...');
f1 = figure('Name', 'Exp H: Prefilter Separability and Complexity', ...
            'Color', viz.Colors.Background, 'Position', [50, 50, 1500, 500]);

% Subplot 1: SVD Singular Values
ax1_1 = subplot(1, 3, 1);
b = bar(ax1_1, 1:5, [s_vals(1:5), s_vals_al(1:5)], 'grouped');
b(1).FaceColor = viz.Colors.Primary;   % Rotated (Non-separable)
b(2).FaceColor = viz.Colors.Secondary; % Axis-aligned (Separable)
set(ax1_1, 'XTick', 1:5);
xlabel(ax1_1, 'Singular Value Index');
ylabel(ax1_1, 'Magnitude');
title(ax1_1, 'SVD Spectrum (Kernel Rank)', 'Color', viz.Colors.Text);
legend(ax1_1, 'Rotated 45 deg', 'Axis-Aligned 0 deg', 'Location', 'northeast', 'TextColor', viz.Colors.Text);
grid(ax1_1, 'on');
viz.styleAxes(ax1_1);

% Subplot 2: Theoretical Operations Count
ax1_2 = subplot(1, 3, 2);
plot(ax1_2, K_sizes, ops_2d, 'Color', viz.Colors.Primary, 'LineWidth', 2.5, 'Marker', 'o', 'DisplayName', '2D (Non-separable): O(K^2)');
hold(ax1_2, 'on');
plot(ax1_2, K_sizes, ops_1d, 'Color', viz.Colors.Secondary, 'LineWidth', 2.5, 'Marker', 'x', 'DisplayName', 'Separable: O(2K)');
hold(ax1_2, 'off');
xlabel(ax1_2, 'Kernel Size (K)');
ylabel(ax1_2, 'Multiplications per Pixel');
title(ax1_2, 'Theoretical Computational Complexity', 'Color', viz.Colors.Text);
legend(ax1_2, 'Location', 'northwest', 'TextColor', viz.Colors.Text);
grid(ax1_2, 'on');
viz.styleAxes(ax1_2);

% Subplot 3: Measured Execution Time
ax1_3 = subplot(1, 3, 3);
plot(ax1_3, K_sizes, times_2d, 'Color', viz.Colors.Primary, 'LineWidth', 2.5, 'Marker', 'o', 'DisplayName', 'Full 2D Filter');
hold(ax1_3, 'on');
plot(ax1_3, K_sizes, times_1d, 'Color', viz.Colors.Secondary, 'LineWidth', 2.5, 'Marker', 'x', 'DisplayName', 'Separable Pass');
hold(ax1_3, 'off');
xlabel(ax1_3, 'Kernel Size (K)');
ylabel(ax1_3, 'Execution Time (seconds)');
title(ax1_3, 'Empirical Execution Time (MATLAB)', 'Color', viz.Colors.Text);
legend(ax1_3, 'Location', 'northwest', 'TextColor', viz.Colors.Text);
grid(ax1_3, 'on');
viz.styleAxes(ax1_3);

drawnow;
print(f1, fig1_path, '-dpng', '-r100');

% --- Figure 2: Spatial Domain Comparison ---
disp('Rendering Figure 2 (Spatial Domain)...');
f2 = figure('Name', 'Exp H: Spatial Domain Comparisons', ...
            'Color', viz.Colors.Background, 'Position', [100, 100, 1400, 800]);

ax2_1 = subplot(2, 2, 1);
imagesc(ax2_1, I_src); colormap(ax2_1, 'gray'); axis(ax2_1, 'image');
title(ax2_1, 'Original Zone Plate (2D Chirp)', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_1);

ax2_2 = subplot(2, 2, 2);
imagesc(ax2_2, I_warped_A); colormap(ax2_2, 'gray'); axis(ax2_2, 'image');
title(ax2_2, 'Warped: No Prefilter (Severe Aliasing)', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_2);

ax2_3 = subplot(2, 2, 3);
imagesc(ax2_3, I_warped_B); colormap(ax2_3, 'gray'); axis(ax2_3, 'image');
title(ax2_3, 'Warped: Separable Approximation (Residual Moiré)', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_3);

ax2_4 = subplot(2, 2, 4);
imagesc(ax2_4, I_warped_C); colormap(ax2_4, 'gray'); axis(ax2_4, 'image');
title(ax2_4, 'Warped: Correct 2D Prefilter (Clean Reconstruction)', 'Color', viz.Colors.Text);
viz.styleAxes(ax2_4);

drawnow;
print(f2, fig2_path, '-dpng', '-r100');

% --- Figure 3: Spectral Domain Comparison (2D FFT) ---
disp('Rendering Figure 3 (Spectral Domain)...');
f3 = figure('Name', 'Exp H: 2D FFT Spectral Comparison', ...
            'Color', viz.Colors.Background, 'Position', [150, 150, 1600, 500]);

ax3_1 = subplot(1, 3, 1);
imagesc(ax3_1, F_A_log); colormap(ax3_1, 'turbo'); axis(ax3_1, 'image');
title(ax3_1, 'FFT: No Prefilter (High-Freq Aliasing Leak)', 'Color', viz.Colors.Text);
xlabel(ax3_1, 'fx'); ylabel(ax3_1, 'fy');
viz.styleAxes(ax3_1);

ax3_2 = subplot(1, 3, 2);
imagesc(ax3_2, F_B_log); colormap(ax3_2, 'turbo'); axis(ax3_2, 'image');
title(ax3_2, 'FFT: Separable Pass (Leaking Corners)', 'Color', viz.Colors.Text);
xlabel(ax3_2, 'fx'); ylabel(ax3_2, 'fy');
viz.styleAxes(ax3_2);

ax3_3 = subplot(1, 3, 3);
imagesc(ax3_3, F_C_log); colormap(ax3_3, 'turbo'); axis(ax3_3, 'image');
title(ax3_3, 'FFT: Correct 2D Anisotropic (Bandlimited Ellipse)', 'Color', viz.Colors.Text);
xlabel(ax3_3, 'fx'); ylabel(ax3_3, 'fy');
viz.styleAxes(ax3_3);

drawnow;
print(f3, fig3_path, '-dpng', '-r100');

%% 10. Print Summary Report
fprintf('\n=== EXPERIMENT H SUMMARY: PREFILTER NON-SEPARABILITY ===\n');
fprintf('  Anisotropic Scales      : s1 = %.1f (45 deg), s2 = %.1f (135 deg)\n', s1, s2);
fprintf('  Prefilter Covariance    : Sigma_src = [%.3f, %.3f; %.3f, %.3f]\n', ...
        Sigma_src(1,1), Sigma_src(1,2), Sigma_src(2,1), Sigma_src(2,2));
fprintf('\n  Kernel Singular Values (Rotated):\n');
fprintf('    s1 (Rank 1 Component) : %.6f\n', s_vals(1));
fprintf('    s2 (Rank 2 Component) : %.6f (Ratio s2/s1 = %.4f%%)\n', s_vals(2), (s_vals(2)/s_vals(1))*100);
fprintf('\n  Kernel Singular Values (Axis-Aligned):\n');
fprintf('    s_al1                 : %.6f\n', s_vals_al(1));
fprintf('    s_al2                 : %.6f (Ratio s2/s1 = %.4e%%)\n', s_vals_al(2), (s_vals_al(2)/s_vals_al(1))*100);
fprintf('\n  Computational Cost (K = %d):\n', K_size);
fprintf('    2D Convolution Cost   : %d operations/pixel\n', ops_2d(K_sizes == K_size));
fprintf('    1D Separable Cost     : %d operations/pixel (%.1fx reduction)\n', ...
        ops_1d(K_sizes == K_size), ops_2d(K_sizes == K_size) / ops_1d(K_sizes == K_size));
fprintf('    Empirical Time 2D     : %.4f seconds\n', times_2d(K_sizes == K_size));
fprintf('    Empirical Time 1D     : %.4f seconds (%.1fx speedup)\n', ...
        times_1d(K_sizes == K_size), times_2d(K_sizes == K_size) / times_1d(K_sizes == K_size));
fprintf('========================================================\n');

if s_vals(2)/s_vals(1) > 0.05
    fprintf('SUCCESS: Rotated elliptical prefilter shown to be non-separable (high Rank-2 content).\n');
else
    fprintf('FAILURE: Rotated kernel is mathematically separable or ratio is too small.\n');
end

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function I_out = run_separable_conv(I_in, g_h, g_v, boundary)
    import OceanMath.Image.Pixels.BoundaryPadder
    I_dbl = double(I_in);
    [H, W, C] = size(I_dbl);
    
    % Horizontal pass (g_h is 1 x Kh)
    Kh = length(g_h);
    pad_cols = (Kh - 1) / 2;
    g_h_conv = fliplr(g_h);
    I_pad_h = BoundaryPadder.pad(I_dbl, [0, pad_cols], boundary);
    I_h = zeros(H, W, C);
    for m = 1:Kh
        I_h = I_h + g_h_conv(m) * I_pad_h(:, m : m + W - 1, :);
    end
    
    % Vertical pass (g_v is Kv x 1)
    Kv = length(g_v);
    pad_rows = (Kv - 1) / 2;
    g_v_conv = flipud(g_v(:));
    I_pad_v = BoundaryPadder.pad(I_h, [pad_rows, 0], boundary);
    I_out = zeros(H, W, C);
    for m = 1:Kv
        I_out = I_out + g_v_conv(m) * I_pad_v(m : m + H - 1, :, :);
    end
end

function I_out = run_2d_conv(I_in, K, boundary)
    import OceanMath.Image.Pixels.BoundaryPadder
    I_dbl = double(I_in);
    [H, W, C] = size(I_dbl);
    
    K_conv = rot90(K, 2);
    [Kh, Kw] = size(K_conv);
    pad_rows = (Kh - 1) / 2;
    pad_cols = (Kw - 1) / 2;
    
    I_pad = BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], boundary);
    I_out = zeros(H, W, C);
    for r = 1:Kh
        for c = 1:Kw
            I_out = I_out + K_conv(r, c) * I_pad(r : r + H - 1, c : c + W - 1, :);
        end
    end
end

function I_dst = run_gather_warp(I_src, xs, ys)
    [N_src, ~] = size(I_src);
    N_tgt = size(xs, 1);
    
    x0 = floor(xs);
    y0 = floor(ys);
    x1 = x0 + 1;
    y1 = y0 + 1;
    
    valid = (x0 >= 1 & x1 <= N_src & y0 >= 1 & y1 <= N_src);
    
    x0(~valid) = 1; x1(~valid) = 1;
    y0(~valid) = 1; y1(~valid) = 1;
    
    dx = xs - x0;
    dy = ys - y0;
    
    I_src_flat = I_src(:);
    idx00 = sub2ind([N_src, N_src], y0, x0);
    idx01 = sub2ind([N_src, N_src], y0, x1);
    idx10 = sub2ind([N_src, N_src], y1, x0);
    idx11 = sub2ind([N_src, N_src], y1, x1);
    
    w00 = (1-dx).*(1-dy);
    w01 = dx.*(1-dy);
    w10 = (1-dx).*dy;
    w11 = dx.*dy;
    
    I_dst = w00 .* I_src_flat(idx00) + w01 .* I_src_flat(idx01) + ...
            w10 .* I_src_flat(idx10) + w11 .* I_src_flat(idx11);
            
    I_dst(~valid) = 0;
end
