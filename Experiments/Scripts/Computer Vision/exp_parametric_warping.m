% EXP_PARAMETRIC_WARPING  Parametric image warping, MIP-mapping, and anisotropic filtering (Ex 3.22).
%
%   DATASET:        elon_musk.jpg (natural portrait) and synthetic checkboard.png
%   MODEL:          Parametric Warping Engine with various interpolants, MipMap, and EWASampler.
%   ASSUMPTIONS:
%     - Input images are double grayscale/RGB normalized to [0, 1].
%     - Forward/inverse mappings are computed via 2D homogeneous coordinate matrices (homographies).
%     - For the anisotropic comparison, the Jacobian of the perspective ground plane
%       mapping is derived analytically to allow precise EWA ellipse footprint calculation.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Antigravity Code Assistant
%   Date: 2026-07-15

clc; clear; rng(42, 'twister');
close all;

% === 1. Setup Paths and Directories ===
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
results_dir  = fullfile(script_dir, 'Results');
addpath(genpath(project_root));

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% === 2. Load and Prepare Images ===
img_path = fullfile(project_root, 'Data', 'images', 'elon_musk.jpg');
if ~exist(img_path, 'file')
    error('elon_musk.jpg not found in Data/images/');
end

I_orig = im2double(imread(img_path));
if size(I_orig, 3) == 3
    I_gray = rgb2gray(I_orig);
else
    I_gray = I_orig;
end
I_gray = imresize(I_gray, [256, 256]); % Standard size for analysis

% Generate high-frequency color checkerboard for anisotropic rendering
[U, V] = meshgrid(1:256, 1:256);
mask = (mod(floor(U/16) + floor(V/16), 2) == 0);
I_checker = zeros(256, 256, 3);
color1 = [0.08, 0.09, 0.12]; % Sleek dark slate
color2 = [0.88, 0.88, 0.92]; % Off-white
for c = 1:3
    I_checker(:,:,c) = mask * color2(c) + (~mask) * color1(c);
end

fprintf('=== EXERCISE 3.22: PARAMETRIC IMAGE WARPING ===\n\n');

% =========================================================================
% PART 1: Affine Warp & Interpolant Comparison
% =========================================================================
fprintf('--- Running Part 1: Affine Warp & Interpolant Comparison ---\n');

% Rotation by 25 degrees and scaling by 0.8 around the image center
theta = 25;
s = 0.8;
cx = 128.5; cy = 128.5;

T1 = [1, 0, -cx; 0, 1, -cy; 0, 0, 1];
R  = [cosd(theta), -sind(theta), 0; sind(theta), cosd(theta), 0; 0, 0, 1];
S  = [s, 0, 0; 0, s, 0; 0, 0, 1];
T2 = [1, 0, cx; 0, 1, cy; 0, 0, 1];

H_forward = T2 * R * S * T1;
H_inv = inv(H_forward);

% Warp using different interpolants and measure round-trip reconstruction error (PSNR)
interpolants = {'nearest', 'linear', 'cubic'};
warp_results = cell(1, 3);
psnr_scores = zeros(1, 3);

% Grid of target coordinates
[X_t, Y_t] = meshgrid(1:256, 1:256);
pts_t = [X_t(:)'; Y_t(:)'; ones(1, 256*256)];

% Inverse map coordinates
pts_s = H_inv * pts_t;
X_s = reshape(pts_s(1, :) ./ pts_s(3, :), 256, 256);
Y_s = reshape(pts_s(2, :) ./ pts_s(3, :), 256, 256);

for k = 1:numel(interpolants)
    method = interpolants{k};
    % Forward warp
    I_warp = interp2(I_gray, X_s, Y_s, method, 0.15); % Fill background with gray (0.15)
    warp_results{k} = I_warp;
    
    % Backward warp to original space for quality metric
    pts_s_back = H_forward * pts_t;
    X_s_back = reshape(pts_s_back(1, :) ./ pts_s_back(3, :), 256, 256);
    Y_s_back = reshape(pts_s_back(2, :) ./ pts_s_back(3, :), 256, 256);
    I_roundtrip = interp2(I_warp, X_s_back, Y_s_back, method, 0.15);
    
    % Compute PSNR inside a central mask to ignore border truncation effects
    mask_center = (X_t - cx).^2 + (Y_t - cy).^2 < 90^2;
    mse = mean((I_gray(mask_center) - I_roundtrip(mask_center)).^2);
    psnr_scores(k) = 10 * log10(1 / (mse + eps));
    fprintf('  Interpolant: %-7s | Round-trip PSNR (center): %.2f dB\n', method, psnr_scores(k));
end

% =========================================================================
% PART 2: MIP-Map Level Analysis (Coarser vs Finer vs Trilinear)
% =========================================================================
fprintf('\n--- Running Part 2: MIP-Map Level Analysis ---\n');

% Construct MIP-map of the luma image
mipmap = OceanMath.Image.MultiResolution.MipMap();
mipmap.build(I_gray, 4);

% Scale down uniformly by 0.35x
scale_factor = 0.35;
W_out = 128; H_out = 128;
[X_out, Y_out] = meshgrid(1:W_out, 1:H_out);

% Coordinates mapping screen coordinates to texture space
U_tex = X_out / scale_factor;
V_tex = Y_out / scale_factor;

% Constant Jacobian for uniform scaling
J_uniform = [1/scale_factor, 0; 0, 1/scale_factor];
d_level = mipmap.computeLevel(J_uniform);
fprintf('  Uniform scale factor: %.2f | Computed MIP level d: %.3f\n', scale_factor, d_level);
fprintf('  Adjacent integer levels (1-indexed): level %d (finer) and level %d (coarser)\n', ...
    floor(d_level)+1, ceil(d_level)+1);

% Sample using Finer, Coarser, and Trilinear modes
I_finer = zeros(H_out, W_out);
I_coarser = zeros(H_out, W_out);
I_trilinear = zeros(H_out, W_out);

for r = 1:H_out
    for c = 1:W_out
        I_finer(r, c) = mipmap.sample(U_tex(r,c), V_tex(r,c), J_uniform, 'Floor');
        I_coarser(r, c) = mipmap.sample(U_tex(r,c), V_tex(r,c), J_uniform, 'Ceil');
        I_trilinear(r, c) = mipmap.sample(U_tex(r,c), V_tex(r,c), J_uniform, 'Trilinear');
    end
end

% =========================================================================
% PART 3: Anisotropic Texture Filtering (Checkerboard Floor Perspective)
% =========================================================================
fprintf('\n--- Running Part 3: Anisotropic Filtering Comparison ---\n');

% Perspective floor mapping parameters
W_scr = 400; H_scr = 300;
xc = W_scr / 2;
yh = 80; % Horizon line on screen
su = 130; sv = 1100;
uc = 128; vc = 5;

% Output images
I_check_nn = zeros(H_scr, W_scr, 3);
I_check_bil = zeros(H_scr, W_scr, 3);
I_check_mip = zeros(H_scr, W_scr, 3);
I_check_ewa = zeros(H_scr, W_scr, 3);

% Instantiate MIP-map and EWA sampler
mip_checker = OceanMath.Image.MultiResolution.MipMap();
mip_checker.build(I_checker, 5);
ewa = OceanMath.Image.Resampling.EWASampler('RadiusSquared', 3.0);

fprintf('  Rendering perspective floor (400x300 pixels)...\n');
for y = 1:H_scr
    if y <= yh
        % Sky / background color (Dark theme background)
        I_check_nn(y, :, :) = repmat(reshape([0.08, 0.09, 0.12], 1, 1, 3), 1, W_scr);
        I_check_bil(y, :, :) = I_check_nn(y, :, :);
        I_check_mip(y, :, :) = I_check_nn(y, :, :);
        I_check_ewa(y, :, :) = I_check_nn(y, :, :);
        continue;
    end
    
    dy = y - yh;
    for x = 1:W_scr
        % 1. Map to texture space
        u = uc + su * (x - xc) / dy;
        v = vc + sv / dy;
        
        % 2. Analytical Jacobian of the mapping: d(u,v)/d(x,y)
        dudx = su / dy;
        dudy = -su * (x - xc) / (dy^2);
        dvdx = 0;
        dvdy = -sv / (dy^2);
        J = [dudx, dudy; dvdx, dvdy];
        
        % 3. Nearest Neighbor
        u_nn = max(1, min(256, round(u)));
        v_nn = max(1, min(256, round(v)));
        I_check_nn(y, x, :) = I_checker(v_nn, u_nn, :);
        
        % 4. Bilinear Interpolation
        u_b = max(1, min(256, u));
        v_b = max(1, min(256, v));
        u1 = floor(u_b); u2 = min(256, u1 + 1);
        v1 = floor(v_b); v2 = min(256, v1 + 1);
        wu = u_b - u1; wv = v_b - v1;
        I_check_bil(y, x, :) = (1-wu)*(1-wv)*I_checker(v1, u1, :) + ...
                               wu*(1-wv)*I_checker(v1, u2, :) + ...
                               (1-wu)*wv*I_checker(v2, u1, :) + ...
                               wu*wv*I_checker(v2, u2, :);
                           
        % 5. Isotropic MIP-Mapping (Trilinear)
        % For 3 channels, sample each channel
        for c = 1:3
            % Build temporary single-channel mipmap for this channel
            mip_c = OceanMath.Image.MultiResolution.MipMap();
            mip_c.Levels = cellfun(@(x) x(:,:,c), mip_checker.Levels, 'UniformOutput', false);
            mip_c.NumLevels = mip_checker.NumLevels;
            I_check_mip(y, x, c) = mip_c.sample(u, v, J, 'Trilinear');
        end
        
        % 6. EWA Anisotropic Filtering
        I_check_ewa(y, x, :) = ewa.sample(I_checker, u, v, J);
    end
end

% =========================================================================
% VISUALIZATION & COMPARISON DASHBOARD
% =========================================================================
fprintf('\nRendering dashboard figure...\n');

viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;
white = colors.Text;

fig = figure('Name', 'Ex 3.22: Parametric Image Warping & Filtering', ...
    'Color', colors.Background, ...
    'Position', [50, 50, 1400, 850]);

% --- Row 1: Affine Interpolant Comparison ---
subplot_titles_r1 = {
    sprintf('Nearest (PSNR: %.1f dB)', psnr_scores(1)), ...
    sprintf('Bilinear (PSNR: %.1f dB)', psnr_scores(2)), ...
    sprintf('Bicubic (PSNR: %.1f dB)', psnr_scores(3))
};

for i = 1:3
    ax = subplot(3, 4, i);
    imshow(warp_results{i}, 'Parent', ax);
    title(ax, subplot_titles_r1{i}, 'Color', white, 'FontSize', 9, 'FontWeight', 'bold');
    viz.styleAxes(ax);
end

% --- Row 1, Col 4: Original reference ---
ax_ref = subplot(3, 4, 4);
imshow(I_gray, 'Parent', ax_ref);
title(ax_ref, 'Original Image (256x256)', 'Color', white, 'FontSize', 9, 'FontWeight', 'bold');
viz.styleAxes(ax_ref);

% --- Row 2: MIP-Map Level Analysis ---
r2_images = {I_gray, I_finer, I_coarser, I_trilinear};
r2_titles = {
    'Reference (Scale 1.0)', ...
    sprintf('Finer Level %d (Aliases)', floor(d_level)+1), ...
    sprintf('Coarser Level %d (Blurry)', ceil(d_level)+1), ...
    sprintf('Trilinear Blend (Clean)', d_level)
};

for i = 1:4
    ax = subplot(3, 4, 4 + i);
    imshow(r2_images{i}, 'Parent', ax);
    title(ax, r2_titles{i}, 'Color', white, 'FontSize', 9, 'FontWeight', 'bold');
    viz.styleAxes(ax);
end

% --- Row 3: Anisotropic Checkerboard Floor Warping ---
r3_images = {I_check_nn, I_check_bil, I_check_mip, I_check_ewa};
r3_titles = {
    'Nearest (Severe Moiré)', ...
    'Bilinear (Jagged / Aliased)', ...
    'Isotropic MIP (Over-blurred Horizon)', ...
    'EWA Anisotropic (Sharp & Anti-aliased)'
};

for i = 1:4
    ax = subplot(3, 4, 8 + i);
    imshow(r3_images{i}, 'Parent', ax);
    title(ax, r3_titles{i}, 'Color', white, 'FontSize', 9, 'FontWeight', 'bold');
    viz.styleAxes(ax);
end

sgtitle(fig, 'Ex 3.22: Parametric Image Warping, MIP-Mapping, & Anisotropic Filtering', ...
    'Color', white, 'FontSize', 14, 'FontWeight', 'bold');

% Save output image
output_path = fullfile(results_dir, 'parametric_warping_results.png');
saveas(fig, output_path);

% =========================================================================
% SUMMARY & TEXTBOOK DISCUSSION
% =========================================================================
fprintf('\n=== EXPERIMENT SUMMARY & DISCUSSION ===\n');
fprintf('  1. Interpolant Round-Trip PSNR (Daubechies-like rotation/scale test):\n');
fprintf('     - Nearest Neighbor : %.2f dB\n', psnr_scores(1));
fprintf('     - Bilinear         : %.2f dB\n', psnr_scores(2));
fprintf('     - Bicubic (Keys)   : %.2f dB\n', psnr_scores(3));
fprintf('     * Discussion: Higher-order interpolants (Bicubic) preserve details much better,\n');
fprintf('       reducing high-frequency attenuation (blurry edges) and pixelation.\n\n');

fprintf('  2. MIP-Map Level Choice (scale factor: %.2f, fractional level: %.3f):\n', scale_factor, d_level);
fprintf('     - Finer Level (Level %d): Selecting only the finer level uses a pre-filter designed\n', floor(d_level)+1);
fprintf('       for less downsampling. High frequencies above the Nyquist rate survive, causing ALIASING.\n');
fprintf('     - Coarser Level (Level %d): Selecting only the coarser level uses a pre-filter designed\n', ceil(d_level)+1);
fprintf('       for more downsampling. It removes valid high-frequency content, causing OVER-BLURRING.\n');
fprintf('     - Trilinear Blend: Interpolating between the two levels eliminates abrupt level transition\n');
fprintf('       artifacts ("popping") and balances smoothing and aliasing in isotropic scaling.\n\n');

fprintf('  3. Anisotropic Resampling Problem & EWA Solution:\n');
fprintf('     - Under extreme perspective tilt (ground plane), the horizontal and vertical resampling\n');
fprintf('       rates are very different (anisotropic). Screen pixels map to highly elongated ellipses.\n');
fprintf('     - Isotropic MIP-mapping computes a single scale factor based on the maximum footprints axis\n');
fprintf('       (the highly compressed axis near the horizon) to avoid aliasing. This leads to extreme\n');
fprintf('       OVER-BLURRING of the other (uncompressed) axis, making the checkerboard wash out completely.\n');
fprintf('     - EWA Anisotropic Filtering computes the exact covariance matrix of the ellipse footprint\n');
fprintf('       analytically using the Jacobian. It filters along the elongated ellipse, preserving sharpness\n');
fprintf('       along the uncompressed axis while perfectly anti-aliasing the compressed axis.\n');
fprintf('     - Other Approaches to Anisotropic Filtering: RIP-maps (separate horizontal/vertical mip-map chains)\n');
fprintf('       and footprint assembly (hardware approximation by taking multiple isotropic samples along the ellipse).\n');
fprintf('  Results saved to: %s\n', output_path);
fprintf('========================================\n');
