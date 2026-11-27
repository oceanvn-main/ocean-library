% exp_rainbow_enhancer.m
% =========================================================================
% Experiment: Additive Rainbow Separation & Enhancement (Ex 3.29)
% =========================================================================
%
% PURPOSE:
%   This script implements the inpainting-based pipeline for Exercise 3.29:
%     1. Synthesizes a realistic test image by adding a physical dispersion
%        rainbow to a natural background image in linear color space.
%     2. Linearizes the composite image into linear light space.
%     3. Generates a binary mask of the rainbow arc region.
%     4. Estimates the background sky behind the rainbow using Laplacian
%        interpolation (regionfill) on each linearized color channel.
%     5. Subtracts the synthesized background to isolate the linear rainbow
%        light energy, clamping negative values to zero.
%     6. Amplifies the isolated signal, composites it back, and applies
%        gamma compression to display the results.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-17
% =========================================================================

clear; close all; clc;
addpath(genpath('c:\Ocean Library\Matlab code practice'));
cd('c:\Ocean Library\Matlab code practice');

fprintf('=== Ex 3.29: Additive Rainbow Separation and Enhancement (Inpainting) ===\n\n');

%% =========================================================================
% STEP 1: LOAD BACKGROUND & SYNTHESIZE ADDITIVE RAINBOW
% =========================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
bg_path      = fullfile(project_root, 'Data', 'images', 'rainbow1.jpg');

I_bg_srgb = im2double(imread(bg_path));
I_bg_srgb = imresize(I_bg_srgb, [512, 512]); % Resize for fast processing
[H, W, C] = size(I_bg_srgb);

% Convert background to linear light space (Gamma = 2.2)
L_bg_gt = max(0, I_bg_srgb) .^ 2.2;

% Define ground truth rainbow geometry
xc_gt = 256;
yc_gt = 512;
r_start_gt = 240;
r_end_gt = 290;
theta_range_gt = [-pi + 0.6, -0.6]; % arch pointing upwards

% Generate ground truth 1D rainbow spectrum profile (7-color interpolation)
num_r_spectrum = 100;
u = linspace(0, 1, num_r_spectrum)';
spectrum_colors = [
    1.0, 0.0, 0.0;   % Red
    1.0, 0.5, 0.0;   % Orange
    1.0, 1.0, 0.0;   % Yellow
    0.0, 1.0, 0.0;   % Green
    0.0, 0.8, 1.0;   % Cyan/Blue
    0.2, 0.0, 0.8;   % Indigo
    0.5, 0.0, 1.0    % Violet
];
xp = linspace(0, 1, size(spectrum_colors, 1));
R_profile_gt = zeros(num_r_spectrum, 3);
for ch = 1:3
    R_profile_gt(:, ch) = interp1(xp, spectrum_colors(:, ch), u, 'linear');
end

% Create 2D coordinates
[X, Y] = meshgrid(1:W, 1:H);
Dist_gt = sqrt((X - xc_gt).^2 + (Y - yc_gt).^2);
Theta_gt = atan2(Y - yc_gt, X - xc_gt);

% Angular mask
angle_mask_gt = (Theta_gt >= theta_range_gt(1)) & (Theta_gt <= theta_range_gt(2));

% Create 2D additive rainbow signal
L_rainbow_gt = zeros(H, W, 3);
r_grid_gt = linspace(r_start_gt, r_end_gt, num_r_spectrum);
for ch = 1:3
    L_rainbow_gt(:, :, ch) = interp1(r_grid_gt, R_profile_gt(:, ch), Dist_gt, 'linear', 0.0) .* angle_mask_gt;
end

% Scale the rainbow strength to be realistic but visible
rainbow_strength = 0.20;
L_rainbow_gt = L_rainbow_gt * rainbow_strength;

% Combine background and rainbow in linear space (additive model)
L_total = L_bg_gt + L_rainbow_gt;
L_total = max(0, min(1, L_total)); % Clamp

% Convert to non-linear display sRGB space
I_composite_srgb = max(0, L_total) .^ (1 / 2.2);

%% =========================================================================
% STEP 2: DEFINE REGION OF INTEREST MASK (Phase 2)
% =========================================================================
fprintf('Generating binary mask of the rainbow arc...\n');
% The mask is slightly wider than the rainbow to encompass all spectral content
M = (Dist_gt >= r_start_gt - 5) & (Dist_gt <= r_end_gt + 5) & angle_mask_gt;

%% =========================================================================
% STEP 3: BACKGROUND SYNTHESIS (Phase 3)
% =========================================================================
fprintf('Solving Laplace equation inside the mask for background synthesis...\n');
% Solve Laplacian interpolation channel-by-channel
L_bg_est = zeros(H, W, 3);
for c = 1:3
    L_bg_est(:,:,c) = regionfill(L_total(:,:,c), M);
end

%% =========================================================================
% STEP 4: MATHEMATICAL EXTRACTION (Phase 4)
% =========================================================================
fprintf('Subtracting background in linear space and clamping...\n');
L_rainbow_est = max(L_total - L_bg_est, 0);

%% =========================================================================
% STEP 5: RE-ENCODING & BOOSTING (Phase 5)
% =========================================================================
fprintf('Boosting signal and applying inverse gamma compression...\n');
k = 3.0; % Enhancement Gain factor
L_enhanced = L_bg_est + k * L_rainbow_est;
L_enhanced = max(0, min(1, L_enhanced));

I_enhanced_srgb = max(0, L_enhanced) .^ (1 / 2.2);
P_isolated_srgb = max(0, k * L_rainbow_est) .^ (1 / 2.2);

%% =========================================================================
% STEP 6: VISUALIZATION (OceanViz Theme)
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');
theme = viz.Colors;

results_dir = fullfile(project_root, 'Results', 'Computer Vision');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

% --- Figure 1: Step-by-Step Pipeline Dashboard ---
fig1 = figure('Name', 'Rainbow Separation Step-by-Step Dashboard', 'Position', [50, 50, 1100, 750], 'Color', theme.Background);
t = tiledlayout(fig1, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% 1. Input sRGB Image
show_img(t, 1, I_composite_srgb, '1. Input Image P_total', theme);

% 2. Binary Mask M
show_img(t, 2, double(M), '2. Binary Mask M(x,y)', theme);

% 3. Synthesized Background L_bg (gamma corrected for display)
show_img(t, 3, L_bg_est .^ (1/2.2), '3. Synthesized Background L_bg', theme);

% 4. Isolated Rainbow L_rainbow (gamma corrected for display)
show_img(t, 4, L_rainbow_est .^ (1/2.2), '4. Isolated Rainbow (sRGB)', theme);

% 5. Final Output (Enhanced Composite)
show_img(t, 5, I_enhanced_srgb, sprintf('5. Enhanced Output (k = %.1f)', k), theme);

% 6. Isolated Enhanced Output
show_img(t, 6, P_isolated_srgb, sprintf('6. Isolated Enhanced Rainbow (k = %.1f)', k), theme);

exportgraphics(fig1, fullfile(results_dir, 'rainbow_inpainting_pipeline.png'), 'Resolution', 150);

fprintf('Evaluation complete. Output images and plots saved to: %s\n', results_dir);

% Launch the interactive app to demonstrate manual tuning (only if in interactive desktop mode)
if usejava('desktop')
    fprintf('\nLaunching interactive RainbowEnhancer app...\n');
    addpath(fullfile(project_root, '+OceanViz', '+App', 'Computer Vision'));
    app = RainbowEnhancer('Image', I_composite_srgb);
    app.show();
end

%% =========================================================================
% LOCAL UTILITY FUNCTION
% =========================================================================
function ax = show_img(tlayout, tileIdx, img, titleStr, theme)
    ax = nexttile(tlayout, tileIdx);
    imshow(img, 'Parent', ax);
    title(ax, titleStr, 'Color', theme.Text, 'FontSize', 11);
    axis(ax, 'image'); axis(ax, 'off');
end
