% EXP_LAPLACIAN_PYRAMID_BLENDING  Burt and Adelson (1983b) Two-Image Laplacian Pyramid Blending (Exercise 3.18).
%
%   DATASET:        apple.jpg and orange.jpg
%   MODEL:          Two-Image LaplacianPyramid blending via mask and complement.
%   ASSUMPTIONS:    Input images are resized to 512x512 to ensure robust power-of-2 dyadic downsampling.
%                   The Mask is a vertical split (left=1, right=0). The complement is 1.0 - Mask.
%                   Gaussian/Laplacian Pyramids are built with 4 levels using a 5x5 Gaussian kernel.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Antigravity Code Assistant
%   Date: 2026-07-13
%
%   =====================================================================
%   DISCUSSION: Weight Renormalization in Pyramid Blending (Part 3)
%   =====================================================================
%   Question: Does the weighted summation stage (step 3) need to keep track of
%   the total weight for renormalization, or does the math just work out?
%
%   Answer:
%   In an ideal continuous setting where the input binary weight masks partition
%   unity at every pixel (i.e., sum_{i=1}^N M_i(x,y) = 1.0), the mathematical sum
%   of the Gaussian pyramid levels of those masks would also partition unity at
%   all levels: sum_{i=1}^N GP_M_i_l(x,y) = 1.0. This holds because reduction is
%   a linear shift-invariant filtering operation.
%
%   However, in a discrete, finite-boundary implementation, two factors cause
%   deviations from unity:
%     1. Boundary Padding: Replicate, symmetric, or circular padding modes distort
%        the sum of weights near the borders of the image during downsampling.
%     2. Numerical Rounding & Downsampling Truncation: Small numerical precision
%        inaccuracies accumulate across multiple decimation/interpolation levels.
%
%   If we do not renormalize, the weights at lower resolution levels will sum to
%   slightly more or less than 1.0 near boundaries, manifesting as noticeable
%   dark or bright halo seams in the final blended image.
%
%   Therefore, to ensure brightness consistency across the entire image (especially
%   near borders), we MUST track the sum of mask weights at each level and divide:
%     L_blend_l = sum_i(GP_M_i_l .* LP_i_l) ./ sum_i(GP_M_i_l)
%   This generalizes the blending math to arbitrary overlapping masks (e.g. feathered
%   exposures) and guarantees brightness-invariant results.
%   =====================================================================

clc; clear; rng(42, 'twister');
close all; % Close any previous windows

% === 1. Data Loading ===
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
apple_path = fullfile(root_dir, 'Data', 'images', 'apple.jpg');
orange_path = fullfile(root_dir, 'Data', 'images', 'orange.jpg');

if ~exist(apple_path, 'file') || ~exist(orange_path, 'file')
    error('Images not found. Ensure apple.jpg and orange.jpg exist in Data/images/');
end

I_apple = imread(apple_path);
I_orange = imread(orange_path);

% Resize to 512x512 for dyadic downsampling
H = 512;
W = 512;
I_apple = imresize(I_apple, [H, W]);
I_orange = imresize(I_orange, [H, W]);

% Convert to double [0, 1]
A = im2double(I_apple);
B = im2double(I_orange);

% === 2. Mask and Complement Generation ===
% Create a sharp vertical split mask
M = zeros(H, W);
M(:, 1:floor(W/2)) = 1;
Mask = double(repmat(M, [1, 1, size(A, 3)]));

% The complement of the mask
MaskComplement = 1.0 - Mask;

% === 3. Generalized Pyramid Blending (4 Levels) ===
num_levels = 4;

% Build Gaussian Pyramids for original images (Apple & Orange)
disp('Building Gaussian Pyramid for Apple...');
GPyrA = OceanMath.Image.MultiResolution.GaussianPyramid();
GPyrA.build(A, num_levels);

disp('Building Gaussian Pyramid for Orange...');
GPyrB = OceanMath.Image.MultiResolution.GaussianPyramid();
GPyrB.build(B, num_levels);

% Build Laplacian Pyramids for original images (Apple & Orange)
disp('Building Laplacian Pyramid for Apple...');
LPyrA = OceanMath.Image.MultiResolution.LaplacianPyramid();
LPyrA.build(A, num_levels);

disp('Building Laplacian Pyramid for Orange...');
LPyrB = OceanMath.Image.MultiResolution.LaplacianPyramid();
LPyrB.build(B, num_levels);

% Build Gaussian Pyramids for Masks
disp('Building Gaussian Pyramid for Mask...');
PyrM = OceanMath.Image.MultiResolution.GaussianPyramid();
PyrM.build(Mask, num_levels);

disp('Building Gaussian Pyramid for Mask Complement...');
PyrMComp = OceanMath.Image.MultiResolution.GaussianPyramid();
PyrMComp.build(MaskComplement, num_levels);

disp('Blending Pyramids with Renormalization...');
PyrBlend = OceanMath.Image.MultiResolution.LaplacianPyramid();
PyrBlend.NumLevels = num_levels;
PyrBlend.Levels = cell(1, num_levels);

for l = 1:num_levels
    LA = LPyrA.Levels{l};
    LB = LPyrB.Levels{l};
    GM = PyrM.Levels{l};
    GMComp = PyrMComp.Levels{l};
    
    % Blended level
    blend_level = GM .* LA + GMComp .* LB;
    % Sum of weights
    weight_sum = GM + GMComp;
    
    % Renormalize to prevent boundary padding brightness artifacts
    PyrBlend.Levels{l} = blend_level ./ (weight_sum + 1e-12);
end

disp('Reconstructing Blended Image...');
I_blend = PyrBlend.reconstruct();
I_blend = max(min(I_blend, 1), 0);

% === 4. Sharp Splice Baseline ===
I_splice = Mask .* A + MaskComplement .* B;

% === 5. Visualization & Figures Organization ===
disp('Generating Output Figures...');
viz = OceanViz.Core('Theme', 'Dark');
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Helper function to plot a pyramid's levels
function plotPyramidLevels(pyramid, is_laplacian, viz_theme)
    num_l = pyramid.NumLevels;
    for l = 1:num_l
        ax = subplot(1, num_l, l);
        level_img = pyramid.Levels{l};
        
        % Laplacian details must be offset by 0.5 to show positive/negative differences,
        % except for the last level which is the low-frequency Gaussian residual.
        if is_laplacian && l < num_l
            I_vis = max(0.0, min(1.0, 2.0 * level_img + 0.5));
        else
            I_vis = level_img;
        end
        
        imshow(I_vis);
        [h_l, w_l, ~] = size(level_img);
        title(sprintf('Level %d\n(%dx%d)', l, w_l, h_l), 'Color', viz_theme.Colors.Text);
        viz_theme.styleAxes(ax);
    end
end

% Figure 1: Gaussian Pyramid of Apple
fig1 = figure('Name', 'Figure 1: Gaussian Pyramid of Apple', 'Position', [100, 100, 1000, 300]);
set(fig1, 'Color', viz.Colors.Background);
plotPyramidLevels(GPyrA, false, viz);
sgtitle('Gaussian Pyramid of Image A (Apple)', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
saveas(fig1, fullfile(results_dir, 'gaussian_pyr_apple.png'));

% Figure 2: Gaussian Pyramid of Orange
fig2 = figure('Name', 'Figure 2: Gaussian Pyramid of Orange', 'Position', [150, 150, 1000, 300]);
set(fig2, 'Color', viz.Colors.Background);
plotPyramidLevels(GPyrB, false, viz);
sgtitle('Gaussian Pyramid of Image B (Orange)', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
saveas(fig2, fullfile(results_dir, 'gaussian_pyr_orange.png'));

% Figure 3: Laplacian Pyramid of Apple
fig3 = figure('Name', 'Figure 3: Laplacian Pyramid of Apple', 'Position', [200, 200, 1000, 300]);
set(fig3, 'Color', viz.Colors.Background);
plotPyramidLevels(LPyrA, true, viz);
sgtitle('Laplacian Pyramid of Image A (Apple)', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
saveas(fig3, fullfile(results_dir, 'laplacian_pyr_apple.png'));

% Figure 4: Laplacian Pyramid of Orange
fig4 = figure('Name', 'Figure 4: Laplacian Pyramid of Orange', 'Position', [250, 250, 1000, 300]);
set(fig4, 'Color', viz.Colors.Background);
plotPyramidLevels(LPyrB, true, viz);
sgtitle('Laplacian Pyramid of Image B (Orange)', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
saveas(fig4, fullfile(results_dir, 'laplacian_pyr_orange.png'));

% Figure 5: Blending Comparison Dashboard
fig5 = figure('Name', 'Figure 5: Blending Comparison Dashboard', 'Position', [300, 100, 1100, 750]);
set(fig5, 'Color', viz.Colors.Background);

ax = subplot(2, 3, 1);
imshow(A);
title('Image A: Apple', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(2, 3, 2);
imshow(B);
title('Image B: Orange', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(2, 3, 3);
imshow(Mask);
title('Mask (Region A)', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(2, 3, 4);
imshow(MaskComplement);
title('Mask Complement (Region B)', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(2, 3, 5);
imshow(I_splice);
title('Sharp Direct Splice', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(2, 3, 6);
imshow(I_blend);
title('Seamless Pyramid Blend', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

sgtitle('Laplacian Pyramid Blending Summary', 'Color', viz.Colors.Text, 'FontWeight', 'bold', 'FontSize', 14);
saveas(fig5, fullfile(results_dir, 'laplacian_blend_2_images.png'));

% === 6. Summary printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset          : apple.jpg and orange.jpg\n');
fprintf('  Images Blended   : 2\n');
fprintf('  Resolution       : %dx%d\n', H, W);
fprintf('  Pyramid Levels   : %d\n', num_levels);
fprintf('  Kernel           : 5x5 Gaussian (Binomial)\n');
fprintf('  Renormalization  : YES (enforced level-wise sum-to-1)\n');
fprintf('  Figures Created  : 5 (Apple GPyr, Orange GPyr, Apple LPyr, Orange LPyr, Dashboard)\n');
fprintf('  Output Folder    : %s\n', results_dir);
fprintf('==========================\n');
