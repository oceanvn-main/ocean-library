% EXP_FACE_MORPHING  Seamless image morphing using Thin Plate Spline warping.
%
%   This script implements face morphing between two images (pair1 and pair2).
%   It compares two methods of image blending:
%     1. Simple Cross-Dissolve: Blends the images directly, producing ghosting.
%     2. Shape-Warped Morphing: Warps both images to a shared intermediate shape
%        (halfway landmarks) using Thin Plate Splines, then blends them.
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-30

clear; close all; clc;

disp('Initializing Experiment: Face Morphing (pair1 <-> pair2)...');
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Create Results folder if not exists
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Paths
img1_path = fullfile(root_dir, 'Data', 'images', 'secrets', 'pair1.jpg');
img2_path = fullfile(root_dir, 'Data', 'images', 'secrets', 'pair3.png');
landmarks_path = fullfile(root_dir, 'Data', 'images', 'secrets', 'pair_landmarks.mat');

out_fig_morph = fullfile(results_dir, 'face_morphing_comparison.png');
out_fig_seq = fullfile(results_dir, 'face_morphing_sequence.png');

% Load images
I1 = imread(img1_path);
I2 = imread(img2_path);

% Downsample to speed up processing and save memory
% The original images are 1705 x 2560 and 595 x 892. Resizing to 512 x 768 is fast and high-quality.
scale1 = 512 / size(I1, 1);
scale2 = 512 / size(I2, 1);

I1_small = imresize(I1, scale1);
I2_small = imresize(I2, [size(I1_small, 1), size(I1_small, 2)]);

[H, W, C] = size(I1_small);

% Load landmarks and scale them accordingly based on their respective original image resolutions
load(landmarks_path, 'pts1', 'pts2');
pts1_scaled = pts1 * scale1;
pts2_scaled = pts2 * scale2;

%% 1. Simple Cross-Dissolve (No Warping)
disp('Computing simple cross-dissolve (ghosting)...');
I_blend_simple = 0.5 * double(I1_small) + 0.5 * double(I2_small);
I_blend_simple = uint8(max(0, min(255, round(I_blend_simple))));

%% 2. Shape-Warped Morphing (Halfway, t = 0.5)
disp('Computing intermediate shape and warping...');
t = 0.5;
pts_inter = (1 - t) * pts1_scaled + t * pts2_scaled;

% Add boundary points to warp the background context cleanly
border_x = [1, W, W, 1]';
border_y = [1, 1, H, H]';
pts_src_boundary = [border_x, border_y];

pts1_all = [pts1_scaled; pts_src_boundary];
pts2_all = [pts2_scaled; pts_src_boundary];
pts_inter_all = [pts_inter; pts_src_boundary];

import OceanMath.Image.Resampling.ThinPlateSpline

% Warp Image 1 to intermediate shape
tps1 = ThinPlateSpline(pts_inter_all, pts1_all, 'thinplate');
I1_warped = tps1.warpImage(I1_small);

% Warp Image 2 to intermediate shape
tps2 = ThinPlateSpline(pts_inter_all, pts2_all, 'thinplate');
I2_warped = tps2.warpImage(I2_small);

% Blend warped images
I_morph_half = (1 - t) * double(I1_warped) + t * double(I2_warped);
I_morph_half = uint8(max(0, min(255, round(I_morph_half))));

%% 3. Generate Morphing Sequence (t = 0.0, 0.25, 0.5, 0.75, 1.0)
disp('Generating morphing sequence...');
t_vals = [0.0, 0.25, 0.5, 0.75, 1.0];
sequence_frames = cell(1, length(t_vals));

for k = 1:length(t_vals)
    tk = t_vals(k);
    pts_k = (1 - tk) * pts1_scaled + tk * pts2_scaled;
    pts_k_all = [pts_k; pts_src_boundary];
    
    tps1_k = ThinPlateSpline(pts_k_all, pts1_all, 'thinplate');
    I1_w_k = tps1_k.warpImage(I1_small);
    
    tps2_k = ThinPlateSpline(pts_k_all, pts2_all, 'thinplate');
    I2_w_k = tps2_k.warpImage(I2_small);
    
    I_morph_k = (1 - tk) * double(I1_w_k) + tk * double(I2_w_k);
    sequence_frames{k} = uint8(max(0, min(255, round(I_morph_k))));
end

%% 4. Visualization & Save Figures
viz = OceanViz.Core('Theme', 'Dark');

% --- Figure 1: Morphing Comparison Dashboard ---
f1 = figure('Name', 'Face Morphing: Warped vs Simple Blend', ...
            'Color', viz.Colors.Background, 'Position', [50, 50, 1500, 900]);

% Top Row: Simple blend
ax1 = subplot(2, 3, 1);
imshow(I1_small);
title(ax1, 'Image 1 (Source)', 'Color', viz.Colors.Text);

ax2 = subplot(2, 3, 2);
imshow(I2_small);
title(ax2, 'Image 2 (Target)', 'Color', viz.Colors.Text);

ax3 = subplot(2, 3, 3);
imshow(I_blend_simple);
title(ax3, 'Simple Blending (Ghosting)', 'Color', viz.Colors.Text);

% Bottom Row: Warped Morph
ax4 = subplot(2, 3, 4);
imshow(I1_warped);
hold(ax4, 'on');
plot(ax4, pts_inter(:, 1), pts_inter(:, 2), 'g.', 'MarkerSize', 8);
hold(ax4, 'off');
title(ax4, 'Warped Image 1 (t = 0.5)', 'Color', viz.Colors.Text);

ax5 = subplot(2, 3, 5);
imshow(I2_warped);
hold(ax5, 'on');
plot(ax5, pts_inter(:, 1), pts_inter(:, 2), 'g.', 'MarkerSize', 8);
hold(ax5, 'off');
title(ax5, 'Warped Image 2 (t = 0.5)', 'Color', viz.Colors.Text);

ax6 = subplot(2, 3, 6);
imshow(I_morph_half);
title(ax6, 'TPS Warped Morphing (Seamless)', 'Color', viz.Colors.Text);

drawnow;
print(f1, out_fig_morph, '-dpng', '-r110');

% --- Figure 2: Morphing Sequence ---
f2 = figure('Name', 'Face Morphing Sequence', ...
            'Color', viz.Colors.Background, 'Position', [100, 100, 1800, 450]);

for k = 1:length(t_vals)
    ax_seq = subplot(1, 5, k);
    imshow(sequence_frames{k});
    title(ax_seq, sprintf('t = %.2f', t_vals(k)), 'Color', viz.Colors.Text);
end

drawnow;
print(f2, out_fig_seq, '-dpng', '-r110');

%% 5. Experiment Summary
fprintf('\n=== FACE MORPHING EXPERIMENT SUMMARY ===\n');
fprintf('  Image 1 size    : %d x %d pixels\n', W, H);
fprintf('  Image 2 size    : %d x %d pixels\n', W, H);
fprintf('  Landmark count  : %d points\n', size(pts1_scaled, 1));
fprintf('  TPS Warping     : Executed with boundary anchor points\n');
fprintf('  Output Figures  : Saved successfully in Results/\n');
fprintf('========================================\n');
disp('SUCCESS: Face morphing comparison complete.');
