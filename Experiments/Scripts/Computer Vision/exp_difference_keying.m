% EXP_DIFFERENCE_KEYING  Statistical Difference Keying (Ex 3.5 Parts 1 & 2)
%
%   Implements Exercise 3.5 using the CDnet 2014 baseline/highway dataset:
%     1. Computes mean & standard variance, and median & robust variance
%        at each pixel in a training sequence.
%     2. For a new frame, classifies each pixel as foreground or background,
%        setting the background pixels to RGBA = 0.
%     3. Saves comparative plots for both parts to the Results folder.
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-07-08

clear; close all; clc;

disp('Initializing CDnet-2014 Difference Keying (Parts 1 & 2)...');
script_dir = fileparts(mfilename('fullpath'));
root_dir   = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Results folder
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

%% 0. Configuration & Paths
category = 'baseline';
sequence = 'highway';
seq_dir  = fullfile(root_dir, 'Data', 'videos', 'dataset', category, sequence);
inp_dir  = fullfile(seq_dir, 'input');
gt_dir   = fullfile(seq_dir, 'groundtruth');
roi_path = fullfile(seq_dir, 'ROI.bmp');
troi_path = fullfile(seq_dir, 'temporalROI.txt');

if ~exist(seq_dir, 'dir')
    error('Dataset sequence not found at: %s', seq_dir);
end

% Parse Temporal ROI to define training and test ranges
roi_data = textscan(fileread(troi_path), '%d %d');
eval_start = double(roi_data{1});
eval_end   = double(roi_data{2});

%% 1. Load Training Sequence (Step 1)
fprintf('\nLoading training frames...\n');
train_end = min(eval_start - 1, 50); % Use first 50 frames to estimate background
train_indices = 1:train_end;

% Load first frame to get dimensions
first_file = fullfile(inp_dir, sprintf('in%06d.jpg', 1));
first_img  = double(imread(first_file)) / 255.0;
[target_h, target_w, target_c] = size(first_img);

% Load ROI mask
roi_mask_raw = imread(roi_path);
roi_mask = imresize(roi_mask_raw > 0, [target_h, target_w], 'nearest') > 0;
if ~any(roi_mask(:))
    roi_mask = true(target_h, target_w);
end

% Allocate and read training frames
n_train = numel(train_indices);
train_frames = zeros(target_h, target_w, target_c, n_train);
for k = 1:n_train
    fpath = fullfile(inp_dir, sprintf('in%06d.jpg', train_indices(k)));
    train_frames(:,:,:,k) = double(imread(fpath)) / 255.0;
end

%% 2. Train Background Models (Step 1)
fprintf('\nTraining background models...\n');
% Method A: Mean and standard variance
bs_mean = OceanMath.Image.Resampling.BackgroundSubtractor(train_frames, 'Method', 'meanstd');

% Method B: Median and robust variance (MAD)
bs_med = OceanMath.Image.Resampling.BackgroundSubtractor(train_frames, 'Method', 'medianmad');

%% 3. Plot Part 1: Background Models & Variance Maps
fprintf('\nGenerating Part 1 plots...\n');
viz = OceanViz.Core('Theme', 'Dark');

fig1 = figure('Name', 'Part 1: Background Model Comparison', ...
    'Color', viz.Colors.Background, 'Position', [100, 100, 1100, 800]);

% Helper to scale variance for visualization
scale_var = @(v) min(1.0, mean(v, 3) * 50);

% Subplot 1: Background Estimate (Mean)
ax1 = subplot(2, 2, 1);
imshow(bs_mean.CentralTendency);
title(ax1, 'Background Central Tendency: Mean', 'Color', viz.Colors.Text, 'FontSize', 12);
xlabel(ax1, 'Standard Arithmetic Mean Estimate', 'Color', viz.Colors.Grid, 'FontSize', 9);

% Subplot 2: Background Estimate (Median)
ax2 = subplot(2, 2, 2);
imshow(bs_med.CentralTendency);
title(ax2, 'Background Central Tendency: Median', 'Color', viz.Colors.Text, 'FontSize', 12);
xlabel(ax2, 'Robust Temporal Median Estimate', 'Color', viz.Colors.Grid, 'FontSize', 9);

% Subplot 3: Variance (Standard)
ax3 = subplot(2, 2, 3);
imshow(scale_var(bs_mean.Variance));
colormap(ax3, 'hot');
colorbar(ax3, 'Color', viz.Colors.Text);
title(ax3, 'Pixel Variance (Standard)', 'Color', viz.Colors.Text, 'FontSize', 12);
xlabel(ax3, 'Scaled variance map (Standard Dev)', 'Color', viz.Colors.Grid, 'FontSize', 9);

% Subplot 4: Variance (Robust MAD)
ax4 = subplot(2, 2, 4);
imshow(scale_var(bs_med.Variance));
colormap(ax4, 'hot');
colorbar(ax4, 'Color', viz.Colors.Text);
title(ax4, 'Robust Variance (MAD-scaled)', 'Color', viz.Colors.Text, 'FontSize', 12);
xlabel(ax4, 'Scaled robust variance map (MAD)', 'Color', viz.Colors.Grid, 'FontSize', 9);

part1_fig_path = fullfile(results_dir, 'ex35_part1_background_models.png');
exportgraphics(fig1, part1_fig_path, 'Resolution', 150);
fprintf('  Part 1 plot saved to: %s\n', part1_fig_path);

%% 4. Apply to a New Frame & Classify (Step 2)
% We select a frame inside the temporal ROI where cars are clearly present (e.g. frame 900)
test_frame_num = 900;
fprintf('\nEvaluating new frame #%d (Step 2)...\n', test_frame_num);

test_img_path = fullfile(inp_dir, sprintf('in%06d.jpg', test_frame_num));
test_frame = double(imread(test_img_path)) / 255.0;

% We use the raw normalized test frame directly. Exposure normalization is not needed
% for the highway sequence and would introduce artifacts due to vehicles changing the mean.
test_frame_norm = test_frame;

% Apply Chi-square threshold (11.34 for 3 DoF at 99% confidence)
threshold = 11.34;
[fg_mask_mean, fg_rgba_mean] = bs_mean.apply(test_frame_norm, threshold);
[fg_mask_med, fg_rgba_med] = bs_med.apply(test_frame_norm, threshold);

% Mask out regions outside spatial ROI
fg_mask_mean = fg_mask_mean & roi_mask;
fg_mask_med = fg_mask_med & roi_mask;

% Zero-out background pixels in final segmented output (RGBA = 0 where mask is false)
fg_rgba_mean(repmat(~fg_mask_mean, [1, 1, 4])) = 0.0;
fg_rgba_med(repmat(~fg_mask_med, [1, 1, 4])) = 0.0;

%% 5. Plot Part 2: Pixel Classification (RGBA = 0 background)
fprintf('\nGenerating Part 2 plots...\n');
fig2 = figure('Name', 'Part 2: Foreground Extraction (RGBA=0 background)', ...
    'Color', viz.Colors.Background, 'Position', [120, 120, 1100, 800]);

% Subplot 1: Input Frame
ax2_1 = subplot(2, 2, 1);
imshow(test_frame);
title(ax2_1, sprintf('Input Frame #%d', test_frame_num), 'Color', viz.Colors.Text, 'FontSize', 12);

% Subplot 2: Classification Mask (MeanStd)
ax2_2 = subplot(2, 2, 2);
imshow(fg_mask_mean);
title(ax2_2, 'Foreground Mask (Mean & Std Dev)', 'Color', viz.Colors.Text, 'FontSize', 12);

% Subplot 3: Classification Mask (Median & MAD)
ax2_3 = subplot(2, 2, 3);
imshow(fg_mask_med);
title(ax2_3, 'Foreground Mask (Median & Robust MAD)', 'Color', viz.Colors.Text, 'FontSize', 12);

% Subplot 4: Segmented Foreground (RGBA=0 Background)
ax2_4 = subplot(2, 2, 4);
% Show RGB channels, transparency is handled by setting background pixels to black/zero
imshow(fg_rgba_med(:,:,1:3));
title(ax2_4, 'Extracted Foreground (RGBA=0 Background)', 'Color', viz.Colors.Text, 'FontSize', 12);
xlabel(ax2_4, 'Background pixels zeroed out', 'Color', viz.Colors.Grid, 'FontSize', 9);

part2_fig_path = fullfile(results_dir, 'ex35_part2_foreground_extraction.png');
exportgraphics(fig2, part2_fig_path, 'Resolution', 150);
fprintf('  Part 2 plot saved to: %s\n', part2_fig_path);

disp('SUCCESS: Exercise 3.5 Parts 1 & 2 completed.');
