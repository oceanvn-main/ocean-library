% exp_histogram_equalization_ex37.m
%
% Scientific Experiment: Histogram Equalization, Gain-Limiting, and Color Ratio Restoration
% (Szeliski Exercise 3.7)
%
% Description:
%   This script implements a complete processing pipeline to address all steps 
%   of Exercise 3.7 from Richard Szeliski's "Computer Vision: Algorithms and Applications":
%     1. Convert the color image to luminance (Rec. 601 coefficients).
%     2. Explicitly compute the histogram, cumulative distribution function (CDF),
%        and the equalization transfer function.
%     3. Optional: Increase the image "punch" by mapping 5% of pixels to pure black and white.
%     4. Optional: Limit the local gain f'(I) of the transfer function by clipping 
%        the histogram and redistributing unaccumulated values uniformly.
%     5. Apply luminance compensation and restore the color channels using color ratios.
%     6. Optional: Detect clipped/saturated colors in the original image and apply a blending
%        mechanism to prevent color shifts in highlights.
%
% Scientific Principles Applied (RULE[experiment.md]):
%   - Explicit Computation: All values are derived step-by-step from input images.
%   - Reproducibility: Fully self-contained path calculation.
%   - No LaTeX: Math formulas and text output use plain text formatting.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-09

clear;
close all;
clc;

% Ensure paths are set
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

% Initialize OceanViz Core Engine (using Scientific theme)
viz = OceanViz.Core('Theme', 'Scientific');

% Set toggle to use synthetic test image or original ocean dataset
use_synthetic = true;

if use_synthetic
    fprintf('Generating synthetic test image for the experiment...\n');
    I_orig = generate_synthetic_test_image();
else
    % Load dataset
    image_path = fullfile(project_root, 'Data', 'images', 'ocean_img.jpg');
    fprintf('Loading image from: %s\n', image_path);
    if ~exist(image_path, 'file')
        error('Target image file does not exist: %s', image_path);
    end
    I_orig = imread(image_path);
end
I_double = double(I_orig);
[height, width, num_channels] = size(I_orig);
num_pixels = height * width;

%% Step 1: Convert Color Image to Luminance (Luma)
% Using Rec. 601 luma coefficients: Y = 0.299*R + 0.587*G + 0.114*B
Y_in = 0.299 * I_double(:,:,1) + 0.587 * I_double(:,:,2) + 0.114 * I_double(:,:,3);
Y_in_uint8 = uint8(max(0, min(255, Y_in)));

%% Step 2: Compute Histogram, CDF, and Standard Equalization
h_orig = zeros(256, 1);
for val = 0:255
    h_orig(val + 1) = sum(Y_in_uint8(:) == val);
end
cdf_orig = cumsum(h_orig);
T_standard = 255.0 * (cdf_orig / num_pixels);
Y_out_standard = T_standard(double(Y_in_uint8) + 1);

%% Step 3: Increase "Punch" by Clipping 5% of Pixels to Black and White
% Map 2.5% of pixels to pure black and 2.5% of pixels to pure white.
low_p = prctile(Y_in(:), 2.5);
high_p = prctile(Y_in(:), 97.5);

Y_punched = (Y_in - low_p) / (high_p - low_p) * 255.0;
Y_punched = max(0, min(255, Y_punched));
Y_punched_uint8 = uint8(Y_punched);

% Compute histogram & CDF of punched luminance
h_punched = zeros(256, 1);
for val = 0:255
    h_punched(val + 1) = sum(Y_punched_uint8(:) == val);
end
cdf_punched = cumsum(h_punched);
T_punched = 255.0 * (cdf_punched / num_pixels);
Y_out_punched = T_punched(double(Y_punched_uint8) + 1);

%% Step 4: Limit Local Gain f'(I) of Transfer Function
% The derivative of the transfer function, f'(I), is proportional to the histogram height.
% To limit f'(I) < gamma, we clip the histogram bins to a maximum limit and redistribute excess.
gamma = 2.0; % maximum allowed contrast enhancement factor (gain limit)
clip_limit = gamma * (num_pixels / 256.0);

h_clipped = min(h_orig, clip_limit);
excess = sum(max(0, h_orig - clip_limit));
h_redist = h_clipped + (excess / 256.0);

cdf_limited = cumsum(h_redist);
T_limited = 255.0 * (cdf_limited / sum(h_redist));
Y_out_limited = T_limited(double(Y_in_uint8) + 1);

%% Step 5: Restore Color Channels using Color Ratios (Equation 2.117)
% Math: C_out = C_in * (Y_out / Y_in)
denom = Y_in;
denom(denom == 0) = 1e-5; % Avoid division by zero
ratio_standard = Y_out_standard ./ denom;
ratio_limited = Y_out_limited ./ denom;

I_ratio_standard = I_double .* cat(3, ratio_standard, ratio_standard, ratio_standard);
I_ratio_standard = uint8(max(0, min(255, I_ratio_standard)));

I_ratio_limited = I_double .* cat(3, ratio_limited, ratio_limited, ratio_limited);
I_ratio_limited = uint8(max(0, min(255, I_ratio_limited)));

%% Step 6: Handle Clipped/Saturated Color Channels to Prevent Color Shifts
% Saturated highlights mapped with ratios can shift hues unnaturally.
% We define a smooth weight mask based on how close any channel is to saturation (255).
max_channel_val = max(I_double, [], 3);
sat_threshold = 245.0;
% Smooth transition weighting from 0.0 (not clipped) to 1.0 (fully clipped)
sat_weight = (max_channel_val - sat_threshold) / (255.0 - sat_threshold);
sat_weight = max(0, min(1, sat_weight));

% Re-blend ratio-compensated image with the original image for saturated areas
I_ratio_clipped_handled = zeros(size(I_double));
for c = 1:3
    I_ratio_clipped_handled(:,:,c) = (1.0 - sat_weight) .* (I_double(:,:,c) .* ratio_limited) + sat_weight .* I_double(:,:,c);
end
I_ratio_clipped_handled = uint8(max(0, min(255, I_ratio_clipped_handled)));

%% Plot Visualizations
fig = viz.createFigure('Exercise 3.7: Histogram Equalization and Color Ratios');
set(fig, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);

% 1. Original RGB Image
ax1 = subplot(3, 4, 1);
imshow(I_orig);
title(ax1, '1. Original RGB Image', 'Color', viz.Colors.Text);
set(ax1, 'Color', viz.Colors.Background);

% 2. Original Luminance Histogram & CDF
ax2 = subplot(3, 4, 2);
hold(ax2, 'on');
yyaxis(ax2, 'left');
p1 = plot(ax2, 0:255, h_orig, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
ylabel(ax2, 'Histogram Count');
set(ax2, 'YColor', viz.Colors.Primary);

yyaxis(ax2, 'right');
p2 = plot(ax2, 0:255, cdf_orig, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
ylabel(ax2, 'Cumulative Sum');
set(ax2, 'YColor', viz.Colors.Secondary);
hold(ax2, 'off');
xlim(ax2, [0, 255]);
viz.styleAxes(ax2);
title(ax2, '2. Original Luma Hist & CDF', 'Color', viz.Colors.Text);
lgd2 = legend(ax2, [p1, p2], {'Histogram', 'CDF'}, 'Location', 'northwest');
set(lgd2, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% 3. Standard Equalized Image
ax3 = subplot(3, 4, 3);
imshow(I_ratio_standard);
title(ax3, '3. Color Ratio Equalized (Std)', 'Color', viz.Colors.Text);
set(ax3, 'Color', viz.Colors.Background);

% 4. Standard Equalized Luma Histogram
ax4 = subplot(3, 4, 4);
h_eq_std = histcounts(Y_out_standard(:), 0:256);
plot(ax4, 0:255, h_eq_std, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
xlim(ax4, [0, 255]);
viz.styleAxes(ax4);
title(ax4, '4. Std Equalized Luma Hist', 'Color', viz.Colors.Text);

% 5. Punched Luminance Image
ax5 = subplot(3, 4, 5);
imshow(uint8(Y_punched));
title(ax5, '5. 5% Punched Luminance', 'Color', viz.Colors.Text);
set(ax5, 'Color', viz.Colors.Background);

% 6. Punched Luma Histogram & CDF
ax6 = subplot(3, 4, 6);
hold(ax6, 'on');
yyaxis(ax6, 'left');
p3 = plot(ax6, 0:255, h_punched, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
ylabel(ax6, 'Histogram Count');
set(ax6, 'YColor', viz.Colors.Primary);

yyaxis(ax6, 'right');
p4 = plot(ax6, 0:255, cdf_punched, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
ylabel(ax6, 'Cumulative Sum');
set(ax6, 'YColor', viz.Colors.Secondary);
hold(ax6, 'off');
xlim(ax6, [0, 255]);
viz.styleAxes(ax6);
title(ax6, '6. Punched Luma Hist & CDF', 'Color', viz.Colors.Text);
lgd6 = legend(ax6, [p3, p4], {'Histogram', 'CDF'}, 'Location', 'northwest');
set(lgd6, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% 7. Punched Equalized Color Ratio Image
I_ratio_punched = I_double .* cat(3, Y_out_punched ./ denom, Y_out_punched ./ denom, Y_out_punched ./ denom);
I_ratio_punched = uint8(max(0, min(255, I_ratio_punched)));
ax7 = subplot(3, 4, 7);
imshow(I_ratio_punched);
title(ax7, '7. Color Ratio Equalized (Punched)', 'Color', viz.Colors.Text);
set(ax7, 'Color', viz.Colors.Background);

% 8. Punched Equalized Luma Histogram
ax8 = subplot(3, 4, 8);
h_eq_punch = histcounts(Y_out_punched(:), 0:256);
plot(ax8, 0:255, h_eq_punch, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
xlim(ax8, [0, 255]);
viz.styleAxes(ax8);
title(ax8, '8. Punched Equalized Luma Hist', 'Color', viz.Colors.Text);

% 9. Contrast/Gain Limited Transfer Function comparison
ax9 = subplot(3, 4, 9);
hold(ax9, 'on');
plot(ax9, 0:255, T_standard, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
plot(ax9, 0:255, T_limited, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
plot(ax9, 0:255, 0:255, 'k--', 'LineWidth', 1.0);
hold(ax9, 'off');
xlim(ax9, [0, 255]);
ylim(ax9, [0, 255]);
xlabel(ax9, 'Input Luma');
ylabel(ax9, 'Mapped Luma');
viz.styleAxes(ax9);
title(ax9, '9. Transfer Function comparison', 'Color', viz.Colors.Text);
lgd = legend(ax9, 'Standard', 'Gain-Limited', 'Identity', 'Location', 'southeast');
set(lgd, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% 10. Gain-Limited Equalized Image (Standard Color Ratio)
ax10 = subplot(3, 4, 10);
imshow(I_ratio_limited);
title(ax10, '10. Equalized (Gain-Limited)', 'Color', viz.Colors.Text);
set(ax10, 'Color', viz.Colors.Background);

% 11. Gain-Limited Equalized Image (Clipped Handler Active)
ax11 = subplot(3, 4, 11);
imshow(I_ratio_clipped_handled);
title(ax11, '11. Equalized (Gain-Lim + Highlight)', 'Color', viz.Colors.Text);
set(ax11, 'Color', viz.Colors.Background);

% 12. Gain-Limited Luma Histogram
ax12 = subplot(3, 4, 12);
h_eq_lim = histcounts(Y_out_limited(:), 0:256);
plot(ax12, 0:255, h_eq_lim, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
xlim(ax12, [0, 255]);
viz.styleAxes(ax12);
title(ax12, '12. Gain-Lim Equalized Luma Hist', 'Color', viz.Colors.Text);

% Super Title
sgtitle(fig, 'Szeliski Exercise 3.7: Complete Pipeline Exploration', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

%% Save Results
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
if use_synthetic
    output_fig_path = fullfile(results_dir, 'exercise_3_7_synthetic_results.png');
else
    output_fig_path = fullfile(results_dir, 'exercise_3_7_results.png');
end
fprintf('Saving results visualization to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

fprintf('Experiment complete!\n');
