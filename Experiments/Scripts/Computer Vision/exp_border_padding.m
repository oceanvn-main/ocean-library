% exp_border_padding.m
%
% Scientific Experiment: Replicating Szeliski's Figure 3.13 (Border Padding & Blurring)
%
% Description:
%   This script pads the 16x16 pixel-art yellow duck image by 4 pixels on all sides
%   using 5 different border-padding strategies:
%     1. Zero Padding (constant 0)
%     2. Cyclic Wrap Padding (periodic toroidal wrapping)
%     3. Clamp Padding (replicate / repeat edge pixels)
%     4. Mirror Padding (symmetrical reflection / reflect)
%     5. Extend Padding (linear extrapolation / extension)
%
%   It then blurs the padded images using a 5x5 box blur kernel. For zero padding,
%   it demonstrates the soft alpha normalization technique to remove the boundary
%   darkening effect.
%
%   The results are rendered in a high-fidelity 2x5 grid matching the dark premium
%   aesthetic and saved as a PNG snapshot.
%
% Author: Antigravity Code Assistant
% Date: 2026-06-12

clear;
close all;
clc;

% Import the boundary padder and convolution operator
import OceanMath.Image.Pixels.BoundaryPadder
import OceanMath.Image.Pixels.TrueConvolution

%% 1. Set Up Directories and File Paths
script_dir = fileparts(mfilename('fullpath'));
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_snapshot_path = fullfile(results_dir, 'border_padding_duck.png');

%% 2. Generate a High-Contrast Checkerboard Image (128x128)
H = 128;
W = 128;
C = 1;
[X, Y] = meshgrid(1:W, 1:H);

% Create a checkerboard pattern with 16x16 squares
square_size = 16;
checker = mod(floor((X - 1) / square_size) + floor((Y - 1) / square_size), 2);

% Overlay a diagonal gradient so that Extend padding's linear extrapolation is visible
I_in = 0.15 + 0.55 * double(checker) + 0.15 * (X + Y) / (H + W);

%% 3. Define the Box Blur Kernel and Padding Size
K_blur = 1/225 * ones(15, 15);
% We use a wide padding P = 32 (rather than the minimum floor(K/2) = 7)
% so that the padded borders are wide enough to clearly visualize the differences.
P = 32;
pad_size = [P, P];

% A. Zero Padding
I_pad_zero = BoundaryPadder.pad(I_in, pad_size, 'zero');

% B. Cyclic Wrap Padding
I_pad_wrap = BoundaryPadder.pad(I_in, pad_size, 'wrap');

% C. Clamp Padding
I_pad_clamp = BoundaryPadder.pad(I_in, pad_size, 'clamp');

% D. Mirror/Reflect Padding
I_pad_mirror = BoundaryPadder.pad(I_in, pad_size, 'mirror');

% E. Extend Padding
I_pad_extend = BoundaryPadder.pad(I_in, pad_size, 'extend');

%% 4. Perform Box Blur and Crop to Original Size (128x128)
blur_op = TrueConvolution('Kernel', K_blur, 'Boundary', 0.0); % 0-padding for the blur itself since we pre-padded

% Helper function to crop the padded image back to original size H x W
crop_to_original = @(I) I(P + (1:H), P + (1:W), :);

% A. Blurred Zero
I_blur_zero = crop_to_original(blur_op.apply(I_pad_zero));

% B. Normalized Zero (Soft Alpha Division)
% Create a binary mask of the image (ones inside original, zeros outside)
M_in = ones(H, W);
M_pad = BoundaryPadder.pad(M_in, pad_size, 'zero');
M_blurred_pad = blur_op.apply(M_pad);
M_blurred = crop_to_original(M_blurred_pad);

% Divide luma channels by the soft alpha mask to remove darkening
I_blur_normalized = zeros(H, W, C);
for ch = 1:C
    I_blur_normalized(:,:,ch) = I_blur_zero(:,:,ch) ./ M_blurred;
end
% Ensure no NaN/Inf from division
I_blur_normalized(isnan(I_blur_normalized)) = 0;
I_blur_normalized(isinf(I_blur_normalized)) = 0;
% Clip to range [0, 1]
I_blur_normalized = min(max(I_blur_normalized, 0), 1);

% C. Blurred Wrap
I_blur_wrap = crop_to_original(blur_op.apply(I_pad_wrap));

% D. Blurred Clamp
I_blur_clamp = crop_to_original(blur_op.apply(I_pad_clamp));

% E. Blurred Mirror
I_blur_mirror = crop_to_original(blur_op.apply(I_pad_mirror));

% F. Blurred Extend
I_blur_extend = crop_to_original(blur_op.apply(I_pad_extend));

%% 5. Render Figure Layout (Dark Premium Theme)
fig = figure('Name', 'Szeliski Figure 3.13 Replicated on Synthetic Image', ...
             'Color', [0.03, 0.04, 0.05], ...
             'Position', [100, 100, 1280, 640]);

% Layout parameters
titles_top = {'Zero padded', 'Wrap padded', 'Clamp padded', 'Mirror padded', 'Extend padded'};
images_top = {I_pad_zero, I_pad_wrap, I_pad_clamp, I_pad_mirror, I_pad_extend};

titles_bot = {'blurred zero', 'normalized zero', 'blurred clamp', 'blurred mirror', 'blurred extend'};
images_bot = {I_blur_zero, I_blur_normalized, I_blur_clamp, I_blur_mirror, I_blur_extend};

% Helper to convert single-channel grayscale to 3-channel truecolor for plotting
to_rgb = @(img) repmat(img, 1, 1, 4 - size(img, 3));

for i = 1:5
    % Top Row (Padded Images)
    ax_top = subplot(2, 5, i);
    image('CData', to_rgb(images_top{i}), 'Parent', ax_top);
    axis(ax_top, 'image');
    set(ax_top, 'Color', [0.03, 0.04, 0.05], 'XColor', [0.3, 0.3, 0.3], 'YColor', [0.3, 0.3, 0.3], ...
                'XTick', [], 'YTick', [], 'FontSize', 6, 'GridColor', [0.2, 0.2, 0.25]);
    grid(ax_top, 'on');
    title(ax_top, titles_top{i}, 'Color', [0.9, 0.9, 0.9], 'FontSize', 12, 'FontWeight', 'bold');
    
    % Add red rectangle highlighting the original image bounds
    rectangle('Parent', ax_top, 'Position', [P+0.5, P+0.5, W, H], ...
              'EdgeColor', [1.0, 0.3, 0.3], 'LineWidth', 1.5, 'LineStyle', '--');
    
    % Bottom Row (Blurred & Cropped results)
    ax_bot = subplot(2, 5, i + 5);
    image('CData', to_rgb(images_bot{i}), 'Parent', ax_bot);
    axis(ax_bot, 'image');
    set(ax_bot, 'Color', [0.03, 0.04, 0.05], 'XColor', [0.3, 0.3, 0.3], 'YColor', [0.3, 0.3, 0.3], ...
                'XTick', [], 'YTick', [], 'FontSize', 6, 'GridColor', [0.2, 0.2, 0.25]);
    grid(ax_bot, 'on');
    title(ax_bot, titles_bot{i}, 'Color', [0.8, 0.8, 0.8], 'FontSize', 11);
end

% Set overall title
annotation(fig, 'textbox', [0.05, 0.93, 0.9, 0.06], ...
           'String', 'BORDER PADDING AND BLURRING EFFECTS (Szeliski Fig 3.13 Replica on Synthetic Image)', ...
           'Color', [1.0, 1.0, 1.0], ...
           'FontSize', 16, ...
           'FontWeight', 'bold', ...
           'HorizontalAlignment', 'center', ...
           'LineStyle', 'none');

drawnow;

% Save the visual result
saveas(fig, output_snapshot_path);

fprintf('Border padding experiment completed successfully.\n');
fprintf('Snapshot generated successfully: %s\n', output_snapshot_path);
