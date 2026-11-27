% exp_linear_filtering_viz.m
%
% Scientific Experiment: Neighborhood Linear Filtering (Correlation vs. Convolution)
%
% Description:
%   This script implements an animation that shows the step-by-step differences
%   between Cross-Correlation (direct neighborhood matching) and True Mathematical 
%   Convolution (180-degree flipped kernel matching). It uses an asymmetric 
%   16x16 pixel-art image of a yellow duck and an asymmetric shift kernel to 
%   clearly demonstrate the difference: Cross-correlation shifts the image to 
%   the left, while True Convolution shifts it to the right.
%
%   The animation is saved as a premium dark-themed MP4 video inside the Results/
%   directory, styled similarly to the textbook animation layout.
%
% Scientific Principles Applied (RULE[experiment.md]):
%   1. Explicit Computation: All sliding-window neighborhood calculations, kernel
%      rotations, and outputs are derived directly from the inputs.
%   2. Verification: The script verifies that the manual frame-by-frame loop output
%      is mathematically identical to the library classes (CrossCorrelation & TrueConvolution).
%   3. Reproducibility: Fully self-contained pathing relative to the script location.
%   4. No LaTeX: Code comments, messages, and command window outputs use plain-text.
%
% Author: Antigravity Code Assistant
% Date: 2026-06-12

clear;
close all;
clc;

% Import the modular linear filtering strategies
import OceanMath.Image.Pixels.CrossCorrelation
import OceanMath.Image.Pixels.TrueConvolution

%% 1. Set Up Directories and File Paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_video_path = fullfile(results_dir, 'linear_filtering_comparison.mp4');
output_snapshot_path = fullfile(results_dir, 'linear_filtering_snapshot.png');

%%% 2. Define the Asymmetric 16x16 Pixel Art Image (Yellow Duck)
% 0: Background (Dark Charcoal/Blue)
% 1: Yellow body
% 2: Orange bill/feet
% 3: Black eye
duck = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0;
    0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0;
    0,0,0,0,0,1,1,3,1,1,1,0,0,0,0,0; % black eye at col 8
    0,0,0,0,0,1,1,1,1,1,1,2,2,0,0,0; % orange bill starts at col 12
    0,0,0,0,0,1,1,1,1,1,1,2,2,2,0,0; % orange bill
    0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0;
    0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0; % body
    0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0;
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0;
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0;
    0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0;
    0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0;
    0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0; % legs
    0,0,0,2,2,2,0,0,0,2,2,2,0,0,0,0; % feet
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
];

% Premium color palette
bg_color = [0.06, 0.07, 0.10]; % Dark navy/charcoal
yellow_color = [0.98, 0.82, 0.08]; % Vibrant yellow
orange_color = [0.95, 0.45, 0.05]; % Vivid orange
black_color = [0.12, 0.12, 0.15]; % Dark slate grey/black

colors = [
    bg_color;      % 0: Background
    yellow_color;  % 1: Yellow
    orange_color;  % 2: Orange
    black_color    % 3: Black
];

[H, W] = size(duck);

% Convert indexed pixel art to RGB double image
I_in = zeros(H, W, 3);
for r = 1:H
    for c = 1:W
        I_in(r, c, :) = colors(duck(r, c) + 1, :);
    end
end

%% 3. Define the Asymmetric Shift Kernel
% This kernel has a single 1.0 on the right side of the center.
% - Cross-Correlation should shift the duck left.
% - True Convolution (rotated 180 degrees) should shift the duck right.
K = [ 0.0,  0.0,  0.0;
      0.0,  0.0,  1.0;
      0.0,  0.0,  0.0 ];

fprintf('Filter Kernel:\n');
disp(K);

%% 4. Apply Filtering via Library Strategies and Verify Output
% Replicate boundary padding is used.
boundary_mode = 'replicate';

xcorr_op = CrossCorrelation('Kernel', K, 'Boundary', boundary_mode);
I_xcorr_expected = xcorr_op.apply(I_in);

conv_op = TrueConvolution('Kernel', K, 'Boundary', boundary_mode);
I_conv_expected = conv_op.apply(I_in);

% Self-contained boundary replication helper (P=1 padding for 3x3 kernel)
I_pad = padReplicate(I_in, 1);

% Perform manual verification of the shift behavior
% Correlation: Output at (r, c) = Input at (r, c+1) (shift left)
% Convolution: Output at (r, c) = Input at (r, c-1) (shift right)
I_xcorr_manual = zeros(H, W, 3);
I_conv_manual = zeros(H, W, 3);

K_flipped = rot90(K, 2);

for r = 1:H
    r_pad = r + 1;
    for c = 1:W
        c_pad = c + 1;
        % Extract 3x3 neighborhood patch
        patch = I_pad(r_pad-1 : r_pad+1, c_pad-1 : c_pad+1, :);
        
        % Multiply by kernels channel-by-channel
        for ch = 1:3
            I_xcorr_manual(r, c, ch) = sum(sum(patch(:,:,ch) .* K));
            I_conv_manual(r, c, ch) = sum(sum(patch(:,:,ch) .* K_flipped));
        end
    end
end

% Verify mathematical correctness
diff_xcorr = max(abs(I_xcorr_manual(:) - I_xcorr_expected(:)));
diff_conv = max(abs(I_conv_manual(:) - I_conv_expected(:)));

fprintf('Verification - Cross-Correlation Difference: %e\n', diff_xcorr);
fprintf('Verification - True Convolution Difference: %e\n', diff_conv);

if diff_xcorr > 1e-12 || diff_conv > 1e-12
    error('Mathematical discrepancy found between manual computation and library strategies.');
else
    fprintf('SUCCESS: Manual computation matches library strategies perfectly.\n');
end

%% 5. Initialize the Premium Figure Layout
fig = figure('Name', 'Linear Filtering: Correlation vs. Convolution', ...
             'Color', [0.03, 0.04, 0.05], ... % Extra dark premium background
             'Units', 'pixels', ...
             'Position', [100, 100, 1280, 720]);

% Left Panel: Large Input Grid
ax_in = axes('Parent', fig, ...
             'Units', 'normalized', ...
             'Position', [0.05, 0.08, 0.45, 0.82], ...
             'Color', bg_color, ...
             'XColor', [0.3, 0.3, 0.3], ...
             'YColor', [0.3, 0.3, 0.3]);
hold(ax_in, 'on');
h_in_img = image('CData', I_in, 'Parent', ax_in);
axis(ax_in, 'image');
xlim(ax_in, [0.5, W + 0.5]);
ylim(ax_in, [0.5, H + 0.5]);
set(ax_in, 'YDir', 'reverse', 'XTick', 1:W, 'YTick', 1:H, 'FontSize', 8, ...
           'GridColor', [0.2, 0.2, 0.25], 'GridAlpha', 0.6);
grid(ax_in, 'on');
title(ax_in, 'Input Image (16x16 Grid)', 'Color', [0.9, 0.9, 0.9], 'FontSize', 14, 'FontWeight', 'bold');

% Active neighborhood 3x3 box
h_box = rectangle('Parent', ax_in, ...
                  'Position', [0.5, 0.5, 3, 3], ...
                  'EdgeColor', [0.3, 0.7, 1.0], ...
                  'LineWidth', 2.5, ...
                  'Curvature', [0.05, 0.05], ...
                  'Visible', 'off');

% Active center pixel dot
h_dot = rectangle('Parent', ax_in, ...
                  'Position', [1.5, 1.5, 1, 1], ...
                  'EdgeColor', [1.0, 0.3, 0.3], ...
                  'LineWidth', 2.0, ...
                  'Visible', 'off');

% Top-Right Panel: Kernel Coefficients & Thumbnail
ax_k = axes('Parent', fig, ...
            'Units', 'normalized', ...
            'Position', [0.56, 0.60, 0.38, 0.30], ...
            'Color', [0.03, 0.04, 0.05], ...
            'XColor', 'none', ...
            'YColor', 'none');
hold(ax_k, 'on');
axis(ax_k, 'equal');
xlim(ax_k, [-4.0, 5.0]);
ylim(ax_k, [0.5, 4.5]);
set(ax_k, 'YDir', 'reverse');

% Thumbnail of image
h_thumb = image('CData', imresize(I_in, [32, 32]), ...
                'XData', [-3.8, -1.8], ...
                'YData', [1.5, 3.5], ...
                'Parent', ax_k);

% Operator text
h_op_text = text(-0.8, 2.5, '\otimes', ...
                 'Color', [0.9, 0.9, 0.9], ...
                 'FontSize', 36, ...
                 'HorizontalAlignment', 'center', ...
                 'Parent', ax_k);

% Draw the 3x3 kernel grid blocks
h_k_boxes = cell(3, 3);
h_k_texts = cell(3, 3);
for r_k = 1:3
    for c_k = 1:3
        pos = [0.8 + c_k, r_k + 0.8, 0.9, 0.9];
        h_k_boxes{r_k, c_k} = rectangle('Parent', ax_k, ...
                                        'Position', pos, ...
                                        'EdgeColor', [0.2, 0.6, 0.9], ...
                                        'LineWidth', 1.5, ...
                                        'Curvature', [0.1, 0.1]);
        h_k_texts{r_k, c_k} = text(0.8 + c_k + 0.45, r_k + 0.8 + 0.45, '0.0', ...
                                   'Color', [1.0, 1.0, 1.0], ...
                                   'FontSize', 12, ...
                                   'FontWeight', 'bold', ...
                                   'HorizontalAlignment', 'center', ...
                                   'Parent', ax_k);
    end
end
title(ax_k, 'Mathematical Operator & Kernel Coefficients', 'Color', [0.9, 0.9, 0.9], 'FontSize', 13, 'FontWeight', 'bold');

% Bottom-Right Panel: Large Output Grid
ax_out = axes('Parent', fig, ...
              'Units', 'normalized', ...
              'Position', [0.56, 0.08, 0.38, 0.45], ...
              'Color', bg_color, ...
              'XColor', [0.3, 0.3, 0.3], ...
              'YColor', [0.3, 0.3, 0.3]);
hold(ax_out, 'on');
I_out_curr = ones(H, W, 3) * 0.05; % Start with a blank dark grey grid
h_out_img = image('CData', I_out_curr, 'Parent', ax_out);
axis(ax_out, 'image');
xlim(ax_out, [0.5, W + 0.5]);
ylim(ax_out, [0.5, H + 0.5]);
set(ax_out, 'YDir', 'reverse', 'XTick', 1:W, 'YTick', 1:H, 'FontSize', 8, ...
            'GridColor', [0.2, 0.2, 0.25], 'GridAlpha', 0.6);
grid(ax_out, 'on');
title_out = title(ax_out, 'Output Image (Empty)', 'Color', [0.9, 0.9, 0.9], 'FontSize', 13, 'FontWeight', 'bold');

% Active output pixel marker
h_out_dot = rectangle('Parent', ax_out, ...
                      'Position', [0.5, 0.5, 1, 1], ...
                      'EdgeColor', [1.0, 0.3, 0.3], ...
                      'LineWidth', 2.0, ...
                      'Visible', 'off');

% Overall title text annotation
h_fig_title = annotation(fig, 'textbox', [0.05, 0.92, 0.9, 0.06], ...
                         'String', 'LINEAR FILTERING DEMONSTRATION', ...
                         'Color', [1.0, 1.0, 1.0], ...
                         'FontSize', 18, ...
                         'FontWeight', 'bold', ...
                         'HorizontalAlignment', 'center', ...
                         'LineStyle', 'none');

drawnow;

%% 6. Video Generation Settings
fps = 15;
writer = VideoWriter(output_video_path, 'MPEG-4');
writer.FrameRate = fps;
writer.Quality = 95;
open(writer);

%% 7. Animation Sequences

% --- PART 1: Cross-Correlation ---
fprintf('Animating Part 1: Cross-Correlation...\n');
set(h_fig_title, 'String', 'LINEAR FILTERING: CROSS-CORRELATION (DIRECT MATCHING)');
set(h_op_text, 'String', '\otimes');
set(title_out, 'String', 'Output Image: Cross-Correlation (Shift LEFT)');

% Set kernel coefficients displayed in the top-right
for r_k = 1:3
    for c_k = 1:3
        set(h_k_texts{r_k, c_k}, 'String', sprintf('%.1f', K(r_k, c_k)));
        if K(r_k, c_k) > 0.0
            set(h_k_boxes{r_k, c_k}, 'FaceColor', [0.1, 0.4, 0.7], 'EdgeColor', [0.3, 0.7, 1.0]);
        else
            set(h_k_boxes{r_k, c_k}, 'FaceColor', 'none', 'EdgeColor', [0.2, 0.6, 0.9]);
        end
    end
end

% Hold on title card (30 frames = 2 seconds)
for f = 1:30
    writeVideo(writer, getframe(fig));
end

set(h_box, 'Visible', 'on');
set(h_dot, 'Visible', 'on');
set(h_out_dot, 'Visible', 'on');

% Reset output grid to blank
I_out_curr = ones(H, W, 3) * 0.05;
set(h_out_img, 'CData', I_out_curr);

% Slide across pixels
for r = 1:H
    for c = 1:W
        % Update sliding window visual coordinates (3x3 box)
        set(h_box, 'Position', [c - 1.5, r - 1.5, 3, 3]);
        set(h_dot, 'Position', [c - 0.5, r - 0.5, 1, 1]);
        set(h_out_dot, 'Position', [c - 0.5, r - 0.5, 1, 1]);
        
        % Fill in the computed pixel value in the current output frame
        I_out_curr(r, c, :) = I_xcorr_expected(r, c, :);
        set(h_out_img, 'CData', I_out_curr);
        
        drawnow;
        writeVideo(writer, getframe(fig));
    end
end

% Hold final cross-correlation image (30 frames)
set(h_box, 'Visible', 'off');
set(h_dot, 'Visible', 'off');
set(h_out_dot, 'Visible', 'off');
drawnow;
for f = 1:30
    writeVideo(writer, getframe(fig));
end


% --- TRANSITION CARD ---
fprintf('Animating Transition...\n');
set(h_fig_title, 'String', 'TRANSITION: PREPARING TRUE MATHEMATICAL CONVOLUTION');
set(title_out, 'String', 'True Convolution: Flipped Kernel Matching');

% Flash a brief transition overlay or empty screen
I_out_curr = ones(H, W, 3) * 0.05;
set(h_out_img, 'CData', I_out_curr);

% Show rotation visual in the top-right kernel box
set(h_op_text, 'String', '*');

for f = 1:30
    if f < 15
        set(h_fig_title, 'String', 'ROTATING KERNEL BY 180 DEGREES...');
        % Display original kernel
        for r_k = 1:3
            for c_k = 1:3
                set(h_k_texts{r_k, c_k}, 'String', sprintf('%.1f', K(r_k, c_k)));
                if K(r_k, c_k) > 0.0
                    set(h_k_boxes{r_k, c_k}, 'FaceColor', [0.7, 0.2, 0.2], 'EdgeColor', [1.0, 0.4, 0.4]);
                else
                    set(h_k_boxes{r_k, c_k}, 'FaceColor', 'none', 'EdgeColor', [0.2, 0.6, 0.9]);
                end
            end
        end
    else
        set(h_fig_title, 'String', 'KERNEL ROTATION COMPLETE: READY TO CONVOLVE');
        % Display flipped kernel coefficients
        for r_k = 1:3
            for c_k = 1:3
                set(h_k_texts{r_k, c_k}, 'String', sprintf('%.1f', K_flipped(r_k, c_k)));
                if K_flipped(r_k, c_k) > 0.0
                    set(h_k_boxes{r_k, c_k}, 'FaceColor', [0.2, 0.6, 0.3], 'EdgeColor', [0.4, 0.8, 0.5]);
                else
                    set(h_k_boxes{r_k, c_k}, 'FaceColor', 'none', 'EdgeColor', [0.2, 0.6, 0.9]);
                end
            end
        end
    end
    drawnow;
    writeVideo(writer, getframe(fig));
end


% --- PART 2: True Convolution ---
fprintf('Animating Part 2: True Convolution...\n');
set(h_fig_title, 'String', 'LINEAR FILTERING: TRUE MATHEMATICAL CONVOLUTION (FLIPPED MATCHING)');
set(title_out, 'String', 'Output Image: True Convolution (Shift RIGHT)');

set(h_box, 'Visible', 'on');
set(h_dot, 'Visible', 'on');
set(h_out_dot, 'Visible', 'on');

% Slide across pixels
for r = 1:H
    for c = 1:W
        set(h_box, 'Position', [c - 1.5, r - 1.5, 3, 3]);
        set(h_dot, 'Position', [c - 0.5, r - 0.5, 1, 1]);
        set(h_out_dot, 'Position', [c - 0.5, r - 0.5, 1, 1]);
        
        % Fill in the computed pixel value in the current output frame
        I_out_curr(r, c, :) = I_conv_expected(r, c, :);
        set(h_out_img, 'CData', I_out_curr);
        
        drawnow;
        writeVideo(writer, getframe(fig));
    end
end

% Hold final convolution image (30 frames)
set(h_box, 'Visible', 'off');
set(h_dot, 'Visible', 'off');
set(h_out_dot, 'Visible', 'off');
drawnow;
for f = 1:30
    writeVideo(writer, getframe(fig));
end

% Save the final frame snapshot
saveas(fig, output_snapshot_path);

close(writer);
close(fig);

fprintf('Video generated successfully: %s\n', output_video_path);
fprintf('Snapshot generated successfully: %s\n', output_snapshot_path);

%% 8. Private Helper Function for Replicate Padding
function I_pad = padReplicate(I, P)
    [H_in, W_in, C_in] = size(I);
    I_pad = zeros(H_in + 2*P, W_in + 2*P, C_in);
    
    % Copy main body
    I_pad(P+1:H_in+P, P+1:W_in+P, :) = I;
    
    % Replicate edges and corners channel-by-channel
    for ch = 1:C_in
        % Replicate top and bottom rows
        for p = 1:P
            I_pad(p, P+1:W_in+P, ch) = I(1, :, ch);
            I_pad(H_in+P+p, P+1:W_in+P, ch) = I(H_in, :, ch);
        end
        % Replicate left and right columns
        for p = 1:P
            I_pad(P+1:H_in+P, p, ch) = I(:, 1, ch);
            I_pad(P+1:H_in+P, W_in+P+p, ch) = I(:, W_in, ch);
        end
        % Replicate corners
        for p = 1:P
            for q = 1:P
                I_pad(p, q, ch) = I(1, 1, ch);
                I_pad(p, W_in+P+q, ch) = I(1, W_in, ch);
                I_pad(H_in+P+p, q, ch) = I(H_in, 1, ch);
                I_pad(H_in+P+p, W_in+P+q, ch) = I(H_in, W_in, ch);
            end
        end
    end
end

