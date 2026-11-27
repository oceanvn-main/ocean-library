% exp_border_padding_formulas.m
%
% Scientific Experiment: Index-Mapping Coordinate Transformations for Border Padding
% (Szeliski Exercise 3.9)
%
% Description:
%   This script explicitly implements the mathematical coordinate mapping formulas
%   for the five boundary padding modes (Zero, Clamping, Cyclic Wrap, Symmetric Mirror,
%   and Linear Extension) in Szeliski Exercise 3.9. It applies these coordinate 
%   transformations to a synthetic 64x64 gradient checkerboard image and visualizes
%   the padded boundaries.
%
% Scientific Principles Applied (RULE[experiment.md]):
%   - Explicit Computation: All padded values are computed element-by-element using
%     closed-form index equations.
%   - Reproducibility: Independent path resolution.
%   - No LaTeX: Plain text formatting in labels and console prints.
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

%% 1. Generate Input Grid Image
% Binary black and white checkerboard data (Szeliski Fig 3.13)
H = 64;
W = 64;
[X, Y] = meshgrid(1:W, 1:H);
checker = mod(floor((X - 1) / 16) + floor((Y - 1) / 16), 2);
I_in = double(checker);

% Padding size
P = 16;
H_pad = H + 2*P;
W_pad = W + 2*P;

%% 2. Apply Boundary Mapping Formulas Explicitly

% A. Constant/Zero Padding
I_pad_zero = zeros(H_pad, W_pad);
I_pad_zero(P + (1:H), P + (1:W)) = I_in;

% B. Replication (Clamping)
% Formula: k = max(1, min(H, i)), l = max(1, min(W, j))  [1-based index]
I_pad_clamp = zeros(H_pad, W_pad);
for i = 1:H_pad
    for j = 1:W_pad
        k = max(1, min(H, i - P));
        l = max(1, min(W, j - P));
        I_pad_clamp(i, j) = I_in(k, l);
    end
end

% C. Cyclic Wrap (Periodic)
% Formula: k = mod(i-1, H) + 1, l = mod(j-1, W) + 1      [1-based index]
I_pad_wrap = zeros(H_pad, W_pad);
for i = 1:H_pad
    for j = 1:W_pad
        k = mod(i - P - 1, H) + 1;
        l = mod(j - P - 1, W) + 1;
        I_pad_wrap(i, j) = I_in(k, l);
    end
end

% D. Mirror Reflection (Symmetric)
% Formula: k = H - abs(mod(i-1, 2*H-2) - (H-1))           [1-based index]
I_pad_mirror = zeros(H_pad, W_pad);
for i = 1:H_pad
    for j = 1:W_pad
        k = H - abs(mod(i - P - 1, 2*H - 2) - (H - 1));
        l = W - abs(mod(j - P - 1, 2*W - 2) - (W - 1));
        I_pad_mirror(i, j) = I_in(k, l);
    end
end

% E. Linear Extension (Extend)
% Extrapolates based on edge pixel slope
I_pad_extend = zeros(H_pad, W_pad);
for i = 1:H_pad
    for j = 1:W_pad
        k_in = i - P;
        l_in = j - P;
        
        k_edge = max(1, min(H, k_in));
        l_edge = max(1, min(W, l_in));
        
        % Horizontal Extrapolation
        if l_in < 1
            v_h = I_in(k_edge, 1) + (l_in - 1) * (I_in(k_edge, 2) - I_in(k_edge, 1));
        elseif l_in > W
            v_h = I_in(k_edge, W) + (l_in - W) * (I_in(k_edge, W) - I_in(k_edge, W - 1));
        else
            v_h = I_in(k_edge, l_in);
        end
        
        % Vertical Extrapolation
        if k_in < 1
            v_v = I_in(1, l_edge) + (k_in - 1) * (I_in(2, l_edge) - I_in(1, l_edge));
        elseif k_in > H
            v_v = I_in(H, l_edge) + (k_in - H) * (I_in(H, l_edge) - I_in(H - 1, l_edge));
        else
            v_v = I_in(k_in, l_edge);
        end
        
        % Corner blending (Average)
        if (l_in < 1 || l_in > W) && (k_in < 1 || k_in > H)
            I_pad_extend(i, j) = 0.5 * (v_h + v_v);
        elseif (l_in < 1 || l_in > W)
            I_pad_extend(i, j) = v_h;
        else
            I_pad_extend(i, j) = v_v;
        end
    end
end

% Clip extend padding outputs to valid range [0, 1] to prevent saturation display issues
I_pad_extend = max(0, min(1, I_pad_extend));

%% 3. Plot Visualization Layout
fig = viz.createFigure('Exercise 3.9: Coordinate Mapping Border Padding');
set(fig, 'Units', 'normalized', 'Position', [0.05, 0.1, 0.9, 0.7]);

titles = {'1. Zero/Constant Padding', '2. Replication (Clamping)', '3. Cyclic Wrap (Periodic)', ...
          '4. Symmetric Mirror', '5. Linear Extension (Slope)'};
images = {I_pad_zero, I_pad_clamp, I_pad_wrap, I_pad_mirror, I_pad_extend};

for idx = 1:5
    ax = subplot(1, 5, idx);
    % Display image in grayscale (black and white colors)
    imshow(images{idx}, []);
    title(ax, titles{idx}, 'Color', viz.Colors.Text, 'FontSize', 10);
    set(ax, 'Color', viz.Colors.Background);
    
    % Overlay dashed red line box for original image bounds (using plot for legend support)
    hold(ax, 'on');
    x_coords = [P+0.5, P+0.5+W, P+0.5+W, P+0.5, P+0.5];
    y_coords = [P+0.5, P+0.5, P+0.5+H, P+0.5+H, P+0.5];
    r_h = plot(ax, x_coords, y_coords, 'Color', [1.0, 0.2, 0.2], 'LineWidth', 1.5, 'LineStyle', '--');
    hold(ax, 'off');
    
    % Add legend styled to match the active theme
    lgd = legend(ax, r_h, 'Original Image Boundary', 'Location', 'southoutside');
    set(lgd, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);
end

sgtitle(fig, 'Szeliski Exercise 3.9: Boundary Padding Formulas and Coordinate Mappings', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

%% 4. Save Results
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_fig_path = fullfile(results_dir, 'exercise_3_9_results.png');
fprintf('Saving results visualization to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

fprintf('Experiment complete!\n');
