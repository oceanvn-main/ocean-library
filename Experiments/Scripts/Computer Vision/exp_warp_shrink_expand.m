% EXP_WARP_SHRINK_EXPAND
%   Demonstrates the Scatter (Forward Mapping) algorithm on two different
%   transformations: one pure shrink (minification) and one pure expand (magnification).
%
%   This explicitly answers the question of what happens under different A matrices.

clear; clc; close all;
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);


% 1. Load Image
img_path = fullfile(root_dir, 'Data', 'images', 'secrets', 'z7970955627476_16ad93554963c82b9572da4c862e052c.jpg');
I_raw = imread(img_path);
I_image = im2double(imresize(I_raw, [200, 200])); % Smaller for fast exact loops
[H_src, W_src, C_src] = size(I_image);

% 2. Define Transformations
% Configuration 1: Pure Shrink
A_shrink = [0.4 0 0; 0 0.4 0; 0 0 1]; 

% Configuration 2: Pure Expand
A_expand = [2.5 0 0; 0 2.5 0; 0 0 1];

transforms = {A_shrink, A_expand};
titles = {'Pure Shrink (A = 0.4x)', 'Pure Expand (A = 2.5x)'};

viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Scatter Warp: Shrink vs Expand', 'Position', [100, 100, 1200, 500]);
set(fig, 'Color', viz.Colors.Background);

for i = 1:2
    A = transforms{i};
    
    % Determine bounds
    corners = [1 1 1; W_src 1 1; 1 H_src 1; W_src H_src 1]';
    mapped_corners = A * corners;
    min_x = floor(min(mapped_corners(1, :))); max_x = ceil(max(mapped_corners(1, :)));
    min_y = floor(min(mapped_corners(2, :))); max_y = ceil(max(mapped_corners(2, :)));
    W_dst = max_x - min_x + 1;
    H_dst = max_y - min_y + 1;
    offset_x = 1 - min_x;
    offset_y = 1 - min_y;

    % Scatter Loop
    I_scatter = zeros(H_dst, W_dst, C_src);
    hit_counter = zeros(H_dst, W_dst);
    
    for y = 1:H_src
        for x = 1:W_src
            p_dst = A * [x; y; 1];
            x_dst = round(p_dst(1) + offset_x);
            y_dst = round(p_dst(2) + offset_y);
            
            if x_dst >= 1 && x_dst <= W_dst && y_dst >= 1 && y_dst <= H_dst
                I_scatter(y_dst, x_dst, :) = I_image(y, x, :);
                hit_counter(y_dst, x_dst) = hit_counter(y_dst, x_dst) + 1;
            end
        end
    end
    
    % Analysis
    valid_mask = false(H_dst, W_dst);
    % Simulate inverse pass just to get valid mask
    A_inv = inv(A);
    for y_dst = 1:H_dst
        for x_dst = 1:W_dst
            p_src = A_inv * [x_dst - offset_x; y_dst - offset_y; 1];
            if p_src(1) >= 1 && p_src(1) <= W_src && p_src(2) >= 1 && p_src(2) <= H_src
                valid_mask(y_dst, x_dst) = true;
            end
        end
    end
    
    valid_hits = hit_counter(valid_mask);
    hole_frac = sum(valid_hits == 0) / numel(valid_hits);
    exact_frac = sum(valid_hits == 1) / numel(valid_hits);
    overlap_frac = sum(valid_hits >= 2) / numel(valid_hits);
    
    % Plot Image
    ax1 = subplot(2, 2, i);
    imshow(I_scatter);
    title(titles{i}, 'Color', viz.Colors.Text);
    viz.styleAxes(ax1);
    
    % Plot Histogram
    ax2 = subplot(2, 2, i + 2);
    bar(ax2, [hole_frac, exact_frac, overlap_frac] * 100, 'FaceColor', viz.Colors.Primary);
    set(ax2, 'XTickLabel', {'Holes (0)', 'Exact (1)', 'Overlaps (2+)'});
    ylabel('Pixels (%)');
    title(sprintf('Holes: %.1f%% | Overlaps: %.1f%%', hole_frac*100, overlap_frac*100));
    viz.styleAxes(ax2);
end
