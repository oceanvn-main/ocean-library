% exp_collage_editor.m
% =========================================================================
% Experiment: 2D Image Collage Compositing (Szeliski Ex 3.26)
% =========================================================================
%
% Programmatically demonstrates the multi-layer homography compositing
% pipeline to construct an image collage.
%
% Steps:
%   1. Load three images: ocean_img.jpg, elon_musk.jpg, and synthetic checkerboard.
%   2. Extract a defined crop region from each.
%   3. Apply different 2D coordinate transformations to position each layer
%      onto an 800x600 canvas:
%        - Layer 1: Panned/translated ocean landscape background.
%        - Layer 2: Similarity-warped face overlay (rotated & scaled).
%        - Layer 3: Perspective-warped checkerboard (projective homography).
%   4. Save the final composite image as proof-of-work.
%
% ASSUMPTIONS:
%   - Canvas resolution is exactly 600 x 800 pixels.
%   - Pixel coordinates are 1-based.
%   - Inverse mapping homography warping uses bilinear interpolation.
%
% REPRODUCIBILITY: All random components are fixed. All outputs are derived
%   directly from inputs through explicit inverse mapping geometry.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-16
% =========================================================================

clear; close all; clc;
addpath(genpath('c:\Ocean Library\Matlab code practice'));
cd('c:\Ocean Library\Matlab code practice');

fprintf('=== Programmatic 2D Collage Compositing Experiment ===\n');

%% =========================================================================
% DATA LOADING & CROPPING
% =========================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
img_dir      = fullfile(project_root, 'Data', 'images');

% Load images
I1_raw = imread(fullfile(img_dir, 'ocean_img.jpg'));
I2_raw = imread(fullfile(img_dir, 'elon_musk.jpg'));

% Create a synthetic checkerboard for layer 3
[X_chk, Y_chk] = meshgrid(1:200, 1:200);
I3_raw = double(mod(floor(X_chk/25) + floor(Y_chk/25), 2));
I3_raw = repmat(I3_raw, 1, 1, 3); % Convert to 3 channels

% Programmatic crop bounds: [y_min, y_max, x_min, x_max]
crop1 = [50, 250,  50, 350];  % ocean portion (201 x 301)
crop2 = [40, 220,  60, 200];  % face portion (181 x 141)
crop3 = [1,  200,  1,  200];  % full checkerboard (200 x 200)

% Extract crops (normalized double)
L1_img = double(I1_raw(crop1(1):crop1(2), crop1(3):crop1(4), :)) / 255;
L2_img = double(I2_raw(crop2(1):crop2(2), crop2(3):crop2(4), :)) / 255;
L3_img = I3_raw;

% Print dimensions
fprintf('Layer 1 (Ocean Landscape) size: %d x %d\n', size(L1_img, 1), size(L1_img, 2));
fprintf('Layer 2 (Face Overlay) size:    %d x %d\n', size(L2_img, 1), size(L2_img, 2));
fprintf('Layer 3 (Checkerboard) size:    %d x %d\n', size(L3_img, 1), size(L3_img, 2));

%% =========================================================================
% LAYER SETUP & HOMOGRAPHY DEFINITION
% =========================================================================
% We define base corners for each crop, and the target coordinates on the
% 800 x 600 canvas.

% --- Layer 1: Ocean Landscape (Translation + Scale) ---
h1 = size(L1_img, 1); w1 = size(L1_img, 2);
L1_base = [0,0; w1,0; w1,h1; 0,h1];
% Scale by 1.5 and position in top-left
L1_dst  = [10, 10; 10+w1*1.5, 10; 10+w1*1.5, 10+h1*1.5; 10, 10+h1*1.5];

% --- Layer 2: Face Overlay (Rotated & Scaled via Similarity) ---
h2 = size(L2_img, 1); w2 = size(L2_img, 2);
L2_base = [0,0; w2,0; w2,h2; 0,h2];
% Displace corners to represent a rotated, scaled pose
theta = deg2rad(15);
scale = 1.2;
tx = 450; ty = 120;
cx2 = w2/2; cy2 = h2/2;
R_mat = scale * [cos(theta), -sin(theta); sin(theta), cos(theta)];
L2_dst = zeros(4, 2);
for k = 1:4
    pt = L2_base(k, :) - [cx2, cy2];
    L2_dst(k, :) = [cx2, cy2] + (R_mat * pt')' + [tx, ty];
end

% --- Layer 3: Checkerboard (Projective Homography) ---
h3 = size(L3_img, 1); w3 = size(L3_img, 2);
L3_base = [0,0; w3,0; w3,h3; 0,h3];
% Non-parallel target boundaries for perspective depth effect
L3_dst  = [200, 350;  450, 380;  400, 560;  150, 520];

% Construct layers cell array matching Sandbox design
layers = cell(1, 3);

% Fit Homographies programmatically
layers{1}.img = L1_img;
layers{1}.base_corners = L1_base;
layers{1}.H = fitHomography_(L1_base, L1_dst);
layers{1}.name = 'Ocean Background';

layers{2}.img = L2_img;
layers{2}.base_corners = L2_base;
layers{2}.H = fitHomography_(L2_base, L2_dst);
layers{2}.name = 'Face Overlay';

layers{3}.img = L3_img;
layers{3}.base_corners = L3_base;
layers{3}.H = fitHomography_(L3_base, L3_dst);
layers{3}.name = 'Checkerboard Floor';

%% =========================================================================
% COMPOSITING & WARPING
% =========================================================================
fprintf('\nCompositing layers onto canvas...\n');
I_canvas = repmat(reshape([0.10, 0.12, 0.16], 1, 1, 3), 600, 800); % Dark background

for i = 1:numel(layers)
    fprintf('  Rendering layer: %s\n', layers{i}.name);
    I_canvas = warpLayer_(I_canvas, layers{i});
end

%% =========================================================================
% VISUALIZATION & OUTPUT
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'Ex 3.26 collage composite result', ...
    'Position', [100, 100, 900, 680], ...
    'Color', [0.11, 0.11, 0.13]);

ax = axes('Parent', fig, 'Position', [0.05, 0.08, 0.90, 0.82]);
imshow(I_canvas, 'Parent', ax);
title(ax, '2D Image Collage Programmatic Composite (Translation, Similarity, Perspective layers)', ...
    'Color', [0.90, 0.90, 0.90], 'FontSize', 12, 'FontWeight', 'bold');
axis(ax, 'image'); axis(ax, 'off');

% Save result
results_dir = fullfile(project_root, 'Results', 'Computer Vision');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
out_path = fullfile(results_dir, 'ex326_collage_demo.png');
exportgraphics(fig, out_path, 'Resolution', 150);
fprintf('\nCollage composite image saved: %s\n', out_path);

%% =========================================================================
% HELPERS
% =========================================================================
function H = fitHomography_(src, dst)
    % Standard Direct Linear Transform (DLT) solver for 3x3 homography
    A = zeros(8, 8);
    B = zeros(8, 1);
    for i = 1:4
        x = src(i,1); y = src(i,2);
        u = dst(i,1); v = dst(i,2);
        A(2*i-1, :) = [x, y, 1, 0, 0, 0, -u*x, -u*y];
        A(2*i, :)   = [0, 0, 0, x, y, 1, -v*x, -v*y];
        B(2*i-1)    = u;
        B(2*i)      = v;
    end
    h = A \ B;
    H = eye(3);
    H(1, 1) = h(1); H(1, 2) = h(2); H(1, 3) = h(3);
    H(2, 1) = h(4); H(2, 2) = h(5); H(2, 3) = h(6);
    H(3, 1) = h(7); H(3, 2) = h(8); H(3, 3) = 1.0;
end

function I_canvas = warpLayer_(I_canvas, layer)
    I_src = layer.img;
    [H_src, W_src, C] = size(I_src);
    H_mat = layer.H;
    
    % Get destination bounding box
    pts_hom = [layer.base_corners, ones(4, 1)]';
    warped = H_mat * pts_hom;
    w = warped(3, :);
    w(abs(w) < 1e-12) = 1e-12;
    cx = warped(1, :) ./ w;
    cy = warped(2, :) ./ w;
    
    min_x = max(1, floor(min(cx)));
    max_x = min(800, ceil(max(cx)));
    min_y = max(1, floor(min(cy)));
    max_y = min(600, ceil(max(cy)));
    
    if min_x > max_x || min_y > max_y, return; end
    
    % Create coordinates grid
    [X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);
    N_pts = numel(X_grid);
    pts_dst = [X_grid(:), Y_grid(:), ones(N_pts, 1)]';
    
    % Backward transform
    H_inv = inv(H_mat);
    pts_src = H_inv * pts_dst;
    ws = pts_src(3, :);
    ws(abs(ws) < 1e-12) = 1e-12;
    src_x = pts_src(1, :) ./ ws;
    src_y = pts_src(2, :) ./ ws;
    
    % Sample and overlay
    inside = (src_x >= 0 & src_x <= W_src-1 & src_y >= 0 & src_y <= H_src-1);
    if ~any(inside), return; end
    
    src_x_1 = src_x(inside) + 1;
    src_y_1 = src_y(inside) + 1;
    x_in = X_grid(inside);
    y_in = Y_grid(inside);
    
    x0 = floor(src_x_1);
    y0 = floor(src_y_1);
    x1 = x0 + 1;
    y1 = y0 + 1;
    
    x0_c = max(1, min(W_src, x0));
    x1_c = max(1, min(W_src, x1));
    y0_c = max(1, min(H_src, y0));
    y1_c = max(1, min(H_src, y1));
    
    dx = src_x_1 - x0;
    dy = src_y_1 - y0;
    
    w00 = (1 - dx) .* (1 - dy);
    w01 = dx .* (1 - dy);
    w10 = (1 - dx) .* dy;
    w11 = dx .* dy;
    
    idx00 = y0_c + (x0_c - 1) * H_src;
    idx01 = y0_c + (x1_c - 1) * H_src;
    idx10 = y1_c + (x0_c - 1) * H_src;
    idx11 = y1_c + (x1_c - 1) * H_src;
    
    canvas_idx = y_in + (x_in - 1) * 600;
    
    for c = 1:C
        Ic = I_src(:, :, c);
        val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
        I_canvas(canvas_idx + (c - 1) * 600 * 800) = val;
    end
end
