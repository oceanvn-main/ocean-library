% EXP_FORWARD_MAPPING_MAGNIFICATION
%   Experiment B: Cracks and Holes Under Magnification
%
%   This script tests the textbook claim regarding the appearance of cracks 
%   and holes when magnifying an image using forward mapping (Scatter).
%   It compares empirical hole fractions against the analytical prediction
%   of (1 - 1/s^2).
clear; clc; close all;

disp('Initializing Cracks and Holes Experiment...');
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

% === 1. Setup Datasets ===
% A) Real Image (For measuring Blur/MSE in Experiment C)
img_path = fullfile(root_dir, 'Data', 'images', 'testimage.jpeg');
I_raw = imread(img_path);
I_real = im2double(imresize(I_raw, 500 / max(size(I_raw, 1), size(I_raw, 2))));
[H_src, W_src, ~] = size(I_real);

% B) Solid Color Image (For unambiguously exposing holes in Experiment B)
% Using a flat bright orange to strongly contrast with the black holes
I_solid = cat(3, ones(H_src, W_src)*1.0, ones(H_src, W_src)*0.4, ones(H_src, W_src)*0.2);

% === 2. Uniform Magnification Sweep ===
disp('Running Uniform Magnification Sweep...');
scales = linspace(1.0, 2.0, 31);
hole_fractions = zeros(size(scales));

for i = 1:length(scales)
    s = scales(i);
    A = [s 0 0; 0 s 0; 0 0 1];
    
    corners = A * [1 1 1; W_src 1 1; 1 H_src 1; W_src H_src 1]';
    min_x = floor(min(corners(1,:))); max_x = ceil(max(corners(1,:)));
    min_y = floor(min(corners(2,:))); max_y = ceil(max(corners(2,:)));
    W_dst = max_x - min_x + 1;
    H_dst = max_y - min_y + 1;
    offset_x = 1 - min_x;
    offset_y = 1 - min_y;
    
    hit_map = zeros(H_dst, W_dst);
    
    for y = 1:H_src
        for x = 1:W_src
            p = A * [x; y; 1];
            xr = round(p(1) + offset_x);
            yr = round(p(2) + offset_y);
            if xr >= 1 && xr <= W_dst && yr >= 1 && yr <= H_dst
                hit_map(yr, xr) = hit_map(yr, xr) + 1;
            end
        end
    end
    
    valid_mask = false(H_dst, W_dst);
    A_inv = inv(A);
    for yd = 1:H_dst
        for xd = 1:W_dst
            ps = A_inv * [xd - offset_x; yd - offset_y; 1];
            if ps(1) >= 1 && ps(1) <= W_src && ps(2) >= 1 && ps(2) <= H_src
                valid_mask(yd, xd) = true;
            end
        end
    end
    
    valid_hits = hit_map(valid_mask);
    hole_fractions(i) = sum(valid_hits == 0) / numel(valid_hits);
end

% Analytical prediction: Holes ~= 1 - (1 / s^2)
expected_holes = 1 - (1 ./ (scales.^2));

% === 3. Visual Snapshots (Holes rendered in Black) ===
disp('Generating Visual Snapshots...');
s_uni = 3.0;
hole_color = [0, 0, 0]; % Standard graphics uninitialized buffer color (Black)
[I_uni_solid, ~, ~] = run_forward_map(I_solid, s_uni, s_uni, hole_color);

s_x = 3.0; s_y = 1.5;
I_non_solid = run_forward_map(I_solid, s_x, s_y, hole_color);

% === 4. Visualization ===
disp('Rendering Dashboard...');
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'Cracks and Holes Analysis', 'Position', [100, 100, 1200, 800]);
set(fig, 'Color', viz.Colors.Background);

% Panel A: Theoretical vs Empirical
ax1 = subplot(2, 2, 1);
plot(ax1, scales, expected_holes * 100, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 2, 'DisplayName', 'Theory: 1 - 1/s^2');
hold(ax1, 'on');
plot(ax1, scales, hole_fractions * 100, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Empirical (Algorithm 3.1)');
hold(ax1, 'off');
xlabel(ax1, 'Uniform Magnification Factor (s)');
ylabel(ax1, 'Hole Fraction (%)');
title(ax1, 'Hole Probability Scaling Law');
legend(ax1, 'Location', 'southeast', 'TextColor', viz.Colors.Text);
grid(ax1, 'on');
viz.styleAxes(ax1);

% Panel B: Original Image (Solid)
ax2 = subplot(2, 2, 2);
imshow(I_solid, 'InitialMagnification', 'fit');
title(ax2, 'Original (Unscaled)', 'Color', viz.Colors.Text);
axis(ax2, 'on'); xlabel(ax2, 'X (pixels)'); ylabel(ax2, 'Y (pixels)');
viz.styleAxes(ax2);

% Panel C: Uniform Holes
ax3 = subplot(2, 2, 3);
imshow(I_uni_solid, 'InitialMagnification', 'fit');
title(ax3, 'Uniform (Sx=3.0, Sy=3.0)', 'Color', viz.Colors.Text);
axis(ax3, 'on'); xlabel(ax3, 'X (pixels)'); ylabel(ax3, 'Y (pixels)');
viz.styleAxes(ax3);

% Panel D: Non-Uniform Holes
ax4 = subplot(2, 2, 4);
imshow(I_non_solid, 'InitialMagnification', 'fit');
title(ax4, 'Non-Uniform (Sx=3.0, Sy=1.5)', 'Color', viz.Colors.Text);
axis(ax4, 'on'); xlabel(ax4, 'X (pixels)'); ylabel(ax4, 'Y (pixels)');
viz.styleAxes(ax4);

% === 5. Experiment C: Hole-Filling Side Effects ===
disp('Running Hole-Filling Experiment (Experiment C)...');
% We must forward map the Real Image here so it has texture for the MSE computation
[I_uni_real, hit_mask_real, valid_mask_real] = run_forward_map(I_real, s_uni, s_uni, hole_color);
gt_uni = inverse_map_bilinear(I_real, s_uni);

% For MSE comparison, crop to match sizes
min_h = min(size(gt_uni,1), size(I_uni_real,1));
min_w = min(size(gt_uni,2), size(I_uni_real,2));
gt_crop = gt_uni(1:min_h, 1:min_w, :);
fwd_crop = I_uni_real(1:min_h, 1:min_w, :);
hit_crop = hit_mask_real(1:min_h, 1:min_w);
val_crop = valid_mask_real(1:min_h, 1:min_w);

nn_crop = fill_nearest_neighbor(fwd_crop, hit_crop, val_crop);
avg_crop = fill_local_averaging(fwd_crop, hit_crop, val_crop);

% Calculate MSE only on valid regions
valid_pixels = val_crop(:);
mse_fwd = mean((fwd_crop(repmat(val_crop, [1,1,3])) - gt_crop(repmat(val_crop, [1,1,3]))).^2);
mse_nn  = mean((nn_crop(repmat(val_crop, [1,1,3]))  - gt_crop(repmat(val_crop, [1,1,3]))).^2);
mse_avg = mean((avg_crop(repmat(val_crop, [1,1,3])) - gt_crop(repmat(val_crop, [1,1,3]))).^2);

fig2 = figure('Name', 'Hole-Filling Quality Degradation', 'Position', [150, 100, 1000, 800]);
set(fig2, 'Color', viz.Colors.Background);

axC1 = subplot(2, 2, 1);
imshow(gt_crop); title('Ground Truth (Inverse Map)', 'Color', viz.Colors.Text);
axis(axC1, 'on'); xlabel(axC1, 'X (pixels)'); ylabel(axC1, 'Y (pixels)');
viz.styleAxes(axC1);

axC2 = subplot(2, 2, 2);
imshow(fwd_crop); title({'Forward Map (Holes)', sprintf('MSE: %.4f', mse_fwd)}, 'Color', viz.Colors.Text);
axis(axC2, 'on'); xlabel(axC2, 'X (pixels)'); ylabel(axC2, 'Y (pixels)');
viz.styleAxes(axC2);

axC3 = subplot(2, 2, 3);
imshow(nn_crop); title({'Nearest-Neighbor Fill', sprintf('MSE: %.4f', mse_nn)}, 'Color', viz.Colors.Text);
axis(axC3, 'on'); xlabel(axC3, 'X (pixels)'); ylabel(axC3, 'Y (pixels)');
viz.styleAxes(axC3);

axC4 = subplot(2, 2, 4);
imshow(avg_crop); title({'Local Averaging Fill', sprintf('MSE: %.4f', mse_avg)}, 'Color', viz.Colors.Text);
axis(axC4, 'on'); xlabel(axC4, 'X (pixels)'); ylabel(axC4, 'Y (pixels)');
viz.styleAxes(axC4);

% === 6. Experiment C (Part 2): Synthetic Aliasing Check ===
disp('Running Synthetic Aliasing Check...');
sz = 256;
[x, y] = meshgrid(-sz/2 : sz/2-1, -sz/2 : sz/2-1);
I_stripes = cos(0.02 * (x.^2 + y.^2)); % Generate a Zone Plate
I_stripes = (I_stripes + 1) / 2;       % Normalize to [0, 1]
I_stripes = cat(3, I_stripes, I_stripes, I_stripes);

gt_str = inverse_map_bilinear(I_stripes, s_uni);
[I_fwd_str, hit_str, val_str] = run_forward_map(I_stripes, s_uni, s_uni, hole_color);

min_h_s = min(size(gt_str,1), size(I_fwd_str,1));
min_w_s = min(size(gt_str,2), size(I_fwd_str,2));
gt_str = gt_str(1:min_h_s, 1:min_w_s, :);
fwd_str = I_fwd_str(1:min_h_s, 1:min_w_s, :);
hit_str = hit_str(1:min_h_s, 1:min_w_s);
val_str = val_str(1:min_h_s, 1:min_w_s);

nn_str = fill_nearest_neighbor(fwd_str, hit_str, val_str);
avg_str = fill_local_averaging(fwd_str, hit_str, val_str);

fig3 = figure('Name', 'Aliasing Assessment: Synthetic Stripes', 'Position', [200, 300, 1200, 600]);
set(fig3, 'Color', viz.Colors.Background);

plot_image_and_fft(1, gt_str, 'Ground Truth', viz);
plot_image_and_fft(2, fwd_str, 'Forward Map (Holes)', viz);
plot_image_and_fft(3, nn_str, 'Nearest-Neighbor', viz);
plot_image_and_fft(4, avg_str, 'Local Averaging', viz);

disp('Experiment Complete!');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function [I_dst, hit_mask, valid_mask] = run_forward_map(I_src, s_x, s_y, hole_color)
    [H_src, W_src, ~] = size(I_src);
    A = [s_x 0 0; 0 s_y 0; 0 0 1];
    
    corners = A * [1 1 1; W_src 1 1; 1 H_src 1; W_src H_src 1]';
    min_x = floor(min(corners(1,:))); max_x = ceil(max(corners(1,:)));
    min_y = floor(min(corners(2,:))); max_y = ceil(max(corners(2,:)));
    W_dst = max_x - min_x + 1;
    H_dst = max_y - min_y + 1;
    offset_x = 1 - min_x;
    offset_y = 1 - min_y;
    
    % Initialize destination with hole color
    I_dst = cat(3, ones(H_dst, W_dst)*hole_color(1), ...
                   ones(H_dst, W_dst)*hole_color(2), ...
                   ones(H_dst, W_dst)*hole_color(3));
                   
    hit_mask = false(H_dst, W_dst);
    
    for y = 1:H_src
        for x = 1:W_src
            p = A * [x; y; 1];
            xr = round(p(1) + offset_x);
            yr = round(p(2) + offset_y);
            if xr >= 1 && xr <= W_dst && yr >= 1 && yr <= H_dst
                hit_mask(yr, xr) = true;
                if size(I_src, 3) == 1
                    val = I_src(y, x);
                    I_dst(yr, xr, 1) = val;
                    I_dst(yr, xr, 2) = val;
                    I_dst(yr, xr, 3) = val;
                else
                    I_dst(yr, xr, :) = I_src(y, x, :);
                end
            end
        end
    end
    
    % Compute valid mask to black-out the out-of-bounds regions
    valid_mask = false(H_dst, W_dst);
    A_inv = inv(A);
    for yd = 1:H_dst
        for xd = 1:W_dst
            ps = A_inv * [xd - offset_x; yd - offset_y; 1];
            if ps(1) >= 1 && ps(1) <= W_src && ps(2) >= 1 && ps(2) <= H_src
                valid_mask(yd, xd) = true;
            end
        end
    end
    
    % Apply mask: outside valid area becomes black
    for c=1:3
       ch = I_dst(:,:,c);
       ch(~valid_mask) = 0;
       I_dst(:,:,c) = ch;
    end
end

function gt = inverse_map_bilinear(I_src, s)
    [H_src, W_src, C] = size(I_src);
    H_dst = floor(H_src * s);
    W_dst = floor(W_src * s);
    gt = zeros(H_dst, W_dst, C);
    for y = 1:H_dst
        for x = 1:W_dst
            xs = x / s;
            ys = y / s;
            
            x0 = floor(xs); x1 = x0 + 1; dx = xs - x0;
            y0 = floor(ys); y1 = y0 + 1; dy = ys - y0;
            
            x0 = max(1, min(W_src, x0)); x1 = max(1, min(W_src, x1));
            y0 = max(1, min(H_src, y0)); y1 = max(1, min(H_src, y1));
            
            w00 = (1-dx)*(1-dy); w01 = dx*(1-dy);
            w10 = (1-dx)*dy;     w11 = dx*dy;
            
            for c = 1:C
                gt(y,x,c) = w00*I_src(y0,x0,c) + w01*I_src(y0,x1,c) + ...
                            w10*I_src(y1,x0,c) + w11*I_src(y1,x1,c);
            end
        end
    end
end

function fwd_nn = fill_nearest_neighbor(fwd, hit_mask, valid_mask)
    C = size(fwd, 3);
    fwd_nn = fwd;
    
    fill_targets = ~hit_mask & valid_mask;
    
    [~, idx] = bwdist(hit_mask); 
    for c = 1:C
        ch = fwd(:,:,c);
        ch_nn = fwd_nn(:,:,c);
        ch_nn(fill_targets) = ch(idx(fill_targets));
        fwd_nn(:,:,c) = ch_nn;
    end
end

function fwd_avg = fill_local_averaging(fwd, hit_mask, valid_mask)
    [H_dst, W_dst, C] = size(fwd);
    fwd_avg = fwd;
    K = ones(5, 5); 
    
    valid_data = zeros(H_dst, W_dst, C);
    for c = 1:C
        ch = fwd(:,:,c);
        ch_v = zeros(H_dst, W_dst);
        ch_v(hit_mask) = ch(hit_mask);
        valid_data(:,:,c) = ch_v;
    end
    
    mask_filtered = imfilter(double(hit_mask), K, 'replicate');
    
    fill_targets = ~hit_mask & valid_mask;
    for c = 1:C
        data_filtered = imfilter(valid_data(:,:,c), K, 'replicate');
        filled_ch = data_filtered ./ (mask_filtered + 1e-6); 
        
        ch_avg = fwd_avg(:,:,c);
        ch_avg(fill_targets) = filled_ch(fill_targets);
        fwd_avg(:,:,c) = ch_avg;
    end
end

function plot_image_and_fft(col_idx, I, title_str, viz)
    ax1 = subplot(2, 4, col_idx);
    imshow(I);
    title(ax1, title_str, 'Color', viz.Colors.Text);
    axis(ax1, 'on'); xlabel(ax1, 'X (pixels)'); ylabel(ax1, 'Y (pixels)');
    viz.styleAxes(ax1);
    
    ax2 = subplot(2, 4, col_idx + 4);
    if size(I,3) == 3
        I_gray = rgb2gray(I);
    else
        I_gray = I;
    end
    F = fftshift(fft2(I_gray));
    P = log(abs(F) + 1);
    imagesc(ax2, P);
    colormap(ax2, 'parula');
    axis(ax2, 'image'); axis(ax2, 'on');
    xlabel(ax2, '\omega_x (Frequency)'); ylabel(ax2, '\omega_y (Frequency)');
    title(ax2, 'Power Spectrum', 'Color', viz.Colors.Secondary);
    viz.styleAxes(ax2);
end
