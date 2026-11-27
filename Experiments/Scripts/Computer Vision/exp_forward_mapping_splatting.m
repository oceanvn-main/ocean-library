% EXP_FORWARD_MAPPING_SPLATTING
%   Experiment A: The Non-Integer Destination Problem (Rounding vs Splatting)
%
%   This script tests the textbook claim regarding the artifacts produced by
%   forward mapping when destination coordinates fall between integer pixel grids.
%   It contrasts the "snapping" jump discontinuity of Rounding against the 
%   "energy blur" variance of bilinear Splatting.
clear; clc; close all;

disp('Initializing Non-Integer Destination Experiment...');
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

% === 1. Setup Synthetic Datasets ===
% A) Single Point Dataset (for exact mathematical tracking)
W_pt = 31; H_pt = 31;
cx_pt = 16; cy_pt = 16;
tx = cx_pt + 8; ty = cy_pt; % Tracking point 8 pixels to the right of center
I_point = zeros(H_pt, W_pt);
I_point(ty, tx) = 1.0;

% B) Real Image Dataset (for visual aliasing vs blur comparison)
% B) Real Image Dataset (for visual aliasing vs blur comparison)
img_path = fullfile(root_dir, 'Data', 'images', 'secrets', 'z7970955627476_16ad93554963c82b9572da4c862e052c.jpg');
I_raw = imread(img_path); %output uint8, unsigned-interger 8 bits, H x W x 3, where each cell is [0, 255]
% Don't force a squashed 150x150 size. Just shrink it proportionally to ~256 pixels for speed
I_check = im2double(imresize(I_raw, 256 / max(size(I_raw, 1), size(I_raw, 2)))); %to double [0.0 1.0]
[H_chk, W_chk, ~] = size(I_check);
cx_chk = W_chk/2 + 0.5; cy_chk = H_chk/2 + 0.5; % Find correct center, avoid the bias top-left (1, 1)

% === 2. Animation Sweep ===
thetas = linspace(0, 45, 91); % 0 to 45 degrees in 0.5 degree sub-pixel increments

% Tracking Arrays
true_x = zeros(size(thetas)); % Continuous coord
round_x = zeros(size(thetas)); % Actual coordinate being stored
splat_x = zeros(size(thetas)); % Coord from Splatting alg

var_round = zeros(size(thetas));
var_splat = zeros(size(thetas));

disp('Running Animation Sweep (Tracking Point)...');
for i = 1:length(thetas)
    th = deg2rad(thetas(i));
    
    % Affine matrix: rotate around center.
    % R only rotate top -left, We move center of image to top-left to align with graphic
    T1 = [1 0 -cx_pt; 0 1 -cy_pt; 0 0 1]; % Translate center to top-left
    R  = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1]; % Apply rotate, now image spin in place
    T2 = [1 0 cx_pt; 0 1 cy_pt; 0 0 1]; % Translate back
    A_pt = T2 * R * T1;
    
    % Forward Warp
    [I_rnd, I_spt] = forward_warp_compare(I_point, A_pt, [H_pt, W_pt]);
    
    % Track True Position
    p_true = A_pt * [tx; ty; 1]; % Continuous coord
    true_x(i) = p_true(1);
    
    % Calculate Empirical Stats
    [com_r, v_r] = calc_stats(I_rnd);
    [com_s, v_s] = calc_stats(I_spt);
    
    round_x(i) = com_r(1);
    splat_x(i) = com_s(1);
    
    var_round(i) = v_r;
    var_splat(i) = v_s;
    
    % Capture visual snapshot at 15 degrees
    if thetas(i) == 15.0
        T1_chk = [1 0 -cx_chk; 0 1 -cy_chk; 0 0 1];
        T2_chk = [1 0 cx_chk; 0 1 cy_chk; 0 0 1];
        A_chk = T2_chk * R * T1_chk;
        [I_chk_rnd, I_chk_spt] = forward_warp_compare(I_check, A_chk, [H_chk, W_chk]);
    end
end

% === 3. Visualization ===
disp('Rendering Metrics and Snapshots...');
viz = OceanViz.Core('Theme', 'Dark');

% Figure 1: Edge Artifact Visuals
fig1 = figure('Name', 'Visual Diff: Aliasing vs Blur', 'Position', [100, 200, 1200, 400]);
set(fig1, 'Color', viz.Colors.Background);

ax0 = subplot(1, 3, 1);
imshow(I_check, 'InitialMagnification', 100);
title('Original', 'Color', viz.Colors.Text);
viz.styleAxes(ax0);

ax1 = subplot(1, 3, 2);
imshow(I_chk_rnd, 'InitialMagnification', 100);
title('Rounding: Jagged / Stair-stepping', 'Color', viz.Colors.Text);
viz.styleAxes(ax1);

ax2 = subplot(1, 3, 3);
imshow(I_chk_spt, 'InitialMagnification', 100);
title('Splatting: Blurred Soft Edges', 'Color', viz.Colors.Text);
viz.styleAxes(ax2);

% Figure 2: Mathematical Tracking
fig2 = figure('Name', 'Mathematical Failure Tracking', 'Position', [150, 150, 1000, 500]);
set(fig2, 'Color', viz.Colors.Background);

% Panel A: Discontinuous Jumps
ax3 = subplot(1, 2, 1);
plot(ax3, thetas, splat_x, '-', 'Color', viz.Colors.Secondary, 'LineWidth', 4, 'DisplayName', 'Splatting (Smooth)');
hold(ax3, 'on');
plot(ax3, thetas, true_x, '--', 'Color', [1 1 1], 'LineWidth', 2, 'DisplayName', 'True Continuous Path');
plot(ax3, thetas, round_x, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Rounding (Jumps/Overwrites)');
hold(ax3, 'off');
xlabel(ax3, 'Rotation Angle \theta (degrees)');
ylabel(ax3, 'Destination X Coordinate');
title(ax3, 'Jitter/Jumping Artifact (Position Track)');
legend(ax3, 'Location', 'best', 'TextColor', viz.Colors.Text);
grid(ax3, 'on');
viz.styleAxes(ax3);

% Panel B: Energy Spread Variance
ax4 = subplot(1, 2, 2);
plot(ax4, thetas, var_round, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Rounding (\sigma^2 = 0)');
hold(ax4, 'on');
plot(ax4, thetas, var_splat, '-', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'DisplayName', 'Splatting (Energy Spread)');
hold(ax4, 'off');
xlabel(ax4, 'Rotation Angle \theta (degrees)');
ylabel(ax4, 'Spatial Variance \sigma^2 (Pixels^2)');
title(ax4, 'Blur Artifact (Energy Variance)');
legend(ax4, 'Location', 'best', 'TextColor', viz.Colors.Text);
grid(ax4, 'on');
viz.styleAxes(ax4);

disp('Experiment Complete!');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function [I_round, I_splat] = forward_warp_compare(I, A, out_size)
    [H, W, C] = size(I);
    W_out = out_size(2); H_out = out_size(1);
    
    I_round = zeros(H_out, W_out, C);
    
    I_splat_accum = zeros(H_out, W_out, C);
    W_splat_accum = zeros(H_out, W_out);
    
    for y = 1:H
        for x = 1:W
            p_dst = A * [x; y; 1];
            xd = p_dst(1);
            yd = p_dst(2);
            val = double(I(y, x, :));
            
            % --- ROUNDING ---
            xr = round(xd); yr = round(yd);
            if xr >= 1 && xr <= W_out && yr >= 1 && yr <= H_out
                I_round(yr, xr, :) = val; % overwrite
            end
            
            % --- SPLATTING ---
            x0 = floor(xd); x1 = x0 + 1; dx = xd - x0;
            y0 = floor(yd); y1 = y0 + 1; dy = yd - y0;
            
            w00 = (1-dx)*(1-dy);
            w01 = dx*(1-dy);
            w10 = (1-dx)*dy;
            w11 = dx*dy;
            
            if x0 >= 1 && x0 <= W_out && y0 >= 1 && y0 <= H_out
                I_splat_accum(y0, x0, :) = I_splat_accum(y0, x0, :) + w00 * val;
                W_splat_accum(y0, x0) = W_splat_accum(y0, x0) + w00;
            end
            if x1 >= 1 && x1 <= W_out && y0 >= 1 && y0 <= H_out
                I_splat_accum(y0, x1, :) = I_splat_accum(y0, x1, :) + w01 * val;
                W_splat_accum(y0, x1) = W_splat_accum(y0, x1) + w01;
            end
            if x0 >= 1 && x0 <= W_out && y1 >= 1 && y1 <= H_out
                I_splat_accum(y1, x0, :) = I_splat_accum(y1, x0, :) + w10 * val;
                W_splat_accum(y1, x0) = W_splat_accum(y1, x0) + w10;
            end
            if x1 >= 1 && x1 <= W_out && y1 >= 1 && y1 <= H_out
                I_splat_accum(y1, x1, :) = I_splat_accum(y1, x1, :) + w11 * val;
                W_splat_accum(y1, x1) = W_splat_accum(y1, x1) + w11;
            end
        end
    end
    
    % Normalize Splat Accumulator
    mask = W_splat_accum > 0;
    I_splat = zeros(size(I_splat_accum));
    for c = 1:C
        c_acc = I_splat_accum(:,:,c);
        c_out = zeros(H_out, W_out);
        c_out(mask) = c_acc(mask) ./ W_splat_accum(mask);
        I_splat(:,:,c) = c_out;
    end
end

function [com, var_spread] = calc_stats(I)
    [H, W] = size(I);
    sum_I = sum(I(:));
    if sum_I < 1e-8
        com = [0 0];
        var_spread = 0;
        return;
    end
    [X, Y] = meshgrid(1:W, 1:H);
    cx = sum(X(:) .* I(:)) / sum_I;
    cy = sum(Y(:) .* I(:)) / sum_I;
    com = [cx, cy];
    
    dx = X - cx;
    dy = Y - cy;
    r2 = dx.^2 + dy.^2;
    var_spread = sum(r2(:) .* I(:)) / sum_I;
end
