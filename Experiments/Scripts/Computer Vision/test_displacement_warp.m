% TEST_DISPLACEMENT_WARP  Simple script to test and understand forward displacement warping.
%
%   This script generates a 256x256 checkerboard, defines a sinusoidal
%   displacement field, and visualizes the internal stages of forward mapping:
%     Stage 1: Input image overlaid with the displacement vector field.
%     Stage 2: The weight accumulator (W_accum) showing collisions (>1) and holes (=0).
%     Stage 3: The final warped image after normalization and hole filling.
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-30

clc; clear; close all;

% Set up paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

fprintf('Generating test image (256x256 checkerboard)...\n');
N = 256;
[X, Y] = meshgrid(1:N, 1:N);
I_src = mod(floor(X / 16) + floor(Y / 16), 2);
I_src(1, :) = 0.5; I_src(end, :) = 0.5;
I_src(:, 1) = 0.5; I_src(:, end) = 0.5;

% Generate a sinusoidal wave displacement field (wave-like deformation)
fprintf('Creating displacement field...\n');
Dx = 18 * sin(2 * pi * Y / 128);  % horizontal wave
Dy = 18 * cos(2 * pi * X / 128);  % vertical wave

% --- Replicate Algorithm Internals for Stage 2 Visualization ---
I_dbl = double(I_src);
I_accum = zeros(N, N);
W_accum = zeros(N, N);

for r = 1:N
    for c = 1:N
        dx = Dx(r, c);
        dy = Dy(r, c);
        x = c + dx;
        y = r + dy;

        if x >= 1 && x <= N && y >= 1 && y <= N
            x0 = floor(x);
            y0 = floor(y);
            x1 = x0 + 1;
            y1 = y0 + 1;

            wx = x - x0;
            wy = y - y0;

            w00 = (1 - wx) * (1 - wy);
            w01 = wx * (1 - wy);
            w10 = (1 - wx) * wy;
            w11 = wx * wy;

            pixel_val = I_dbl(r, c);

            if y0 >= 1 && y0 <= N && x0 >= 1 && x0 <= N
                I_accum(y0, x0) = I_accum(y0, x0) + w00 * pixel_val;
                W_accum(y0, x0) = W_accum(y0, x0) + w00;
            end
            if y0 >= 1 && y0 <= N && x1 >= 1 && x1 <= N
                I_accum(y0, x1) = I_accum(y0, x1) + w01 * pixel_val;
                W_accum(y0, x1) = W_accum(y0, x1) + w01;
            end
            if y1 >= 1 && y1 <= N && x0 >= 1 && x0 <= N
                I_accum(y1, x0) = I_accum(y1, x0) + w10 * pixel_val;
                W_accum(y1, x0) = W_accum(y1, x0) + w10;
            end
            if y1 >= 1 && y1 <= N && x1 >= 1 && x1 <= N
                I_accum(y1, x1) = I_accum(y1, x1) + w11 * pixel_val;
                W_accum(y1, x1) = W_accum(y1, x1) + w11;
            end
        end
    end
end

fprintf('Warping image using DisplacementFieldWarp...\n');
tic;
I_warped = OceanMath.Image.Resampling.DisplacementFieldWarp.warp(I_src, Dx, Dy);
time_taken = toc;
fprintf('Warp completed in %.4f seconds.\n', time_taken);

% --- Stage-by-Stage Visualization ---
fprintf('Displaying stage-by-stage results...\n');
figure('Name', 'Forward Displacement Warp Stages', 'Position', [100, 100, 1500, 550]);

% Stage 1: Original Image & Vector Field
subplot(1, 3, 1);
imshow(I_src);
hold on;
% Sample quiver points to avoid cluttering
sample_step = 16;
[X_samp, Y_samp] = meshgrid(1:sample_step:N, 1:sample_step:N);
Dx_samp = Dx(1:sample_step:end, 1:sample_step:end);
Dy_samp = Dy(1:sample_step:end, 1:sample_step:end);
h_vec = quiver(X_samp, Y_samp, Dx_samp, Dy_samp, 0, 'Color', 'r', 'LineWidth', 1.2);
hold off;
legend(h_vec, {'Displacement Vectors'}, 'Location', 'southoutside');
title('Stage 1: Input & Vector Field');
axis on;

% Stage 2: Weight Accumulator Map (Collisions and Holes)
subplot(1, 3, 2);
% Display W_accum. Areas with value 0 are black (holes), values > 1 are bright (collisions)
imagesc(W_accum);
colorbar;
title('Stage 2: Weight Accumulator (W_{accum})');
xlabel('Black (0) = Gaps/Holes | Yellow (>1) = Collisions');
axis image;

% Stage 3: Normalized & Hole-Filled Final Output
subplot(1, 3, 3);
imshow(I_warped);
title('Stage 3: Warped Output');
axis on;

fprintf('Success: DisplacementFieldWarp stage-by-stage test completed.\n');
