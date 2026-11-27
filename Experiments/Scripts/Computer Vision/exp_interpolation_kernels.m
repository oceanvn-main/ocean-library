% EXP_INTERPOLATION_KERNELS
%   Experiment F: Verifying Eq. 3.75 as a Pure Interpolation Operation
%
%   This script explicitly proves that Inverse Mapping (Gathering) quality 
%   depends strictly on the choice of the reconstruction filter h(x). 
%   By reconstructing a known continuous signal from discrete integer samples 
%   at fractional coordinates, we measure the RMS error as frequency approaches 
%   Nyquist. We also plot the frequency response of the kernels to mathematically 
%   justify their performance ranking.

clear; clc; close all;

disp('Initializing Experiment F: Interpolation Kernels...');
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

% === 1. Setup Signal Parameters ===
N = 256; % Number of integer samples
n = 1:N; % Integer sample coordinates

% Query points: High-res fractional coordinates (avoid edges to prevent boundary artifacts)
margin = 4;
xq = linspace(1+margin, N-margin, 1000); 

% Frequencies to sweep (from DC to Nyquist limit 0.5)
f0_sweep = linspace(0.01, 0.49, 40);

% === 2. Define Kernels ===
% h(s) where s is the distance |x - k|
h_nearest = @(s) double(abs(s) <= 0.5);
h_linear  = @(s) max(0, 1 - abs(s));
h_cubic   = @(s) (abs(s) <= 1) .* (1.5*abs(s).^3 - 2.5*abs(s).^2 + 1) + ...
                 (abs(s) > 1 & abs(s) <= 2) .* (-0.5*abs(s).^3 + 2.5*abs(s).^2 - 4*abs(s) + 2);
h_lanczos = @(s) (abs(s) < 3) .* sinc(s) .* sinc(s/3); % Note: MATLAB sinc is sin(pi*x)/(pi*x)

kernels = {
    struct('name', 'Nearest Neighbor', 'func', h_nearest, 'radius', 1, 'color', '#E74C3C'), ...
    struct('name', 'Bilinear (Triangle)', 'func', h_linear, 'radius', 1, 'color', '#F39C12'), ...
    struct('name', 'Bicubic (Keys a=-0.5)', 'func', h_cubic, 'radius', 2, 'color', '#3498DB'), ...
    struct('name', 'Lanczos-3 (Windowed Sinc)', 'func', h_lanczos, 'radius', 3, 'color', '#2ECC71')
};

num_kernels = length(kernels);
rms_errors = zeros(num_kernels, length(f0_sweep));

% === 3. Run Reconstruction Sweep ===
disp('Running Frequency Sweep and Measuring RMS Errors...');
for f_idx = 1:length(f0_sweep)
    f0 = f0_sweep(f_idx);
    
    % True continuous signal
    y_true = cos(2 * pi * f0 * xq);
    
    % Discrete integer samples
    y_samp = cos(2 * pi * f0 * n);
    
    % Reconstruct for each kernel
    for k_idx = 1:num_kernels
        K = kernels{k_idx};
        y_rec = zeros(size(xq));
        
        % Algorithm 3.2 (1D Gather): g(x) = sum f(k) h(x-k)
        for i = 1:length(xq)
            x_val = xq(i);
            % Find integer samples within the kernel's support radius
            k_start = ceil(x_val - K.radius);
            k_end = floor(x_val + K.radius);
            
            val = 0;
            for k_samp = k_start:k_end
                % h(x - k)
                weight = K.func(x_val - k_samp);
                val = val + y_samp(k_samp) * weight;
            end
            y_rec(i) = val;
        end
        
        % Calculate RMS error
        rms_errors(k_idx, f_idx) = sqrt(mean((y_rec - y_true).^2));
    end
end

% === 4. Calculate Frequency Responses ===
disp('Computing Kernel Frequency Responses...');
% High-res continuous kernel spatial evaluation
s_eval = linspace(-4, 4, 1024);
ds = s_eval(2) - s_eval(1);
Fs = 1 / ds; % Spatial sampling frequency

N_fft = 8192; % Heavy zero-padding for smooth FFT
% The true frequency axis resulting from the FFT:
freq_axis = linspace(-Fs/2, Fs/2, N_fft);

H_mag = zeros(num_kernels, N_fft);
for k_idx = 1:num_kernels
    K = kernels{k_idx};
    h_spatial = K.func(s_eval);
    % We must scale the FFT by the spatial step size to get the true continuous DTFT magnitude
    H_val = abs(fftshift(fft(h_spatial, N_fft))) * ds;
    H_mag(k_idx, :) = H_val;
end

% === 5. Render Dashboard ===
disp('Rendering Dashboard...');
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'Exp F: Interpolation Kernels', 'Position', [100, 100, 1400, 600]);
set(fig, 'Color', viz.Colors.Background);

% Plot 1: RMS Error vs Frequency
ax1 = subplot(1, 2, 1);
hold(ax1, 'on');
for k_idx = 1:num_kernels
    plot(ax1, f0_sweep, rms_errors(k_idx, :), 'Color', kernels{k_idx}.color, ...
        'LineWidth', 2.5, 'DisplayName', kernels{k_idx}.name);
end
hold(ax1, 'off');
viz.styleAxes(ax1);
set(ax1, 'YScale', 'log'); % Log scale to see the massive differences
xlabel(ax1, 'Frequency f_0 (cycles/pixel)');
ylabel(ax1, 'RMS Error (Log Scale)');
title(ax1, 'Reconstruction Quality vs Frequency', 'Color', viz.Colors.Text);
legend(ax1, 'Location', 'northwest', 'TextColor', viz.Colors.Text);
xlim(ax1, [0, 0.5]);
grid(ax1, 'on');
xline(ax1, 0.5, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'DisplayName', 'Nyquist Limit');

% Plot 2: Frequency Response
ax2 = subplot(1, 2, 2);
hold(ax2, 'on');

% Ideal Brick-Wall Response (Rect function in frequency)
% It passes frequencies up to Nyquist (0.5) perfectly, then cuts to 0.
ideal_resp = abs(freq_axis) <= 0.5;
plot(ax2, freq_axis, ideal_resp, '--', 'Color', '#BDC3C7', 'LineWidth', 2, 'DisplayName', 'Ideal Brick-Wall');

for k_idx = 1:num_kernels
    plot(ax2, freq_axis, H_mag(k_idx, :), 'Color', kernels{k_idx}.color, ...
        'LineWidth', 2, 'DisplayName', kernels{k_idx}.name);
end
hold(ax2, 'off');
viz.styleAxes(ax2);
xlabel(ax2, 'Frequency (cycles/pixel)');
ylabel(ax2, 'Magnitude |H(f)|');
title(ax2, 'Kernel Frequency Response', 'Color', viz.Colors.Text);
legend(ax2, 'Location', 'northeast', 'TextColor', viz.Colors.Text);
xlim(ax2, [0, 1.5]); % Show passband and the first side-lobes
ylim(ax2, [-0.1, 1.2]);
grid(ax2, 'on');

% === 6. Step Edge Response (Ringing/Overshoot Analysis) ===
disp('Simulating Step Edge Response...');
N_step = 32;
n_step = 1:N_step;
y_step = double(n_step > N_step/2); % True step function

% High-res query points to see continuous interpolation curve
xq_step = linspace(N_step/2 - 4, N_step/2 + 5, 500);
y_true_step = double(xq_step > N_step/2);

fig2 = figure('Name', 'Spatial Step Edge Response', 'Position', [200, 200, 1200, 600]);
set(fig2, 'Color', viz.Colors.Background);

ax_step = axes(fig2);
hold(ax_step, 'on');

% Plot true discrete samples
stem(ax_step, n_step, y_step, 'Color', viz.Colors.Text, 'MarkerFaceColor', viz.Colors.Text, 'LineWidth', 2, 'DisplayName', 'Discrete Samples');
plot(ax_step, xq_step, y_true_step, '--', 'Color', '#BDC3C7', 'LineWidth', 2, 'DisplayName', 'Ideal Continuous Step');

for k_idx = 1:num_kernels
    K = kernels{k_idx};
    y_rec = zeros(size(xq_step));
    
    for i = 1:length(xq_step)
        x_val = xq_step(i);
        k_start = ceil(x_val - K.radius);
        k_end = floor(x_val + K.radius);
        
        val = 0;
        for k_samp = k_start:k_end
            if k_samp >= 1 && k_samp <= N_step
                weight = K.func(x_val - k_samp);
                val = val + y_step(k_samp) * weight;
            end
        end
        y_rec(i) = val;
    end
    
    plot(ax_step, xq_step, y_rec, 'Color', kernels{k_idx}.color, 'LineWidth', 2, 'DisplayName', kernels{k_idx}.name);
end
hold(ax_step, 'off');
viz.styleAxes(ax_step);
xlim(ax_step, [N_step/2 - 4, N_step/2 + 5]);
ylim(ax_step, [-0.2, 1.2]);
xlabel(ax_step, 'Spatial Coordinate (x)');
ylabel(ax_step, 'Interpolated Value g(x)');
title(ax_step, 'Step Edge Response (Ringing & Overshoot)', 'Color', viz.Colors.Text);
legend(ax_step, 'Location', 'southeast', 'TextColor', viz.Colors.Text);
grid(ax_step, 'on');

% === 7. Visual Image Reconstruction (Zoneplate) ===
disp('Rendering Visual 2D Image Comparison...');
img_path = fullfile(root_dir, 'Data', 'images', 'zoneplate.png');
if exist(img_path, 'file')
    I_zoneplate = im2double(imread(img_path));
    
    % Crop a small 80x80 patch (offset from center to capture high-frequency rings)
    [H_img, W_img, ~] = size(I_zoneplate);
    crop_size = 80;
    cy = round(H_img * 0.25); 
    cx = round(W_img * 0.75);
    
    cy = max(1 + crop_size/2, min(H_img - crop_size/2, cy));
    cx = max(1 + crop_size/2, min(W_img - crop_size/2, cx));
    
    I_crop = I_zoneplate(cy-crop_size/2 : cy+crop_size/2-1, cx-crop_size/2 : cx+crop_size/2-1, :);
    
    % Upscale massively (8x) using MATLAB's built-in optimized kernel functions
    scale_factor = 8;
    I_nn      = imresize(I_crop, scale_factor, 'nearest');
    I_lin     = imresize(I_crop, scale_factor, 'bilinear');
    I_cub     = imresize(I_crop, scale_factor, 'bicubic');
    I_lanczos = imresize(I_crop, scale_factor, 'lanczos3');
    
    fig3 = figure('Name', 'Visual Kernel Comparison (Zoneplate)', 'Position', [250, 250, 1600, 350]);
    set(fig3, 'Color', viz.Colors.Background);
    
    ax_c1 = subplot(1, 5, 1); imshow(I_crop); title(ax_c1, 'Original (80x80 Crop)', 'Color', viz.Colors.Text); viz.styleAxes(ax_c1);
    ax_c2 = subplot(1, 5, 2); imshow(I_nn); title(ax_c2, 'Nearest Neighbor', 'Color', viz.Colors.Text); viz.styleAxes(ax_c2);
    ax_c3 = subplot(1, 5, 3); imshow(I_lin); title(ax_c3, 'Bilinear', 'Color', viz.Colors.Text); viz.styleAxes(ax_c3);
    ax_c4 = subplot(1, 5, 4); imshow(I_cub); title(ax_c4, 'Bicubic', 'Color', viz.Colors.Text); viz.styleAxes(ax_c4);
    ax_c5 = subplot(1, 5, 5); imshow(I_lanczos); title(ax_c5, 'Lanczos-3', 'Color', viz.Colors.Text); viz.styleAxes(ax_c5);
end

disp('Experiment Complete!');
