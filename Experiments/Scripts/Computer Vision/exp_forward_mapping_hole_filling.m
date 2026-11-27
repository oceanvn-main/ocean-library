% EXP_FORWARD_MAPPING_HOLE_FILLING
%   Experiment C: Hole-Filling Side Effects (Further Aliasing and Blurring)
%
%   This script evaluates post-hoc hole-filling strategies (Nearest-Neighbor
%   and Local Averaging) for Forward Mapping, proving that they introduce 
%   unacceptable Aliasing and Blurring compared to a proper Inverse Mapping approach.

disp('Initializing Hole-Filling Experiment...');
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

% === 1. Setup Datasets ===
% A) Synthetic Stripes (To aggressively expose Aliasing)
sz = 128;
[X, Y] = meshgrid(1:sz, 1:sz);
% Vertical stripes, period = 4 pixels
I_stripes = double(mod(X, 4) < 2);

% B) Real Photograph (To measure Blur and SSIM degradation)
img_path = fullfile(root_dir, 'Data', 'images', 'apple.jpg');
if exist(img_path, 'file')
    I_raw = imread(img_path);
else
    % Fallback to checkerboard if image not found
    I_raw = uint8(kron(ones(64), [255 0; 0 255])); 
end
% Resize to 200x200 for fast processing
I_real = im2double(imresize(I_raw, [200, 200]));

% === 2. Experiment Setup ===
scale = 3.0;

disp('Processing Synthetic Stripes Dataset...');
% Run mapping on Stripes
[gt_s, fwd_s, nn_s, avg_s] = run_filling_experiment(I_stripes, scale);

disp('Processing Real Image Dataset...');
% Run mapping on Real Image
[gt_r, fwd_r, nn_r, avg_r] = run_filling_experiment(I_real, scale);

% Calculate quantitative metrics on the Real Image
mse_fwd = mean((fwd_r(:) - gt_r(:)).^2);
mse_nn  = mean((nn_r(:)  - gt_r(:)).^2);
mse_avg = mean((avg_r(:) - gt_r(:)).^2);

% === 3. Visualization ===
disp('Rendering Dashboard...');
viz = OceanViz.Core('Theme', 'Dark');

% Figure 1: Synthetic Stripes & Spectral Aliasing
fig1 = figure('Name', 'Aliasing Assessment: Synthetic Stripes', 'Position', [100, 400, 1200, 600]);
set(fig1, 'Color', viz.Colors.Background);

plot_image_and_fft(1, gt_s, 'Ground Truth (Inverse Map)', viz);
plot_image_and_fft(2, fwd_s, 'Forward Map (Holes)', viz);
plot_image_and_fft(3, nn_s, 'Nearest-Neighbor Fill', viz);
plot_image_and_fft(4, avg_s, 'Local Averaging Fill', viz);

% Figure 2: Real Image Quality Degradation
fig2 = figure('Name', 'Blur & Quality Degradation: Real Image', 'Position', [150, 100, 1000, 800]);
set(fig2, 'Color', viz.Colors.Background);

ax1 = subplot(2, 2, 1);
imshow(gt_r); title('Ground Truth (Inverse Map)', 'Color', viz.Colors.Text);
viz.styleAxes(ax1);

ax2 = subplot(2, 2, 2);
imshow(fwd_r); title(sprintf('Forward Map (Holes)\\nMSE: %.4f', mse_fwd), 'Color', viz.Colors.Text);
viz.styleAxes(ax2);

ax3 = subplot(2, 2, 3);
imshow(nn_r); title(sprintf('Nearest-Neighbor Fill\\nMSE: %.4f', mse_nn), 'Color', viz.Colors.Text);
viz.styleAxes(ax3);

ax4 = subplot(2, 2, 4);
imshow(avg_r); title(sprintf('Local Averaging Fill\\nMSE: %.4f', mse_avg), 'Color', viz.Colors.Text);
viz.styleAxes(ax4);

disp('Experiment Complete!');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function [gt, fwd, fwd_nn, fwd_avg] = run_filling_experiment(I_src, s)
    [H_src, W_src, C] = size(I_src);
    H_dst = floor(H_src * s);
    W_dst = floor(W_src * s);
    
    % 1. Ground Truth (Inverse Mapping with Bilinear Interpolation)
    gt = zeros(H_dst, W_dst, C);
    for y = 1:H_dst
        for x = 1:W_dst
            xs = x / s;
            ys = y / s;
            
            % Bilinear bounds
            x0 = floor(xs); x1 = x0 + 1; dx = xs - x0;
            y0 = floor(ys); y1 = y0 + 1; dy = ys - y0;
            
            % Bound checking
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
    
    % 2. Forward Mapping (Scatter)
    fwd = zeros(H_dst, W_dst, C);
    hit_mask = false(H_dst, W_dst);
    
    for y = 1:H_src
        for x = 1:W_src
            xd = round(x * s);
            yd = round(y * s);
            if xd >= 1 && xd <= W_dst && yd >= 1 && yd <= H_dst
                hit_mask(yd, xd) = true;
                fwd(yd, xd, :) = I_src(y, x, :);
            end
        end
    end
    
    % Make holes pure blue for visualization if C==3
    if C == 3
        for y = 1:H_dst
            for x = 1:W_dst
                if ~hit_mask(y, x)
                    fwd(y,x,1) = 0; fwd(y,x,2) = 0.5; fwd(y,x,3) = 1; % Blue holes
                end
            end
        end
    end
    
    % 3. Nearest Neighbor Hole-Fill
    fwd_nn = fwd;
    [~, idx] = bwdist(hit_mask); % idx contains the linear index of the nearest non-zero pixel
    for c = 1:C
        ch = fwd(:,:,c);
        ch_nn = fwd_nn(:,:,c);
        ch_nn(~hit_mask) = ch(idx(~hit_mask));
        fwd_nn(:,:,c) = ch_nn;
    end
    
    % 4. Local Averaging Hole-Fill
    % Normalized convolution over a 3x3 window
    fwd_avg = fwd;
    K = ones(5, 5); % 5x5 mean filter
    
    % We must only average the valid (hit) pixels, ignoring the hole values (blue)
    valid_data = zeros(H_dst, W_dst, C);
    for c = 1:C
        ch = fwd(:,:,c);
        ch_v = zeros(H_dst, W_dst);
        ch_v(hit_mask) = ch(hit_mask);
        valid_data(:,:,c) = ch_v;
    end
    
    mask_filtered = imfilter(double(hit_mask), K, 'replicate');
    for c = 1:C
        data_filtered = imfilter(valid_data(:,:,c), K, 'replicate');
        filled_ch = data_filtered ./ (mask_filtered + 1e-6); % Avoid div by zero
        
        ch_avg = fwd_avg(:,:,c);
        ch_avg(~hit_mask) = filled_ch(~hit_mask);
        fwd_avg(:,:,c) = ch_avg;
    end
end

function plot_image_and_fft(col_idx, I, title_str, viz)
    % Image Panel
    ax1 = subplot(2, 4, col_idx);
    imshow(I);
    title(ax1, title_str, 'Color', viz.Colors.Text);
    viz.styleAxes(ax1);
    
    % FFT Panel
    ax2 = subplot(2, 4, col_idx + 4);
    
    % Compute Power Spectrum
    if size(I,3) == 3
        I_gray = rgb2gray(I);
    else
        I_gray = I;
    end
    
    F = fftshift(fft2(I_gray));
    P = log(abs(F) + 1);
    
    imagesc(ax2, P);
    colormap(ax2, 'parula');
    axis(ax2, 'image'); axis(ax2, 'off');
    title(ax2, 'Power Spectrum', 'Color', viz.Colors.Secondary);
    viz.styleAxes(ax2);
end
