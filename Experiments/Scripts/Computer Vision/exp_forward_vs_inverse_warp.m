% EXP_FORWARD_VS_INVERSE_WARP  Phase 1: Scatter vs Gather Mapping Coverage
%
%   DATASET:        Synthetic 2D Chirp Image (256x256).
%   MODEL:          Explicit Scatter (Forward) and Gather (Inverse) warping.
%   ASSUMPTIONS:    - Scatter uses nearest-cell deposit (no anti-aliasing/blending).
%                   - Gather uses bilinear interpolation.
%                   - Holes are computed strictly inside the valid mapped area.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading (Real Image) ===
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
img_path = fullfile(root_dir, 'Data', 'images', 'secrets', 'z7970955627476_16ad93554963c82b9572da4c862e052c.jpg');

I_raw = imread(img_path);
% Resize to 256x256 for performance in explicit loops
I_image = im2double(imresize(I_raw, [256, 256]));
[H_src, W_src, C_src] = size(I_image);

% === 2. Transformation Matrix ===
% Non-uniform scaling: magnify X, minify Y, plus slight rotation
s_x = 1.0;
s_y = 1.0;
theta = deg2rad(15);

S = [s_x 0 0; 0 s_y 0; 0 0 1];
R_mat = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
A = R_mat * S;
A_inv = inv(A);

% Determine destination bounding box
corners = [1 1 1; W_src 1 1; 1 H_src 1; W_src H_src 1]';
mapped_corners = A * corners;
min_x = floor(min(mapped_corners(1, :))); max_x = ceil(max(mapped_corners(1, :)));
min_y = floor(min(mapped_corners(2, :))); max_y = ceil(max(mapped_corners(2, :)));

W_dst = max_x - min_x + 1;
H_dst = max_y - min_y + 1;
offset_x = 1 - min_x;
offset_y = 1 - min_y;

% === 3. Scatter (Forward Mapping) ===
disp('Running Scatter (Forward Mapping)...');
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

% === 4. Gather (Inverse Mapping) ===
disp('Running Gather (Inverse Mapping)...');
I_gather = zeros(H_dst, W_dst, C_src);
valid_mask = false(H_dst, W_dst);

for y_dst = 1:H_dst
    for x_dst = 1:W_dst
        p_src = A_inv * [x_dst - offset_x; y_dst - offset_y; 1];
        x_src = p_src(1);
        y_src = p_src(2);
        
        if x_src >= 1 && x_src <= W_src && y_src >= 1 && y_src <= H_src
            valid_mask(y_dst, x_dst) = true;
            
            % Bilinear Interpolation
            x0 = floor(x_src); x1 = ceil(x_src);
            y0 = floor(y_src); y1 = ceil(y_src);
            
            x0 = max(1, min(W_src, x0)); x1 = max(1, min(W_src, x1));
            y0 = max(1, min(H_src, y0)); y1 = max(1, min(H_src, y1));
            
            dx = x_src - x0;
            dy = y_src - y0;
            
            v00 = I_image(y0, x0, :);
            v01 = I_image(y0, x1, :);
            v10 = I_image(y1, x0, :);
            v11 = I_image(y1, x1, :);
            
            v0 = v00 * (1 - dx) + v01 * dx;
            v1 = v10 * (1 - dx) + v11 * dx;
            I_gather(y_dst, x_dst, :) = v0 * (1 - dy) + v1 * dy;
        end
    end
end

% === 5. Hit Counter Analysis ===
valid_hits = hit_counter(valid_mask);
hole_fraction = sum(valid_hits == 0) / numel(valid_hits);
exact_fraction = sum(valid_hits == 1) / numel(valid_hits);
overlap_fraction = sum(valid_hits >= 2) / numel(valid_hits);

% === 6. Minification Sweep ===
disp('Running Minification Sweep...');
sweep_sy = linspace(0.1, 1.0, 10);
sweep_holes = zeros(size(sweep_sy));

% Using smaller image for faster sweep
H_sweep = 100; W_sweep = 100;
[X_s, Y_s] = meshgrid(linspace(-1, 1, W_sweep), linspace(-1, 1, H_sweep));
I_sweep = sin(50 * (X_s.^2 + Y_s.^2));

for i = 1:length(sweep_sy)
    cur_sy = sweep_sy(i);
    cur_A = R_mat * [s_x 0 0; 0 cur_sy 0; 0 0 1];
    cur_A_inv = inv(cur_A);
    
    % Get bounds
    corn = [1 1 1; W_sweep 1 1; 1 H_sweep 1; W_sweep H_sweep 1]';
    m_corn = cur_A * corn;
    mx_x = ceil(max(m_corn(1, :))); mn_x = floor(min(m_corn(1, :)));
    mx_y = ceil(max(m_corn(2, :))); mn_y = floor(min(m_corn(2, :)));
    
    W_d = mx_x - mn_x + 1; H_d = mx_y - mn_y + 1;
    ox = 1 - mn_x; oy = 1 - mn_y;
    
    hc = zeros(H_d, W_d);
    for y = 1:H_sweep
        for x = 1:W_sweep
            pd = cur_A * [x; y; 1];
            xd = round(pd(1) + ox); yd = round(pd(2) + oy);
            if xd >= 1 && xd <= W_d && yd >= 1 && yd <= H_d
                hc(yd, xd) = hc(yd, xd) + 1;
            end
        end
    end
    
    vm = false(H_d, W_d);
    for yd = 1:H_d
        for xd = 1:W_d
            ps = cur_A_inv * [xd - ox; yd - oy; 1];
            xs = ps(1); ys = ps(2);
            if xs >= 1 && xs <= W_sweep && ys >= 1 && ys <= H_sweep
                vm(yd, xd) = true;
            end
        end
    end
    
    vh = hc(vm);
    sweep_holes(i) = sum(vh == 0) / numel(vh);
end

% === 7. Visualization ===
disp('Rendering Figures...');
viz = OceanViz.Core('Theme', 'Dark');

% Figure 1: Visual Diff
fig1 = figure('Name', 'Scatter vs Gather Visual Diff', 'Position', [100, 100, 1200, 500]);
set(fig1, 'Color', viz.Colors.Background);

ax1 = subplot(1, 3, 1);
imshow(I_image);
title('Original Image', 'Color', viz.Colors.Text);
viz.styleAxes(ax1);

ax2 = subplot(1, 3, 2);
imshow(I_scatter);
title(sprintf('Scatter (Holes = %.1f%%)', hole_fraction * 100), 'Color', viz.Colors.Text);
viz.styleAxes(ax2);

ax3 = subplot(1, 3, 3);
imshow(I_gather);
title('Gather (Clean)', 'Color', viz.Colors.Text);
viz.styleAxes(ax3);

% Figure 2: Analysis Plots
fig2 = figure('Name', 'Hit Counter & Minification Analysis', 'Position', [150, 150, 1000, 400]);
set(fig2, 'Color', viz.Colors.Background);

ax4 = subplot(1, 2, 1);
categories = {'Holes (0)', 'Exact (1)', 'Overlaps (2+)'};
fractions = [hole_fraction, exact_fraction, overlap_fraction] * 100;
bar(ax4, fractions, 'FaceColor', viz.Colors.Primary);
set(ax4, 'XTickLabel', categories);
ylabel('Percentage of Destination Pixels (%)');
title('Scatter Hit Distribution');
viz.styleAxes(ax4);

ax5 = subplot(1, 2, 2);
plot(ax5, sweep_sy, sweep_holes * 100, '-o', 'LineWidth', 2, 'Color', viz.Colors.Secondary);
xlabel('Minification Factor (S_y)');
ylabel('Hole Fraction (%)');
title('Hole Fraction vs Minification Strength');
grid on;
viz.styleAxes(ax5);

% === 8. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset          : Real Image (Apple) (256x256)\n');
fprintf('  Affine Transform : Sx = %.1f, Sy = %.1f, Theta = %.1f deg\n', s_x, s_y, rad2deg(theta));
fprintf('  Forward (Scatter): Nearest-cell deposit\n');
fprintf('  Inverse (Gather) : Bilinear Interpolation\n');
fprintf('  --- Hit Analysis ---\n');
fprintf('  Holes (0 hits)   : %5.2f%%\n', hole_fraction * 100);
fprintf('  Exact (1 hit)    : %5.2f%%\n', exact_fraction * 100);
fprintf('  Overlaps (>=2)   : %5.2f%%\n', overlap_fraction * 100);
fprintf('==========================\n');
