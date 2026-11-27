% EXP_FORWARD_MAPPING_HIERARCHY
%   Experiment D: Parametric Transform Hierarchy
%
%   A rigorous validation of how transform classes (Translation, Euclidean, 
%   Similarity, Affine, Projective) dictate Forward Mapping holes.
%   Validates 5 strict falsifiable hypotheses regarding hole density and orientation.

clear; clc; close all;
disp('Initializing Experiment D: Parametric Transform Hierarchy...');
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

% =========================================================================
% BLOCK 1: Setup Dataset
% =========================================================================
% We use a 200x400 non-square solid orange image.
% The elongated Y-axis is critical for clearly demonstrating the band-wise 
% spatial density variance in the Projective condition (Hypothesis 5).
H_src = 400; W_src = 200; 
I_solid = cat(3, ones(H_src, W_src)*1.0, ones(H_src, W_src)*0.4, ones(H_src, W_src)*0.2);
hole_color = [0, 0, 0];

% Number of horizontal bands for spatial variance testing
num_bands = 10;

% Storage for results
results = struct();

% =========================================================================
% BLOCK 2: Data Collection (The 6 Conditions)
% =========================================================================

% --- Condition 1: Translation Control ---
disp('Condition 1: Translation Control');
H_trans = OceanMath.Geometry.Transform2D.translation(0.5, 0.3); % Fractional
[~, mask_trans, ~, results.H_trans, ~] = run_forward_warp_projective(I_solid, H_trans, num_bands);

% --- Condition 2: Euclidean Sweep ---
disp('Condition 2: Euclidean Sweep');
angles = [0, 15, 30, 45, 60, 75, 90];
results.H_euc_sweep = zeros(size(angles));
for i = 1:length(angles)
    H_euc = OceanMath.Geometry.Transform2D.rigid(0, 0, angles(i));
    [I_euc_tmp, mask_euc_tmp, ~, results.H_euc_sweep(i), ~] = run_forward_warp_projective(I_solid, H_euc, num_bands);
    if angles(i) == 30
        mask_euc_30 = mask_euc_tmp; % Save for gallery
        I_euc_30 = I_euc_tmp;
    end
end
results.H_euc_avg = mean(results.H_euc_sweep);

% --- Condition 3: Similarity ---
disp('Condition 3: Similarity');
H_sim = OceanMath.Geometry.Transform2D.similarity(0, 0, 30, 2.0);
[I_sim, mask_sim, ~, results.H_sim, ~] = run_forward_warp_projective(I_solid, H_sim, num_bands);

% --- Condition 4: Affine (No Shear) ---
disp('Condition 4: Affine (No Shear)');
H_aff_noshear = OceanMath.Geometry.Transform2D.affine(0, 0, 30, 4.0, 1.0, 0, 0);
[I_aff_noshear, mask_aff_noshear, ~, results.H_aff_noshear, results.bands_aff_noshear] = run_forward_warp_projective(I_solid, H_aff_noshear, num_bands);

% --- Condition 5: Affine (Shear) Sweep ---
disp('Condition 5: Affine (Shear) Sweep');
shears = [0, 0.5, 1.0, 1.5, 2.0];
results.H_shear_sweep = zeros(size(shears));
results.theta_measured = zeros(size(shears));
results.theta_predicted = zeros(size(shears));
results.fft_masks = cell(size(shears));

for i = 1:length(shears)
    k = shears(i);
    H_sh = OceanMath.Geometry.Transform2D.affine(0, 0, 0, 2.0, 2.0, k, 0);
    [I_tmp, mask_tmp, ~, results.H_shear_sweep(i), ~] = run_forward_warp_projective(I_solid, H_sh, num_bands);
    
    if k == 1.5
        mask_aff_shear = mask_tmp;
    end
    
    % Predicted angle: The structural streaks follow the sheared vertical basis vectors.
    % A vertical line [0; 1] is mapped to [s*k; s]. Its angle from the vertical is atand(sk/s) = atand(k).
    results.theta_predicted(i) = atand(shears(i));
    
    % 1. FFT for Visualization (keep this for the gallery)
    binary_mask = double(mask_tmp);
    binary_mask_centered = binary_mask - mean(binary_mask(:));
    F = fftshift(abs(fft2(binary_mask_centered)));
    F(floor(end/2)+1, floor(end/2)+1) = 0; % zero out DC
    results.fft_masks{i} = F;
    
    % 2. Robust Angle Measurement using Radon Transform
    theta_sweep = -90:0.5:89.5;
    [R, ~] = radon(binary_mask, theta_sweep);
    
    % The projection angle that maximizes variance corresponds to the streak orientation
    proj_vars = var(R, 0, 1);
    [~, max_idx] = max(proj_vars);
    
    measured_angle = theta_sweep(max_idx);
    
    % Align signs perfectly with our prediction framework (abs for magnitude tracking)
    results.theta_measured(i) = abs(measured_angle);
    results.theta_predicted(i) = abs(results.theta_predicted(i));
end

% --- Condition 6: Projective Warp ---
disp('Condition 6: Projective Warp');
% We want a perspective warp that magnifies heavily at the bottom.
% Scale is approx 2.0 at center so det ~ 4.
H_proj = [2.0,  0,     0; 
          0,    2.0,   0; 
          0.000, 0.003, 1];
[I_proj, mask_proj, ~, results.H_proj, results.bands_proj] = run_forward_warp_projective(I_solid, H_proj, num_bands);


% =========================================================================
% BLOCK 3: Dashboard Rendering
% =========================================================================
disp('Rendering Dashboards...');
viz = OceanViz.Core('Theme', 'Dark');

% --- PLOT 1: Global Hole Fractions (Bar chart & Euclidean Sweep) ---
fig1 = figure('Name', 'Exp D: Global Hole Fractions', 'Position', [50, 50, 1000, 500]);
set(fig1, 'Color', viz.Colors.Background);

ax1 = subplot(1, 2, 1);
categories = {'Sim', 'Aff (No Sh)', 'Aff (Sh, k=0)', 'Proj'};
vals = [results.H_sim, results.H_aff_noshear, results.H_shear_sweep(1), results.H_proj] * 100;
b = bar(ax1, vals, 'FaceColor', viz.Colors.Primary);
hold(ax1, 'on');
% Theory line
yline(ax1, 75.0, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'DisplayName', 'Theory: 1 - 1/det(A)');
ylabel(ax1, 'Hole Fraction (%)');
set(ax1, 'XTickLabel', categories);
title(ax1, 'H1: Matched det(A)=4 Hole Fractions', 'Color', viz.Colors.Text);
ylim(ax1, [0 100]);
legend(ax1, 'Location', 'southwest');
viz.styleAxes(ax1);

ax2 = subplot(1, 2, 2);
plot(ax2, angles, results.H_euc_sweep * 100, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 2);
xlabel(ax2, 'Rotation Angle (deg)'); ylabel(ax2, 'Hole Fraction (%)');
title(ax2, 'H2: Euclidean Sweep (det=1)', 'Color', viz.Colors.Text);
ylim(ax2, [0 30]); % Should stay bounded well below 75%
grid(ax2, 'on');
viz.styleAxes(ax2);


% --- PLOT 2: Hit-Mask Gallery ---
fig2 = figure('Name', 'Exp D: Hit-Mask Gallery', 'Position', [100, 100, 1200, 700]);
set(fig2, 'Color', viz.Colors.Background);

panels = {mask_trans, mask_euc_30, mask_sim, mask_aff_noshear, mask_aff_shear, mask_proj};
titles = {'Translation (H4)', 'Euclidean', 'Similarity', 'Affine (No Shear)', 'Affine (Shear=1.5)', 'Projective (H5)'};

for i = 1:6
    ax = subplot(2, 3, i);
    imshow(panels{i});
    title(ax, titles{i}, 'Color', viz.Colors.Text);
    viz.styleAxes(ax);
end

% === Figure 3: Affine Shear Analysis ===
fig3 = figure('Name', 'Exp D: Affine Shear Analysis', 'Position', [150, 150, 1400, 450]);
set(fig3, 'Color', viz.Colors.Background);

% Plot 1: Angle vs Shear (The Orientation Sub-Part)
ax1 = subplot(1, 4, 1);
plot(ax1, shears, results.theta_measured, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 2, 'DisplayName', 'Measured (Radon)');
hold(ax1, 'on');
plot(ax1, shears, results.theta_predicted, '--x', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'DisplayName', 'Predicted (atan(k))');
hold(ax1, 'off');
xlabel(ax1, 'Shear factor k'); ylabel(ax1, 'Streak Angle (deg)');
title(ax1, 'H3: Orientation Accuracy', 'Color', viz.Colors.Text);
legend(ax1, 'Location', 'southeast', 'TextColor', viz.Colors.Text);
viz.styleAxes(ax1);

% Plot 2: Hole Fraction vs Shear (The Amount Sub-Part)
ax2 = subplot(1, 4, 2);
plot(ax2, shears, results.H_shear_sweep * 100, '-s', 'Color', viz.Colors.Primary, 'LineWidth', 2, 'DisplayName', 'Measured Fraction');
hold(ax2, 'on');
yline(ax2, (1 - 1/4.0)*100, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'DisplayName', 'Theoretical (1-1/detA)');
hold(ax2, 'off');
ylim(ax2, [0, 100]);
xlabel(ax2, 'Shear factor k'); ylabel(ax2, 'Hole Fraction (%)');
title(ax2, 'H3: Constant Hole Fraction', 'Color', viz.Colors.Text);
legend(ax2, 'Location', 'northeast', 'TextColor', viz.Colors.Text);
viz.styleAxes(ax2);

% Plot 3 & 4: FFT Mask Gallery
ax3 = subplot(1, 4, 3);
imagesc(ax3, log(abs(results.fft_masks{3}) + 1));
colormap(ax3, 'parula'); axis(ax3, 'image'); axis(ax3, 'off');
title(ax3, 'FFT Mask (k=1.0)', 'Color', viz.Colors.Text);

ax4 = subplot(1, 4, 4);
imagesc(ax4, log(abs(results.fft_masks{5}) + 1));
colormap(ax4, 'parula'); axis(ax4, 'image'); axis(ax4, 'off');
title(ax4, 'FFT Mask (k=2.0)', 'Color', viz.Colors.Text);

% --- PLOT 4: Projective Spatial Variance (H5) ---
fig4 = figure('Name', 'Exp D: Projective Spatial Variance', 'Position', [200, 200, 1000, 500]);
set(fig4, 'Color', viz.Colors.Background);

ax4_1 = subplot(1, 2, 1);
b_idx = 1:num_bands;
plot(ax4_1, b_idx, results.bands_proj * 100, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 2, 'DisplayName', 'Projective');
hold(ax4_1, 'on');
plot(ax4_1, b_idx, results.bands_aff_noshear * 100, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 2, 'DisplayName', 'Affine (Baseline)');
xlabel(ax4_1, 'Horizontal Band Index (Top to Bottom)'); ylabel(ax4_1, 'Hole Fraction Hb (%)');
title(ax4_1, 'H5: Spatial Variance (Monotonicity)', 'Color', viz.Colors.Text);
legend(ax4_1, 'Location', 'northeast');
grid(ax4_1, 'on');
viz.styleAxes(ax4_1);

ax4_2 = subplot(1, 2, 2);
imshow(I_proj);
hold(ax4_2, 'on');
[Hp, Wp, ~] = size(I_proj);
for i = 1:num_bands-1
    y_line = (i / num_bands) * Hp;
    yline(ax4_2, y_line, '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
end
title(ax4_2, 'Projective Hit-Map with Band Boundaries', 'Color', viz.Colors.Text);
viz.styleAxes(ax4_2);

disp('Experiment Complete!');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function [I_dst, hit_mask, valid_mask, hole_frac, hole_frac_bands] = run_forward_warp_projective(I_src, H_mat, num_bands)
    [H_src, W_src, C] = size(I_src);
    hole_color = [0, 0, 0];
    
    % 1. Find bounding box using OceanMath's Transform2D component
    corners_in = [1 W_src 1 W_src; 1 1 H_src H_src];
    corners_out = OceanMath.Geometry.Transform2D.transformPoints(H_mat, corners_in);
    
    min_x = floor(min(corners_out(1,:))); max_x = ceil(max(corners_out(1,:)));
    min_y = floor(min(corners_out(2,:))); max_y = ceil(max(corners_out(2,:)));
    
    W_dst = max_x - min_x + 1;
    H_dst = max_y - min_y + 1;
    offset_x = 1 - min_x;
    offset_y = 1 - min_y;
    
    % 2. Initialize destination with hole color
    I_dst = cat(3, ones(H_dst, W_dst)*hole_color(1), ...
                   ones(H_dst, W_dst)*hole_color(2), ...
                   ones(H_dst, W_dst)*hole_color(3));
    
    % 3. Vectorized Forward Map using transformPoints
    [X_src, Y_src] = meshgrid(1:W_src, 1:H_src);
    pts_in = [X_src(:)'; Y_src(:)'];
    pts_out = OceanMath.Geometry.Transform2D.transformPoints(H_mat, pts_in);
    
    xr = round(pts_out(1,:) + offset_x);
    yr = round(pts_out(2,:) + offset_y);
    
    % Filter valid hits
    valid = xr >= 1 & xr <= W_dst & yr >= 1 & yr <= H_dst;
    dst_idx = sub2ind([H_dst, W_dst], yr(valid), xr(valid));
    src_idx = find(valid);
    
    for c = 1:C
        ch_src = I_src(:,:,c);
        ch_dst = I_dst(:,:,c);
        ch_dst(dst_idx) = ch_src(src_idx);
        I_dst(:,:,c) = ch_dst;
    end
    
    % 4. Inverse Map to correctly clip out-of-bounds regions to black
    H_inv = inv(H_mat);
    [X_dst, Y_dst] = meshgrid(1:W_dst, 1:H_dst);
    pts_dst = [X_dst(:)' - offset_x; Y_dst(:)' - offset_y];
    pts_back = OceanMath.Geometry.Transform2D.transformPoints(H_inv, pts_dst);
    
    out_of_bounds = pts_back(1,:) < 1 | pts_back(1,:) > W_src | ...
                    pts_back(2,:) < 1 | pts_back(2,:) > H_src;
                    
    oob_idx = find(out_of_bounds);
    for c = 1:C
        ch_dst = I_dst(:,:,c);
        ch_dst(oob_idx) = 0; % Force areas outside the source canvas to black
        I_dst(:,:,c) = ch_dst;
    end
    
    % 5. Calculate precise hole fraction
    hit_mask = false(H_dst, W_dst);
    hit_mask(dst_idx) = true;
    
    valid_mask = true(H_dst, W_dst);
    valid_mask(oob_idx) = false;
    
    valid_hits = hit_mask(valid_mask);
    
    if isempty(valid_hits)
        hole_frac = 0;
    else
        hole_frac = sum(valid_hits == 0) / numel(valid_hits);
    end
    
    % 6. Band-wise spatial variance analysis
    hole_frac_bands = zeros(num_bands, 1);
    band_height = floor(H_dst / num_bands);
    
    for b = 1:num_bands
        y_start = (b - 1) * band_height + 1;
        y_end = b * band_height;
        if b == num_bands
            y_end = H_dst;
        end
        
        band_valid = valid_mask(y_start:y_end, :);
        band_hit = hit_mask(y_start:y_end, :);
        
        valid_band_hits = band_hit(band_valid);
        if isempty(valid_band_hits)
            hole_frac_bands(b) = NaN;
        else
            hole_frac_bands(b) = sum(valid_band_hits == 0) / numel(valid_band_hits);
        end
    end
end
