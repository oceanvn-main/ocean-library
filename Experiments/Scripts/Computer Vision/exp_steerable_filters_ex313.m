% exp_steerable_filters_ex313.m
%
% Scientific Experiment: Steerable Filter Orientations and Edge/Corner Detection
% (Szeliski Exercise 3.13)
%
% Description:
%   This script implements and validates Szeliski Exercise 3.13 using Freeman and
%   Adelson's (1991) steerable filter algorithm. It:
%     1. Generates a synthetic geometric calibration image.
%     2. Computes the steerable basis responses for Order 1 (G1) and Order 2 (G2).
%     3. Validates O(1) steering responses at arbitrary angles (0, 45, 90 deg).
%     4. Computes dominant orientations and magnitudes.
%     5. Evaluates 1D angular response profiles (steerability curves) at key points.
%
% Scientific Principles (RULE[experiment.md]):
%   - Explicit Computation: All values derived programmatically.
%   - Reproducibility: Clean path initialization.
%   - No LaTeX: Labels and legends use ASCII plain-text notations.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-13

clear;
close all;
clc;

% Ensure paths are set
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

% Initialize OceanViz Core Engine (using Scientific theme)
viz = OceanViz.Core('Theme', 'Scientific');

%% 1. Generate Synthetic Geometric Calibration Image (256x256)
H = 256;
W = 256;
[X, Y] = meshgrid(1:W, 1:H);
I = zeros(H, W);

% Add Circle
I = I + double((X - 128).^2 + (Y - 128).^2 <= 50^2);
% Add Square
I = I + double(X >= 64 & X <= 192 & Y >= 64 & Y <= 192);
% Add Diagonal Line (45 deg)
I = I + double(abs(X - Y) <= 2);
% Add Steeper Diagonal Line (approx 26.5 deg)
I = I + double(abs(X - 2*Y + 128) <= 2);

I_in = max(0, min(1, I));

%% 2. Instantiate and Compute Steerable Responses
fprintf('Applying Steerable Filters...\n');

% First-order (G1) Filter
sf1 = OceanMath.Image.Pixels.SteerableFilter('Sigma', 2.0, 'Order', 1, 'Size', [15, 15]);
sf1.apply(I_in);

% Second-order (G2) Filter
sf2 = OceanMath.Image.Pixels.SteerableFilter('Sigma', 2.0, 'Order', 2, 'Size', [15, 15]);
sf2.apply(I_in);

% Extract Steered Responses for G1
R1_0 = sf1.steer(0);
R1_45 = sf1.steer(45);
R1_90 = sf1.steer(90);

% Compute dominant orientation and magnitude fields
[theta_g1, mag_g1] = sf1.computeDominantOrientation();
[theta_g2, mag_g2] = sf2.computeDominantOrientation();

%% 3. Evaluate Angular Responses (Steerability Curves) at Key Points
% Let's sample three coordinates:
% Point A: Horizontal edge point (y = 64, x = 128) -> expects max response at 90 deg (y-deriv)
% Point B: Vertical edge point (y = 128, x = 64)   -> expects max response at 0 deg (x-deriv)
% Point C: Diagonal line point (y = 128, x = 128)   -> expects max response at 45 deg
ptA = [64, 128];
ptB = [128, 64];
ptC = [128, 128];

angles = 0:360;
respA = zeros(size(angles));
respB = zeros(size(angles));
respC = zeros(size(angles));

for a_idx = 1:length(angles)
    R_steer = sf1.steer(angles(a_idx));
    respA(a_idx) = abs(R_steer(ptA(1), ptA(2)));
    respB(a_idx) = abs(R_steer(ptB(1), ptB(2)));
    respC(a_idx) = abs(R_steer(ptC(1), ptC(2)));
end

%% 4. Render Dashboard (3x3 Layout)
fig = viz.createFigure('Exercise 3.13: Steerable Filter Analysis');
set(fig, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);

% Colormap for rendering response fields
cmap = jet(256);

% Subplot 1: Original Image
ax1 = subplot(3, 3, 1); imshow(I_in);
title(ax1, '1. Original Image (Geometric Grid)', 'Color', viz.Colors.Text);
set(ax1, 'Color', viz.Colors.Background);

% Subplot 2: G1 response at 0 degrees (vertical edges / x-deriv)
ax2 = subplot(3, 3, 2); imshow(R1_0, []); colormap(ax2, cmap);
title(ax2, '2. G1 Response (0 deg - Vertical Edges)', 'Color', viz.Colors.Text);
set(ax2, 'Color', viz.Colors.Background);

% Subplot 3: G1 response at 45 degrees
ax3 = subplot(3, 3, 3); imshow(R1_45, []); colormap(ax3, cmap);
title(ax3, '3. G1 Response (45 deg - Diagonal Edges)', 'Color', viz.Colors.Text);
set(ax3, 'Color', viz.Colors.Background);

% Subplot 4: G1 response at 90 degrees (horizontal edges / y-deriv)
ax4 = subplot(3, 3, 4); imshow(R1_90, []); colormap(ax4, cmap);
title(ax4, '4. G1 Response (90 deg - Horizontal Edges)', 'Color', viz.Colors.Text);
set(ax4, 'Color', viz.Colors.Background);

% Subplot 5: G1 Dominant Orientation (gradient direction)
ax5 = subplot(3, 3, 5); imshow(theta_g1, [-180, 180]); colormap(ax5, hsv(256));
title(ax5, '5. G1 Dominant Orientation Field', 'Color', viz.Colors.Text);
set(ax5, 'Color', viz.Colors.Background);

% Subplot 6: G1 Response Magnitude (gradient magnitude)
ax6 = subplot(3, 3, 6); imshow(mag_g1, []); colormap(ax6, cmap);
title(ax6, '6. G1 Gradient Magnitude', 'Color', viz.Colors.Text);
set(ax6, 'Color', viz.Colors.Background);

% Subplot 7: G2 Dominant Orientation
ax7 = subplot(3, 3, 7); imshow(theta_g2, [-90, 90]); colormap(ax7, hsv(256));
title(ax7, '7. G2 Dominant Orientation Field', 'Color', viz.Colors.Text);
set(ax7, 'Color', viz.Colors.Background);

% Subplot 8: G2 Magnitude
ax8 = subplot(3, 3, 8); imshow(mag_g2, []); colormap(ax8, cmap);
title(ax8, '8. G2 Filter Magnitude (Corners & Lines)', 'Color', viz.Colors.Text);
set(ax8, 'Color', viz.Colors.Background);

% Subplot 9: Steerability Curves (1D Angular Responses)
ax9 = subplot(3, 3, 9);
hold(ax9, 'on');
p_horiz = plot(ax9, angles, respA, 'Color', viz.Colors.Primary, 'LineWidth', 1.5);
p_vert  = plot(ax9, angles, respB, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5);
p_diag  = plot(ax9, angles, respC, 'Color', viz.Colors.Boundary, 'LineWidth', 1.5);
hold(ax9, 'off');
xlim(ax9, [0, 360]);
xlabel(ax9, 'Steering Angle (Degrees)');
ylabel(ax9, 'Absolute Response');
viz.styleAxes(ax9);
title(ax9, '9. 1D Angular Response at Key Points', 'Color', viz.Colors.Text);

lgd = legend(ax9, [p_horiz, p_vert, p_diag], ...
    {'Horizontal Edge Point', 'Vertical Edge Point', 'Diagonal Line Point'}, ...
    'Location', 'southoutside');
set(lgd, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% Super Title
sgtitle(fig, 'Szeliski Exercise 3.13: Freeman and Adelson Steerable Filter Verification', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

%% 5. Save Results
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_fig_path = fullfile(results_dir, 'exercise_3_13_results.png');
fprintf('Saving results visualization to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

fprintf('Experiment complete!\n');
