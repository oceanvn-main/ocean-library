% exp_feature_morphing.m
% =========================================================================
% Experiment: Feature-Based Image Morphing (Szeliski Ex 3.25)
% =========================================================================
%
% Implements the 4-step cross-dissolve morphing pipeline (Sec 3.6.3):
%   Step 1. Build Delaunay triangulation from control points
%   Step 2. Interpolate: pts_mid(t) = (1-t)*pts_A + t*pts_B
%   Step 3. Inverse-warp both images to the intermediate skeleton
%   Step 4. Cross-dissolve: I_morph = (1-t)*I_A_warp + t*I_B_warp
%
% Part 1: Cross-dissolve morph across 3 warp engines (9 frames each)
% Part 2: Image rotation via morphing -- chord-vs-arc artifact analysis
%
% ASSUMPTION: Images are resized to 256x256. Control points are defined
%   as pixel coordinates [x, y] (column, row) in the image space.
%
% REPRODUCIBILITY: All random components are deterministic. No training
%   data is used; all morphs are derived purely from input images and
%   explicit geometric correspondences.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-16
% =========================================================================

clear; close all; clc;
addpath(genpath('c:\Ocean Library\Matlab code practice'));
cd('c:\Ocean Library\Matlab code practice');

rng(42);  % For reproducibility

%% =========================================================================
% DATA LOADING
% =========================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
img_dir      = fullfile(project_root, 'Data', 'images');

TARGET_SZ = [256, 256];

I_A_raw = imread(fullfile(img_dir, 'ocean_img.jpg'));
I_B_raw = imread(fullfile(img_dir, 'elon_musk.jpg'));

I_A = double(imresize(I_A_raw, TARGET_SZ)) / 255;
I_B = double(imresize(I_B_raw, TARGET_SZ)) / 255;

% If grayscale, replicate to 3 channels
if size(I_A, 3) == 1, I_A = repmat(I_A, 1, 1, 3); end
if size(I_B, 3) == 1, I_B = repmat(I_B, 1, 1, 3); end

fprintf('Images loaded: I_A %dx%dx%d, I_B %dx%dx%d\n', ...
    size(I_A, 1), size(I_A, 2), size(I_A, 3), ...
    size(I_B, 1), size(I_B, 2), size(I_B, 3));

%% =========================================================================
% PART 1: CROSS-DISSOLVE MORPHING DASHBOARD
% =========================================================================
% Generate a 4x4 uniform correspondence grid as source points.
% Both source and target start identical; 4 central points are manually
% offset to demonstrate a non-trivial deformation.
%
% NOTE: In the interactive app (Tab 9), the user drags these points
%   interactively. Here we define a fixed example for reproducibility.
% =========================================================================

% --- Build correspondence grid (pts_A == uniform grid) ---
[pts_A, ~] = OceanMath.Image.Resampling.ImageMorpher.uniformGridPoints( ...
    TARGET_SZ, 4);

% --- Define target points (pts_B) by manually offsetting 4 central points ---
% This simulates a user dragging control points in Image B
% to warp the nose/eye region of a face (illustrative, not semantic).
pts_B = pts_A;  % start identical
% Points near image center (approximately index 6, 7, 10, 11 in a 4x4 grid)
center_indices = [6, 7, 10, 11];
offsets = [0, 20; 0, 20; 0, -20; 0, -20];  % [dx, dy]
for k = 1:numel(center_indices)
    idx = center_indices(k);
    pts_B(idx, :) = pts_B(idx, :) + offsets(k, :);
end

fprintf('Control pairs: %d\n', size(pts_A, 1));

% --- Morph configurations ---
warp_modes = {'PiecewiseAffine', 'ThinPlateSpline', 'BSplineGrid'};
mode_labels = {'Piecewise Affine', 'Thin Plate Spline', 'B-Spline FFD'};
N_frames    = 9;
t_vals      = linspace(0, 1, N_frames);
grid_res    = [4, 4];

%% Compute morph sequences
fprintf('\n[Part 1] Computing morph sequences...\n');
morph_frames = cell(numel(warp_modes), N_frames);

for m = 1:numel(warp_modes)
    fprintf('  Mode: %-20s ', mode_labels{m});
    t_start = tic;
    for f = 1:N_frames
        morph_frames{m, f} = OceanMath.Image.Resampling.ImageMorpher.morphFrame( ...
            I_A, I_B, pts_A, pts_B, t_vals(f), ...
            'WarpMode', warp_modes{m}, ...
            'GridRes',  grid_res);
    end
    fprintf('[%.2f s]\n', toc(t_start));
end

%% Plot Part 1 Dashboard
viz = OceanViz.Core('Theme', 'Dark');

fig1 = figure('Name', 'Ex 3.25 Part 1 - Cross-Dissolve Morphing Dashboard', ...
    'Position', [50, 50, 1400, 480], ...
    'Color', [0.11, 0.11, 0.13]);

n_modes = numel(warp_modes);

for m = 1:n_modes
    for f = 1:N_frames
        ax = subplot(n_modes, N_frames, (m-1)*N_frames + f);
        imshow(morph_frames{m, f}, 'Parent', ax);
        
        if f == 1
            % Row label on left
            ylabel(ax, mode_labels{m}, ...
                'FontSize', 8, 'FontWeight', 'bold', ...
                'Color', [0.90, 0.90, 0.90], ...
                'Rotation', 90, 'HorizontalAlignment', 'center');
        end
        if m == 1
            % Column label on top (t value)
            title(ax, sprintf('t=%.2f', t_vals(f)), ...
                'FontSize', 7, 'Color', [0.20, 0.60, 1.00]);
        else
            title(ax, '');
        end
        ax.XTick = []; ax.YTick = [];
        ax.Color = [0.11, 0.11, 0.13];
    end
end

sgtitle(fig1, 'Ex 3.25 — Feature-Based Morphing: Cross-Dissolve (9 frames, 3 warp engines)', ...
    'Color', [0.90, 0.90, 0.90], 'FontSize', 13, 'FontWeight', 'bold');

%% Save Part 1 figure
results_dir = fullfile(project_root, 'Results', 'Computer Vision');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
out1 = fullfile(results_dir, 'ex325_part1_crossdissolve_dashboard.png');
exportgraphics(fig1, out1, 'Resolution', 150);
fprintf('\n[Part 1] Dashboard saved: %s\n', out1);

%% =========================================================================
% PART 2: IMAGE ROTATION VIA MORPHING — ARTIFACT ANALYSIS
% =========================================================================
% Hypothesis: Linear interpolation of Cartesian positions traces straight-
%   line chords between source and target positions, not circular arcs.
%   At t = 0.5, the chord midpoint for a rotation of theta is at radius
%       r_chord = r * cos(theta/2)
%   while the true midpoint on the arc is at radius r (unchanged).
%   This causes pixels to collapse inward ("implode") at t = 0.5.
% =========================================================================

ROT_ANGLES  = [30, 60, 90];    % degrees to test
N_ROT_FRAMES = 5;              % t = 0, 0.25, 0.5, 0.75, 1
t_rot = linspace(0, 1, N_ROT_FRAMES);
rot_grid_res = 8;

fprintf('\n[Part 2] Computing rotation morphs...\n');

% Use a single source image (Image A)
I_rot = I_A;

% Compute error analysis at t = 0.5 for all angles
error_maps = cell(1, numel(ROT_ANGLES));
chord_radii = zeros(1, numel(ROT_ANGLES));
true_radii  = zeros(1, numel(ROT_ANGLES));

fig2 = figure('Name', 'Ex 3.25 Part 2 - Rotation via Morphing', ...
    'Position', [50, 600, 1400, 680], ...
    'Color', [0.11, 0.11, 0.13]);

n_angles = numel(ROT_ANGLES);

for a = 1:n_angles
    theta_deg = ROT_ANGLES(a);
    fprintf('  Rotation: %d deg\n', theta_deg);
    
    [pA_rot, pB_rot] = OceanMath.Image.Resampling.ImageMorpher.rotationControlPoints( ...
        TARGET_SZ, theta_deg, rot_grid_res);
    
    % Compute morph frames at 5 t values
    rot_frames = cell(1, N_ROT_FRAMES);
    for f = 1:N_ROT_FRAMES
        rot_frames{f} = OceanMath.Image.Resampling.ImageMorpher.morphFrame( ...
            I_rot, I_rot, pA_rot, pB_rot, t_rot(f), ...
            'WarpMode', 'PiecewiseAffine');
    end
    
    % ---- Radial error analysis at t = 0.5 ----
    % For each control point, compute:
    %   true_mid   = center + r (ideally: stays at radius r)
    %   chord_mid  = (pA + pB) / 2 = center + r*cos(theta/2) * unit_direction
    cx = (TARGET_SZ(2) + 1) / 2;
    cy = (TARGET_SZ(1) + 1) / 2;
    dx = pA_rot(:,1) - cx;
    dy = pA_rot(:,2) - cy;
    r  = sqrt(dx.^2 + dy.^2);
    
    theta_rad = deg2rad(theta_deg);
    pts_mid_actual = (pA_rot + pB_rot) / 2;
    d_mid  = sqrt((pts_mid_actual(:,1) - cx).^2 + (pts_mid_actual(:,2) - cy).^2);
    r_ideal = r;  % true arc midpoint radius equals source radius
    
    radial_error = abs(d_mid - r_ideal);
    chord_radii(a) = mean(d_mid(r > 5));    % average chord midpoint radius
    true_radii(a)  = mean(r_ideal(r > 5));  % average true radius
    
    % ---- Row a: morph frames ----
    for f = 1:N_ROT_FRAMES
        ax = subplot(n_angles, N_ROT_FRAMES + 1, (a-1)*(N_ROT_FRAMES+1) + f);
        imshow(rot_frames{f}, 'Parent', ax);
        title(ax, sprintf('t=%.2f', t_rot(f)), 'FontSize', 7, 'Color', [0.20, 0.60, 1.00]);
        if f == 1
            ylabel(ax, sprintf('%d deg', theta_deg), ...
                'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.90, 0.90, 0.90]);
        end
        ax.XTick = []; ax.YTick = [];
        ax.Color = [0.11, 0.11, 0.13];
    end
    
    % ---- Rightmost column: radial error scatter at t = 0.5 ----
    ax_err = subplot(n_angles, N_ROT_FRAMES + 1, (a-1)*(N_ROT_FRAMES+1) + N_ROT_FRAMES + 1);
    scatter(ax_err, r_ideal, d_mid, 40, radial_error, 'filled');
    colormap(ax_err, hot);
    colorbar(ax_err, 'Color', [0.55, 0.58, 0.63], 'FontSize', 7);
    hold(ax_err, 'on');
    % Ideal line: chord_radius = r * cos(theta/2)
    r_line = linspace(0, max(r_ideal) * 1.1, 50);
    plot(ax_err, r_line, r_line * cos(theta_rad/2), '--', ...
        'Color', [1.00, 0.40, 0.40], 'LineWidth', 1.5);
    plot(ax_err, r_line, r_line, '-', 'Color', [0.20, 0.60, 1.00], 'LineWidth', 1.0);
    hold(ax_err, 'off');
    
    xlabel(ax_err, 'r_{true} (px)', 'Color', [0.55, 0.58, 0.63], 'FontSize', 7);
    ylabel(ax_err, 'r_{chord} (px)', 'Color', [0.55, 0.58, 0.63], 'FontSize', 7);
    title(ax_err, sprintf('Radial error at t=0.5\nchord/true = %.3f', cos(theta_rad/2)), ...
        'FontSize', 7, 'Color', [1.00, 0.85, 0.20]);
    ax_err.Color        = [0.11, 0.11, 0.13];
    ax_err.XColor       = [0.55, 0.58, 0.63];
    ax_err.YColor       = [0.55, 0.58, 0.63];
    ax_err.FontSize     = 7;
    
    legend(ax_err, {'samples', sprintf('theory: r*cos(theta/2)', theta_deg), 'ideal (r=r)'}, ...
        'FontSize', 6, 'TextColor', [0.90, 0.90, 0.90], ...
        'Color', [0.17, 0.19, 0.23], 'EdgeColor', [0.35, 0.37, 0.42], ...
        'Location', 'northwest');
end

sgtitle(fig2, 'Ex 3.25 Part 2 — Rotation via Morphing: Chord-vs-Arc Artifact Analysis', ...
    'Color', [0.90, 0.90, 0.90], 'FontSize', 12, 'FontWeight', 'bold');

%% Save Part 2 figure
out2 = fullfile(results_dir, 'ex325_part2_rotation_artifact.png');
exportgraphics(fig2, out2, 'Resolution', 150);
fprintf('[Part 2] Dashboard saved: %s\n', out2);

%% =========================================================================
% PART 2: Discussion — printed to console
% =========================================================================
fprintf('\n\n== Discussion: Rotation via Morphing ==\n');
fprintf('========================================\n');
fprintf('Rotation angle (deg) | Chord/True radius at t=0.5 | cos(theta/2)\n');
fprintf('------------------------------------------------------------------\n');
for a = 1:n_angles
    theta_deg = ROT_ANGLES(a);
    fprintf('  %3d deg             | %.4f                    | %.4f\n', ...
        theta_deg, chord_radii(a)/true_radii(a), cos(deg2rad(theta_deg)/2));
end
fprintf('\n');
fprintf('Observation:\n');
fprintf('  The morphing algorithm linearly interpolates CARTESIAN coordinates,\n');
fprintf('  so at t=0.5 pixels follow the midpoint of the straight chord between\n');
fprintf('  source and rotated positions rather than the midpoint of the arc.\n');
fprintf('  The chord midpoint lies at radius r*cos(theta/2) < r,\n');
fprintf('  causing all pixels to pull INWARD at the middle of the morph.\n\n');
fprintf('  For small angles (30 deg): chord/true ~ %.3f (barely visible)\n', cos(deg2rad(30)/2));
fprintf('  For 90 deg:  chord/true ~ %.3f (strong implosion at t=0.5)\n', cos(deg2rad(90)/2));
fprintf('  For 180 deg: chord/true ~ %.3f (all pixels collapse to center)\n\n', cos(deg2rad(180)/2));
fprintf('  Fix: Interpolate the rotation angle directly (theta(t) = t*theta)\n');
fprintf('  and apply the rotation analytically, rather than linearly mixing\n');
fprintf('  Cartesian coordinates of the control grid.\n');
fprintf('  This is equivalent to geodesic interpolation on SO(2).\n');

fprintf('\n[Experiment Complete]\n');
