% EXP_BLUE_SCREEN_MATTING  Blue/Green Screen Matting and Compositing.
%
%   This script implements Exercise 3.4:
%     1. Loads a known background image (_DSC5156.JPG) and a composite image (_DSC5157.JPG).
%     2. Pulls the alpha matte using the Smith-Blinn / Vlahos difference formulation.
%     3. Performs color despilling to eliminate color fringes at the borders.
%     4. Blends the extracted foreground onto a new scenic background (ocean_img.jpg).
%     5. Visualizes and saves the matting dashboard.
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-07-01

clear; close all; clc;

disp('Initializing Blue Screen Matting Experiment...');
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Create Results folder if not exists
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Paths
bg_path = fullfile(root_dir, 'Data', 'images', '_DSC5156.JPG');
obj_path = fullfile(root_dir, 'Data', 'images', '_DSC5157.JPG');
scenic_path = fullfile(root_dir, 'Data', 'images', 'pepsi.png');
out_fig_path = fullfile(results_dir, 'matting_results.png');

if ~exist(bg_path, 'file') || ~exist(obj_path, 'file')
    error('Input images _DSC5156.JPG or _DSC5157.JPG are missing in Data/images/');
end

% 1. Load images
img_bg = double(imread(bg_path)) / 255.0;
img_obj = double(imread(obj_path)) / 255.0;

% Downsample to a reasonable resolution to prevent Out of Memory errors
img_bg = imresize(img_bg, [1000, NaN]);
img_obj = imresize(img_obj, [1000, NaN]);

% Check if scenic background exists, otherwise create a synthetic one
if exist(scenic_path, 'file')
    img_scenic = double(imread(scenic_path)) / 255.0;
else
    % Create a gradient sky-sea background
    [X, Y] = meshgrid(1:size(img_bg, 2), 1:size(img_bg, 1));
    img_scenic = cat(3, Y/size(img_bg,1), 0.5*ones(size(img_bg)), 1.0 - Y/size(img_bg,1));
end

%% 2. Perform Matting
disp('Instantiating matting engine...');
bsm = OceanMath.Image.Resampling.BlueScreenMatting(img_bg);

% Use Euclidean distance mode with tuned thresholds for background subtraction
alpha_min = 0.15;
alpha_max = 0.25;
despill_strength = 1.0;

disp('Pulling the alpha matte using Euclidean distance...');
[alpha, ~] = bsm.pullMatte(img_obj, ...
    'Mode', 'Euclidean', ...
    'AlphaMin', alpha_min, ...
    'AlphaMax', alpha_max);

% Clean up non-green border regions using a soft garbage matte
disp('Applying soft garbage matte to isolate the Coke can...');
[H, W, ~] = size(img_obj);
garbage_mask = zeros(H, W);
% Scale of the original bounding box (1:2800, 1200:4500) divided by 3.63
garbage_mask(1:770, 330:1238) = 1.0;
garbage_mask = imgaussfilt(garbage_mask, 5); % Scale down sigma accordingly
alpha = alpha .* garbage_mask;

% Re-calculate pre-multiplied foreground after garbage mask subtraction
foreground = img_obj - (1.0 - alpha) .* img_bg;
foreground = max(0.0, min(1.0, foreground));

% Apply green color despilling on the final foreground
R_fg = foreground(:,:,1);
G_fg = foreground(:,:,2);
B_fg = foreground(:,:,3);
limit = max(R_fg, B_fg);
excess = max(0.0, G_fg - limit);
G_fg = G_fg - despill_strength * excess;
foreground(:,:,2) = max(0.0, G_fg);

%% 3. Blend onto Scenic Background
disp('Compositing foreground onto the new background...');
composite_img = bsm.blendOn(foreground, alpha, img_scenic);

%% 4. Visualization using OceanViz Core
disp('Creating visualization dashboard...');
viz = OceanViz.Core('Theme', 'Dark');
f1 = figure('Name', 'Blue/Green Screen Matting Results', ...
            'Color', viz.Colors.Background, 'Position', [50, 50, 1600, 1000]);

% Panel 1: Original input image
ax1 = subplot(2, 2, 1);
imshow(img_obj);
title(ax1, 'Original Input Image with Object', 'Color', viz.Colors.Text);

% Panel 2: Extracted Alpha Matte
ax2 = subplot(2, 2, 2);
imshow(alpha);
colormap(ax2, 'gray');
title(ax2, sprintf('Extracted Alpha Matte (Min=%.2f, Max=%.2f)', alpha_min, alpha_max), 'Color', viz.Colors.Text);

% Panel 3: Despilled pre-multiplied foreground on black
ax3 = subplot(2, 2, 3);
imshow(foreground);
title(ax3, 'Extracted Foreground (Despilled)', 'Color', viz.Colors.Text);

% Panel 4: Final composite on new background
ax4 = subplot(2, 2, 4);
imshow(composite_img);
title(ax4, 'Composited Image on Pepsi Background', 'Color', viz.Colors.Text);

drawnow;
print(f1, out_fig_path, '-dpng', '-r110');

%% 5. Print Results Summary
fprintf('\n=== MATTING EXPERIMENT SUMMARY ===\n');
fprintf('  Background Image : _DSC5156.JPG\n');
fprintf('  Composite Image  : _DSC5157.JPG\n');
fprintf('  Image Dimensions : %d x %d\n', size(img_bg, 1), size(img_bg, 2));
fprintf('  Keying Thresholds: AlphaMin=%.2f, AlphaMax=%.2f\n', alpha_min, alpha_max);
fprintf('  Despilling       : Enabled (Strength=%.1f)\n', despill_strength);
fprintf('  Output Figures   : Saved successfully in Results/matting_results.png\n');
fprintf('==================================\n');
disp('SUCCESS: Blue screen matting completed.');
