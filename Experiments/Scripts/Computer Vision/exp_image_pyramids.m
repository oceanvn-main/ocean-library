% exp_image_pyramids.m
%
% Scientific Experiment: Multi-resolution Image Pyramids (Gaussian & Laplacian)
%
% Description:
%   This script evaluates and visualizes Gaussian and Laplacian pyramids
%   on the 'elon_musk.jpg' image. It verifies that the Laplacian pyramid
%   achieves perfect reconstruction and plots the scale levels on a dark-themed
%   dashboard.
%
% Author: Antigravity Code Assistant
%   Date: 2026-06-19

clear;
close all;
clc;

% Import library classes
import OceanMath.Image.MultiResolution.GaussianPyramid
import OceanMath.Image.MultiResolution.LaplacianPyramid
import OceanViz.Core

%% 1. Set Up Directories and File Paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
data_dir = fullfile(project_root, 'Data', 'images');
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

image_path = fullfile(data_dir, 'elon_musk.jpg');
output_path = fullfile(results_dir, 'image_pyramids_dashboard.png');

if ~isfile(image_path)
    error('exp_image_pyramids:ImageNotFound', 'Could not locate Elon Musk image at: %s', image_path);
end

%% 2. Load and Prepare Image
I_raw = imread(image_path);
I_dbl = double(I_raw) / 255.0;

% Resize to height 512 for manageable processing and visualization
I_resized = imresize(I_dbl, [512, NaN]);
[H, W, C] = size(I_resized);

fprintf('Loaded image dimensions: %dx%dx%d\n', H, W, C);

%% 3. Build Pyramids
num_levels = 5;
fprintf('Building %d-level Gaussian and Laplacian pyramids...\n', num_levels);

% Build Gaussian Pyramid
gp = GaussianPyramid('Boundary', 'replicate');
gp.build(I_resized, num_levels);

% Build Laplacian Pyramid
lp = LaplacianPyramid('Boundary', 'replicate');
lp.build(I_resized, num_levels);

%% 4. Reconstruct and Verify Perfect Reconstruction
fprintf('Reconstructing original image from Laplacian pyramid...\n');
I_rec = lp.reconstruct();

% Compute reconstruction error metrics
max_err = max(abs(I_resized(:) - I_rec(:)));
mean_sq_err = mean((I_resized(:) - I_rec(:)).^2);

fprintf('Reconstruction Verification:\n');
fprintf('  Max Absolute Error: %e\n', max_err);
fprintf('  Mean Squared Error: %e\n', mean_sq_err);

if max_err < 1e-12
    fprintf('  SUCCESS: Perfect reconstruction verified (error < 1e-12).\n');
else
    warning('exp_image_pyramids:ReconstructionError', 'Reconstruction error is higher than expected.');
end

%% 5. Render OceanViz Dashboard
fprintf('Rendering pyramids dashboard...\n');
viz = Core('Theme', 'Dark');
fig = viz.createFigure('Multi-resolution Image Pyramids: Gaussian & Laplacian');
set(fig, 'Position', [100, 100, 1400, 800]);

% Layout: 2 rows, 5 columns
% Row 1: Gaussian levels
% Row 2: Laplacian levels
ax_gp = cell(1, num_levels);
ax_lp = cell(1, num_levels);

for level = 1:num_levels
    % Gaussian subplots
    ax_gp{level} = subplot(2, num_levels, level);
    viz.styleAxes(ax_gp{level});
    
    imshow(gp.Levels{level}, 'Parent', ax_gp{level});
    [h_l, w_l, ~] = size(gp.Levels{level});
    title(ax_gp{level}, sprintf('Gaussian L%d (%dx%d)', level, h_l, w_l), ...
          'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Laplacian subplots
    ax_lp{level} = subplot(2, num_levels, num_levels + level);
    viz.styleAxes(ax_lp{level});
    
    % For detail levels (1 to num_levels-1), offset by 0.5 and boost contrast
    % to make detail bands clearly visible.
    if level < num_levels
        % Laplacian detail scaled by 2.0 and offset to 0.5 mid-gray
        I_lp_viz = max(0.0, min(1.0, 2.0 * lp.Levels{level} + 0.5));
        imshow(I_lp_viz, 'Parent', ax_lp{level});
        title(ax_lp{level}, sprintf('Laplacian L%d Detail', level), ...
              'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    else
        % The last level is a low-frequency Gaussian residual
        imshow(lp.Levels{level}, 'Parent', ax_lp{level});
        title(ax_lp{level}, sprintf('Laplacian L%d Residual', level), ...
              'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    end
end

% Add main window header with reconstruction error info
header_str = sprintf('IMAGE PYRAMIDS STUDY (Max Reconstruction Error: %e)', max_err);
annotation(fig, 'textbox', [0.05, 0.94, 0.9, 0.05], ...
    'String', header_str, ...
    'Color', 'w', 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'LineStyle', 'none');

drawnow;

% Save Figure
saveas(fig, output_path);
fprintf('Pyramids dashboard saved successfully to: %s\n', output_path);
