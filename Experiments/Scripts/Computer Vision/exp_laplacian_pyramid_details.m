% EXP_LAPLACIAN_PYRAMID_DETAILS  Burt and Adelson (1983b) Laplacian pyramid details (Figure 3.42).
%
%   DATASET:        Data/images/apple.jpg and Data/images/orange.jpg
%   MODEL:          LaplacianPyramid details visualization.
%   ASSUMPTIONS:    Input images are resized to 512x512.
%                   Levels 1, 3, and 5 are visualized after expanding them 
%                   back to the original resolution to match the paper's grid.
%                   The bottom row shows the fully reconstructed left and right 
%                   pyramid branches to illustrate the effective weighting function.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading & Alignment ===
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
apple_path = fullfile(root_dir, 'Data', 'images', 'apple.jpg');
orange_path = fullfile(root_dir, 'Data', 'images', 'orange.jpg');

I_apple = imread(apple_path);
I_orange = imread(orange_path);

H = 512; W = 512;
I_apple = imresize(I_apple, [H, W]);
I_orange = imresize(I_orange, [H, W]);

A = im2double(I_apple);
B = im2double(I_orange);

% Align to fix the contour gap
A = imtranslate(A, [0, -5], 'FillValues', 1);
B = imtranslate(B, [0, 0], 'FillValues', 1);

% === 2. Mask Generation ===
Mask = zeros(H, W, size(A, 3));
Mask(:, 1:floor(W/2), :) = 1;

% === 3. Pyramid Construction ===
num_levels = 6;
PyrA = OceanMath.Image.MultiResolution.LaplacianPyramid();
PyrA.build(A, num_levels);

PyrB = OceanMath.Image.MultiResolution.LaplacianPyramid();
PyrB.build(B, num_levels);

PyrM = OceanMath.Image.MultiResolution.GaussianPyramid();
PyrM.build(Mask, num_levels);

% === 4. Branch Pyramids (Left and Right isolated) ===
% We build two separate Laplacian pyramids that represent the masked contributions
BranchA = OceanMath.Image.MultiResolution.LaplacianPyramid();
BranchA.NumLevels = num_levels;
BranchA.Levels = cell(1, num_levels);

BranchB = OceanMath.Image.MultiResolution.LaplacianPyramid();
BranchB.NumLevels = num_levels;
BranchB.Levels = cell(1, num_levels);

for l = 1:num_levels
    GM = PyrM.Levels{l};
    BranchA.Levels{l} = GM .* PyrA.Levels{l};
    BranchB.Levels{l} = (1 - GM) .* PyrB.Levels{l};
end

% === 5. Expansion Helper ===
% To visualize level L at full resolution, we must expand it L-1 times
expand_to_base = @(img, level) expand_loop(img, level, 'replicate', PyrA.Kernel);

% === 6. Visualization Grid ===
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Figure 3.42 Laplacian Pyramid Details', 'Position', [50, 50, 1000, 1000]);
set(fig, 'Color', viz.Colors.Background);

levels_to_show = [1, 3, 5]; % Corresponding to levels 0, 2, 4 in the paper (0-indexed)
plot_idx = 1;

for i = 1:length(levels_to_show)
    l = levels_to_show(i);
    
    % Get the masked laplacian levels
    LA_masked = BranchA.Levels{l};
    LB_masked = BranchB.Levels{l};
    LBlend = LA_masked + LB_masked;
    
    % Expand them to full resolution for visualization
    LA_full = expand_to_base(LA_masked, l);
    LB_full = expand_to_base(LB_masked, l);
    LBlend_full = expand_to_base(LBlend, l);
    
    % Add 0.5 to center the zero-mean laplacian around gray
    LA_vis = max(min(LA_full + 0.5, 1), 0);
    LB_vis = max(min(LB_full + 0.5, 1), 0);
    LBlend_vis = max(min(LBlend_full + 0.5, 1), 0);
    
    % Left Column: Apple masked
    ax = subplot(4, 3, plot_idx); plot_idx = plot_idx + 1;
    imshow(LA_vis);
    if i == 1, title('Masked Apple (L_A)', 'Color', viz.Colors.Text); end
    ylabel(sprintf('Level %d', l-1), 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    viz.styleAxes(ax);
    
    % Middle Column: Orange masked
    ax = subplot(4, 3, plot_idx); plot_idx = plot_idx + 1;
    imshow(LB_vis);
    if i == 1, title('Masked Orange (L_B)', 'Color', viz.Colors.Text); end
    viz.styleAxes(ax);
    
    % Right Column: Blended
    ax = subplot(4, 3, plot_idx); plot_idx = plot_idx + 1;
    imshow(LBlend_vis);
    if i == 1, title('Blended (L_{blend})', 'Color', viz.Colors.Text); end
    viz.styleAxes(ax);
end

% 4th Row: Reconstructed Full Images
I_rec_A = max(min(BranchA.reconstruct(), 1), 0);
I_rec_B = max(min(BranchB.reconstruct(), 1), 0);
I_rec_Blend = max(min(I_rec_A + I_rec_B, 1), 0);

ax = subplot(4, 3, 10);
imshow(I_rec_A);
title('Reconstructed Apple Branch', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(4, 3, 11);
imshow(I_rec_B);
title('Reconstructed Orange Branch', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

ax = subplot(4, 3, 12);
imshow(I_rec_Blend);
title('Final Blended Orapple', 'Color', viz.Colors.Text);
viz.styleAxes(ax);

fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Script           : exp_laplacian_pyramid_details.m\n');
fprintf('  Output           : Replicates Figure 3.42 grid of bandpass details\n');
fprintf('==========================\n');

% Helper function to expand an image recursively
function out = expand_loop(img, current_level, boundary, kernel)
    out = img;
    for k = current_level:-1:2
        % Target size doubles at each step
        sz = size(out);
        target_size = [sz(1)*2, sz(2)*2, sz(3)];
        out = OceanMath.Image.MultiResolution.PyramidOperator.expand(out, target_size, boundary, kernel);
    end
end
