% EXP_BLEND_FOOTBALL_TEAMS  Laplacian Pyramid blending of two football team photos.
%
%   DATASET:        Data/images/blend1.jfif and Data/images/blend2.jfif
%   MODEL:          LaplacianPyramid blending via multiresolution masks.
%   ASSUMPTIONS:    Both images are resized to preserve aspect ratio.
%                   They are placed side-by-side with a smooth overlap zone.
%                   8 pyramid levels for a wide, smooth seam.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Ocean Library / Antigravity
%   Date: 2026-06-25

clc; clear; rng(42, 'twister');

% === 1. Path Setup ===
script_dir = fileparts(mfilename('fullpath'));
root_dir   = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
out_path = fullfile(results_dir, 'football_blend.png');

% === 2. Load Images ===
p1 = fullfile(root_dir, 'Data', 'images', 'blend1.jfif');
p2 = fullfile(root_dir, 'Data', 'images', 'blend2.jfif');

if ~exist(p1,'file') || ~exist(p2,'file')
    error('Images blend1.jfif / blend2.jfif not found in Data/images/');
end

I1 = im2double(imread(p1));
I2 = im2double(imread(p2));

% === 3. Dimensions and Canvas Setup ===
% Standard size per image (preserving 3:4 aspect ratio of the 2890x2168 input)
H = 1024;
W = 768;

A = imresize(I1, [H, W]);
B = imresize(I2, [H, W]);

% Overlap width (256 pixels is 1/3 of the image width)
O = 256;
W_total = 2 * W - O; % 2 * 768 - 256 = 1280

% Canvas A (Left image, padded on the right with replicated edge)
A_canvas = [A, repmat(A(:, end, :), [1, W_total - W, 1])];

% Canvas B (Right image, padded on the left with replicated edge)
B_canvas = [repmat(B(:, 1, :), [1, W - O, 1]), B];

% === 4. Mask: Linear Gradient inside Overlap Zone ===
grad = linspace(1, 0, O); % 1 x O gradient
M = [ones(H, W - O), repmat(grad, [H, 1]), zeros(H, W_total - W)];
Mask = repmat(M, [1, 1, 3]);

% === 5. Build Pyramids ===
num_levels = 8;
disp('Building Laplacian Pyramid for Canvas A (Left)...');
PyrA = OceanMath.Image.MultiResolution.LaplacianPyramid();
PyrA.build(A_canvas, num_levels);

disp('Building Laplacian Pyramid for Canvas B (Right)...');
PyrB = OceanMath.Image.MultiResolution.LaplacianPyramid();
PyrB.build(B_canvas, num_levels);

disp('Building Gaussian Pyramid for Mask...');
PyrM = OceanMath.Image.MultiResolution.GaussianPyramid();
PyrM.build(Mask, num_levels);

% === 6. Blend at Each Level ===
disp('Blending pyramids level-by-level...');
PyrBlend = OceanMath.Image.MultiResolution.LaplacianPyramid();
PyrBlend.NumLevels = num_levels;
PyrBlend.Levels    = cell(1, num_levels);

for l = 1:num_levels
    LA = PyrA.Levels{l};
    LB = PyrB.Levels{l};
    GM = PyrM.Levels{l};
    PyrBlend.Levels{l} = GM .* LA + (1 - GM) .* LB;
end

% === 7. Reconstruct ===
disp('Reconstructing blended image...');
I_blend  = max(min(PyrBlend.reconstruct(), 1), 0);
I_splice = Mask .* A_canvas + (1 - Mask) .* B_canvas;  % Naive sharp splice for comparison

% === 8. Visualize ===
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Football Team Side-by-Side Blend', ...
             'Position', [100, 100, 1400, 800], ...
             'Color', viz.Colors.Background);

ax1 = subplot(2, 1, 1);
imshow(I_splice);
title('Naive Splice (Sharp Boundary)', 'Color', viz.Colors.Text, 'FontSize', 13);
viz.styleAxes(ax1);

ax2 = subplot(2, 1, 2);
imshow(I_blend);
title(sprintf('Laplacian Pyramid Panorama Blend (%d levels, %dpx Overlap)', num_levels, O), 'Color', viz.Colors.Text, 'FontSize', 13);
viz.styleAxes(ax2);

drawnow;
print(fig, out_path, '-dpng', '-r120');
fprintf('\nSaved: %s\n', out_path);

% === 9. Summary ===
fprintf('\n=== PANORAMA BLEND SUMMARY ===\n');
fprintf('  Image Size       : %dx%d each\n', W, H);
fprintf('  Overlap Width    : %d pixels\n', O);
fprintf('  Output Size      : %dx%d (Double-width Panorama)\n', W_total, H);
fprintf('  Pyramid Levels   : %d\n', num_levels);
fprintf('==============================\n');

