% EXP_LOCAL_LAPLACIAN  Local Laplacian contrast manipulation, edge-preserving filtering, and tone manipulation (Ex 3.20).
%
%   DATASET:        ongtrung.jpg
%   MODEL:          Local Laplacian Filters (Paris et al. 2011) via fast discretized bin interpolation.
%   ASSUMPTIONS:    Input images are normalized to [0, 1] double.
%                   For color images, the filter is applied only to the L* (luminance) channel in CIE L*a*b* space.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Antigravity Code Assistant
%   Date: 2026-07-15

clc; clear; rng(42, 'twister');
close all;

% === 1. Setup Paths and Data Loading ===
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

image_path = fullfile(project_root, 'Data', 'images', 'ongtrung.jpg');
if ~exist(image_path, 'file')
    error('ongtrung.jpg not found in Data/images/');
end

I_orig = imread(image_path);
I_orig = im2double(I_orig);

% === 2. Instantiate and Run Local Laplacian Configurations ===

% Config 1: Detail Enhancement (Alpha < 1, Beta = 1, Sigma = 0.1)
% This boosts fine textures and micro-contrast without introducing halos or noise.
disp('Running Local Laplacian: Detail Enhancement (Alpha = 0.25)...');
llf_enhance = OceanMath.Image.MultiResolution.LocalLaplacianFilter(...
    'NumLevels', 4, 'NumBins', 8, 'Sigma', 0.1, 'Alpha', 0.5, 'Beta', 1.0);
I_enhance = llf_enhance.filter(I_orig);

% Config 2: Detail Smoothing / Edge-Preserving Denoising (Alpha > 1, Beta = 1, Sigma = 0.15)
% This smooths out skin and small details while preserving sharp outline boundaries (similar to bilateral filtering).
disp('Running Local Laplacian: Detail Smoothing / Denoising (Alpha = 2.0)...');
llf_smooth = OceanMath.Image.MultiResolution.LocalLaplacianFilter(...
    'NumLevels', 4, 'NumBins', 8, 'Sigma', 0.15, 'Alpha', 2.0, 'Beta', 1.0);
I_smooth = llf_smooth.filter(I_orig);

% Config 3: Tone Mapping / HDR Dynamic Range Compression (Alpha = 1, Beta < 1, Sigma = 0.2)
% This compresses large-scale edge contrast to bring out shadow and highlight details while maintaining local textures.
disp('Running Local Laplacian: Tone Mapping / Edge Compression (Beta = 0.25)...');
llf_tone = OceanMath.Image.MultiResolution.LocalLaplacianFilter(...
    'NumLevels', 4, 'NumBins', 8, 'Sigma', 0.2, 'Alpha', 1.0, 'Beta', 0.25);
I_tone = llf_tone.filter(I_orig);

% === 3. Visualization Dashboard ===
disp('Rendering Local Laplacian Dashboard...');
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'Local Laplacian Contrast Manipulation (Ex 3.20)', 'Position', [100, 100, 1000, 850]);
set(fig, 'Color', viz.Colors.Background);

% Subplot (a): Original Image
ax1 = subplot(2, 2, 1);
imshow(I_orig);
title('(a) Original Image', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
viz.styleAxes(ax1);

% Subplot (b): Detail Enhancement
ax2 = subplot(2, 2, 2);
imshow(I_enhance);
title('(b) Detail Enhancement (\alpha = 0.25)', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
viz.styleAxes(ax2);

% Subplot (c): Edge-Preserving Smoothing / Denoising
ax3 = subplot(2, 2, 3);
imshow(I_smooth);
title('(c) Edge-Preserving Smoothing (\alpha = 2.0)', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
viz.styleAxes(ax3);

% Subplot (d): Tone Mapping
ax4 = subplot(2, 2, 4);
imshow(I_tone);
title('(d) Tone Mapping / Edge Compression (\beta = 0.25)', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
viz.styleAxes(ax4);

% Add main figure supertitle
sgtitle('Local Laplacian Filters (Paris et al. 2011) — Exercise 3.20', ...
        'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% Save Dashboard output
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_path = fullfile(results_dir, 'local_laplacian_results.png');
saveas(fig, output_path);

% === 4. Plot Intermediate Pyramids from the Filter Object ===
disp('Plotting Local Laplacian pyramids for Detail Enhancement...');
fig_pyr = llf_enhance.plotPyramids();
pyr_output_path = fullfile(results_dir, 'local_laplacian_pyramids.png');
saveas(fig_pyr, pyr_output_path);

% Call report on the input Laplacian pyramid to describe its overcompleteness
llf_enhance.InputLaplacianPyramid.report();

% === 5. Summary printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset          : ongtrung.jpg\n');
fprintf('  Resolution       : %dx%d\n', size(I_orig, 1), size(I_orig, 2));
fprintf('  Pyramid Levels   : %d\n', llf_enhance.NumLevels);
fprintf('  Intensity Bins   : %d\n', llf_enhance.NumBins);
fprintf('  Luminance Space  : CIE L*a*b* (L channel)\n');
fprintf('  Results Saved to : %s\n', output_path);
fprintf('  Pyramids Plot    : %s\n', pyr_output_path);
fprintf('==========================\n');
