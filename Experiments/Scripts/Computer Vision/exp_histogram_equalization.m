% exp_histogram_equalization.m
%
% Scientific Experiment: Histogram Analysis and Equalization (Szeliski Figure 3.7)
%
% Description:
%   This script implements the complete point operator processing pipeline for
%   histogram analysis, full histogram equalization, and partial histogram 
%   equalization. It applies these operations to the ocean image (ocean_img.jpg) 
%   to replicate the analysis in Figure 3.7 of Richard Szeliski's textbook 
%   "Computer Vision: Algorithms and Applications" (2nd edition).
%
%   This implementation utilizes the Ocean Library's modular pixel transformation
%   strategies (OceanMath.Image.Pixels.*) instead of manual calculations.
%
% Scientific Principles Applied (RULE[experiment.md]):
%   1. Explicit Computation: All histograms, CDFs, transfer functions, and 
%      equalized images are derived programmatically from the input image.
%   2. Reproducibility: All steps are self-contained and path-independent.
%   3. No Special-Casing: The algorithm generalizes to any 8-bit RGB image.
%   4. No LaTeX: Code comments and text output utilize plain-text formatting.
%
% Author: Antigravity Code Assistant
% Date: 2026-06-11

clear;
close all;
clc;

% Import strategies
import OceanMath.Image.Pixels.ColorSpaceConverter
import OceanMath.Image.Pixels.HistogramEqualization

%% 1. Locate and Load the Dataset
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
image_path = fullfile(project_root, 'Data', 'images', 'ocean_img.jpg');

fprintf('Loading image from: %s\n', image_path);
if ~exist(image_path, 'file')
    error('Target image file does not exist: %s', image_path);
end

I_orig = imread(image_path);
if size(I_orig, 3) ~= 3
    error('Input image must be a 3-channel color image (RGB).');
end

[height, width, ~] = size(I_orig);
num_pixels = height * width;
fprintf('Loaded RGB image of size %d x %d (%d pixels).\n', width, height, num_pixels);

%% 2. Calculate Luminance (Intensity Channel) using ColorSpaceConverter
grayConv = ColorSpaceConverter('RGB2Gray');
I_gray = grayConv.apply(I_orig);

%% 3. Apply Full Histogram Equalization to Color & Gray Images
% Initialize the strategy with Alpha = 1.0 (Full Equalization)
eqColor = HistogramEqualization('Alpha', 1.0);
I_eq = eqColor.apply(I_orig);

eqGray = HistogramEqualization('Alpha', 1.0);
I_gray_eq = eqGray.apply(I_gray);

%% 4. Retrieve Computed Metrics from the Objects
% Histograms
h_R = eqColor.Histogram(:, 1);
h_G = eqColor.Histogram(:, 2);
h_B = eqColor.Histogram(:, 3);
h_Y = eqGray.Histogram(:, 1);

% Cumulative Distribution Functions (CDFs)
cdf_R = eqColor.CDF(:, 1);
cdf_G = eqColor.CDF(:, 2);
cdf_B = eqColor.CDF(:, 3);
cdf_Y = eqGray.CDF(:, 1);

% Transfer Functions
T_R = eqColor.TransferFunction(:, 1);
T_G = eqColor.TransferFunction(:, 2);
T_B = eqColor.TransferFunction(:, 3);
T_Y = eqGray.TransferFunction(:, 1);

%% 5. Apply Partial Histogram Equalization using the Strategy
% Set blending coefficient alpha (0.5 = 50% equalization blend)
alpha = 0.5;
fprintf('Applying Partial Histogram Equalization with alpha = %.2f\n', alpha);

eqColorPartial = HistogramEqualization('Alpha', alpha);
I_partial = eqColorPartial.apply(I_orig);

% Compute channel histograms for panels (e) and (f) insets
h_R_eq = zeros(256, 1);
h_G_eq = zeros(256, 1);
h_B_eq = zeros(256, 1);
h_R_part = zeros(256, 1);
h_G_part = zeros(256, 1);
h_B_part = zeros(256, 1);
for val = 0:255
    h_R_eq(val + 1) = sum(sum(I_eq(:, :, 1) == val));
    h_G_eq(val + 1) = sum(sum(I_eq(:, :, 2) == val));
    h_B_eq(val + 1) = sum(sum(I_eq(:, :, 3) == val));
    
    h_R_part(val + 1) = sum(sum(I_partial(:, :, 1) == val));
    h_G_part(val + 1) = sum(sum(I_partial(:, :, 2) == val));
    h_B_part(val + 1) = sum(sum(I_partial(:, :, 3) == val));
end

%% 6. Visualization (2x4 Subplot Grid)
fig = figure('Name', 'Histogram Analysis and Equalization', ...
             'Units', 'normalized', ...
             'Position', [0.05, 0.1, 0.9, 0.8]);

% Panel (a): Original Image
subplot(2, 4, 1);
imshow(I_orig);
title('(a) Original Image', 'FontSize', 12);
axis on;

% Panel (b): Color Channel & Intensity Histograms
subplot(2, 4, 2);
hold on;
plot(0:255, h_B, 'b', 'LineWidth', 1.5);
plot(0:255, h_G, 'g', 'LineWidth', 1.5);
plot(0:255, h_R, 'r', 'LineWidth', 1.5);
plot(0:255, h_Y, 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.5); % Dark grey for Luma
hold off;
grid on;
xlim([0, 255]);
xlabel('Intensity Value');
ylabel('Pixel Count');
title('(b) Color & Luma Histograms', 'FontSize', 12);
legend('B', 'G', 'R', 'Y', 'Location', 'northeast');

% Panel (c): Cumulative Distribution Functions (CDFs)
subplot(2, 4, 3);
hold on;
plot(0:255, cdf_B, 'b', 'LineWidth', 1.5);
plot(0:255, cdf_G, 'g', 'LineWidth', 1.5);
plot(0:255, cdf_R, 'r', 'LineWidth', 1.5);
plot(0:255, cdf_Y, 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.5);
hold off;
grid on;
xlim([0, 255]);
ylim([0, num_pixels]);
xlabel('Intensity Value');
ylabel('Cumulative Pixel Count');
title('(c) Cumulative Distribution Functions', 'FontSize', 12);
legend('B', 'G', 'R', 'Y', 'Location', 'southeast');

% Panel (d): Equalization (Transfer) Functions
subplot(2, 4, 4);
hold on;
plot(0:255, T_B, 'b', 'LineWidth', 1.5);
plot(0:255, T_G, 'g', 'LineWidth', 1.5);
plot(0:255, T_R, 'r', 'LineWidth', 1.5);
plot(0:255, T_Y, 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.5);
plot(0:255, 0:255, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.0); % Identity mapping line
hold off;
grid on;
xlim([0, 255]);
ylim([0, 255]);
xlabel('Input Intensity');
ylabel('Output Intensity');
title('(d) Transfer Functions T(r)', 'FontSize', 12);
legend('B', 'G', 'R', 'Y', 'Identity', 'Location', 'southeast');

% Panel (e): Full Histogram Equalization Image
subplot(2, 4, 5);
imshow(I_eq);
title('(e) Full Equalization Image', 'FontSize', 12);
axis on;

% Panel (f): Fully Equalized Histograms (smoothed)
subplot(2, 4, 6);
hold on;
plot(0:255, movmean(h_B_eq, 9), 'b', 'LineWidth', 1.5);
plot(0:255, movmean(h_G_eq, 9), 'g', 'LineWidth', 1.5);
plot(0:255, movmean(h_R_eq, 9), 'r', 'LineWidth', 1.5);
hold off;
grid on;
xlim([0, 255]);
xlabel('Intensity Value');
ylabel('Pixel Count');
title('(f) Fully Equalized Histograms', 'FontSize', 12);
legend('B', 'G', 'R', 'Location', 'northeast');

% Panel (g): Partial Histogram Equalization Image (alpha = 0.5)
subplot(2, 4, 7);
imshow(I_partial);
title(sprintf('(g) Partial Equalization Image (\\alpha = %.1f)', alpha), 'FontSize', 12);
axis on;

% Panel (h): Partially Equalized Histograms (smoothed)
subplot(2, 4, 8);
hold on;
plot(0:255, movmean(h_B_part, 9), 'b', 'LineWidth', 1.5);
plot(0:255, movmean(h_G_part, 9), 'g', 'LineWidth', 1.5);
plot(0:255, movmean(h_R_part, 9), 'r', 'LineWidth', 1.5);
hold off;
grid on;
xlim([0, 255]);
xlabel('Intensity Value');
ylabel('Pixel Count');
title('(h) Partially Equalized Histograms', 'FontSize', 12);
legend('B', 'G', 'R', 'Location', 'northeast');

% Add overall figure title
sgtitle('Szeliski Figure 3.7 Replication: Histogram Equalization on Ocean Image', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% 7. Save Results
results_dir = fullfile(project_root, 'Experiments', 'Scripts', 'Computer Vision', 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_fig_path = fullfile(results_dir, 'histogram_equalization_results.png');
fprintf('Saving results plot to: %s\n', output_fig_path);
saveas(fig, output_fig_path);

fprintf('Experiment complete!\n');
