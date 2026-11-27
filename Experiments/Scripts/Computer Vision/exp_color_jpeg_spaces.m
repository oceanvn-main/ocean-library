% EXP_COLOR_JPEG_SPACES  Color space transformations & JPEG compression quality sweep.
%
%   DATASET:        ocean_img.jpg (size determined dynamically)
%   MODEL:          Color Space transformations (RGB, rgb, L*a*b*, HSV) & JPEG quantization quality sweep (Q=95, 30, 5)
%   ASSUMPTIONS:
%     - The input image exists at the specified path and is in RGB color format.
%     - Built-in color conversion functions rgb2lab and rgb2hsv are available.
%     - L*a*b* channels are scaled to [0, 1] for visual plotting. a* and b* channels are scaled via min-max.
%     - H channel (Hue) is colorized by setting Saturation and Value to 1 to visualize hue angles clearly.
%     - Zoom coordinates focus on the face/glasses area to highlight compression block artifacts.
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%

clc; clear; close all; rng(42, 'twister');

fprintf('=== Starting Color Spaces and JPEG Compression Scientific Experiment ===\n');

%% Step 1: Load Image
imagePath = fullfile('c:', 'Ocean Library', 'Matlab code practice', 'Data', 'images', 'consu.jpg');
if ~isfile(imagePath)
    error("Input image not found: %s", imagePath);
end

img = imread(imagePath);
[h, w, c] = size(img);
fprintf('Loaded image: %s (Height=%d, Width=%d, Channels=%d)\n', 'conde.jpg', h, w, c);

img_d = double(img) / 255.0;

%% Step 2: Compute Color Spaces (Figure 2.33)
fprintf('Computing color space transformations...\n');

% Row 1: RGB
R = img_d(:,:,1);
G = img_d(:,:,2);
B = img_d(:,:,3);

% Row 2: Normalized rgb
sum_rgb = sum(img_d, 3);
sum_rgb(sum_rgb == 0) = 1e-8; % Avoid division by zero
r = img_d(:,:,1) ./ sum_rgb;
g = img_d(:,:,2) ./ sum_rgb;
b = img_d(:,:,3) ./ sum_rgb;
rgb_norm = cat(3, r, g, b);

% Row 3: L*a*b*
lab = rgb2lab(img_d);
L_star = lab(:,:,1) / 100.0; % Lightness is in [0, 100], scale to [0, 1]

a_star = lab(:,:,2);
a_min = min(a_star(:));
a_max = max(a_star(:));
a_star_scaled = (a_star - a_min) / (a_max - a_min);

b_star = lab(:,:,3);
b_min = min(b_star(:));
b_max = max(b_star(:));
b_star_scaled = (b_star - b_min) / (b_max - b_min);

% Row 4: HSV
hsv = rgb2hsv(img_d);
H = hsv(:,:,1);
S = hsv(:,:,2);
V = hsv(:,:,3);

% Colorize H channel: set Saturation and Value to 1 to show the hue colors
H_color = hsv2rgb(cat(3, H, ones(size(H)), ones(size(H))));

%% Step 3: JPEG Compression Quality Simulation (Figure 2.34)
fprintf('Simulating JPEG compression at Q=95, Q=30, and Q=5...\n');

resultsDir = fullfile('c:', 'Ocean Library', 'Matlab code practice', 'Results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

tempFile95 = fullfile(resultsDir, 'temp_q95.jpg');
tempFile30 = fullfile(resultsDir, 'temp_q30.jpg');
tempFile5  = fullfile(resultsDir, 'temp_q5.jpg');

imwrite(img, tempFile95, 'jpeg', 'Quality', 95);
imwrite(img, tempFile30, 'jpeg', 'Quality', 30);
imwrite(img, tempFile5, 'jpeg', 'Quality', 5);

% Get compressed file sizes in KB
dir95 = dir(tempFile95); size95_kb = dir95.bytes / 1024;
dir30 = dir(tempFile30); size30_kb = dir30.bytes / 1024;
dir5 = dir(tempFile5);   size5_kb = dir5.bytes / 1024;

fprintf('Compressed sizes: Q=95: %.2f KB, Q=30: %.2f KB, Q=5: %.2f KB\n', size95_kb, size30_kb, size5_kb);

img_q95 = imread(tempFile95);
img_q30 = imread(tempFile30);
img_q5  = imread(tempFile5);

% Clean up temp files
if isfile(tempFile95); delete(tempFile95); end
if isfile(tempFile30); delete(tempFile30); end
if isfile(tempFile5);  delete(tempFile5);  end

% Define zoom region around glasses/face area
yStart = round(h * 0.15);
yEnd   = round(h * 0.30);
xStart = round(w * 0.40);
xEnd   = round(w * 0.75);

crop_q95 = img_q95(yStart:yEnd, xStart:xEnd, :);
crop_q30 = img_q30(yStart:yEnd, xStart:xEnd, :);
crop_q5  = img_q5(yStart:yEnd, xStart:xEnd, :);

%% Step 4: Render Figure 2.33 using OceanViz
fprintf('Rendering and saving Figure 2.33...\n');
viz = OceanViz.Core('Theme', 'Dark');
fig1 = viz.createFigure('Figure 2.33: Color Space Transformations');
set(fig1, 'Position', [50, 50, 1200, 900]);

% Row 1: RGB channels
ax1 = subplot(4, 4, 1, 'Parent', fig1); imshow(img_d, 'Parent', ax1); title(ax1, '(a) RGB', 'Color', viz.Colors.Text);
ax2 = subplot(4, 4, 2, 'Parent', fig1); imshow(R, 'Parent', ax2); title(ax2, '(b) R', 'Color', viz.Colors.Text);
ax3 = subplot(4, 4, 3, 'Parent', fig1); imshow(G, 'Parent', ax3); title(ax3, '(c) G', 'Color', viz.Colors.Text);
ax4 = subplot(4, 4, 4, 'Parent', fig1); imshow(B, 'Parent', ax4); title(ax4, '(d) B', 'Color', viz.Colors.Text);

% Row 2: Normalized rgb channels
ax5 = subplot(4, 4, 5, 'Parent', fig1); imshow(rgb_norm, 'Parent', ax5); title(ax5, '(e) rgb', 'Color', viz.Colors.Text);
ax6 = subplot(4, 4, 6, 'Parent', fig1); imshow(r, 'Parent', ax6); title(ax6, '(f) r', 'Color', viz.Colors.Text);
ax7 = subplot(4, 4, 7, 'Parent', fig1); imshow(g, 'Parent', ax7); title(ax7, '(g) g', 'Color', viz.Colors.Text);
ax8 = subplot(4, 4, 8, 'Parent', fig1); imshow(b, 'Parent', ax8); title(ax8, '(h) b', 'Color', viz.Colors.Text);

% Row 3: L*a*b* channels
ax9  = subplot(4, 4, 9, 'Parent', fig1);  imshow(L_star, 'Parent', ax9); title(ax9, '(i) L*', 'Color', viz.Colors.Text);
ax10 = subplot(4, 4, 10, 'Parent', fig1); imshow(a_star_scaled, 'Parent', ax10); title(ax10, '(j) a*', 'Color', viz.Colors.Text);
ax11 = subplot(4, 4, 11, 'Parent', fig1); imshow(b_star_scaled, 'Parent', ax11); title(ax11, '(k) b*', 'Color', viz.Colors.Text);

% Row 4: HSV channels
ax12 = subplot(4, 4, 13, 'Parent', fig1); imshow(H_color, 'Parent', ax12); title(ax12, '(l) H', 'Color', viz.Colors.Text);
ax13 = subplot(4, 4, 14, 'Parent', fig1); imshow(S, 'Parent', ax13); title(ax13, '(m) S', 'Color', viz.Colors.Text);
ax14 = subplot(4, 4, 15, 'Parent', fig1); imshow(V, 'Parent', ax14); title(ax14, '(n) V', 'Color', viz.Colors.Text);

% Style all active axes
activeAxes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14];
for k = 1:numel(activeAxes)
    ax = activeAxes(k);
    axis(ax, 'image');
    set(ax, 'XTick', [], 'YTick', []);
    grid(ax, 'off');
end

saveas(fig1, fullfile(resultsDir, 'color_space_transformation.png'));
fprintf('Saved color space figure to Results/color_space_transformation.png\n');

%% Step 5: Render Figure 2.34 using OceanViz
fprintf('Rendering and saving Figure 2.34...\n');
fig2 = viz.createFigure('Figure 2.34: JPEG Quality Settings');
set(fig2, 'Position', [100, 100, 1100, 700]);

% Row 1: Full Images
ax15 = subplot(2, 3, 1, 'Parent', fig2); imshow(img_q95, 'Parent', ax15); title(ax15, sprintf('Full: Q=95 (%.1f KB)', size95_kb), 'Color', viz.Colors.Text);
ax16 = subplot(2, 3, 2, 'Parent', fig2); imshow(img_q30, 'Parent', ax16); title(ax16, sprintf('Full: Q=30 (%.1f KB)', size30_kb), 'Color', viz.Colors.Text);
ax17 = subplot(2, 3, 3, 'Parent', fig2); imshow(img_q5, 'Parent', ax17);  title(ax17, sprintf('Full: Q=5 (%.1f KB)', size5_kb), 'Color', viz.Colors.Text);

% Row 2: Cropped Zoom Insets
ax18 = subplot(2, 3, 4, 'Parent', fig2); imshow(crop_q95, 'Parent', ax18); title(ax18, sprintf('Zoom: Q=95 (%.1f KB)', size95_kb), 'Color', viz.Colors.Text);
ax19 = subplot(2, 3, 5, 'Parent', fig2); imshow(crop_q30, 'Parent', ax19); title(ax19, sprintf('Zoom: Q=30 (%.1f KB)', size30_kb), 'Color', viz.Colors.Text);
ax20 = subplot(2, 3, 6, 'Parent', fig2); imshow(crop_q5, 'Parent', ax20);  title(ax20, sprintf('Zoom: Q=5 (%.1f KB)', size5_kb), 'Color', viz.Colors.Text);

activeAxes2 = [ax15, ax16, ax17, ax18, ax19, ax20];
for k = 1:numel(activeAxes2)
    ax = activeAxes2(k);
    axis(ax, 'image');
    set(ax, 'XTick', [], 'YTick', []);
    grid(ax, 'off');
end

saveas(fig2, fullfile(resultsDir, 'jpeg_quality_comparison.png'));
fprintf('Saved JPEG quality figure to Results/jpeg_quality_comparison.png\n');

%% Step 6: Print Experiment Summary
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : %s (Height=%d, Width=%d, Channels=%d)\n', 'ocean_img.jpg', h, w, c);
fprintf('  Experiment    : Color Space Conversion & JPEG Artifact Analysis\n');
fprintf('  JPEG Sizes    : Q=95: %.2f KB | Q=30: %.2f KB | Q=5: %.2f KB\n', size95_kb, size30_kb, size5_kb);
fprintf('  Output 1      : Results/color_space_transformation.png\n');
fprintf('  Output 2      : Results/jpeg_quality_comparison.png\n');
fprintf('==========================\n');

fprintf('=== Experiment Completed Successfully ===\n');
