% exp_image_vector_field.m
%
% Scientific Experiment: Image Gradient Vector Field Visualization
%
% Available images (from Data/images):
%   'apple.jpg'          'corgi.jpg'         'duck.jpg'
%   'earth.jpg'          'marilyn.jpg'       'ocean_img.jpg'
%   'orange.jpg'         'pepsi.png'         'rainbow1.jpg'
%   'sea.jpg'            'zoneplate.png'     'Squid-game-symbols.jpg'
%   'conde.jpg'          'checkboard.png'    'elon_musk.jpg'
%   (or set IMAGE_NAME = '' to use built-in synthetic test image)
%
% Description:
%   This script computes and visualizes the spatial gradient vector field
%   of a grayscale image using MATLAB Image Processing Toolbox built-ins.
%
%   Built-in functions used:
%     imread        - load image from file
%     rgb2gray      - convert colour to grayscale
%     im2double     - convert any image class to double [0,1]
%     imgaussfilt   - Gaussian smoothing filter
%     imgradientxy  - compute directional gradients Gx, Gy (Sobel)
%     imgradient    - compute gradient magnitude M and angle Theta
%     quiver        - plot vector field arrows
%     streamslice   - plot smooth streamlines through the vector field
%     imshow        - display image
%     imagesc       - scaled colormapped image display
%     colormap      - apply colormap to axes
%     colorbar      - add colorbar
%     sgtitle       - super-title across subplots
%
%   Vector Field Definition:
%     Gx     = d(I)/dx  (horizontal — Sobel X, via imgradientxy)
%     Gy     = d(I)/dy  (vertical   — Sobel Y, via imgradientxy)
%     M      = sqrt(Gx^2 + Gy^2)     (magnitude, via imgradient)
%     Theta  = atan2d(Gy, Gx)        (degrees,   via imgradient)
%
% Controls (CONFIG section below):
%   IMAGE_NAME    : filename in Data/images  ('' = built-in synthetic image)
%   DENSITY       : quiver arrow spacing in pixels (8-32 recommended)
%   SCALE         : arrow length scale factor (0 = MATLAB auto)
%   SIGMA         : imgaussfilt sigma  (0 = skip)
%   THRESHOLD_PCT : hide bottom X% magnitude vectors  (0-100)
%   USE_PERP      : true = edge-tangent field  |  false = gradient field
%   SHOW_STREAM   : true = overlay streamslice lines on vector field panel
%
% Scientific Principles:
%   - All computations via documented MATLAB built-ins.
%   - Reproducibility: deterministic synthetic image generation.
%   - No LaTeX: plain-text ASCII labels throughout.
%
% Author: Antigravity Code Assistant
% Date: 2026-08-04

clear;
close all;
clc;

% =====================================================================
% CONFIG
% =====================================================================
IMAGE_NAME    = 'ocean_img.jpg'; % filename from Data/images  ('' = synthetic)
DENSITY       = 14;   % Arrow grid spacing in pixels
SCALE         = 0;    % Quiver scale  (0 = MATLAB auto-scale)
SIGMA         = 1.5;  % imgaussfilt sigma  (0 = no smoothing)
THRESHOLD_PCT = 20;   % Suppress lowest X% magnitude arrows
USE_PERP      = false; % false = gradient field  |  true = edge-tangent field
SHOW_STREAM   = true;  % Overlay streamslice lines on vector field panel

% =====================================================================
% PATH SETUP
% =====================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

% Resolve full image path from filename
DATA_IMAGES_DIR = fullfile(project_root, 'Data', 'images');
if ~isempty(IMAGE_NAME)
    IMAGE_PATH = fullfile(DATA_IMAGES_DIR, IMAGE_NAME);
else
    IMAGE_PATH = '';
end

% Initialize OceanViz visualization engine
viz = OceanViz.Core('Theme', 'Scientific');

fprintf('=== Image Gradient Vector Field Visualization ===\n\n');

% =====================================================================
% 1. LOAD OR GENERATE IMAGE
%    Built-ins: imread, rgb2gray, im2double
% =====================================================================
if ~isempty(IMAGE_PATH) && isfile(IMAGE_PATH)
    fprintf('[1] Loading image: %s\n', IMAGE_PATH);
    img_raw = imread(IMAGE_PATH);           % MATLAB built-in: imread
    if size(img_raw, 3) == 3
        img_gray = rgb2gray(img_raw);       % MATLAB built-in: rgb2gray
    else
        img_gray = img_raw;
    end
    I = im2double(img_gray);                % MATLAB built-in: im2double  -> [0,1] double
    [H, W] = size(I);
    fprintf('    Image size: %d x %d pixels\n', H, W);
else
    fprintf('[1] No image specified — using built-in synthetic test image.\n');
    H = 320; W = 320;
    [X, Y] = meshgrid(linspace(-1, 1, W), linspace(-1, 1, H));
    r  = sqrt(X.^2 + Y.^2);
    I  = 0.5 * cos(6 * pi * r).^2 .* exp(-r.^2 / 0.6);
    Xr = X * cosd(25) + Y * sind(25);
    Yr = -X * sind(25) + Y * cosd(25);
    I  = I + 0.6 * double(abs(Xr) < 0.35 & abs(Yr) < 0.20);
    I  = I + 0.8 * exp(-((X + 0.5).^2 + (Y + 0.4).^2) / (2 * 0.04^2));
    I  = I + 0.2 * (X + Y + 2) / 4;
    I  = max(0, min(1, I));
    fprintf('    Synthetic image size: %d x %d pixels\n', H, W);
end

% =====================================================================
% 2. GAUSSIAN PRE-SMOOTHING
%    Built-in: imgaussfilt
% =====================================================================
if SIGMA > 0
    fprintf('[2] Gaussian pre-smoothing via imgaussfilt (sigma = %.1f).\n', SIGMA);
    I_smooth = imgaussfilt(I, SIGMA);       % MATLAB built-in: imgaussfilt
else
    fprintf('[2] Pre-smoothing skipped (SIGMA = 0).\n');
    I_smooth = I;
end

% =====================================================================
% 3. COMPUTE GRADIENT VECTOR FIELD
%    Built-ins: imgradientxy, imgradient
% =====================================================================
fprintf('[3] Computing gradient field via imgradientxy + imgradient.\n');

% imgradientxy returns [Gx, Gy] using Sobel operator by default
[Gx, Gy] = imgradientxy(I_smooth, 'sobel');    % MATLAB built-in: imgradientxy

% imgradient returns magnitude and direction angle (degrees, CCW from +x)
[M, Theta] = imgradient(Gx, Gy);               % MATLAB built-in: imgradient

fprintf('    Magnitude range : [%.4f, %.4f]\n', min(M(:)), max(M(:)));
fprintf('    Mean magnitude  :  %.4f\n',        mean(M(:)));

% =====================================================================
% 4. OPTIONAL: PERPENDICULAR (EDGE-TANGENT) FIELD
% =====================================================================
if USE_PERP
    fprintf('[4] Switching to edge-tangent (perpendicular) field.\n');
    Vx = -Gy;  Vy = Gx;
    field_label = 'Edge-Tangent Vector Field';
else
    Vx = Gx;   Vy = Gy;
    field_label = 'Gradient Vector Field';
end

% =====================================================================
% 5. BUILD SUBSAMPLED QUIVER GRID
% =====================================================================
fprintf('[5] Building quiver grid (density = %d px).\n', DENSITY);

rows = (1 + round(DENSITY/2)) : DENSITY : H;
cols = (1 + round(DENSITY/2)) : DENSITY : W;
[Cg, Rg] = meshgrid(cols, rows);

Vx_grid = Vx(rows, cols);
Vy_grid = Vy(rows, cols);
M_grid  = M(rows, cols);

% Threshold: suppress lowest THRESHOLD_PCT% magnitude vectors
M_thresh = prctile(M(:), THRESHOLD_PCT);        % MATLAB built-in: prctile
mask = M_grid >= M_thresh;

% Zero-out masked arrows
Vx_plot = Vx_grid .* mask;
Vy_plot = Vy_grid .* mask;

n_arrows = sum(mask(:));

% =====================================================================
% 6. RENDER DASHBOARD  (2 x 3 layout)
%    Built-ins: imshow, imagesc, colormap, colorbar, quiver, streamslice
% =====================================================================
fprintf('[6] Rendering visualization dashboard.\n\n');

fig = viz.createFigure('Image Gradient Vector Field Analysis');
set(fig, 'Units', 'normalized', 'Position', [0.02, 0.05, 0.96, 0.88]);

% Colormaps (all MATLAB built-ins)
cmap_gray = gray(256);
cmap_hot  = hot(256);
cmap_hsv  = hsv(256);
cmap_cool = cool(256);

% Build Blue-White-Red diverging colormap (inline, no helper function)
n_half    = 128;
cmap_bwr  = [ [linspace(0,1,n_half)', linspace(0,1,n_half)', ones(n_half,1)]; ...
              [ones(n_half,1), linspace(1,0,n_half)', linspace(1,0,n_half)'] ];

% ---- Panel 1: Original / Smoothed Image  (imshow) ----
ax1 = subplot(2, 3, 1);
imshow(I_smooth, 'Parent', ax1);               % MATLAB built-in: imshow
if SIGMA > 0
    title(ax1, sprintf('1. Input Image  (imgaussfilt sigma=%.1f)', SIGMA), ...
          'Color', viz.Colors.Text, 'FontWeight', 'bold');
else
    title(ax1, '1. Input Image  (No Pre-smoothing)', ...
          'Color', viz.Colors.Text, 'FontWeight', 'bold');
end
set(ax1, 'Color', viz.Colors.Background);

% ---- Panel 2: Horizontal Gradient Gx  (imagesc) ----
ax2 = subplot(2, 3, 2);
imagesc(ax2, Gx);                              % MATLAB built-in: imagesc
colormap(ax2, cmap_bwr);                       % MATLAB built-in: colormap
colorbar(ax2);                                 % MATLAB built-in: colorbar
title(ax2, '2. Gx = dI/dx  (imgradientxy Sobel-X)', ...
      'Color', viz.Colors.Text, 'FontWeight', 'bold');
axis(ax2, 'equal', 'tight', 'ij');
viz.styleAxes(ax2);

% ---- Panel 3: Vertical Gradient Gy  (imagesc) ----
ax3 = subplot(2, 3, 3);
imagesc(ax3, Gy);
colormap(ax3, cmap_bwr);
colorbar(ax3);
title(ax3, '3. Gy = dI/dy  (imgradientxy Sobel-Y)', ...
      'Color', viz.Colors.Text, 'FontWeight', 'bold');
axis(ax3, 'equal', 'tight', 'ij');
viz.styleAxes(ax3);

% ---- Panel 4: Gradient Magnitude  (imagesc + hot colormap) ----
ax4 = subplot(2, 3, 4);
imagesc(ax4, M);
colormap(ax4, cmap_hot);
cb4 = colorbar(ax4);
cb4.Label.String = 'Magnitude';
cb4.Color = viz.Colors.Text;
title(ax4, '4. Gradient Magnitude  M  (imgradient)', ...
      'Color', viz.Colors.Text, 'FontWeight', 'bold');
axis(ax4, 'equal', 'tight', 'ij');
viz.styleAxes(ax4);

% ---- Panel 5: Gradient Orientation  (imagesc + hsv colormap) ----
ax5 = subplot(2, 3, 5);
imagesc(ax5, Theta);                           % Theta in degrees from imgradient
colormap(ax5, cmap_hsv);
cb5 = colorbar(ax5);
cb5.Label.String = 'Angle (deg)';
cb5.Color = viz.Colors.Text;
clim(ax5, [-180, 180]);
title(ax5, '5. Gradient Orientation Theta  (imgradient)', ...
      'Color', viz.Colors.Text, 'FontWeight', 'bold');
axis(ax5, 'equal', 'tight', 'ij');
viz.styleAxes(ax5);

% ---- Panel 6: Vector Field Overlay  (quiver / streamslice) ----
ax6 = subplot(2, 3, 6);

imshow(I_smooth, 'Parent', ax6);              % background: grayscale image
colormap(ax6, cmap_gray);
hold(ax6, 'on');

if SHOW_STREAM
    % --- streamslice: smooth streamlines through the full vector field ---
    %     Built-in: streamslice
    %     Note: streamslice expects (X, Y, U, V) with X=cols, Y=rows
    [Xs, Ys] = meshgrid(1:W, 1:H);
    h_stream = streamslice(ax6, Xs, Ys, Vx, Vy, 2);   % MATLAB built-in: streamslice
    set(h_stream, 'Color', [0.2 0.8 1.0], 'LineWidth', 0.7);
    legend_str = {'Streamlines'};
end

% --- quiver: plot subsampled gradient arrows ---
%     Built-in: quiver
h_quiv = quiver(ax6, Cg, Rg, Vx_plot, Vy_plot, SCALE, ...  % MATLAB built-in: quiver
                'Color', [1.0 0.85 0.1], ...
                'LineWidth', 0.8, 'MaxHeadSize', 0.5, ...
                'AutoScaleFactor', 0.8);

hold(ax6, 'off');

title(ax6, sprintf('6. %s  (quiver%s)\n[density=%dpx  threshold=%d%%]', ...
      field_label, ternary(SHOW_STREAM, ' + streamslice', ''), ...
      DENSITY, THRESHOLD_PCT), ...
      'Color', viz.Colors.Text, 'FontWeight', 'bold');
axis(ax6, 'equal', 'tight', 'ij');
viz.styleAxes(ax6);

% ---- Global super-title ----
sgtitle(fig, 'Image Gradient Vector Field  —  MATLAB Built-in Pipeline', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', viz.Colors.Text);

% =====================================================================
% 7. PRINT SUMMARY STATISTICS
% =====================================================================
fprintf('--- Summary Statistics ---\n');
fprintf('  Image size          : %d x %d px\n',   H, W);
fprintf('  Pre-smooth          : imgaussfilt  sigma=%.1f\n', SIGMA);
fprintf('  Gradient method     : imgradientxy + imgradient (Sobel)\n');
fprintf('  Vector plot         : quiver%s\n', ternary(SHOW_STREAM, ' + streamslice', ''));
fprintf('  Vectors plotted     : %d / %d grid cells\n', n_arrows, numel(Cg));
fprintf('  Magnitude (min/max) : %.4f / %.4f\n', min(M(:)), max(M(:)));
fprintf('  Magnitude threshold : %.4f (%d-th percentile)\n', M_thresh, THRESHOLD_PCT);
fprintf('  Field type          : %s\n', field_label);
fprintf('\n');

% =====================================================================
% 8. SAVE RESULT
% =====================================================================
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_path = fullfile(results_dir, 'exp_image_vector_field_result.png');
saveas(fig, output_path);                      % MATLAB built-in: saveas
fprintf('Result saved to: %s\n', output_path);
fprintf('Experiment complete!\n');

% =====================================================================
% LOCAL HELPER  (not a custom implementation — just syntactic sugar)
% =====================================================================
function out = ternary(cond, a, b)
% ternary  Inline ternary operator.
    if cond; out = a; else; out = b; end
end
