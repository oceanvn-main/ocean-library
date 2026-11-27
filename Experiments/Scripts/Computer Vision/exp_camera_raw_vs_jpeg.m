% EXP_CAMERA_RAW_VS_JPEG  Experiment comparing Sony Camera RAW (.ARW) vs 3 JPEG versions.
%
%   PURPOSE:
%     Real-world validation of custom JpegSyntheticCompressor against:
%       1. Sony Camera RAW matrix (.ARW) processed via demosaic + WB + gamma
%       2. In-Camera native JPEG (.JPG)
%       3. MATLAB built-in JPEG conversion (imwrite)
%       4. Custom JpegSyntheticCompressor engine
%
%   DATASET:
%     Sony Alpha RAW + JPEG pair:
%       RAW:  Data/images/raw/_DSC5165.ARW  (or _DSC5166.ARW)
%       JPEG: Data/images/_DSC5165.JPG      (or _DSC5166.JPG)
%
%   MEASUREMENTS:
%     - Exact File Size on disk (MB) for all 4 representations.
%     - PSNR & SSIM metrics vs processed RAW ground truth.
%     - 4-panel OceanViz Dark-theme visual dashboard.
%
%   Ref: Szeliski "Computer Vision" 2e, §4.3 & §10.1 (Image Formation & Codecs)

clc; clear; close all;
rng(42, 'twister');

fprintf('=== Experiment: Camera RAW (.ARW) vs Native, MATLAB & Custom JPEG ===\n\n');

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 1] Path & Self-Location Setup
%  ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = fileparts(mfilename('fullpath'));
EXP_DIR     = fileparts(SCRIPT_DIR);
SCRIPTS_DIR = fileparts(EXP_DIR);
WORKSPACE   = fileparts(SCRIPTS_DIR);
RESULTS_DIR = fullfile(WORKSPACE, 'Results');

if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

RAW_FILE_PATH    = fullfile(WORKSPACE, 'Data', 'images', 'raw', '_DSC5165.ARW');
CAMERA_JPG_PATH  = fullfile(WORKSPACE, 'Data', 'images', '_DSC5165.JPG');
MATLAB_JPG_PATH  = fullfile(RESULTS_DIR, '_DSC5165_matlab_q90.jpg');
CUSTOM_JPG_PATH  = fullfile(RESULTS_DIR, '_DSC5165_custom_q90.jpg');

QUALITY_FACTOR   = 90;  % Standard high quality used by cameras

assert(isfile(RAW_FILE_PATH),   'RAW file not found: %s', RAW_FILE_PATH);
assert(isfile(CAMERA_JPG_PATH), 'Camera JPG not found: %s', CAMERA_JPG_PATH);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 2] Process Sony Camera RAW (.ARW)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 1] Processing Sony Camera RAW file (_DSC5165.ARW) ...\n');
size_raw_bytes = dir(RAW_FILE_PATH).bytes;
size_raw_mb    = size_raw_bytes / (1024 * 1024);
fprintf('  RAW File Size: %.2f MB\n', size_raw_mb);

% Read RAW sensor matrix via rawread & rawinfo
raw_cfa = rawread(RAW_FILE_PATH);
info    = rawinfo(RAW_FILE_PATH);

% Extract black/white levels and normalize sensor linear values
blk = double(info.ColorInfo.BlackLevel(1));
wht = double(info.ColorInfo.WhiteLevel(1));
raw_sub = max(0.0, double(raw_cfa) - blk);
raw_norm = min(1.0, raw_sub / (wht - blk));

% Demosaic RGGB Bayer matrix to 16-bit RGB
raw_u16 = uint16(round(raw_norm * 65535.0));
cfa_layout = lower(info.CFALayout);
rgb_demosaiced = double(demosaic(raw_u16, cfa_layout)) / 65535.0;

% Apply Camera White Balance multipliers (As Taken)
wb = double(info.ColorInfo.CameraAsTakenWhiteBalance);
wb_r = wb(1) / wb(2);
wb_b = wb(4) / wb(3);
rgb_wb = rgb_demosaiced;
rgb_wb(:,:,1) = min(1.0, rgb_wb(:,:,1) * wb_r);
rgb_wb(:,:,3) = min(1.0, rgb_wb(:,:,3) * wb_b);

% Apply sRGB gamma curve (approx 1/2.2) to match display space
img_raw_full = rgb_wb .^ (1 / 2.2);
img_raw_full = max(0.0, min(1.0, img_raw_full));

% Scale working resolution to 25% (1096x1641) to prevent MATLAB OOM on 20MP matrices
PROC_SCALE  = 0.25;
img_raw_rgb = imresize(img_raw_full, PROC_SCALE, 'bilinear');
[H, W, ~]   = size(img_raw_rgb);
fprintf('  Working dimensions for experiment: %d x %d x 3\n', H, W);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 3] Load In-Camera Native JPEG (.JPG)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 2] Loading In-Camera native JPEG (_DSC5165.JPG) ...\n');
size_camera_bytes = dir(CAMERA_JPG_PATH).bytes;
size_camera_mb    = size_camera_bytes / (1024 * 1024);
fprintf('  Camera JPEG File Size: %.2f MB\n', size_camera_mb);

img_camera_raw = imread(CAMERA_JPG_PATH);
img_camera     = double(img_camera_raw) / 255.0;

% If dimensions differ slightly due to camera cropping, resize camera image to match RAW
if size(img_camera, 1) ~= H || size(img_camera, 2) ~= W
    img_camera = imresize(img_camera, [H, W], 'bilinear');
end

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 4] MATLAB Built-in JPEG Conversion
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 3] Converting RAW to JPEG using MATLAB built-in imwrite (Quality=%d) ...\n', QUALITY_FACTOR);
img_raw_u8 = uint8(round(img_raw_rgb * 255.0));
imwrite(img_raw_u8, MATLAB_JPG_PATH, 'jpg', 'Quality', QUALITY_FACTOR);

size_matlab_bytes = dir(MATLAB_JPG_PATH).bytes;
size_matlab_mb    = size_matlab_bytes / (1024 * 1024);
fprintf('  MATLAB JPEG File Size: %.2f MB\n', size_matlab_mb);

img_matlab = double(imread(MATLAB_JPG_PATH)) / 255.0;

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 5] Custom JpegSyntheticCompressor Conversion
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 4] Converting RAW to JPEG using Custom JpegSyntheticCompressor (Quality=%d) ...\n', QUALITY_FACTOR);

comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', QUALITY_FACTOR);

% Convert RAW RGB to YCbCr space for standard JPEG quantization
YCbCr_raw = rgb2ycbcr(img_raw_rgb);
[YCbCr_custom, q_all, Q_table] = comp.compress(YCbCr_raw);
img_custom = max(0.0, min(1.0, ycbcr2rgb(YCbCr_custom)));

% Save custom output to disk to measure exact file payload
img_custom_u8 = uint8(round(img_custom * 255.0));
imwrite(img_custom_u8, CUSTOM_JPG_PATH, 'jpg', 'Quality', QUALITY_FACTOR);

size_custom_bytes = dir(CUSTOM_JPG_PATH).bytes;
size_custom_mb    = size_custom_bytes / (1024 * 1024);
fprintf('  Custom Engine JPEG File Size: %.2f MB\n', size_custom_mb);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 6] Compute Quality Metrics (PSNR & SSIM vs Processed RAW)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 5] Computing Quality Metrics vs Processed RAW Ground Truth ...\n');

psnr_camera = OceanStats.Image.ImageQualityMetrics.psnr(img_camera, img_raw_rgb);
ssim_camera = OceanStats.Image.ImageQualityMetrics.ssim(img_camera, img_raw_rgb);

psnr_matlab = OceanStats.Image.ImageQualityMetrics.psnr(img_matlab, img_raw_rgb);
ssim_matlab = OceanStats.Image.ImageQualityMetrics.ssim(img_matlab, img_raw_rgb);

psnr_custom = OceanStats.Image.ImageQualityMetrics.psnr(img_custom, img_raw_rgb);
ssim_custom = OceanStats.Image.ImageQualityMetrics.ssim(img_custom, img_raw_rgb);

metrics(1).Label = sprintf('In-Camera JPEG (%.2f MB)', size_camera_mb);
metrics(1).PSNR  = psnr_camera; metrics(1).SSIM = ssim_camera;

metrics(2).Label = sprintf('MATLAB imwrite (%.2f MB)', size_matlab_mb);
metrics(2).PSNR  = psnr_matlab; metrics(2).SSIM = ssim_matlab;

metrics(3).Label = sprintf('Custom Engine  (%.2f MB)', size_custom_mb);
metrics(3).PSNR  = psnr_custom; metrics(3).SSIM = ssim_custom;

OceanStats.Image.ImageQualityMetrics.printTable(metrics);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 7] Render OceanViz Comparison Dashboard
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 6] Rendering 4-panel comparison dashboard using OceanViz Engine ...\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Take 1: Sony RAW (_DSC5165.ARW) vs 3 JPEG Representations');
set(fig, 'Position', [30, 30, 1380, 850]);
set(fig, 'Color', viz.Colors.Background);

% Downsample for high-speed, high-contrast dashboard rendering (display size 800x1200)
disp_scale = 800 / H;
img_raw_disp    = imresize(img_raw_rgb, disp_scale, 'bilinear');
img_camera_disp = imresize(img_camera,  disp_scale, 'bilinear');
img_matlab_disp = imresize(img_matlab,  disp_scale, 'bilinear');
img_custom_disp = imresize(img_custom,  disp_scale, 'bilinear');

% --- Panel 1: Processed Sony Camera RAW (.ARW) ---
ax1 = subplot(2, 2, 1, 'Parent', fig);
imshow(img_raw_disp, 'Parent', ax1);
title_str1 = sprintf('1. Sony Camera RAW (_DSC5165.ARW)\nFile Size: %.2f MB  |  Res: %dx%d', ...
    size_raw_mb, W, H);
title(ax1, title_str1, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax1, 'off');

% --- Panel 2: In-Camera Native JPEG (.JPG) ---
ax2 = subplot(2, 2, 2, 'Parent', fig);
imshow(img_camera_disp, 'Parent', ax2);
title_str2 = sprintf('2. In-Camera JPEG (_DSC5165.JPG)\nFile Size: %.2f MB  |  PSNR=%.2f dB  SSIM=%.4f', ...
    size_camera_mb, psnr_camera, ssim_camera);
title(ax2, title_str2, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax2, 'off');

% --- Panel 3: MATLAB Built-in JPEG (imwrite) ---
ax3 = subplot(2, 2, 3, 'Parent', fig);
imshow(img_matlab_disp, 'Parent', ax3);
title_str3 = sprintf('3. MATLAB Built-in imwrite (Q=%d)\nFile Size: %.2f MB  |  PSNR=%.2f dB  SSIM=%.4f', ...
    QUALITY_FACTOR, size_matlab_mb, psnr_matlab, ssim_matlab);
title(ax3, title_str3, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax3, 'off');

% --- Panel 4: Custom JpegSyntheticCompressor ---
ax4 = subplot(2, 2, 4, 'Parent', fig);
imshow(img_custom_disp, 'Parent', ax4);
title_str4 = sprintf('4. Custom JpegSyntheticCompressor (Q=%d)\nFile Size: %.2f MB  |  PSNR=%.2f dB  SSIM=%.4f', ...
    QUALITY_FACTOR, size_custom_mb, psnr_custom, ssim_custom);
title(ax4, title_str4, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax4, 'off');

% Style backgrounds
for ax = [ax1, ax2, ax3, ax4]
    set(ax, 'Color', viz.Colors.Background);
end

% Save figure artifact
fig_path = fullfile(RESULTS_DIR, 'exp_camera_raw_vs_jpeg.png');
saveas(fig, fig_path);
fprintf('\n[Step 7] Saved comparison figure to:\n  → %s\n', fig_path);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Summary Table Output]
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('═', 1, 72));
fprintf('  TAKE 1 SUMMARY — SONY RAW (_DSC5165.ARW) vs JPEG CODECS\n');
fprintf('%s\n', repmat('─', 1, 72));
fprintf('  Representation                       File Size (MB)   PSNR (dB)   SSIM\n');
fprintf('%s\n', repmat('─', 1, 72));
fprintf('  1. Sony Camera RAW (.ARW)              %6.2f MB       [Reference Ground Truth]\n', size_raw_mb);
fprintf('  2. In-Camera JPEG (.JPG)               %6.2f MB        %6.2f     %.4f\n', size_camera_mb, psnr_camera, ssim_camera);
fprintf('  3. MATLAB Built-in JPEG                %6.2f MB        %6.2f     %.4f\n', size_matlab_mb, psnr_matlab, ssim_matlab);
fprintf('  4. Custom JpegSyntheticCompressor      %6.2f MB        %6.2f     %.4f\n', size_custom_mb, psnr_custom, ssim_custom);
fprintf('%s\n', repmat('═', 1, 72));

fprintf('\n=== Experiment Completed Successfully ===\n');
