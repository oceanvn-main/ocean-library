% EXP_VERIFY_JPEG_COMPRESSOR  Verification of custom JpegSyntheticCompressor vs MATLAB built-in imwrite.
%
%   PURPOSE:
%     Scientifically verify the custom 8x8 DCT quantization compressor
%     (JpegSyntheticCompressor) against MATLAB's native JPEG engine (imwrite).
%
%   DATASET:
%     Procedurally generated synthetic image (SyntheticImageGenerator, 256x256 mixed).
%
%   EXPERIMENTAL METHOD:
%     1. Generate pristine raw image (img_clean).
%     2. Compress using MATLAB built-in: imwrite(img, path, 'Quality', Q).
%     3. Compress using custom engine: JpegSyntheticCompressor('QualityFactor', Q).
%     4. Compute PSNR & SSIM metrics (vs Ground Truth, and Custom vs MATLAB).
%     5. Visualize via OceanViz Dark-theme engine (Ground Truth, MATLAB JPEG, Custom JPEG, Absolute Difference).
%
%   NOTE ON JPEG DIFFERENCES:
%     - MATLAB's imwrite() applies RGB->YCbCr conversion + 4:2:0 chroma subsampling + Huffman coding.
%     - JpegSyntheticCompressor applies pure 8x8 DCT-II quantization directly per channel.
%     - Both use the identical ISO/IEC 10918-1 Annex K quantization scaling formula.
%
%   Author: Antigravity Code Assistant
%   Date:   2026-08-19

clc; clear; close all;
rng(42, 'twister');

fprintf('=== Verification: JpegSyntheticCompressor vs MATLAB Built-in JPEG ===\n\n');

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 1] Self-location & Setup
%  ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = fileparts(mfilename('fullpath'));
EXP_DIR     = fileparts(SCRIPT_DIR);
SCRIPTS_DIR = fileparts(EXP_DIR);
WORKSPACE   = fileparts(SCRIPTS_DIR);
RESULTS_DIR = fullfile(WORKSPACE, 'Results');
TEMP_JPG    = fullfile(RESULTS_DIR, 'temp_matlab_q10.jpg');

if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

QUALITY_FACTOR = 10;  % Severe compression to highlight blocking grid behavior

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 2] Generate Synthetic Raw Image
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 1] Generating synthetic raw image (256x256 mixed mode) ...\n');
gen = OceanData.Generators.SyntheticImageGenerator(...
    'Height', 256, 'Width', 256, 'Mode', 'mixed', 'Seed', 42);
img_raw = gen.generate();  % (256 x 256 x 3) double [0, 1]

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 3] MATLAB Built-in JPEG Compression
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 2] Compressing using MATLAB built-in imwrite (Quality=%d) ...\n', QUALITY_FACTOR);
% Write double [0,1] to JPEG using uint8 conversion
img_uint8 = uint8(round(img_raw * 255.0));
imwrite(img_uint8, TEMP_JPG, 'jpg', 'Quality', QUALITY_FACTOR);

% Read back the MATLAB compressed image
img_matlab_raw = imread(TEMP_JPG);
img_matlab = double(img_matlab_raw) / 255.0;

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 4] Custom JpegSyntheticCompressor
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 3] Compressing using custom JpegSyntheticCompressor (Quality=%d) ...\n', QUALITY_FACTOR);
comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', QUALITY_FACTOR);
[img_custom, q_all, Q_table] = comp.compress(img_raw);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 5] Quantitative Quality Evaluation (PSNR & SSIM)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 4] Computing Quality Metrics ...\n');

psnr_matlab = OceanStats.Image.ImageQualityMetrics.psnr(img_matlab, img_raw);
ssim_matlab = OceanStats.Image.ImageQualityMetrics.ssim(img_matlab, img_raw);

psnr_custom = OceanStats.Image.ImageQualityMetrics.psnr(img_custom, img_raw);
ssim_custom = OceanStats.Image.ImageQualityMetrics.ssim(img_custom, img_raw);

% Metric comparing Custom vs MATLAB JPEG output
psnr_diff   = OceanStats.Image.ImageQualityMetrics.psnr(img_custom, img_matlab);
ssim_diff   = OceanStats.Image.ImageQualityMetrics.ssim(img_custom, img_matlab);

metrics(1).Label = 'MATLAB imwrite()';       metrics(1).PSNR = psnr_matlab; metrics(1).SSIM = ssim_matlab;
metrics(2).Label = 'Custom Synthetic Engine'; metrics(2).PSNR = psnr_custom; metrics(2).SSIM = ssim_custom;
metrics(3).Label = 'Custom vs MATLAB Diff';   metrics(3).PSNR = psnr_diff;   metrics(3).SSIM = ssim_diff;

OceanStats.Image.ImageQualityMetrics.printTable(metrics);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 6] OceanViz Plotting & Dashboard Rendering
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 5] Rendering comparison dashboard using OceanViz Engine ...\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('JPEG Compressor Verification: MATLAB vs Custom Engine');
set(fig, 'Position', [40, 40, 1350, 750]);
set(fig, 'Color', viz.Colors.Background);

% --- Panel 1: Raw Synthetic Image (Ground Truth) ---
ax1 = subplot(2, 3, 1, 'Parent', fig);
imshow(img_raw, 'Parent', ax1);
title(ax1, '1. Raw Synthetic Image (Ground Truth)', ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax1, 'off');

% --- Panel 2: MATLAB Built-in JPEG ---
ax2 = subplot(2, 3, 2, 'Parent', fig);
imshow(img_matlab, 'Parent', ax2);
title(ax2, sprintf('2. MATLAB imwrite (Q=%d)\nPSNR=%.2f dB | SSIM=%.4f', ...
    QUALITY_FACTOR, psnr_matlab, ssim_matlab), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax2, 'off');

% --- Panel 3: Custom JpegSyntheticCompressor ---
ax3 = subplot(2, 3, 3, 'Parent', fig);
imshow(img_custom, 'Parent', ax3);
title(ax3, sprintf('3. Custom Compressor (Q=%d)\nPSNR=%.2f dB | SSIM=%.4f', ...
    QUALITY_FACTOR, psnr_custom, ssim_custom), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax3, 'off');

% --- Panel 4: Absolute Difference Map (Custom vs MATLAB) ---
ax4 = subplot(2, 3, 4, 'Parent', fig);
diff_map = mean(abs(img_custom - img_matlab), 3);
imagesc(ax4, diff_map);
axis(ax4, 'image'); axis(ax4, 'off');
colormap(ax4, 'hot');
cb4 = colorbar(ax4);
cb4.Color = viz.Colors.Text;
cb4.Label.String = 'Abs Difference';
cb4.Label.Color  = viz.Colors.Text;
title(ax4, sprintf('4. Abs Diff Map (Custom vs MATLAB)\nCross-PSNR=%.2f dB', psnr_diff), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');

% --- Panel 5: Zoomed Inset - MATLAB Blocking (Center Patch 64x64) ---
ax5 = subplot(2, 3, 5, 'Parent', fig);
crop_r = 96:160; crop_c = 96:160;
imshow(img_matlab(crop_r, crop_c, :), 'Parent', ax5);
title(ax5, '5. Zoomed 64x64 Patch (MATLAB)', ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax5, 'off');

% --- Panel 6: Zoomed Inset - Custom Blocking (Center Patch 64x64) ---
ax6 = subplot(2, 3, 6, 'Parent', fig);
imshow(img_custom(crop_r, crop_c, :), 'Parent', ax6);
title(ax6, '6. Zoomed 64x64 Patch (Custom)', ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax6, 'off');

% Apply OceanViz background styling
for ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    set(ax, 'Color', viz.Colors.Background);
end

% Save figure artifact
fig_path = fullfile(RESULTS_DIR, 'exp_verify_jpeg_compressor.png');
saveas(fig, fig_path);
fprintf('\n[Step 6] Saved verification dashboard figure to:\n  → %s\n', fig_path);

% Clean up temporary file
if exist(TEMP_JPG, 'file'), delete(TEMP_JPG); end

fprintf('\n=== Verification Script Completed Successfully ===\n');
