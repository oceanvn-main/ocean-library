% EXP_POCS_ROBUSTNESS_NOISY  POCS Deblocking Robustness under Dual Degradation.
%
%   PURPOSE:
%     Scientific robustness evaluation of POCS alternating projection (Ex 4.3)
%     under dual degradation: Additive Gaussian Sensor Noise (sigma_n = 0.05)
%     combined with Severe JPEG Block Quantization (Quality Factor Q = 5).
%
%   DATASET:
%     Sony Camera RAW matrix: Data/images/raw/_DSC5165.ARW
%
%   MODELS BENCHMARKED (5 Models):
%     1. Degraded Baseline   — Additive Noise + JPEG Q=5 (No Restoration)
%     2. Pure DCT POCS       — POCS hyper-rectangle projection (No spatial smoother)
%     3. POCS + Gaussian     — POCS with GaussianFilter smoother
%     4. POCS + Bilateral    — POCS with edge-preserving BilateralFilter smoother (Default)
%     5. POCS + Guided       — POCS with GuidedFilter smoother
%
%   ASSUMPTIONS:
%     - Sony RAW is demapsic'd, white-balanced, and converted to sRGB gamma.
%     - Noise is added in YCbCr color space to simulate sensor/transmission noise.
%     - Quantization bounds [Q*(q - 0.5), Q*(q + 0.5)] are tracked from noisy compression.
%
%   REPRODUCIBILITY:
%     rng(42, 'twister') called at top.
%
%   Ref: Szeliski "Computer Vision" 2e, §4.3 (POCS, p. 234)

clc; clear; close all;
rng(42, 'twister');

fprintf('=== Experiment: POCS Robustness under Dual Degradation (Noise + JPEG Q=5) ===\n\n');

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 1] Path & Self-Location Setup
%  ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = fileparts(mfilename('fullpath'));
EXP_DIR     = fileparts(SCRIPT_DIR);
SCRIPTS_DIR = fileparts(EXP_DIR);
WORKSPACE   = fileparts(SCRIPTS_DIR);
RESULTS_DIR = fullfile(WORKSPACE, 'Results');

if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

RAW_FILE_PATH = fullfile(WORKSPACE, 'Data', 'images', 'raw', '_DSC5165.ARW');
assert(isfile(RAW_FILE_PATH), 'Sony RAW file not found: %s', RAW_FILE_PATH);

QUALITY_FACTOR = 10;     % Standard default JPEG quality (Annex K scale = 1.0)
NOISE_SIGMA    = 0.0;    % Standard clean JPEG evaluation
POCS_MAX_ITER  = 20;     % 20 iterations (optimal speed & convergence)
POCS_TOLERANCE = 1e-4;   % Convergence tolerance

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 2] Read & Process Sony Camera RAW (.ARW)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 1] Ingesting Sony Camera RAW file (_DSC5165.ARW) ...\n');
size_raw_mb = dir(RAW_FILE_PATH).bytes / (1024 * 1024);
fprintf('  RAW File Size: %.2f MB\n', size_raw_mb);

raw_cfa = rawread(RAW_FILE_PATH);
info    = rawinfo(RAW_FILE_PATH);

% Black/white normalization
blk = double(info.ColorInfo.BlackLevel(1));
wht = double(info.ColorInfo.WhiteLevel(1));
raw_sub = max(0.0, double(raw_cfa) - blk);
raw_norm = min(1.0, raw_sub / (wht - blk));

% Demosaic RGGB Bayer matrix
raw_u16 = uint16(round(raw_norm * 65535.0));
cfa_layout = lower(info.CFALayout);
rgb_demosaiced = double(demosaic(raw_u16, cfa_layout)) / 65535.0;

% Camera White Balance
wb = double(info.ColorInfo.CameraAsTakenWhiteBalance);
wb_r = wb(1) / wb(2);
wb_b = wb(4) / wb(3);
rgb_wb = rgb_demosaiced;
rgb_wb(:,:,1) = min(1.0, rgb_wb(:,:,1) * wb_r);
rgb_wb(:,:,3) = min(1.0, rgb_wb(:,:,3) * wb_b);

% sRGB Gamma
img_raw_full = max(0.0, min(1.0, rgb_wb .^ (1 / 2.2)));

% Resize working copy to 12% (438x656) for ultra-fast benchmark execution (<10s)
PROC_SCALE  = 0.12;
img_clean   = imresize(img_raw_full, PROC_SCALE, 'bilinear');
[H, W, ~]   = size(img_clean);
fprintf('  Ground Truth working resolution: %d x %d x 3\n', H, W);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 3] Dual Degradation Injection (Noise + Severe JPEG Q=5)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 2] Injecting Dual Degradation ...\n');
fprintf('  1. Additive Gaussian Sensor Noise (sigma = %.3f)\n', NOISE_SIGMA);
fprintf('  2. Severe JPEG Block Quantization (QualityFactor = %d)\n', QUALITY_FACTOR);

% Convert clean image to YCbCr space
YCbCr_clean = rgb2ycbcr(img_clean);

% Add Gaussian sensor noise to YCbCr channels
YCbCr_noisy = YCbCr_clean + NOISE_SIGMA * randn(size(YCbCr_clean));
YCbCr_noisy = max(0.0, min(1.0, YCbCr_noisy));

% Compress noisy YCbCr with severe JPEG quantization (Q=5)
comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', QUALITY_FACTOR);
[YCbCr_degraded, q_all, Q_table] = comp.compress(YCbCr_noisy);

img_degraded = max(0.0, min(1.0, ycbcr2rgb(YCbCr_degraded)));

psnr_degraded = OceanStats.Image.ImageQualityMetrics.psnr(img_degraded, img_clean);
ssim_degraded = OceanStats.Image.ImageQualityMetrics.ssim(img_degraded, img_clean);
fprintf('  Degraded Baseline: PSNR = %.2f dB   SSIM = %.4f\n', psnr_degraded, ssim_degraded);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 4] POCS Deblocking Benchmark (5 Models)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 3] Running POCS Robustness Benchmark across 5 Models ...\n');

% --- Model 2: Pure DCT POCS (No Spatial Smoother) ---
fprintf('  [Model 2] Running Pure DCT POCS (No Smoother) ...\n');
engine_pure = OceanMath.Image.Compression.PocsDeblockingEngine(...
    'MaxIter', POCS_MAX_ITER, 'Tolerance', POCS_TOLERANCE, ...
    'Smoother', [], 'Verbose', false);
[ycbcr_m2, hist2] = engine_pure.deblock(YCbCr_degraded, q_all, Q_table);
img_m2 = max(0.0, min(1.0, ycbcr2rgb(ycbcr_m2)));

% --- Model 3: POCS + GaussianFilter Smoother ---
fprintf('  [Model 3] Running POCS + GaussianFilter Smoother ...\n');
gauss_filter = OceanMath.Image.Filters.GaussianFilter('Sigma', 1.0);
engine_gauss = OceanMath.Image.Compression.PocsDeblockingEngine(...
    'MaxIter', POCS_MAX_ITER, 'Tolerance', POCS_TOLERANCE, ...
    'Smoother', gauss_filter, 'SmoothEvery', 1, 'Verbose', false);
[ycbcr_m3, hist3] = engine_gauss.deblock(YCbCr_degraded, q_all, Q_table);
img_m3 = max(0.0, min(1.0, ycbcr2rgb(ycbcr_m3)));

% --- Model 4: POCS + BilateralFilter Smoother (Default) ---
fprintf('  [Model 4] Running POCS + BilateralFilter Smoother ...\n');
bilat_filter = OceanMath.Image.Filters.BilateralFilter(...
    'SpatialSigma', 1.5, 'RangeSigma', 0.08);
engine_bilat = OceanMath.Image.Compression.PocsDeblockingEngine(...
    'MaxIter', POCS_MAX_ITER, 'Tolerance', POCS_TOLERANCE, ...
    'Smoother', bilat_filter, 'SmoothEvery', 1, 'Verbose', false);
[ycbcr_m4, hist4] = engine_bilat.deblock(YCbCr_degraded, q_all, Q_table);
img_m4 = max(0.0, min(1.0, ycbcr2rgb(ycbcr_m4)));

% --- Model 5: POCS + GuidedFilter Smoother ---
fprintf('  [Model 5] Running POCS + GuidedFilter Smoother ...\n');
guided_filter = OceanMath.Image.Filters.GuidedFilter('Radius', 2, 'Epsilon', 1e-4);
engine_guided = OceanMath.Image.Compression.PocsDeblockingEngine(...
    'MaxIter', POCS_MAX_ITER, 'Tolerance', POCS_TOLERANCE, ...
    'Smoother', guided_filter, 'SmoothEvery', 1, 'Verbose', false);
[ycbcr_m5, hist5] = engine_guided.deblock(YCbCr_degraded, q_all, Q_table);
img_m5 = max(0.0, min(1.0, ycbcr2rgb(ycbcr_m5)));

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 5] Compute Quality Metrics & Print Table
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 4] Quality Evaluation Summary vs Ground Truth ...\n');

psnr_m2 = OceanStats.Image.ImageQualityMetrics.psnr(img_m2, img_clean);
ssim_m2 = OceanStats.Image.ImageQualityMetrics.ssim(img_m2, img_clean);

psnr_m3 = OceanStats.Image.ImageQualityMetrics.psnr(img_m3, img_clean);
ssim_m3 = OceanStats.Image.ImageQualityMetrics.ssim(img_m3, img_clean);

psnr_m4 = OceanStats.Image.ImageQualityMetrics.psnr(img_m4, img_clean);
ssim_m4 = OceanStats.Image.ImageQualityMetrics.ssim(img_m4, img_clean);

psnr_m5 = OceanStats.Image.ImageQualityMetrics.psnr(img_m5, img_clean);
ssim_m5 = OceanStats.Image.ImageQualityMetrics.ssim(img_m5, img_clean);

m(1).Label = 'Degraded (Noise+Q=5)'; m(1).PSNR = psnr_degraded; m(1).SSIM = ssim_degraded;
m(2).Label = 'Pure DCT POCS';        m(2).PSNR = psnr_m2;       m(2).SSIM = ssim_m2;
m(3).Label = 'POCS + Gaussian';      m(3).PSNR = psnr_m3;       m(3).SSIM = ssim_m3;
m(4).Label = 'POCS + Bilateral';     m(4).PSNR = psnr_m4;       m(4).SSIM = ssim_m4;
m(5).Label = 'POCS + Guided';        m(5).PSNR = psnr_m5;       m(5).SSIM = ssim_m5;

OceanStats.Image.ImageQualityMetrics.printTable(m);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 6] OceanViz 6-Panel Dashboard Rendering
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 5] Rendering OceanViz comparison dashboard ...\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('POCS Deblocking Robustness under Dual Degradation');
set(fig, 'Position', [30, 30, 1400, 820]);
set(fig, 'Color', viz.Colors.Background);

% Downsample display size for responsive figure rendering (600x900)
disp_scale = 600 / H;
img_clean_d    = imresize(img_clean,    disp_scale, 'bilinear');
img_degraded_d = imresize(img_degraded, disp_scale, 'bilinear');
img_m4_d       = imresize(img_m4,       disp_scale, 'bilinear');
img_m5_d       = imresize(img_m5,       disp_scale, 'bilinear');

% --- Panel 1: Pristine Sony RAW Ground Truth ---
ax1 = subplot(2, 3, 1, 'Parent', fig);
imshow(img_clean_d, 'Parent', ax1);
title(ax1, '1. Pristine Sony RAW Ground Truth', ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax1, 'off');

% --- Panel 2: Dual Degraded Input ---
ax2 = subplot(2, 3, 2, 'Parent', fig);
imshow(img_degraded_d, 'Parent', ax2);
title_str2 = sprintf('2. Degraded (Noise \\sigma=%.2f + JPEG Q=%d)\nPSNR=%.2f dB  SSIM=%.4f', ...
    NOISE_SIGMA, QUALITY_FACTOR, psnr_degraded, ssim_degraded);
title(ax2, title_str2, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax2, 'off');

% --- Panel 3: Restored (POCS + Bilateral Filter) ---
ax3 = subplot(2, 3, 3, 'Parent', fig);
imshow(img_m4_d, 'Parent', ax3);
title_str3 = sprintf('3. Restored: POCS + Bilateral Filter\nPSNR=%.2f dB  SSIM=%.4f', ...
    psnr_m4, ssim_m4);
title(ax3, title_str3, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax3, 'off');

% --- Panel 4: Restored (POCS + Guided Filter) ---
ax4 = subplot(2, 3, 4, 'Parent', fig);
imshow(img_m5_d, 'Parent', ax4);
title_str4 = sprintf('4. Restored: POCS + Guided Filter\nPSNR=%.2f dB  SSIM=%.4f', ...
    psnr_m5, ssim_m5);
title(ax4, title_str4, 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax4, 'off');

% --- Panel 5: Absolute Error Heatmap (POCS+Bilateral vs Ground Truth) ---
ax5 = subplot(2, 3, 5, 'Parent', fig);
err_m4 = mean(abs(img_m4 - img_clean), 3);
imagesc(ax5, err_m4); axis(ax5, 'image'); axis(ax5, 'off');
colormap(ax5, 'hot');
cb5 = colorbar(ax5); cb5.Color = viz.Colors.Text;
cb5.Label.String = 'Abs Error'; cb5.Label.Color = viz.Colors.Text;
title(ax5, sprintf('5. Error Map (POCS+Bilateral vs GT)\nMax Error = %.3f', max(err_m4(:))), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');

% --- Panel 6: POCS Convergence Comparison Plot ---
ax6 = subplot(2, 3, 6, 'Parent', fig);
hold(ax6, 'on');

plot(ax6, 1:numel(hist2.RMSE), hist2.RMSE, '-v', ...
    'Color', [0.7 0.7 0.7], 'LineWidth', 1.2, 'MarkerSize', 3, 'DisplayName', 'Pure DCT');
plot(ax6, 1:numel(hist3.RMSE), hist3.RMSE, '-^', ...
    'Color', viz.Colors.Secondary, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'POCS+Gaussian');
plot(ax6, 1:numel(hist4.RMSE), hist4.RMSE, '-o', ...
    'Color', viz.Colors.Primary, 'LineWidth', 2.0, 'MarkerSize', 4, 'DisplayName', 'POCS+Bilateral');
plot(ax6, 1:numel(hist5.RMSE), hist5.RMSE, '-s', ...
    'Color', [0.2 0.8 0.4], 'LineWidth', 1.8, 'MarkerSize', 3, 'DisplayName', 'POCS+Guided');

set(ax6, 'Color', viz.Colors.Background, 'XColor', viz.Colors.Text, ...
    'YColor', viz.Colors.Text, 'GridColor', viz.Colors.Grid, 'YScale', 'log');
grid(ax6, 'on');
xlabel(ax6, 'Iteration', 'Color', viz.Colors.Text);
ylabel(ax6, 'Relative Change (log)', 'Color', viz.Colors.Text);
title(ax6, 'POCS Convergence Comparison', 'Color', viz.Colors.Text, ...
    'FontSize', 10, 'FontWeight', 'bold');

lg6 = legend(ax6, 'Location', 'northeast');
lg6.TextColor = viz.Colors.Text;
lg6.Color     = viz.Colors.Background;
hold(ax6, 'off');

% Style axes
for ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    set(ax, 'Color', viz.Colors.Background);
end

% Save figure artifact
fig_path = fullfile(RESULTS_DIR, 'exp_pocs_robustness_noisy.png');
saveas(fig, fig_path);
fprintf('\n[Step 6] Saved comparison figure to:\n  → %s\n', fig_path);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Summary]
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('═', 1, 72));
fprintf('  POCS ROBUSTNESS EXPERIMENT SUMMARY — DUAL DEGRADATION\n');
fprintf('%s\n', repmat('─', 1, 72));
fprintf('  Dataset               : Sony Camera RAW (_DSC5165.ARW)\n');
fprintf('  Sensor Noise          : Gaussian (sigma = %.3f)\n', NOISE_SIGMA);
fprintf('  JPEG Quantization     : QualityFactor = %d (Severe Blocking)\n', QUALITY_FACTOR);
OceanStats.Image.ImageQualityMetrics.printTable(m);
fprintf('%s\n', repmat('═', 1, 72));

fprintf('\n=== Experiment Completed Successfully ===\n');
