% EXP_IMAGE_DEBLOCKING  Exercise 4.3 — Image Deblocking via POCS + Guided Restoration.
%
%   DATASET:
%     Synthetic or user-supplied image (see IMAGE_SOURCE below).
%     Synthetic: procedurally generated 256x256 mixed image (gradient + shapes + texture).
%     Real:      any image file readable by imread().
%
%   MODEL:
%     1. JpegSyntheticCompressor  — Software DCT encoder with custom quantization table
%     2. PocsDeblockingEngine     — POCS alternating projection (spatial smooth + DCT clamp)
%     3. ChromaSubsampler         — YCbCr 4:2:0 subsampling + GuidedFilter upsampling
%     4. ImageQualityMetrics      — PSNR, SSIM comparison table
%
%   ASSUMPTIONS:
%     - Input image is in RGB colour space.
%     - Synthetic images: double [0,1]. Real images: uint8 -> normalised to [0,1].
%     - POCS operates on double [0,1] images.
%     - YCbCr conversion uses MATLAB's built-in rgb2ycbcr / ycbcr2rgb.
%     - Guided Filter upsampling uses luminance Y as guidance for Cb, Cr.
%
%   REPRODUCIBILITY:
%     rng(42, 'twister') is called at the top. Synthetic image seeded independently.
%
%   EXERCISE PARTS:
%     Part 1 — Spatial deblocking: POCS on Y channel only
%     Part 2 — Full DCT deblocking: POCS on all channels in YCbCr space
%     Part 3 — Chroma subsampling + Guided upsampling
%     Part 4 — Quality evaluation: PSNR/SSIM table + OceanViz dashboard
%
%   Ref: Szeliski "Computer Vision: Algorithms and Applications" 2e, §4.3, p. 234.

clc; clear; close all;
rng(42, 'twister');

fprintf('=== Ex 4.3 — Image Deblocking: POCS + Guided Restoration ===\n\n');

%% ─────────────────────────────────────────────────────────────────────────────
%  SELF-LOCATE: derive workspace root from this script's absolute path
%  This fixes MATLAB run() path issues — works from any current directory.
%  ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = fileparts(mfilename('fullpath'));           % .../Experiments/Scripts/Computer Vision
EXP_DIR     = fileparts(SCRIPT_DIR);                     % .../Experiments/Scripts
SCRIPTS_DIR = fileparts(EXP_DIR);                        % .../Experiments
WORKSPACE   = fileparts(SCRIPTS_DIR);                    % c:/Ocean Library/Matlab code practice

%% ─────────────────────────────────────────────────────────────────────────────
%  CONFIGURATION  (edit these to switch modes)
%  ─────────────────────────────────────────────────────────────────────────────

% IMAGE_SOURCE: 'synthetic' or an absolute file path to a real image
IMAGE_SOURCE    = 'synthetic';
% IMAGE_SOURCE  = fullfile(WORKSPACE, 'Data', 'images', 'consu.jpg');

QUALITY_FACTOR  = 10;     % JPEG quality [1=worst, 100=best]; low → severe blocking
POCS_MAX_ITER   = 40;     % Maximum POCS iterations
POCS_TOLERANCE  = 1e-5;   % Convergence threshold
CHROMA_SCHEME   = '4:2:0'; % Chrominance subsampling scheme
UPSAMPLE_METHOD = 'guided'; % 'guided' | 'jbf' | 'nearest'
RESULTS_DIR     = fullfile(WORKSPACE, 'Results');
SAVE_FIGURES    = true;


%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 1] Load or generate the clean input image
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 1] Loading / generating input image ...\n');

if strcmp(IMAGE_SOURCE, 'synthetic')
    gen = OceanData.Generators.SyntheticImageGenerator(...
        'Height', 256, 'Width', 256, 'Mode', 'mixed', 'Seed', 42);
    img_clean = gen.generate();   % (H x W x 3) double [0, 1]
    source_label = 'Synthetic (256x256, mixed)';
    fprintf('  Mode: SYNTHETIC  |  Size: %dx%dx%d\n', size(img_clean,1), size(img_clean,2), size(img_clean,3));
else
    assert(isfile(IMAGE_SOURCE), 'Image file not found: %s', IMAGE_SOURCE);
    img_raw = imread(IMAGE_SOURCE);
    if size(img_raw, 3) == 1
        img_raw = cat(3, img_raw, img_raw, img_raw);  % Greyscale → RGB
    end
    img_clean = double(img_raw) / 255.0;
    [~, fname, fext] = fileparts(IMAGE_SOURCE);
    source_label = sprintf('Real image (%s%s)', fname, fext);
    fprintf('  Mode: REAL FILE  |  %s  |  Size: %dx%dx%d\n', ...
        source_label, size(img_clean,1), size(img_clean,2), size(img_clean,3));
end

[H, W, ~] = size(img_clean);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 2] Synthetic JPEG compression (in YCbCr space per JPEG standard)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 2] Applying synthetic JPEG compression (QualityFactor=%d) ...\n', QUALITY_FACTOR);

comp = OceanMath.Image.Compression.JpegSyntheticCompressor(...
    'QualityFactor', QUALITY_FACTOR);

% Convert clean image to YCbCr space first so DCT indices (q_all) match Y, Cb, Cr
YCbCr_clean = rgb2ycbcr(img_clean);
[YCbCr_compressed, q_all, Q_table] = comp.compress(YCbCr_clean);
img_compressed = max(0, min(1, ycbcr2rgb(YCbCr_compressed)));

fprintf('  Quantization table (luma):\n');
disp(Q_table);

% Sanity check: PSNR should be low at severe quality
psnr_compressed = OceanStats.Image.ImageQualityMetrics.psnr(img_compressed, img_clean);
ssim_compressed = OceanStats.Image.ImageQualityMetrics.ssim(img_compressed, img_clean);
fprintf('  Compressed PSNR = %.2f dB   SSIM = %.4f\n', psnr_compressed, ssim_compressed);
assert(psnr_compressed < 35, ...
    'PSNR=%.1f dB: blocking may not be severe enough for demo (try lower QualityFactor)', psnr_compressed);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 3] Part 1 — POCS Deblocking (Y channel / spatial domain only)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 3] Part 1: POCS Deblocking (Y channel only) ...\n');

Y_comp  = YCbCr_compressed(:,:,1);

% Build engine (BilateralFilter smoother, default)
engine_part1 = OceanMath.Image.Compression.PocsDeblockingEngine(...
    'MaxIter',     POCS_MAX_ITER, ...
    'Tolerance',   POCS_TOLERANCE, ...
    'BlockSize',   8, ...
    'SmoothEvery', 2, ...       % Apply smoothing every 2 iterations (less aggressive)
    'Verbose',     true);

% q_all{1} is the Y channel quantization indices
[Y_deblocked, history_part1] = engine_part1.deblock(Y_comp, q_all(1), Q_table);

% Reconstruct full RGB from deblocked Y + original compressed Cb/Cr
YCbCr_part1        = YCbCr_compressed;
YCbCr_part1(:,:,1) = Y_deblocked;
img_part1          = max(0, min(1, ycbcr2rgb(YCbCr_part1)));

psnr_part1 = OceanStats.Image.ImageQualityMetrics.psnr(img_part1, img_clean);
ssim_part1 = OceanStats.Image.ImageQualityMetrics.ssim(img_part1, img_clean);
fprintf('  [Part 1] PSNR = %.2f dB   SSIM = %.4f   Iters = %d\n', ...
    psnr_part1, ssim_part1, history_part1.NumIter);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 4] Part 2 — POCS Deblocking (Full YCbCr DCT space)
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 4] Part 2: POCS Deblocking (all YCbCr channels) ...\n');

engine_part2 = OceanMath.Image.Compression.PocsDeblockingEngine(...
    'MaxIter',     POCS_MAX_ITER, ...
    'Tolerance',   POCS_TOLERANCE, ...
    'BlockSize',   8, ...
    'SmoothEvery', 1, ...
    'Verbose',     true);

% De-block YCbCr channels using matching YCbCr q_all indices
[img_part2_ycbcr, history_part2] = engine_part2.deblock(YCbCr_compressed, q_all, Q_table);
img_part2 = max(0, min(1, ycbcr2rgb(img_part2_ycbcr)));

psnr_part2 = OceanStats.Image.ImageQualityMetrics.psnr(img_part2, img_clean);
ssim_part2 = OceanStats.Image.ImageQualityMetrics.ssim(img_part2, img_clean);
fprintf('  [Part 2] PSNR = %.2f dB   SSIM = %.4f   Iters = %d\n', ...
    psnr_part2, ssim_part2, history_part2.NumIter);


%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 5] Part 3 — Chroma Subsampling + Guided Upsampling
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 5] Part 3: Chroma subsampling (%s) + %s upsampling ...\n', ...
    CHROMA_SCHEME, UPSAMPLE_METHOD);

% Subsample on the POCS-deblocked YCbCr (Part 2 output is best starting point)
subsampler = OceanMath.Image.Compression.ChromaSubsampler(...
    'Scheme',           CHROMA_SCHEME, ...
    'UpsamplingMethod', UPSAMPLE_METHOD, ...
    'GuidedRadius',     2, ...
    'GuidedEpsilon',    1e-4);

[Y_full, Cb_sub, Cr_sub, sub_info] = subsampler.subsample(img_part2_ycbcr);
fprintf('  Original Cb/Cr size: %dx%d  →  Subsampled: %dx%d\n', ...
    H, W, size(Cb_sub,1), size(Cb_sub,2));

YCbCr_guided = subsampler.upsample(Y_full, Cb_sub, Cr_sub);
img_part3 = max(0, min(1, ycbcr2rgb(YCbCr_guided)));

% Also compute nearest-neighbour baseline for comparison
subsampler_nn = OceanMath.Image.Compression.ChromaSubsampler(...
    'Scheme', CHROMA_SCHEME, 'UpsamplingMethod', 'nearest');
YCbCr_nn = subsampler_nn.upsample(Y_full, Cb_sub, Cr_sub);
img_nn = max(0, min(1, ycbcr2rgb(YCbCr_nn)));

psnr_guided = OceanStats.Image.ImageQualityMetrics.psnr(img_part3, img_clean);
ssim_guided = OceanStats.Image.ImageQualityMetrics.ssim(img_part3, img_clean);
psnr_nn     = OceanStats.Image.ImageQualityMetrics.psnr(img_nn,    img_clean);
ssim_nn     = OceanStats.Image.ImageQualityMetrics.ssim(img_nn,    img_clean);

fprintf('  Guided upsampling:   PSNR = %.2f dB   SSIM = %.4f\n', psnr_guided, ssim_guided);
fprintf('  Nearest (baseline):  PSNR = %.2f dB   SSIM = %.4f\n', psnr_nn,     ssim_nn);

% Sanity check: guided should outperform nearest
if psnr_guided >= psnr_nn
    fprintf('  [PASS] Guided PSNR (%.2f) >= Nearest PSNR (%.2f)\n', psnr_guided, psnr_nn);
else
    warning('EXP:QualityRegression', ...
        'Guided PSNR (%.2f) < Nearest PSNR (%.2f) - check parameters.', psnr_guided, psnr_nn);
end

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 6] Part 4 — Quality Evaluation Summary
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 6] Part 4: Quality evaluation summary\n');

metrics(1).Label = 'JPEG Compressed';     metrics(1).PSNR = psnr_compressed; metrics(1).SSIM = ssim_compressed;
metrics(2).Label = 'POCS (Y only)';       metrics(2).PSNR = psnr_part1;      metrics(2).SSIM = ssim_part1;
metrics(3).Label = 'POCS (Full YCbCr)';  metrics(3).PSNR = psnr_part2;      metrics(3).SSIM = ssim_part2;
metrics(4).Label = 'POCS + Guided Up';   metrics(4).PSNR = psnr_guided;     metrics(4).SSIM = ssim_guided;
metrics(5).Label = 'POCS + Nearest Up';  metrics(5).PSNR = psnr_nn;         metrics(5).SSIM = ssim_nn;

OceanStats.Image.ImageQualityMetrics.printTable(metrics);

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 7] OceanViz Dashboard
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[Step 7] Rendering OceanViz deblocking dashboard ...\n');

viz = OceanViz.Core('Theme', 'Dark');

% Primary 6-panel dashboard (Part 1 deblocked)
viz_data.Original      = img_clean;
viz_data.Compressed    = img_compressed;
viz_data.Deblocked     = img_part2;   % Full POCS (Part 2) as primary result
viz_data.GuidedOut     = img_part3;
viz_data.History       = history_part2;
viz_data.Metrics       = metrics;
viz_data.QualityFactor = QUALITY_FACTOR;
viz_data.Tolerance     = POCS_TOLERANCE;

fig1 = viz.createFigure('Ex 4.3 — POCS Deblocking Dashboard');
set(fig1, 'Position', [30, 30, 1400, 820]);
set(fig1, 'Color', viz.Colors.Background);

ax1 = subplot(2, 3, 1, 'Parent', fig1);
imshow(img_clean,      'Parent', ax1);
title(ax1, 'Ground Truth', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
axis(ax1, 'off');

ax2 = subplot(2, 3, 2, 'Parent', fig1);
imshow(img_compressed, 'Parent', ax2);
title(ax2, sprintf('JPEG Compressed [Q=%d]', QUALITY_FACTOR), ...
    'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
axis(ax2, 'off');

ax3 = subplot(2, 3, 3, 'Parent', fig1);
imshow(img_part2,      'Parent', ax3);
title(ax3, 'POCS Deblocked (Full YCbCr)', ...
    'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
axis(ax3, 'off');

% Error heatmaps
ax4 = subplot(2, 3, 4, 'Parent', fig1);
err_comp = mean(abs(img_compressed - img_clean), 3);
imagesc(ax4, err_comp); axis(ax4, 'image'); axis(ax4, 'off');
colormap(ax4, 'hot');
cb4 = colorbar(ax4); cb4.Color = viz.Colors.Text;
title(ax4, sprintf('Error: Compressed  PSNR=%.1f dB', psnr_compressed), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');

ax5 = subplot(2, 3, 5, 'Parent', fig1);
err_pocs = mean(abs(img_part2 - img_clean), 3);
imagesc(ax5, err_pocs); axis(ax5, 'image'); axis(ax5, 'off');
colormap(ax5, 'hot');
cb5 = colorbar(ax5); cb5.Color = viz.Colors.Text;
title(ax5, sprintf('Error: POCS  PSNR=%.1f dB', psnr_part2), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');

% Convergence curve
ax6 = subplot(2, 3, 6, 'Parent', fig1);
hold(ax6, 'on');
iters2 = 1:numel(history_part2.RMSE);
plot(ax6, iters2, history_part2.RMSE, '-o', ...
    'Color', viz.Colors.Primary, 'LineWidth', 2, ...
    'MarkerSize', 4, 'MarkerFaceColor', viz.Colors.Primary, ...
    'DisplayName', 'POCS Full YCbCr');

iters1 = 1:numel(history_part1.RMSE);
plot(ax6, iters1, history_part1.RMSE, '-s', ...
    'Color', viz.Colors.Secondary, 'LineWidth', 1.5, ...
    'MarkerSize', 4, 'MarkerFaceColor', viz.Colors.Secondary, ...
    'DisplayName', 'POCS Y only');

set(ax6, 'Color', viz.Colors.Background, 'XColor', viz.Colors.Text, ...
    'YColor', viz.Colors.Text, 'GridColor', viz.Colors.Grid, 'YScale', 'log');
grid(ax6, 'on');
xlabel(ax6, 'Iteration', 'Color', viz.Colors.Text);
ylabel(ax6, 'Relative Change (log)', 'Color', viz.Colors.Text);
title(ax6, 'POCS Convergence', 'Color', viz.Colors.Text, ...
    'FontSize', 10, 'FontWeight', 'bold');
lg6 = legend(ax6, 'Location', 'northeast');
lg6.TextColor = viz.Colors.Text;
lg6.Color     = viz.Colors.Background;
hold(ax6, 'off');

% Style all axes backgrounds
for ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    set(ax, 'Color', viz.Colors.Background);
end

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 8] Second figure: Chroma upsampling comparison
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('[Step 8] Rendering chroma upsampling comparison ...\n');

fig2 = viz.createFigure('Ex 4.3 — Chrominance Upsampling Comparison');
set(fig2, 'Position', [60, 60, 1100, 400]);
set(fig2, 'Color', viz.Colors.Background);

ax_a = subplot(1, 3, 1, 'Parent', fig2);
imshow(img_part2, 'Parent', ax_a);
title(ax_a, 'POCS (Before Subsampling)', 'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax_a, 'off');

ax_b = subplot(1, 3, 2, 'Parent', fig2);
imshow(img_nn, 'Parent', ax_b);
title(ax_b, sprintf('Nearest Up  PSNR=%.1f dB', psnr_nn), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax_b, 'off');

ax_c = subplot(1, 3, 3, 'Parent', fig2);
imshow(img_part3, 'Parent', ax_c);
title(ax_c, sprintf('Guided Up  PSNR=%.1f dB', psnr_guided), ...
    'Color', viz.Colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
axis(ax_c, 'off');

for ax = [ax_a, ax_b, ax_c]
    set(ax, 'Color', viz.Colors.Background);
end

%% ─────────────────────────────────────────────────────────────────────────────
%  [Step 9] Save results
%  ─────────────────────────────────────────────────────────────────────────────
if SAVE_FIGURES
    if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end
    saveas(fig1, fullfile(RESULTS_DIR, 'exp_deblocking_dashboard.png'));
    saveas(fig2, fullfile(RESULTS_DIR, 'exp_deblocking_chroma_comparison.png'));
    fprintf('\n[Step 9] Saved figures to Results/\n');
    fprintf('  → Results/exp_deblocking_dashboard.png\n');
    fprintf('  → Results/exp_deblocking_chroma_comparison.png\n');
end

%% ─────────────────────────────────────────────────────────────────────────────
%  [Summary]
%  ─────────────────────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('═', 1, 65));
fprintf('  EXP SUMMARY — Ex 4.3 Image Deblocking\n');
fprintf('%s\n', repmat('─', 1, 65));
fprintf('  Dataset        : %s\n', source_label);
fprintf('  Image size     : %d x %d x 3\n', H, W);
fprintf('  JPEG Q factor  : %d\n', QUALITY_FACTOR);
fprintf('  Chroma scheme  : %s | Upsampler: %s\n', CHROMA_SCHEME, UPSAMPLE_METHOD);
OceanStats.Image.ImageQualityMetrics.printTable(metrics);
fprintf('%s\n', repmat('═', 1, 65));
fprintf('\n=== Experiment Completed Successfully ===\n');
