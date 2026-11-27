% EXP_WAVELET_COMPRESSION  Wavelet construction and applications — compression & entropy (Ex 3.21).
%
%   DATASET:        elon_musk.jpg (portrait, 8-bit grayscale luminance channel)
%   MODEL:          Three multiresolution representations:
%                     1. Laplacian Pyramid        (overcomplete, 4 levels)
%                     2. Haar Wavelet Lifting      (critically sampled, 4 levels)
%                     3. CDF 5/3 Wavelet Lifting   (critically sampled, 4 levels)
%   ASSUMPTIONS:
%     - Images are normalized to [0, 1] double. All entropy is in bits (log base 2).
%     - Quantization step q is applied uniformly to ALL coefficients in a subband.
%     - Entropy is computed from the INDICES of quantized coefficients
%       (i.e., X_idx = round(X / q)), not the reconstructed values.
%       This models the number of distinct symbols the entropy coder must encode.
%     - For the Laplacian pyramid, bpp is measured relative to the ORIGINAL image
%       size (N_orig), not the total pyramid size. This correctly penalizes the
%       33% redundancy overhead of the overcomplete representation.
%     - For critically-sampled wavelets, N_subband sums exactly to N_orig,
%       so the weighted entropy equals the true average bits per pixel.
%     - Only the luminance (L*) channel is analyzed. Color channels are ignored.
%     - The rate-distortion sweep uses q from 0.25/255 (near-lossless) to 64/255
%       (highly lossy), covering the full practical compression range.
%     - CDF 5/3 lifting uses integer arithmetic (floor/predict/update steps).
%       The image is therefore scaled to [0, 255] integer space before forward
%       transform and scaled back after reconstruction. Quantization steps for
%       CDF 5/3 are specified in gray-level units (e.g. q_int = 0.25 for 1/4
%       gray level). Entropy and bpp accounting is identical to the other methods.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Antigravity Code Assistant
%   Date: 2026-07-15

clc; clear; rng(42, 'twister');
close all;

% === 1. Data Loading ===
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
results_dir  = fullfile(script_dir, 'Results');
addpath(genpath(project_root));

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

image_name = 'elon_musk.jpg';
image_path = fullfile(project_root, 'Data', 'images', image_name);
if ~exist(image_path, 'file')
    error('exp_wavelet_compression:ImageNotFound', ...
          'Test image not found: %s', image_path);
end

I_rgb = im2double(imread(image_path));

% Convert to luminance only (CIE L*a*b*, normalized to [0, 1])
% Assumption: This isolates luma for a fair single-channel compression comparison.
if size(I_rgb, 3) == 3
    I_lab = rgb2lab(I_rgb);
    I_luma = I_lab(:,:,1) / 100.0;   % L* is in [0,100]; normalize to [0,1]
else
    I_luma = I_rgb;
end

[H_orig, W_orig] = size(I_luma);
N_orig = H_orig * W_orig;

fprintf('Image: %s | Resolution: %d x %d | Luminance pixels: %d\n', ...
    image_name, W_orig, H_orig, N_orig);

% === 2. Experiment Configuration ===

% Quantization steps to sweep (in [0, 1] pixel value space)
% 1/4 gray level (as specified in Ex 3.21) = 0.25/255 ≈ 0.00098
% We sweep from near-lossless to coarsely lossy
q_steps_gray = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];
q_steps      = q_steps_gray / 255.0;       % for [0,1] space (Laplacian + Haar)
q_steps_int  = q_steps_gray;               % for [0,255] integer space (CDF 5/3)
n_q = numel(q_steps);

% Pyramid/Wavelet decomposition levels
num_levels = 4;

% CDF 5/3: scale image to integer space to satisfy integer-lifting contract.
% floor() in CDF 5/3 predict/update steps requires integer-valued samples;
% otherwise the round(X/q) quantization maps near-zero floats to a near-uniform
% distribution — maximum entropy and meaningless PSNR.
I_luma_int = round(I_luma * 255);   % [0, 255] integer (double type)

% Pre-allocate rate-distortion result matrices
% Rows = quantization steps, Cols = [LaplacianPyr, Haar, CDF53]
bpp_results  = zeros(n_q, 3);
psnr_results = zeros(n_q, 3);

% Method labels
method_names = {'Laplacian Pyramid', 'Haar Wavelet', 'CDF 5/3 Wavelet'};

% === Helper: Compute entropy of a band in bits ===
% Flattens X to a column vector, quantizes to integer indices, then calls
% OceanMath.Information.Entropy with Discrete mode and Base=2.
computeBandEntropy = @(X, q) ...
    OceanMath.Information.Entropy( ...
        round(X(:) / q), ...
        'TypeAsString', 'Discrete', ...
        'Base', 2);

% === Helper: PSNR (peak=1.0 since image is in [0,1]) ===
computePSNR = @(orig, recon) ...
    10 * log10(1.0 / mean((orig(:) - recon(:)).^2 + eps));

fprintf('\n--- Beginning rate-distortion sweep (%d q-steps x 3 methods) ---\n', n_q);

for qi = 1:n_q
    q = q_steps(qi);
    fprintf('  q = %.4f (%.2f/255)...  ', q, q_steps_gray(qi));

    % ------------------------------------------------------------------ %
    % Method 1: Laplacian Pyramid
    % ------------------------------------------------------------------ %
    pyramid = OceanMath.Image.MultiResolution.LaplacianPyramid();
    pyramid.build(I_luma, num_levels);

    total_bits_pyr = 0;
    quantized_levels = cell(1, num_levels);
    for lvl = 1:num_levels
        band = pyramid.Levels{lvl};
        % Quantize
        idx  = round(band / q);
        quantized_levels{lvl} = idx * q;   % dequantize for reconstruction
        % Entropy (bits) of this level's symbols
        H_band = computeBandEntropy(band, q);
        N_band = numel(band);
        total_bits_pyr = total_bits_pyr + N_band * H_band;
    end
    % bpp relative to original image size (penalizes overcompleteness)
    bpp_results(qi, 1) = total_bits_pyr / N_orig;

    % Reconstruct from quantized levels
    pyramid.Levels = quantized_levels;
    I_recon_pyr = pyramid.reconstruct();
    I_recon_pyr = min(max(I_recon_pyr, 0), 1);
    psnr_results(qi, 1) = computePSNR(I_luma, I_recon_pyr);

    % ------------------------------------------------------------------ %
    % Method 2: Haar Wavelet
    % ------------------------------------------------------------------ %
    haar = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'Haar', 'Levels', num_levels);
    coeffs_haar = haar.forward(I_luma);

    total_bits_haar = 0;
    % Quantize LL subband
    idx_LL = round(coeffs_haar.LL / q);
    coeffs_haar_q = coeffs_haar;
    coeffs_haar_q.LL = idx_LL * q;
    H_LL = computeBandEntropy(coeffs_haar.LL, q);
    total_bits_haar = total_bits_haar + numel(coeffs_haar.LL) * H_LL;

    % Quantize all detail subbands
    for lvl = 1:num_levels
        det = coeffs_haar.Levels{lvl};
        det_q = struct();
        for fn = {'LH', 'HL', 'HH'}
            fname = fn{1};
            band  = det.(fname);
            idx   = round(band / q);
            det_q.(fname) = idx * q;
            H_band = computeBandEntropy(band, q);
            total_bits_haar = total_bits_haar + numel(band) * H_band;
        end
        coeffs_haar_q.Levels{lvl} = det_q;
    end
    bpp_results(qi, 2) = total_bits_haar / N_orig;

    % Reconstruct
    I_recon_haar = haar.inverse(coeffs_haar_q);
    I_recon_haar = min(max(I_recon_haar, 0), 1);
    psnr_results(qi, 2) = computePSNR(I_luma, I_recon_haar);

    % ------------------------------------------------------------------ %
    % Method 3: CDF 5/3 Wavelet  [integer lifting in [0,255] space]
    % ------------------------------------------------------------------ %
    q_int = q_steps_int(qi);    % quantization step in gray-level units [0,255]
    cdf53 = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'CDF53', 'Levels', num_levels);
    coeffs_cdf = cdf53.forward(I_luma_int);   % forward on integer-scaled image

    total_bits_cdf = 0;
    % Quantize LL
    idx_LL = round(coeffs_cdf.LL / q_int);
    coeffs_cdf_q = coeffs_cdf;
    coeffs_cdf_q.LL = idx_LL * q_int;
    H_LL = computeBandEntropy(coeffs_cdf.LL, q_int);
    total_bits_cdf = total_bits_cdf + numel(coeffs_cdf.LL) * H_LL;

    % Quantize all detail subbands
    for lvl = 1:num_levels
        det = coeffs_cdf.Levels{lvl};
        det_q = struct();
        for fn = {'LH', 'HL', 'HH'}
            fname = fn{1};
            band  = det.(fname);
            idx   = round(band / q_int);
            det_q.(fname) = idx * q_int;
            H_band = computeBandEntropy(band, q_int);
            total_bits_cdf = total_bits_cdf + numel(band) * H_band;
        end
        coeffs_cdf_q.Levels{lvl} = det_q;
    end
    bpp_results(qi, 3) = total_bits_cdf / N_orig;

    % Reconstruct: inverse in integer space, then scale back to [0,1]
    I_recon_cdf_int = cdf53.inverse(coeffs_cdf_q);
    I_recon_cdf = min(max(I_recon_cdf_int / 255.0, 0), 1);
    psnr_results(qi, 3) = computePSNR(I_luma, I_recon_cdf);

    fprintf('bpp=[%.2f, %.2f, %.2f]  PSNR=[%.1f, %.1f, %.1f] dB\n', ...
        bpp_results(qi,1), bpp_results(qi,2), bpp_results(qi,3), ...
        psnr_results(qi,1), psnr_results(qi,2), psnr_results(qi,3));
end

fprintf('--- Sweep complete ---\n\n');

% === 3. Entropy Audit at the 1/4 Gray-Level Quantization Step ===
% Report the per-band entropy breakdown at q = 1/(4*255) ≈ 0.00098
% This is the specific quantization level called out in the exercise.
fprintf('=== Per-Band Entropy Audit at q = 1/4 gray level (%.5f) ===\n', 1/(4*255));

q_audit = 1 / (4 * 255);

% --- Laplacian Pyramid ---
fprintf('\n[Laplacian Pyramid]\n');
pyramid_audit = OceanMath.Image.MultiResolution.LaplacianPyramid();
pyramid_audit.build(I_luma, num_levels);
total_pyr_bits = 0;
for lvl = 1:num_levels
    band = pyramid_audit.Levels{lvl};
    H_band = computeBandEntropy(band, q_audit);
    N_band = numel(band);
    [bh, bw] = size(band);
    fprintf('  Level %d (%4dx%4d): H = %.4f bits  |  pixels = %7d  |  contribution = %.4f bpp\n', ...
        lvl, bw, bh, H_band, N_band, N_band * H_band / N_orig);
    total_pyr_bits = total_pyr_bits + N_band * H_band;
end
fprintf('  >> Total bpp (incl. overcompleteness overhead): %.4f bits/pixel\n', total_pyr_bits / N_orig);

% --- Haar Wavelet ---
fprintf('\n[Haar Wavelet]\n');
haar_audit = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'Haar', 'Levels', num_levels);
c_haar = haar_audit.forward(I_luma);
total_haar_bits = 0;
H_LL = computeBandEntropy(c_haar.LL, q_audit);
N_LL = numel(c_haar.LL);
fprintf('  LL  (%4dx%4d): H = %.4f bits  |  pixels = %7d\n', ...
    size(c_haar.LL,2), size(c_haar.LL,1), H_LL, N_LL);
total_haar_bits = total_haar_bits + N_LL * H_LL;
for lvl = 1:num_levels
    det = c_haar.Levels{lvl};
    for fn = {'LH', 'HL', 'HH'}
        fname = fn{1};
        band  = det.(fname);
        H_band = computeBandEntropy(band, q_audit);
        N_band = numel(band);
        fprintf('  L%d %s (%4dx%4d): H = %.4f bits  |  pixels = %7d\n', ...
            lvl, fname, size(band,2), size(band,1), H_band, N_band);
        total_haar_bits = total_haar_bits + N_band * H_band;
    end
end
fprintf('  >> Total bpp (critically sampled): %.4f bits/pixel\n', total_haar_bits / N_orig);

% --- CDF 5/3 Wavelet --- [integer space: q_audit_int = 0.25 gray levels]
q_audit_int = 0.25;    % 1/4 gray level in integer [0,255] units
fprintf('\n[CDF 5/3 Wavelet]  (integer space, q_int = %.2f gray levels)\n', q_audit_int);
cdf_audit = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'CDF53', 'Levels', num_levels);
c_cdf = cdf_audit.forward(I_luma_int);    % forward on [0,255] integer image
total_cdf_bits = 0;
H_LL = computeBandEntropy(c_cdf.LL, q_audit_int);
N_LL = numel(c_cdf.LL);
fprintf('  LL  (%4dx%4d): H = %.4f bits  |  pixels = %7d\n', ...
    size(c_cdf.LL,2), size(c_cdf.LL,1), H_LL, N_LL);
total_cdf_bits = total_cdf_bits + N_LL * H_LL;
for lvl = 1:num_levels
    det = c_cdf.Levels{lvl};
    for fn = {'LH', 'HL', 'HH'}
        fname = fn{1};
        band  = det.(fname);
        H_band = computeBandEntropy(band, q_audit_int);
        N_band = numel(band);
        fprintf('  L%d %s (%4dx%4d): H = %.4f bits  |  pixels = %7d\n', ...
            lvl, fname, size(band,2), size(band,1), H_band, N_band);
        total_cdf_bits = total_cdf_bits + N_band * H_band;
    end
end
fprintf('  >> Total bpp (critically sampled): %.4f bits/pixel\n', total_cdf_bits / N_orig);

% === 4. Visualization Dashboard ===
fprintf('\nRendering rate-distortion dashboard...\n');

viz = OceanViz.Core('Theme', 'Dark');
colors    = viz.Colors;
accent    = colors.Primary;    % ocean blue  — Laplacian Pyramid
secondary = colors.Secondary;  % coral/warm  — Haar
green     = [0.20, 0.75, 0.45]; % custom green — CDF 5/3
white     = colors.Text;

fig = figure('Name', 'Ex 3.21: Wavelet Compression Rate-Distortion', ...
    'Color', [0.11, 0.11, 0.13], ...
    'Position', [80, 80, 1300, 620]);

% ---- Panel 1: Rate-Distortion Curve (bpp vs PSNR) ----
ax1 = subplot(1, 2, 1);
viz.styleAxes(ax1);
hold(ax1, 'on');

% Plot with markers
plot(ax1, bpp_results(:,1), psnr_results(:,1), '-o', ...
    'Color', accent, 'LineWidth', 2.2, 'MarkerSize', 7, ...
    'MarkerFaceColor', accent, 'DisplayName', 'Laplacian Pyramid (overcomplete)');
plot(ax1, bpp_results(:,2), psnr_results(:,2), '-s', ...
    'Color', secondary, 'LineWidth', 2.2, 'MarkerSize', 7, ...
    'MarkerFaceColor', secondary, 'DisplayName', 'Haar Wavelet');
plot(ax1, bpp_results(:,3), psnr_results(:,3), '-^', ...
    'Color', green, 'LineWidth', 2.2, 'MarkerSize', 7, ...
    'MarkerFaceColor', green, 'DisplayName', 'CDF 5/3 Wavelet');

% Mark the 1/4 gray-level operating point
q_1_4 = 1 / (4 * 255);
% Find closest q step
[~, idx_audit] = min(abs(q_steps - q_1_4));
xline(ax1, bpp_results(idx_audit, 1), '--', 'Color', [0.6, 0.6, 0.6], ...
    'LineWidth', 1.0, 'HandleVisibility', 'off');
text(ax1, bpp_results(idx_audit, 1) + 0.05, min(psnr_results(:)) + 2, ...
    'q = 1/4 gray', 'Color', [0.6, 0.6, 0.6], 'FontSize', 8);

xlabel(ax1, 'Average Bits per Pixel (bpp)', 'Color', white, 'FontSize', 11);
ylabel(ax1, 'PSNR (dB)', 'Color', white, 'FontSize', 11);
title(ax1, 'Rate-Distortion Comparison', 'Color', white, 'FontSize', 13, 'FontWeight', 'bold');

leg = legend(ax1, 'Location', 'southeast', 'FontSize', 9);
leg.TextColor = white;
leg.Color     = [0.17, 0.19, 0.23];
leg.EdgeColor = [0.35, 0.37, 0.42];

grid(ax1, 'on');
hold(ax1, 'off');

% ---- Panel 2: Visual Reconstruction Comparison (high-q lossy case) ----
ax2 = subplot(1, 2, 2);
viz.styleAxes(ax2);
hold(ax2, 'on');

% Use the second-to-coarsest quantization (large q = visible degradation)
q_demo = q_steps(end - 2);   % e.g. 16/255

% Laplacian Pyramid reconstruction at q_demo
pyr_demo = OceanMath.Image.MultiResolution.LaplacianPyramid();
pyr_demo.build(I_luma, num_levels);
for lvl = 1:num_levels
    idx = round(pyr_demo.Levels{lvl} / q_demo);
    pyr_demo.Levels{lvl} = idx * q_demo;
end
I_pyr_demo = min(max(pyr_demo.reconstruct(), 0), 1);

% Haar Wavelet reconstruction at q_demo
haar_demo = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'Haar', 'Levels', num_levels);
c_h = haar_demo.forward(I_luma);
c_h.LL = round(c_h.LL / q_demo) * q_demo;
for lvl = 1:num_levels
    for fn = {'LH', 'HL', 'HH'}
        c_h.Levels{lvl}.(fn{1}) = round(c_h.Levels{lvl}.(fn{1}) / q_demo) * q_demo;
    end
end
I_haar_demo = min(max(haar_demo.inverse(c_h), 0), 1);

% CDF 5/3 reconstruction at q_demo  [integer space]
q_demo_int = round(q_demo * 255);   % convert to gray-level units
cdf_demo = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'CDF53', 'Levels', num_levels);
c_c = cdf_demo.forward(I_luma_int);   % forward on [0,255] integer image
c_c.LL = round(c_c.LL / q_demo_int) * q_demo_int;
for lvl = 1:num_levels
    for fn = {'LH', 'HL', 'HH'}
        c_c.Levels{lvl}.(fn{1}) = round(c_c.Levels{lvl}.(fn{1}) / q_demo_int) * q_demo_int;
    end
end
I_cdf_demo = min(max(cdf_demo.inverse(c_c) / 255.0, 0), 1);

% Concatenate horizontally for side-by-side view
strip_h = 20;   % separator row height
sep_row = ones(strip_h, W_orig);
label_rows = [ones(strip_h, W_orig) * 0.15; sep_row];

mosaic = [I_luma; sep_row; I_pyr_demo; sep_row; I_haar_demo; sep_row; I_cdf_demo];

imshow(mosaic, 'Parent', ax2);
title(ax2, sprintf('Reconstructions at q=%.0f/255\n(Original / Pyramid / Haar / CDF53)', ...
    round(q_demo * 255)), 'Color', white, 'FontSize', 10, 'FontWeight', 'bold');

hold(ax2, 'off');

% Compute final PSNRs for the demo q
psnr_pyr_demo  = computePSNR(I_luma, I_pyr_demo);
psnr_haar_demo = computePSNR(I_luma, I_haar_demo);
psnr_cdf_demo  = computePSNR(I_luma, I_cdf_demo);

% Annotate PSNR values on the reconstruction mosaic
text(ax2, 5, strip_h + H_orig * 1.5, sprintf('PSNR: %.1f dB', psnr_pyr_demo), ...
    'Color', accent, 'FontSize', 8, 'FontWeight', 'bold');
text(ax2, 5, strip_h + H_orig * 2.5 + strip_h, sprintf('PSNR: %.1f dB', psnr_haar_demo), ...
    'Color', secondary, 'FontSize', 8, 'FontWeight', 'bold');
text(ax2, 5, strip_h + H_orig * 3.5 + strip_h*2, sprintf('PSNR: %.1f dB', psnr_cdf_demo), ...
    'Color', green, 'FontSize', 8, 'FontWeight', 'bold');

% Save figure
output_path = fullfile(results_dir, 'wavelet_compression_results.png');
saveas(fig, output_path);

% === 5. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset          : %s (%dx%d)\n', image_name, W_orig, H_orig);
fprintf('  Pyramid Levels   : %d\n', num_levels);
fprintf('  Quantization (q) : swept from %.2f to %.2f /255\n', q_steps_gray(1), q_steps_gray(end));
fprintf('\n  At q = 1/4 gray level:\n');
fprintf('    Laplacian Pyramid  : %.4f bpp\n', total_pyr_bits / N_orig);
fprintf('    Haar Wavelet       : %.4f bpp\n', total_haar_bits / N_orig);
fprintf('    CDF 5/3 Wavelet    : %.4f bpp  [q_int=%.2f gray]\n', total_cdf_bits / N_orig, q_audit_int);
fprintf('\n  At q = %d/255 (demo reconstruction):\n', round(q_demo * 255));
fprintf('    Laplacian Pyramid  : PSNR = %.2f dB\n', psnr_pyr_demo);
fprintf('    Haar Wavelet       : PSNR = %.2f dB\n', psnr_haar_demo);
fprintf('    CDF 5/3 Wavelet    : PSNR = %.2f dB\n', psnr_cdf_demo);
fprintf('\n  Results saved to : %s\n', output_path);
fprintf('==========================\n');
