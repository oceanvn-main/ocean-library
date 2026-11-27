% EXP_GAUSSIAN_KERNEL_VERIFICATION
%   Verifies that GaussianKernel (binomial & sampled modes) matches
%   MATLAB built-in equivalents, and that the Gaussian/Laplacian pyramid
%   structure (Burt & Adelson 1983, Figure 3.33 in Szeliski) is correctly
%   implemented.
%
%   HYPOTHESES:
%     H1 – Binomial kernel (N=5) coefficients equal Pascal row C(4,k)/2^4.
%     H2 – Binomial kernel approximates fspecial('gaussian', N, sigma_eff)
%          with effective sigma derived from N (sigma_eff = sqrt(N-1)/2).
%     H3 – GaussianPyramid.build produces levels whose sizes halve each step.
%     H4 – LaplacianPyramid.build produces bandpass layers: each level equals
%          G_l  -  upsample(G_{l+1})  (verified against OceanMath reduce/expand).
%     H5 – Perfect reconstruction: collapsing the Laplacian pyramid recovers
%          the original image with max absolute error < 1e-10.
%
%   MATLAB BUILT-INS USED FOR COMPARISON:
%     fspecial('gaussian', N, sigma)   – reference Gaussian kernel
%     pascal(N)                        – Pascal's triangle matrix
%     imread, rgb2gray                 – load and convert real image
%
%   ASSUMPTIONS:
%     - Real grayscale image (elon_musk.jpg) cropped to 128x128 centre patch.
%     - Pyramid uses default 5x5 binomial GaussianKernel (no sigma).
%     - Comparison tolerances are empirically meaningful, not textbook-copied.
%     - rng seed fixed only for reproducibility of any future random steps.
%
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Author: Antigravity Code Assistant
%   Date:   2026-06-22

clc; clear; close all;
rng(42, 'twister');

fprintf('=== EXPERIMENT: GaussianKernel Verification ===\n\n');

% =========================================================================
% 1. CONFIGURATION
% =========================================================================
KERNEL_SIZE   = 5;          % kernel size to test
N_LEVELS      = 4;          % pyramid depth
RECON_TOL     = 1e-10;      % acceptable reconstruction error (absolute)
BINOM_TOL     = 1e-12;      % tolerance for exact binomial coefficient check
GAUSS_APPROX_TOL = 0.05;    % max allowed max-abs-diff vs fspecial gaussian
                             % (binomial is exact discrete Gaussian — sampled
                             %  continuous Gaussian differs slightly by design)

% =========================================================================
% 2. LOAD REAL IMAGE  (grayscale, double, fixed crop — deterministic)
% =========================================================================
IMG_PATH = fullfile('C:', 'Ocean Library', 'Matlab code practice', ...
    'Data', 'images', 'elon_musk.jpg');

if ~isfile(IMG_PATH)
    error('exp:ImageNotFound', 'Elon image not found at:\n  %s', IMG_PATH);
end

I_rgb  = imread(IMG_PATH);                    % load as uint8 RGB
I_gray = im2double(rgb2gray(I_rgb));          % convert to double grayscale [0,1]

% Resize to 128x128 (preserves full face — avoids landing on chest/background)
IMG_H = 128;  IMG_W = 128;
[H_full, W_full] = size(I_gray);
I_synth = imresize(I_gray, [IMG_H, IMG_W]);   % bilinear by default

fprintf('--- Image loaded: %dx%d -> resized to %dx%d grayscale\n', ...
    H_full, W_full, IMG_H, IMG_W);
fprintf('    Intensity range: [%.4f, %.4f]\n', min(I_synth(:)), max(I_synth(:)));

% =========================================================================
% 3. H1 — Binomial coefficients match Pascal's triangle row
% =========================================================================
fprintf('\n[H1] Binomial kernel coefficients vs Pascal''s triangle\n');

k_obj   = OceanMath.Signal.Kernels.GaussianKernel('Size', KERNEL_SIZE);
[h, ~]  = k_obj.get1DDecomposition();         % our implementation
h_unnorm = h * sum(h) / sum(h);               % already normalised — recover counts
h_counts = h * 2^(KERNEL_SIZE-1);             % un-normalise: multiply by 2^(N-1)

% Built-in Pascal's triangle row: C(N-1, k) for k = 0..N-1
% NOTE: pascal(N) in MATLAB returns a symmetric positive-definite matrix
%       where entry (i,j) = C(i+j-2, i-1) — NOT Pascal triangle rows.
%       We must compute C(N-1, k) directly.
expected_row = arrayfun(@(k) nchoosek(KERNEL_SIZE-1, k), 0:(KERNEL_SIZE-1));

err_h1 = max(abs(h_counts - expected_row));
fprintf('  Our kernel (un-normalised) : %s\n', mat2str(round(h_counts)));
fprintf('  Pascal row C(4,k)          : %s\n', mat2str(expected_row));
fprintf('  Max abs difference         : %.2e  (tol = %.2e)\n', err_h1, BINOM_TOL);
h1_pass = err_h1 < BINOM_TOL;
fprintf('  H1 result: %s\n', tf2str(h1_pass));

% =========================================================================
% 4. H2 — Binomial kernel approximates fspecial('gaussian')
% =========================================================================
fprintf('\n[H2] Binomial kernel vs fspecial gaussian\n');

% Effective sigma of an N-tap binomial kernel: sigma_eff = sqrt(N-1)/2
sigma_eff = sqrt(KERNEL_SIZE - 1) / 2;
h_builtin = fspecial('gaussian', [1, KERNEL_SIZE], sigma_eff);

fprintf('  Effective sigma (CLT estimate): %.4f\n', sigma_eff);
fprintf('  Our kernel   : %s\n', mat2str(h, 4));
fprintf('  fspecial ref : %s\n', mat2str(h_builtin, 4));

err_h2 = max(abs(h - h_builtin));
fprintf('  Max abs difference : %.4f  (tol = %.4f)\n', err_h2, GAUSS_APPROX_TOL);
h2_pass = err_h2 < GAUSS_APPROX_TOL;
fprintf('  H2 result: %s\n', tf2str(h2_pass));
fprintf('  (Note: CLT sigma_eff is approximate — some difference is expected.)\n');

% =========================================================================
% 5. H3 — GaussianPyramid level sizes halve each step
% =========================================================================
fprintf('\n[H3] GaussianPyramid level sizes halve each step\n');

% NOTE: GaussianPyramid.build() stores levels in obj.Levels — returns nothing.
gp = OceanMath.Image.MultiResolution.GaussianPyramid();
gp.build(I_synth, N_LEVELS);
G = gp.Levels;   % cell array {1..N_LEVELS}

h3_pass = true;
for lv = 2:N_LEVELS
    [Hprev, Wprev] = size(G{lv-1}, [1 2]);
    [Hcurr, Wcurr] = size(G{lv},   [1 2]);
    expected_H = ceil(Hprev / 2);
    expected_W = ceil(Wprev / 2);
    ok = (Hcurr == expected_H) && (Wcurr == expected_W);
    fprintf('  Level %d→%d : %dx%d → %dx%d  (expected %dx%d)  %s\n', ...
        lv-1, lv, Hprev, Wprev, Hcurr, Wcurr, expected_H, expected_W, tf2str(ok));
    h3_pass = h3_pass && ok;
end
fprintf('  H3 result: %s\n', tf2str(h3_pass));

% =========================================================================
% 6. H4 — Laplacian pyramid bandpass: L_l = G_l - upsample(G_{l+1})
% =========================================================================
fprintf('\n[H4] LaplacianPyramid bandpass verification\n');

% NOTE: LaplacianPyramid.build() also stores internally in obj.Levels.
% The coarsest Gaussian is stored in Levels{N_LEVELS}.
lp = OceanMath.Image.MultiResolution.LaplacianPyramid();
lp.build(I_synth, N_LEVELS);
L = lp.Levels;   % Levels{1..N_LEVELS-1} are bandpass; Levels{N_LEVELS} is coarsest Gaussian

h4_pass = true;
for lv = 1:(N_LEVELS-1)
    % Recompute expected bandpass from our separate Gaussian pyramid (gp.Levels)
    tgt_size = size(G{lv});
    G_up  = OceanMath.Image.MultiResolution.PyramidOperator.expand( ...
                G{lv+1}, tgt_size);
    L_expected = G{lv} - G_up;

    err_lv = max(abs(L{lv}(:) - L_expected(:)));
    ok = err_lv < RECON_TOL;
    fprintf('  Level %d: max|L_ours - L_expected| = %.2e  %s\n', lv, err_lv, tf2str(ok));
    h4_pass = h4_pass && ok;
end
fprintf('  H4 result: %s\n', tf2str(h4_pass));

% =========================================================================
% 7. H5 — Perfect reconstruction
% =========================================================================
fprintf('\n[H5] Perfect reconstruction from Laplacian pyramid\n');

% reconstruct() uses the internally stored Levels — no arguments needed.
I_recon = lp.reconstruct();
err_recon = max(abs(I_recon(:) - I_synth(:)));
h5_pass = err_recon < RECON_TOL;
fprintf('  Max absolute error (recon vs original): %.2e  (tol = %.2e)\n', ...
    err_recon, RECON_TOL);
fprintf('  H5 result: %s\n', tf2str(h5_pass));

% =========================================================================
% 8. VISUALISATION
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'GaussianKernel Verification', ...
    'Color', [0.11 0.13 0.16], 'Position', [50 50 1400 860]);

% --- Row 1: Kernel comparison ---
ax1 = subplot(3, 4, 1); hold(ax1, 'on');
bar(ax1, h_counts, 'FaceColor', viz.Colors.Primary, 'EdgeColor', 'none');
title(ax1, 'Binomial Counts', 'Color','w');
xlabel(ax1,'k'); ylabel(ax1,'C(N-1,k)');
viz.styleAxes(ax1);

ax2 = subplot(3, 4, 2); hold(ax2, 'on');
plot(ax2, h,       '-o', 'Color', viz.Colors.Primary,   'LineWidth', 1.8);
plot(ax2, h_builtin, '--s', 'Color', viz.Colors.Secondary, 'LineWidth', 1.8);
legend(ax2, {'Our Binomial','fspecial Gaussian'}, 'TextColor','w', 'Location','north');
title(ax2, sprintf('1D Kernel (N=%d)', KERNEL_SIZE), 'Color','w');
viz.styleAxes(ax2);

ax3 = subplot(3, 4, 3);
K2D = k_obj.getMatrix();
imagesc(ax3, K2D); colorbar(ax3);
title(ax3, '2D Binomial Kernel', 'Color','w');
colormap(ax3, 'hot'); axis(ax3, 'image');
viz.styleAxes(ax3);

ax4 = subplot(3, 4, 4);
imagesc(ax4, I_synth); colormap(ax4, 'gray'); axis(ax4, 'image');
title(ax4, 'Elon Musk (resized 128x128)', 'Color','w');
viz.styleAxes(ax4);

% --- Row 2: Gaussian pyramid levels ---
for lv = 1:N_LEVELS
    ax = subplot(3, 4, 4 + lv);
    imagesc(ax, G{lv}); colormap(ax, 'gray'); axis(ax, 'image');
    title(ax, sprintf('G level %d (%dx%d)', lv, size(G{lv},1), size(G{lv},2)), 'Color','w');
    viz.styleAxes(ax);
end

% --- Row 3: Laplacian pyramid levels ---
for lv = 1:(N_LEVELS-1)
    ax = subplot(3, 4, 8 + lv);
    L_disp = L{lv};
    imagesc(ax, L_disp); colormap(ax, 'gray'); axis(ax, 'image');
    title(ax, sprintf('L level %d', lv), 'Color','w');
    viz.styleAxes(ax);
end
ax_r = subplot(3, 4, 12);
imagesc(ax_r, I_recon); colormap(ax_r, 'gray'); axis(ax_r, 'image');
title(ax_r, sprintf('Recon (err=%.1e)', err_recon), 'Color','w');
viz.styleAxes(ax_r);

sgtitle(fig, 'GaussianKernel Verification — Binomial Approximation & Pyramid Structure', ...
    'Color', 'w', 'FontSize', 13, 'FontWeight', 'bold');

% =========================================================================
% 9. SUMMARY
% =========================================================================
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Image size      : %dx%d\n', IMG_H, IMG_W);
fprintf('  Kernel size     : %d\n', KERNEL_SIZE);
fprintf('  Pyramid levels  : %d\n', N_LEVELS);
fprintf('  sigma_eff (CLT) : %.4f\n', sigma_eff);
fprintf('\n  Hypothesis Results:\n');
fprintf('    H1 – Binomial vs Pascal row        : %s  (err=%.2e)\n', tf2str(h1_pass), err_h1);
fprintf('    H2 – Binomial vs fspecial gaussian : %s  (err=%.4f)\n', tf2str(h2_pass), err_h2);
fprintf('    H3 – Level sizes halve             : %s\n', tf2str(h3_pass));
fprintf('    H4 – Bandpass structure            : %s\n', tf2str(h4_pass));
fprintf('    H5 – Perfect reconstruction        : %s  (err=%.2e)\n', tf2str(h5_pass), err_recon);
all_pass = h1_pass && h2_pass && h3_pass && h4_pass && h5_pass;
fprintf('\n  Overall: %s\n', tf2str_full(all_pass));
fprintf('==========================\n');

% =========================================================================
% LOCAL HELPER
% =========================================================================
function s = tf2str(tf)
    if tf; s = 'PASS'; else; s = 'FAIL'; end
end

function s = tf2str_full(tf)
    if tf; s = 'ALL HYPOTHESES PASSED'; else; s = 'ONE OR MORE HYPOTHESES FAILED'; end
end
