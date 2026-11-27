% EXP_WAVELET_DECOMPOSITION
%   2D Discrete Wavelet Transform decomposition of the Elon Musk image.
%   Reproduces conceptually Figure 3.36 from Szeliski (2021):
%     (a) Single-level DWT — LL, LH, HL, HH subbands
%     (b) Multi-level coefficient magnitude mosaic
%
%   HYPOTHESES:
%     H1 – Single-level DWT decomposes image into 4 subbands:
%          LL (approx), LH (horizontal edges), HL (vertical edges), HH (diagonal).
%     H2 – Perfect reconstruction: inverse(forward(I)) recovers original
%          image with max absolute error < 1e-10 (Haar, float space).
%     H3 – Multi-level DWT: LL subband halves in size at each level
%          (exact halvings, since our lifting pads to power-of-2 multiples).
%     H4 – Energy compaction: LL subband holds >85% of total image energy.
%
%   NOTE ON WAVELET CHOICE:
%     The original script used MATLAB's Wavelet Toolbox ('db4').
%     We replace all Toolbox calls (dwt2, idwt2, wavedec2, appcoef2, detcoef2)
%     with OceanMath.Signal.Lifting.LiftingScheme.
%     'Haar' is used for H2 (perfect reconstruction test) because it operates
%     in floating-point without integer-rounding error. 'CDF53' requires
%     integer-space images for lossless reconstruction; a note is added.
%     For analysis (H1, H3, H4, visualization), CDF53 in float space gives
%     a good approximation of the subband structure.
%
%   TOOL USED:
%     OceanMath.Signal.Lifting.LiftingScheme  (replaces dwt2/wavedec2 etc.)
%       - No MATLAB Wavelet Toolbox required.
%
%   ASSUMPTIONS:
%     - Elon Musk image loaded from Data/images/elon_musk.jpg
%     - Resized to 256x256 (must be even at every decomposition level)
%     - Wavelet: 'CDF53' for analysis/visualization; 'Haar' for recon test
%     - N_LEVELS = 4 for multi-level decomposition
%     - Images in [0,1] double float
%
%   REPRODUCIBILITY: fully deterministic (no random elements)
%
%   Reference: Szeliski, Computer Vision: Algorithms & Applications, 2nd ed.,
%              Figure 3.36, p.160
%   Author: Ocean Library Experiment (updated 2026-07-15 — Toolbox-free)
%   Date:   2026-06-23

clc; clear; close all;

fprintf('=== EXPERIMENT: 2D Wavelet Decomposition (Szeliski Fig 3.36) ===\n');
fprintf('    Using OceanMath.Signal.Lifting.LiftingScheme (no Wavelet Toolbox).\n\n');

% =========================================================================
% 1. CONFIGURATION
% =========================================================================
WAVELET_ANALYSIS = 'CDF53';  % CDF 5/3 — smooth, good energy compaction (JPEG 2000)
WAVELET_RECON    = 'Haar';   % Haar — lossless in float space (for H2 perfect-recon test)
N_LEVELS  = 4;
RECON_TOL = 1e-10;   % Perfect reconstruction tolerance (Haar, float)
IMG_SIZE  = 256;     % Resize target — 256 = 2^8 so all 4 levels divide exactly

% =========================================================================
% 2. LOAD IMAGE
% =========================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end

IMG_PATH = fullfile(project_root, 'Data', 'images', 'elon_musk.jpg');
if ~isfile(IMG_PATH)
    error('exp:ImageNotFound', 'Image not found at:\n  %s', IMG_PATH);
end

I_rgb  = imread(IMG_PATH);
I_gray = im2double(rgb2gray(I_rgb));          % double grayscale [0,1]
I      = imresize(I_gray, [IMG_SIZE, IMG_SIZE]);  % 256x256

fprintf('--- Image: Elon Musk, resized to %dx%d grayscale\n', IMG_SIZE, IMG_SIZE);
fprintf('    Intensity range: [%.4f, %.4f]\n\n', min(I(:)), max(I(:)));

% =========================================================================
% 3. H1 + H2 — Single-Level DWT and Perfect Reconstruction
%    Replaces: [cA,cH,cV,cD] = dwt2(I, WAVELET)
%              I_recon = idwt2(cA,cH,cV,cD, WAVELET, size(I))
% =========================================================================
fprintf('[H1/H2] Single-level 2D DWT (analysis wavelet: %s)\n', WAVELET_ANALYSIS);

% Forward single-level analysis transform (CDF53 for subband structure)
ls1_analysis = OceanMath.Signal.Lifting.LiftingScheme(...
    'WaveletType', WAVELET_ANALYSIS, 'Levels', 1);
c1 = ls1_analysis.forward(I);

% Extract the 4 subbands (struct layout: c1.LL, c1.Levels{1}.{LH,HL,HH})
cA = c1.LL;                   % LL  — approximation (low-low)
cH = c1.Levels{1}.LH;         % LH  — horizontal edges (high-frequency rows)
cV = c1.Levels{1}.HL;         % HL  — vertical edges   (high-frequency cols)
cD = c1.Levels{1}.HH;         % HH  — diagonal/mixed

fprintf('  Input size      : %dx%d\n', size(I,1), size(I,2));
fprintf('  Subband size    : %dx%d (each)\n', size(cA,1), size(cA,2));

E_total = sum(I(:).^2);
E_LL    = sum(cA(:).^2);
E_LH    = sum(cH(:).^2);
E_HL    = sum(cV(:).^2);
E_HH    = sum(cD(:).^2);

fprintf('  Subband energies (single level, %s):\n', WAVELET_ANALYSIS);
fprintf('    LL approx (cA)       : %.4f  (%.1f%% of total)\n', E_LL, E_LL/E_total*100);
fprintf('    LH horiz edges (cH)  : %.4f  (%.1f%%)\n', E_LH, E_LH/E_total*100);
fprintf('    HL vert  edges (cV)  : %.4f  (%.1f%%)\n', E_HL, E_HL/E_total*100);
fprintf('    HH diagonal    (cD)  : %.4f  (%.1f%%)\n', E_HH, E_HH/E_total*100);

h1_pass = ~isempty(cA) && ~isempty(cH) && ~isempty(cV) && ~isempty(cD) && ...
          isequal(size(cA), size(cH), size(cV), size(cD));
fprintf('  H1 result: %s\n', tf2str(h1_pass));

% Perfect reconstruction using Haar (lossless in float space)
% Note: CDF53 uses floor() -> integer-only lossless. Haar is float-safe.
fprintf('\n  Perfect reconstruction test (wavelet: %s, float space)\n', WAVELET_RECON);
ls1_haar = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', WAVELET_RECON, 'Levels', 1);
c1_haar  = ls1_haar.forward(I);
I_recon  = ls1_haar.inverse(c1_haar);
% Crop to original size (forward pads to even; crop back)
I_recon  = I_recon(1:IMG_SIZE, 1:IMG_SIZE);
err_recon = max(abs(I_recon(:) - I(:)));
h2_pass   = err_recon < RECON_TOL;
fprintf('  Reconstruction max |error| : %.2e  (tol = %.2e)\n', err_recon, RECON_TOL);
fprintf('  H2 result: %s\n', tf2str(h2_pass));

% =========================================================================
% 4. H3 — Multi-Level: LL halves each level
%    Replaces: [C, S] = wavedec2(I, N, WAVELET)
%              then reading S matrix for sizes
% =========================================================================
fprintf('\n[H3] Multi-level DWT (%d levels, %s)\n', N_LEVELS, WAVELET_ANALYSIS);

ls_multi = OceanMath.Signal.Lifting.LiftingScheme(...
    'WaveletType', WAVELET_ANALYSIS, 'Levels', N_LEVELS);
cN = ls_multi.forward(I);    % coeffs.LL is coarsest LL; .Levels{1..N} are detail levels

% Our LiftingScheme pads to multiples of 2^N, so sizes halve exactly.
% Level 1 detail: IMG_SIZE/2 x IMG_SIZE/2 ... Level N: IMG_SIZE/2^N x IMG_SIZE/2^N
h3_pass = true;
fprintf('  Level size progression (detail subbands):\n');
for lv = 1:N_LEVELS
    det = cN.Levels{lv};
    actual_sz = size(det.LH, 1);          % rows of LH at this level
    expected_sz = IMG_SIZE / 2^lv;
    ok = (actual_sz == expected_sz);
    fprintf('    Level %d detail LH: %dx%d  (expected %dx%d)  %s\n', ...
        lv, size(det.LH,1), size(det.LH,2), expected_sz, expected_sz, tf2str(ok));
    h3_pass = h3_pass && ok;
end
ll_sz = size(cN.LL, 1);
fprintf('    Coarsest LL: %dx%d  (= IMG_SIZE/2^%d = %d)  %s\n', ...
    ll_sz, ll_sz, N_LEVELS, IMG_SIZE/2^N_LEVELS, tf2str(ll_sz == IMG_SIZE/2^N_LEVELS));
fprintf('  H3 result: %s\n', tf2str(h3_pass));

% =========================================================================
% 5. H4 — Energy Compaction
% =========================================================================
fprintf('\n[H4] Energy compaction (single-level LL from CDF53)\n');

frac_ll = E_LL / E_total;
h4_pass = frac_ll > 0.85;
fprintf('  Total image energy : %.4f\n', E_total);
fprintf('  LL energy fraction : %.1f%%  (threshold: >85%%)\n', frac_ll*100);
fprintf('  H4 result: %s\n', tf2str(h4_pass));

% =========================================================================
% 6. VISUALISATION
% =========================================================================
fprintf('\nRendering visualization...\n');

viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;

fig = figure('Name', 'Wavelet Decomposition — Elon Musk', ...
    'Color', colors.Background, 'Position', [30 30 1500 920]);

% Helper: normalise detail band for display (abs + scale to [0,1])
show_detail = @(X) mat2gray(abs(X));

% --- Row 1: Single-level subbands (Fig 3.36a) ---
panels_r1 = {I,           mat2gray(cA),      show_detail(cH), show_detail(cV), show_detail(cD)};
titles_r1 = {'Original',  'LL (Approx)',      'LH (Horiz Edges)', 'HL (Vert Edges)', 'HH (Diagonal)'};

for k = 1:5
    ax = subplot(3, 5, k);
    imagesc(ax, panels_r1{k});
    colormap(ax, 'gray'); axis(ax, 'image off');
    title(ax, titles_r1{k}, 'Color', colors.Text, 'FontSize', 9, 'FontWeight', 'bold');
    viz.styleAxes(ax);
end

% --- Row 2: LH (horizontal edge) at each of 4 levels + coarsest LL ---
%   Replaces: detcoef2('h', C, S, lv) and appcoef2(C, S, WAVELET, N_LEVELS)
for lv = 1:N_LEVELS
    ax  = subplot(3, 5, 5 + lv);
    dH  = cN.Levels{lv}.LH;    % LH detail at this level
    imagesc(ax, show_detail(dH));
    colormap(ax, 'gray'); axis(ax, 'image off');
    title(ax, sprintf('LH Level %d\n(%dx%d)', lv, size(dH,1), size(dH,2)), ...
        'Color', colors.Text, 'FontSize', 8);
    viz.styleAxes(ax);
end

% Last panel of row 2: coarsest LL approximation
ax_ll = subplot(3, 5, 10);
imagesc(ax_ll, mat2gray(cN.LL));
colormap(ax_ll, 'gray'); axis(ax_ll, 'image off');
title(ax_ll, sprintf('LL Level %d\n(%dx%d)', N_LEVELS, size(cN.LL,1), size(cN.LL,2)), ...
    'Color', colors.Text, 'FontSize', 8);
viz.styleAxes(ax_ll);

% --- Row 3: Multi-level coefficient magnitude mosaic (Fig 3.36b) ---
ax_mosaic = subplot(3, 5, [11 12 13 14 15]);
mosaic = build_wavelet_mosaic(cN, N_LEVELS, IMG_SIZE);
imagesc(ax_mosaic, mat2gray(abs(mosaic)));
colormap(ax_mosaic, 'gray'); axis(ax_mosaic, 'image off');
title(ax_mosaic, ...
    sprintf('Multi-Level Coefficient Magnitude Mosaic — %d levels, %s  [cf. Szeliski Fig 3.36b]', ...
    N_LEVELS, WAVELET_ANALYSIS), 'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
viz.styleAxes(ax_mosaic);

sgtitle(fig, ...
    sprintf('Wavelet Decomposition — Elon Musk (Szeliski 2021, Fig 3.36)\n[%s analysis | %s recon | OceanMath.Signal.Lifting — no Wavelet Toolbox]', ...
    WAVELET_ANALYSIS, WAVELET_RECON), ...
    'Color', colors.Text, 'FontSize', 13, 'FontWeight', 'bold');

output_path = fullfile(results_dir, 'wavelet_decomposition.png');
saveas(fig, output_path);

% =========================================================================
% 7. SUMMARY
% =========================================================================
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Image           : Elon Musk %dx%d\n', IMG_SIZE, IMG_SIZE);
fprintf('  Analysis wavelet: %s (CDF 5/3)\n', WAVELET_ANALYSIS);
fprintf('  Recon wavelet   : %s (for lossless float-space H2)\n', WAVELET_RECON);
fprintf('  Levels          : %d\n', N_LEVELS);
fprintf('\n  Hypothesis Results:\n');
fprintf('    H1 – 4-subband single-level DWT : %s\n',             tf2str(h1_pass));
fprintf('    H2 – Perfect reconstruction     : %s  (err=%.2e)\n', tf2str(h2_pass), err_recon);
fprintf('    H3 – LL halves each level        : %s\n',             tf2str(h3_pass));
fprintf('    H4 – Energy compaction >85%%     : %s  (%.1f%% in LL)\n', tf2str(h4_pass), frac_ll*100);

all_pass = h1_pass && h2_pass && h3_pass && h4_pass;
fprintf('\n  Overall: %s\n', tf2str_full(all_pass));
fprintf('  Results saved to: %s\n', output_path);
fprintf('==========================\n');

% =========================================================================
% LOCAL HELPERS
% =========================================================================
function s = tf2str(tf)
    if tf; s = 'PASS'; else; s = 'FAIL'; end
end

function s = tf2str_full(tf)
    if tf; s = 'ALL HYPOTHESES PASSED'; else; s = 'ONE OR MORE HYPOTHESES FAILED'; end
end

function mosaic = build_wavelet_mosaic(coeffs, N, full_size)
    % BUILD_WAVELET_MOSAIC  Assembles multi-level wavelet coefficient mosaic.
    %
    %   Uses the OceanMath.Signal.Lifting struct layout:
    %     coeffs.LL              — coarsest approximation
    %     coeffs.Levels{lv}.LH  — LH detail at level lv  (lv=1 is finest)
    %     coeffs.Levels{lv}.HL  — HL detail at level lv
    %     coeffs.Levels{lv}.HH  — HH detail at level lv
    %
    %   Level numbering: Levels{1} = finest (largest), Levels{N} = coarsest.
    %
    %   Standard quadrant mosaic layout (left=low-freq, top=low-freq):
    %     Top-left corner  = coarsest LL
    %     Top-right of LL  = LH at level N  (horizontal edges, coarsest)
    %     Below LL         = HL at level N  (vertical edges, coarsest)
    %     Diagonal of LL   = HH at level N  (diagonal, coarsest)
    %     ... expanding outward for finer levels ...

    mosaic = zeros(full_size, full_size);

    % Expected slot size for each level (level 1 = full_size/2, level N = full_size/2^N)
    slot_sizes = zeros(1, N);
    for lv = 1:N
        slot_sizes(lv) = full_size / 2^lv;
    end

    % Coarsest LL — top-left corner
    slot_N = slot_sizes(N);
    mosaic(1:slot_N, 1:slot_N) = imresize(coeffs.LL, [slot_N, slot_N]);

    % Fill quadrants level by level from coarsest (N) to finest (1)
    % At level lv: the 3 detail bands occupy 3 quadrants of a 2*slot_lv x 2*slot_lv block
    % at the top-left of the mosaic. The top-left slot is already filled by coarser levels.
    for lv = N:-1:1
        s = slot_sizes(lv);          % slot size at this level

        dH = imresize(coeffs.Levels{lv}.LH, [s, s]);   % LH horizontal edges
        dV = imresize(coeffs.Levels{lv}.HL, [s, s]);   % HL vertical edges
        dD = imresize(coeffs.Levels{lv}.HH, [s, s]);   % HH diagonal

        % LH — top-right of the 2s x 2s block
        mosaic(1:s,       s+1:2*s) = dH;
        % HL — bottom-left of the 2s x 2s block
        mosaic(s+1:2*s,   1:s)     = dV;
        % HH — bottom-right of the 2s x 2s block
        mosaic(s+1:2*s,   s+1:2*s) = dD;
    end
end
