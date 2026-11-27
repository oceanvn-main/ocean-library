% EXP_PDA_DIGITS_FIG12_11_12  ESL Figures 12.11 & 12.12 — PDA on Digit Recognition
%
% Ref: ESL Section 12.6, Figures 12.11 and 12.12
% Author: Ocean Library
% Date: 2026-03-16
%
% Description:
%   Reproduces ESL Figure 12.11 (LDA vs PDA discriminant coefficient images)
%   and Figure 12.12 (first two PDA canonical variate scatter plot) using
%   the handwritten digit zip code dataset.
%
%   Pipeline follows the 8-step Ocean Library ML Pipeline.

clc; clear; close all;
rng(42, 'twister');
RNG_SEED = 42;
DATASET  = 'zip_digits';
IMG_ROWS = 16;
IMG_COLS = 16;
LAMBDA   = 4000;  % PDA penalty strength — calibrated for UNNORMALIZED X^T X
                  % ESL uses lambda in normalized form (rel. to within-class cov).
                  % Our PenalizedRegression solves (X'X + lam*Omega)b = X'z (raw sum).
                  % X^T X diagonal ~ N * var(pixel) ~ 7291 * 0.09 ~ 656.
                  % Omega_2D mean eigenvalue ~ 10.5 => for df~4: lam ~ 64*656/10.5 ~ 4000.
                  % Tune: 500=~15df (lightly smooth), 4000=~4df (heavily smooth as textbook).

fprintf('Producing ESL Figures 12.11 & 12.12 — PDA on Digit Recognition\n');
fprintf('================================================================\n');

%% === Step 1: Ingestion ===
% The zip dataset provides a CANONICAL train/test split from ESL.
% Step 1.5 (DataSplitter) is NOT needed — using the pre-split files directly.
% [SKIP - 1.5] Reason: zip.train and zip.test are the ESL-defined splits.
fprintf('\nStep 1: Ingestion - Loading Zip Digit Dataset...\n');

gen_train = OceanData.Generators.ZipGenerator('Type', 'train');
gen_test  = OceanData.Generators.ZipGenerator('Type', 'test');
[X_train, y_train] = gen_train.generate();
[X_test,  y_test]  = gen_test.generate();

X_train = double(X_train);  y_train = double(y_train);
X_test  = double(X_test);   y_test  = double(y_test);

[N_train, p] = size(X_train);
N_test  = size(X_test, 1);
K = numel(unique(y_train));

fprintf('  Train: N=%d, p=%d, K=%d classes\n', N_train, p, K);
fprintf('  Test:  N=%d\n', N_test);

%% === Step 2: DataLake (DA-1) ===
fprintf('\nStep 2: DataLake (DA-1) - Raw Persistence...\n');
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    meta = struct('source', DATASET, 'seed', RNG_SEED);
    lake.put('raw/zip/X_train', X_train, meta);
    lake.put('raw/zip/y_train', y_train, meta);
    lake.put('raw/zip/X_test',  X_test,  meta);
    lake.put('raw/zip/y_test',  y_test,  meta);
    fprintf('  [PASS] DataLake ready.\n');
catch
    fprintf('  [SKIP] DataLake unavailable.\n');
end

%% === Step 3: DataWarehouse (DA-2/DA-3) ===
fprintf('\nStep 3: DataWarehouse - Structured Storage...\n');
try
    dw = OceanData.DataArch.DataWarehouse('rootDir', './dw/');
    pixelNames = arrayfun(@(i) sprintf('pix%d', i), 1:p, 'UniformOutput', false);
    colNames = [pixelNames, {'digit'}];
    colTypes = [repmat({'double'}, 1, p), {'double'}];
    schema = OceanData.DataArch.DWSchema(colNames, colTypes);
    dw.createTable(DATASET, schema);
    trainTable = array2table([X_train, y_train], 'VariableNames', colNames);
    dw.insertBatch(DATASET, trainTable);
    fprintf('  [PASS] DataWarehouse ready.\n');
catch
    fprintf('  [SKIP] DataWarehouse unavailable.\n');
end

%% === Step 4: CachingWarehouse (DA-4) ===
fprintf('\nStep 4: CachingWarehouse (DA-4) - Read-Through Cache...\n');
try
    redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
    cdw   = OceanData.Adapters.CachingWarehouse(dw, redis);
    fprintf('  [PASS] CachingWarehouse connected to Redis.\n');
catch
    fprintf('  [SKIP] Redis unavailable.\n');
end

%% === Step 5 & 6: FeatureStore & FeatureCache (DA-5/DA-6) ===
fprintf('\nStep 5 & 6: FeatureStore & FeatureCache (DA-5/DA-6)...\n');
try
    fs = OceanData.DataArch.FeatureStore('rootDir', './fs/');
    feat_key = sprintf('%s_raw_seed%d', DATASET, RNG_SEED);
    if ~fs.exists(feat_key)
        fs.put(feat_key, X_train);
    end
    entry = fs.get(feat_key);
    X_feat = entry.data; %#ok<NASGU>
    fprintf('  [PASS] FeatureStore ready.\n');
catch
    fprintf('  [SKIP] FeatureStore/Cache unavailable.\n');
end

%% === Step 7: Model Training & Evaluation ===
fprintf('\nStep 7: Training LDA and PDA...\n');

% Model cache: train-once, load-on-rerun
modelStore = OceanData.DataArch.FeatureStore('rootDir', './cache/models/');
cachePrefix = sprintf('%s/seed%d', DATASET, RNG_SEED);

% --- 7a. Standard LDA ---
fprintf('  Fitting LDA...\n');
lda = OceanTools.cachedFit(modelStore, sprintf('%s/LDA', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.LDA(), X_train, y_train);

y_pred_lda = lda.predict(X_test);
err_lda = mean(y_pred_lda ~= y_test);
fprintf('  LDA Test Error: %.3f\n', err_lda);

% --- 7b. PDA (FDA + PenalizedRegression + 2D roughness penalty) ---
fprintf('  Building 2D roughness penalty (Omega_2D for 16x16 pixels)...\n');
Omega2D = OceanMath.LinearAlgebra.Optimization.generateRoughnessPenalty2D(IMG_ROWS, IMG_COLS);

penRegBuilder = @() OceanStats.Supervised.Regression.PenalizedRegression(...
    'PenaltyMatrix', Omega2D, 'Lambda', LAMBDA);

fprintf('  Fitting PDA (FDA + PenalizedRegression, Lambda=%.1f)...\n', LAMBDA);
pda = OceanTools.cachedFit(modelStore, sprintf('%s/PDA_lam%d', cachePrefix, LAMBDA), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', penRegBuilder), X_train, y_train);

y_pred_pda = pda.predict(X_test);
err_pda = mean(y_pred_pda ~= y_test);
fprintf('  PDA Test Error: %.3f\n', err_pda);

% --- Effective df estimate (diagonal approximation) ---
% df(lambda) = tr{ X (X'X + lam*Omega)^{-1} X' } ~ sum sigma_i^2 / (sigma_i^2 + lam*omega_i)
% Approximated cheaply from column norms of X and diagonal of Omega.
try
    Omega2D_diag = full(diag(Omega2D));          % 256 x 1
    XtX_diag = sum(X_train .^ 2, 1)';            % 256 x 1 diagonal of X'X
    df_est = sum(XtX_diag ./ (XtX_diag + LAMBDA .* Omega2D_diag + eps));
    fprintf('  Estimated effective df: %.1f (of %d features)\n', df_est, p);
catch
    fprintf('  [SKIP] df estimate unavailable.\n');
end

fprintf('\n  --- Classification Summary ---\n');
fprintf('  %-20s  Test Error\n', 'Method');
fprintf('  %-20s  ----------\n', '------');
fprintf('  %-20s  %.3f\n', 'LDA', err_lda);
fprintf('  %-20s  %.3f (Lambda=%.1f)\n', 'PDA', err_pda, LAMBDA);

%% === Step 8: Visualization ===
fprintf('\nStep 8: Producing Figures 12.11 and 12.12...\n');

% Extract discriminant coefficient vectors (K-1 = 9 directions)
S_lda = lda.Scaling;          % 256 x 9
S_pda = pda.getScaling();     % 256 x 9

num_directions = min(size(S_lda, 2), 9);

% ── Re-sort PDA directions to match LDA ordering ─────────────────────────
% pda.getScaling() = B_mat * InternalLDA.Scaling may have different
% direction ordering due to the non-isotropic penalty. Re-order PDA
% columns by correlation with matching LDA columns.
S_pda_sorted = S_pda;
used = false(1, size(S_pda, 2));
for d = 1:num_directions
    best_corr = -Inf;
    best_j    = d;
    for j = 1:size(S_pda, 2)
        if used(j), continue; end
        c = abs(corr(S_lda(:,d), S_pda(:,j)));
        if c > best_corr
            best_corr = c;
            best_j    = j;
        end
    end
    S_pda_sorted(:, d) = S_pda(:, best_j);
    used(best_j) = true;
end
S_pda = S_pda_sorted;

% ── Compute class means for sign convention ───────────────────────────────
class_labels_v = unique(y_train);
Mu_X = zeros(p, K);
for kk = 1:K
    mask = (y_train == class_labels_v(kk));
    Mu_X(:, kk) = mean(X_train(mask, :), 1)';
end

% ── Sign convention: match PDA sign to LDA sign ──────────────────────────
% For each direction d, ensure PDA has the same sign orientation as LDA
% by checking the correlation of their coefficient vectors.
% Then apply a consistent sign via max-abs-pixel in LDA (negated to match
% the ESL textbook convention where dominant pixel = red).
%
% SIGN_OVERRIDE: directions where MATLAB's eig() picked the opposite sign
% from R's eigen(). Flip those to match the textbook exactly.
%                      1   2   3   4   5   6   7   8   9
SIGN_OVERRIDE =      [1,  1,  1, -1,  1,  1, -1,  1,  1];

for d = 1:num_directions
    % Ensure PDA matches LDA sign (same direction, not opposite)
    if corr(S_lda(:,d), S_pda(:,d)) < 0
        S_pda(:, d) = -S_pda(:, d);
    end
    
    % Then: max-abs-pixel in LDA is NEGATIVE (matches ESL textbook convention)
    [~, idx] = max(abs(S_lda(:, d)));
    s = -sign(S_lda(idx, d)) * SIGN_OVERRIDE(d);
    if s ~= 0
        S_lda(:, d) = s * S_lda(:, d);
        S_pda(:, d) = s * S_pda(:, d);
    end
end


% ── Diagnostic: display sample digits to verify orientation ───────────────
fig_diag = figure('Name', 'Digit Display Diagnostic', ...
    'Position', [50 50 600 200], 'Color', 'white');
% Show one sample of digit 0 and digit 1 using same reshape convention
sample_digits = [0, 1, 7, 4];
for si = 1:length(sample_digits)
    sd = sample_digits(si);
    sample_idx = find(y_train == sd, 1);
    ax_d = subplot(1, length(sample_digits), si);
    img = reshape(X_train(sample_idx, :), IMG_COLS, IMG_ROWS)';
    imagesc(ax_d, img); colormap(ax_d, gray); axis(ax_d, 'image', 'off');
    title(ax_d, sprintf('Digit %d', sd), 'FontSize', 10);
end
sgtitle(fig_diag, 'Verify: digits should look correct (1 = vertical stroke, hat at top)', ...
    'FontSize', 10);
fprintf('  [DIAG] Check digit display orientation in diagnostic figure.\n');

% ── Figure 12.11: Coefficient Images ───────────────────────────────────────
fig11 = figure('Name', 'Figure 12.11 — LDA vs PDA Discriminant Coefficients', ...
    'Position', [50 50 1300 700], 'Color', [0.97 0.97 0.97]);

% Red-green diverging colormap (negative = red, zero = black, positive = green)
n_colors = 256;
cmap = [linspace(1,0,n_colors/2)', linspace(0,0,n_colors/2)', linspace(0,0,n_colors/2)'; ...
        linspace(0,0,n_colors/2)', linspace(0,1,n_colors/2)', linspace(0,0,n_colors/2)'];

nrows_plot = 3;
ncols_plot = 6;  % 2 images per direction (LDA | PDA), 3 directions per row

for d = 1:num_directions
    % Each direction takes 2 subplot positions
    row = ceil(d / 3);
    col_pair = mod(d-1, 3) * 2 + 1;  % 1,3,5 in each row
    
    % LDA coefficient image
    ax_lda = subplot(nrows_plot, ncols_plot, (row-1)*ncols_plot + col_pair);
    % Row-major storage + R-convention y-axis flip:
    % reshape fills column-major → transpose for correct image.
    % flipud matches R's image() which plots row 1 at BOTTOM.
    coef_lda = flipud(reshape(S_lda(:, d), IMG_COLS, IMG_ROWS)');
    clim = max(abs(coef_lda(:)));
    imagesc(ax_lda, coef_lda, [-clim, clim]);
    colormap(ax_lda, cmap);
    axis(ax_lda, 'off'); axis(ax_lda, 'image');
    title(ax_lda, sprintf('LDA: Coef %d', d), 'FontSize', 8);
    
    % PDA coefficient image
    ax_pda = subplot(nrows_plot, ncols_plot, (row-1)*ncols_plot + col_pair + 1);
    coef_pda = flipud(reshape(S_pda(:, d), IMG_COLS, IMG_ROWS)');
    clim = max(abs(coef_pda(:)));
    imagesc(ax_pda, coef_pda, [-clim, clim]);
    colormap(ax_pda, cmap);
    axis(ax_pda, 'off'); axis(ax_pda, 'image');
    title(ax_pda, sprintf('PDA: Coef %d', d), 'FontSize', 8);
end

sgtitle(fig11, 'Figure 12.11 — Discriminant Coefficient Images (LDA vs PDA)', ...
    'FontSize', 12, 'FontWeight', 'bold');
fprintf('  [PASS] Figure 12.11 displayed.\n');

% ── Figure 12.12: Canonical Variate Scatter ─────────────────────────────────
fig12 = figure('Name', 'Figure 12.12 — PDA Canonical Variate Scatter', ...
    'Position', [100 100 700 680], 'Color', 'white');
ax12 = axes(fig12);
ax12.Color = 'white';
ax12.Box   = 'on';
hold(ax12, 'on');

% Visually distinct colors matching ESL figure (digits 0-9)
digit_colors = [
    0.12 0.47 0.71;   % 0 - blue
    0.84 0.15 0.16;   % 1 - red
    0.17 0.63 0.17;   % 2 - green
    0.58 0.40 0.74;   % 3 - purple
    0.55 0.34 0.29;   % 4 - brown
    0.89 0.47 0.76;   % 5 - pink
    0.50 0.50 0.50;   % 6 - gray
    0.74 0.74 0.13;   % 7 - olive/yellow
    0.09 0.75 0.81;   % 8 - cyan
    1.00 0.50 0.05;   % 9 - orange
];

coords_test = pda.project(X_test, 2);   % N_test x 2

% Sign convention for axes: textbook places digit 1 on POSITIVE Coordinate 1.
% Detect and flip axes signs if our convention is inverted.
if any(y_test == 1) && mean(coords_test(y_test == 1, 1)) < 0
    coords_test(:, 1) = -coords_test(:, 1);
    fprintf('  [INFO] Flipped Coordinate 1 sign (digit 1 centroid was negative).\n');
end
% Textbook places digit 6 on POSITIVE Coordinate 2 (upper half).
if any(y_test == 6) && mean(coords_test(y_test == 6, 2)) < 0
    coords_test(:, 2) = -coords_test(:, 2);
    fprintf('  [INFO] Flipped Coordinate 2 sign (digit 6 centroid was negative).\n');
end

% Determine axis limits from data (with padding), cap at ±7
pad = 0.5;
xl = [max(-7, min(coords_test(:,1)) - pad), min(7, max(coords_test(:,1)) + pad)];
yl = [max(-7, min(coords_test(:,2)) - pad), min(7, max(coords_test(:,2)) + pad)];
xlim(ax12, xl);
ylim(ax12, yl);  % Set BEFORE adding text so Clipping works

digits_sorted = unique(y_test);
class_centroids = zeros(K, 2);

for ki = 1:K
    dv = digits_sorted(ki);
    mask = (y_test == dv);
    x_pts = coords_test(mask, 1);
    y_pts = coords_test(mask, 2);
    col   = digit_colors(ki, :);

    % Only plot points inside axis limits to avoid crowding
    in_bounds = x_pts >= xl(1) & x_pts <= xl(2) & ...
                y_pts >= yl(1) & y_pts <= yl(2);

    % Plot digit characters (small, clipped)
    text(ax12, x_pts(in_bounds), y_pts(in_bounds), ...
        repmat({num2str(dv)}, sum(in_bounds), 1), ...
        'Color', col, 'FontSize', 6, 'FontWeight', 'normal', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'Clipping', 'on');

    class_centroids(ki, :) = [mean(x_pts), mean(y_pts)];
end

% Overlay class centroids as bold circled numbers
for ki = 1:K
    cx = class_centroids(ki, 1);
    cy = class_centroids(ki, 2);
    dv = digits_sorted(ki);
    plot(ax12, cx, cy, 'o', ...
        'MarkerSize', 20, 'LineWidth', 2.0, ...
        'MarkerFaceColor', 'white', 'MarkerEdgeColor', 'black');
    text(ax12, cx, cy, num2str(dv), ...
        'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'Clipping', 'on');
end

xlabel(ax12, 'PDA: Discriminant Coordinate 1', 'FontSize', 11);
ylabel(ax12, 'PDA: Discriminant Coordinate 2', 'FontSize', 11);
title(ax12, 'Figure 12.12 — First Two Penalized Canonical Variates (Test Data)', ...
    'FontSize', 11, 'FontWeight', 'bold');
grid(ax12, 'on');
hold(ax12, 'off');
fprintf('  [PASS] Figure 12.12 displayed.\n');

fprintf('\nDone.\n');
