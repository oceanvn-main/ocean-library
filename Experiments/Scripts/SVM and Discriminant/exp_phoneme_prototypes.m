%% exp_phoneme_prototypes.m — ESL Exercise 12.5 (Parts a & b)
%  Phoneme Classification via Smooth Prototypes (Natural Spline Basis).
%
%  THEORY:
%    The phoneme dataset contains 256-dim log-periodogram curves for 5
%    phonemes. Part (a) visualizes the raw functional data. Part (b)
%    projects curves into a low-dimensional natural spline coefficient
%    space, runs per-class K-means for smooth prototypes, and classifies
%    by nearest prototype.
%
%  Pipeline: Ocean Library 8-Step Data Science Pipeline
%  Author:  Ocean Library / ESL Chapter 12 Exercises
%  Date:    2026-03-24
%  Ref:     ESL Section 12.5, Exercise 12.5(a,b)
% =========================================================================

clc; clear; close all;
rng(42, 'twister');
RNG_SEED = 42;
DATASET  = 'phoneme';

fprintf('=== ESL Exercise 12.5(a): Phoneme Log-Periodograms ===\n\n');

%% === 1. Data Ingestion ===
% Real dataset: Phoneme log-periodograms (Hastie et al.)
% 4509 samples, 256 frequency features, 5 classes, ~60 speakers.

% Resolve project root robustly (works even if run from editor temp)
scriptPath = mfilename('fullpath');
if contains(scriptPath, 'Temp') || isempty(scriptPath)
    % Fallback: assume working directory is the project root
    projectRoot = pwd;
else
    projectRoot = fullfile(fileparts(scriptPath), '..', '..', '..');
end
dataPath = fullfile(projectRoot, 'Data', 'Phoneme', 'Phoneme.data.txt');
T = readtable(dataPath);

% Extract features (columns x_1 ... x_256)
featureCols = strncmp(T.Properties.VariableNames, 'x_', 2);
X = table2array(T(:, featureCols));    % (N x 256)
[N, P] = size(X);
freq = 0:(P-1);                         % frequency axis 0-255

% Encode class labels as integers
classNamesCell = {'aa', 'ao', 'dcl', 'iy', 'sh'};
K = numel(classNamesCell);
g = string(T.g);
y = zeros(N, 1);
for ci = 1:K
    y(strcmp(g, classNamesCell{ci})) = ci;
end

% Speaker IDs (needed for stratified split in Part c)
speakers = string(T.speaker);

% Print fundamental stats
fprintf('[Step 1] Dataset: %s\n', DATASET);
fprintf('  N = %d samples, P = %d features, K = %d classes\n', N, P, K);
for ci = 1:K
    fprintf('  Class %d (%s): %d samples (%.1f%%)\n', ...
        ci, classNamesCell{ci}, sum(y == ci), 100 * sum(y == ci) / N);
end
fprintf('  Unique speakers: %d\n', numel(unique(speakers)));

%% === 1.5. Train/Test Split (MANDATORY — Anti-Leakage) ===
% [SKIP] Part (a) is visualization-only — no model training.
% The full dataset is used for exploratory plotting.
% Speaker-stratified split will be applied in Part (c) when training.
fprintf('\n[Step 1.5] [SKIP] No split needed for Part (a) — visualization only.\n');

%% === 2. DataLake (DA-1) — Raw Persistence ===
% [SKIP] Real CSV file serves as raw immutable source.
fprintf('[Step 2]  [SKIP] Raw data persisted on disk: %s\n', dataPath);

%% === 3. DataWarehouse (DA-2/DA-3) — Structured Storage ===
% [SKIP] No schema-validated storage needed for EDA visualization.
fprintf('[Step 3]  [SKIP] No warehouse needed for EDA.\n');

%% === 4. CachingWarehouse (DA-4) — Read-Through Cache ===
% [SKIP] No repeated analytic queries.
fprintf('[Step 4]  [SKIP] No caching layer needed.\n');

%% === 5 & 6. FeatureStore & FeatureCache (DA-5/DA-6) ===
% [SKIP] No feature engineering for Part (a).
% Raw 256-dim log-periodograms are plotted directly.
fprintf('[Step 5-6] [SKIP] Raw features used directly — no transforms.\n');

%% === 6.5. Model Selection ===
% [SKIP] No model selection for visualization-only Part (a).
%
% 6.5a Feature Selection: N/A — all 256 frequency bins are plotted.
% 6.5b Cross-Validation:  N/A — no model to tune.
% 6.5c Model Comparison:  N/A — no competing models.
fprintf('[Step 6.5] [SKIP] No model selection — visualization task.\n');

%% === 7. Model Training & Evaluation ===
% [SKIP] Part (a) is purely descriptive — no training or prediction.
fprintf('[Step 7]  [SKIP] No training — Part (a) is descriptive.\n');

%% === 8. Visualization ===
% Part (a): Overlay all phoneme log-periodogram curves grouped by class.
% Each class gets its own panel showing individual curves (transparent)
% with a bold class mean overlay.
fprintf('\n[Step 8] Visualization: Phoneme Curves by Class\n');

viz = OceanViz.Core('Theme', 'Scientific');
fig_a = figure('Name', 'Phoneme Curves', 'Color', viz.Colors.Background);
set(fig_a, 'Position', [50, 150, 1800, 500]);

for ci = 1:K
    ax = subplot(1, K, ci, 'Parent', fig_a);
    mask = (y == ci);
    dataViz = struct('X', X(mask,:), ...
        'Y', ci * ones(sum(mask),1), ...       % use class index for palette mapping
        'xAxis', freq, ...
        'classNames', {{classNamesCell{ci}}}, ...
        'NumPaletteColors', K, ...              % request 5-color palette
        'ShowQuantiles', true);                 % show 10th-90th band
    viz.plot(ax, 'FunctionalData', dataViz, ...
        'Title', classNamesCell{ci}, ...
        'XLabel', 'Frequency', ...
        'YLabel', '', ...                       % only leftmost panel needs label
        'ShowLegend', true);
    if ci == 1
        ylabel(ax, 'Log-periodogram', 'Color', viz.Colors.Text);
    end
end

sgtitle(fig_a, 'ESL Ex 12.5(a): Phoneme Log-Periodograms by Class', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

resultsDir = fullfile(projectRoot, 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig_a, fullfile(resultsDir, 'ex12_5a_phoneme_curves.png'));
fprintf('[Step 8] Saved: Results/ex12_5a_phoneme_curves.png\n');

fprintf('\n=== Part (a) Done ===\n');

%% ========================================================================
%                     PART (b): Smooth Prototype Classification
%  ========================================================================
%
%  THEORY (ESL 12.5):
%    Each 256-dim log-periodogram curve is projected into a low-dimensional
%    space of natural cubic spline coefficients.  K-means is run IN this
%    coefficient space (per class), giving smooth prototypes.
%    Classification assigns each curve to the nearest smooth prototype.
%
%  PROCEDURE:
%    1. Build basis B (256 x J) using natural cubic spline with J knots.
%       Prepend intercept: B = [1, N_basis].
%    2. QR factorize:  B = Q*R.  Q is (256 x J) orthonormal.
%       This avoids the costly (B'B)^{-1} inversion.
%    3. Project each sample: theta_i = Q' * x_i  (J x 1 coefficients).
%    4. Per-class K-means in theta-space -> R prototypes per class.
%    5. Classify: argmin_k || Q'*x - theta_k ||_2 (nearest in coeff space).
%    6. Reconstruct smooth prototypes: m = Q * theta_k (back to 256-dim).
%
% =========================================================================

fprintf('\n=== ESL Exercise 12.5(b): Smooth Prototype Classification ===\n\n');

%% === 1.5. Train/Test Split (MANDATORY — Anti-Leakage) ===
% Speaker-stratified 50-50 split: no speaker appears in both sets.
% This prevents the model from memorizing speaker-specific patterns.

uniqueSpeakers = unique(speakers);
nSpeakers = numel(uniqueSpeakers);
rng(RNG_SEED);
perm = randperm(nSpeakers);
halfSp = floor(nSpeakers / 2);
trainSpeakers = uniqueSpeakers(perm(1:halfSp));
testSpeakers  = uniqueSpeakers(perm(halfSp+1:end));

trainIdx = ismember(speakers, trainSpeakers);
testIdx  = ismember(speakers, testSpeakers);

X_train = X(trainIdx, :);  y_train = y(trainIdx);
X_test  = X(testIdx, :);   y_test  = y(testIdx);

fprintf('[Step 1.5] Speaker-stratified split:\n');
fprintf('  Train: %d samples (%d speakers)\n', sum(trainIdx), halfSp);
fprintf('  Test:  %d samples (%d speakers)\n', sum(testIdx), nSpeakers - halfSp);

%% === 2-4. DataLake / DataWarehouse / CachingWarehouse ===
% [SKIP] Not needed for this single-run prototype experiment.
fprintf('[Step 2-4] [SKIP] No persistence layers for prototype experiment.\n');

%% === 5 & 6. Feature Engineering: Natural Spline Basis Projection ===
% This IS the core feature engineering step.  We transform 256-dim
% raw log-periodograms into J-dim spline coefficients via QR-based
% projection: theta_i = Q' * x_i.  This enforces smoothness.

J_BASIS = 12;  % number of spline knots (including boundaries)
R_PROTO = 5;   % R: number of prototypes per class (distinct from K = number of classes)

% Build natural spline basis over the frequency domain
interiorKnots = linspace(0, P-1, J_BASIS + 2);  % includes boundaries
knots = interiorKnots(:);
N_basis = OceanStats.NonParametric.getNSBasis(freq(:), knots);
B = [ones(P, 1), N_basis];   % (256 x J) with intercept
J = size(B, 2);              % actual dimension after intercept

% QR factorization for fast, stable projection
[Q, ~] = qr(B, 0);  % thin QR: Q is (256 x J), Q'Q = I

% Project training and test data into spline coefficient space
Theta_train = X_train * Q;   % (N_train x J)
Theta_test  = X_test  * Q;   % (N_test  x J)

fprintf('[Step 5-6] Spline basis projection:\n');
fprintf('  Knots: %d (J=%d basis dim), R=%d prototypes/class\n', ...
    J_BASIS, J, R_PROTO);
fprintf('  Projection: 256-dim -> %d-dim via QR(B)\n', J);

%% === 6.5. Model Selection ===
% [SKIP for Part b] Fixed configuration R=5, J=12.
% Part (c) will do the full grid search.
fprintf('[Step 6.5] [SKIP] Fixed config for Part (b): R=%d, J=%d. Grid search in Part (c).\n', R_PROTO, J_BASIS);

%% === 7. Model Training & Evaluation ===
% Per-class K-means in coefficient space -> nearest-prototype classification.

fprintf('[Step 7] Training: K-means in spline coefficient space\n');

rng(RNG_SEED);
allProtos      = [];    % will be (K*R_PROTO x J)  — K classes x R prototypes each
allProtoLabels = [];    % (K*R_PROTO x 1)

for ci = 1:K
    classMask = (y_train == ci);
    ThetaClass = Theta_train(classMask, :);

    if R_PROTO == 1
        proto = mean(ThetaClass, 1);
    else
        km = OceanStats.Unsupervised.KMeans();
        km.fit(ThetaClass, R_PROTO);
        proto = km.Centroids;
    end

    allProtos = [allProtos; proto]; %#ok<AGROW>
    allProtoLabels = [allProtoLabels; ci * ones(size(proto, 1), 1)]; %#ok<AGROW>
end

% Classify test set: nearest prototype in theta-space
D2 = sum(Theta_test.^2, 2) + sum(allProtos.^2, 2)' ...
    - 2 * (Theta_test * allProtos');
D2 = max(D2, 0);
[~, nearest] = min(D2, [], 2);
y_pred = allProtoLabels(nearest);

testErr = mean(y_pred ~= y_test);
fprintf('  Test Error: %.1f%% (R=%d prototypes/class, J=%d basis)\n', ...
    testErr * 100, R_PROTO, J);

%% === 8. Visualization: Smooth Prototypes ===
fprintf('[Step 8] Visualization: Smooth Prototypes by Class\n');

palette = viz.getColorSequence(K);

fig_b = figure('Name', 'Smooth Prototypes', 'Color', viz.Colors.Background);
set(fig_b, 'Position', [50, 80, 1800, 500]);

for ci = 1:K
    ax = subplot(1, K, ci, 'Parent', fig_b);
    hold(ax, 'on'); viz.styleAxes(ax);
    classMask = (y_train == ci);

    % Faded raw curves (subsample for speed)
    Xc = X_train(classMask, :);
    nShow = min(100, size(Xc, 1));
    rng(ci);
    showIdx = randperm(size(Xc, 1), nShow);
    for si = 1:nShow
        plot(ax, freq, Xc(showIdx(si), :), ...
            'Color', [palette(ci,:), 0.06], 'LineWidth', 0.3, ...
            'HandleVisibility', 'off');
    end

    % Smooth prototypes (reconstructed from Q * theta)
    % Each prototype gets a distinct color within the panel
    protoPalette = viz.getColorSequence(R_PROTO);
    classProtoIdx = find(allProtoLabels == ci);
    for pi_idx = 1:numel(classProtoIdx)
        smoothProto = Q * allProtos(classProtoIdx(pi_idx), :)';
        pCol = protoPalette(pi_idx, :);
        % Dark outline
        plot(ax, freq, smoothProto, 'Color', [0 0 0], ...
            'LineWidth', 3.0, 'HandleVisibility', 'off');
        % Colored line
        plot(ax, freq, smoothProto, 'Color', pCol, ...
            'LineWidth', 2.0, ...
            'DisplayName', sprintf('Proto %d', pi_idx));
    end

    title(ax, sprintf('%s (R=%d)', classNamesCell{ci}, R_PROTO), ...
        'Color', viz.Colors.Text, 'FontSize', 11);
    xlabel(ax, 'Frequency');
    if ci == 1
        ylabel(ax, 'Log-periodogram', 'Color', viz.Colors.Text);
    end
    legend(ax, 'Location', 'southwest', 'FontSize', 7, ...
        'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
        'Color', [viz.Colors.Background, 0.85]);
    hold(ax, 'off');
end

sgtitle(fig_b, sprintf('ESL Ex 12.5(b): Smooth Prototypes (R=%d/class) — Test Error: %.1f%%', R_PROTO, testErr * 100), ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig_b, fullfile(resultsDir, 'ex12_5b_smooth_prototypes.png'));
fprintf('[Step 8] Saved: Results/ex12_5b_smooth_prototypes.png\n');

fprintf('\n=== Part (b) Done ===\n');

%% ========================================================================
%                     PART (c): Grid Search — R x J
%  ========================================================================
%
%  ESL 12.5(c): Vary R (prototypes per class) in {1,3,5,7} and
%  J (spline basis dimension) in {5,10,15}.  Evaluate each combination
%  on the same speaker-stratified train/test split from Part (b).
%  Report test error rates as a R x J error table.
%  (K = 5 classes is fixed throughout; R is the tunable prototype count.)
%
%  Pipeline mapping:
%    Step 1.5  — REUSE speaker-stratified split from Part (b) (no re-split)
%    Step 5-6  — Recompute QR projection once per J value (outer loop)
%    Step 6.5b — Grid sweep (R x J) acts as our hyperparameter search
%    Step 7    — Final evaluation on held-out test set per config
%    Step 8    — Error heatmap
%
% =========================================================================

fprintf('\n=== ESL Exercise 12.5(c): Grid Search R x J (K=%d classes fixed) ===\n\n', K);
fprintf('[Step 1.5] Reusing speaker-stratified split from Part (b).\n');
fprintf('[Step 2-4] [SKIP] No persistence layers.\n');

%% === 6.5b. Grid Search: R x J ===
% Outer loop over J (recomputes QR projection — O(P*J) per sample).
% Inner loop over R (reruns K-means per class — fast).

Rs = [1, 3, 5, 7];   % R: prototypes per class — controls representational capacity
Js = [5, 10, 15];    % spline knots: controls dimension of the function space
                      % Larger J -> more degrees of freedom -> prototypes can
                      % represent finer local spectral variation, but are less
                      % regularized (higher risk of fitting speaker-specific noise)

errTable = zeros(numel(Rs), numel(Js));

fprintf('[Step 6.5b] Grid search: R in {%s}, J in {%s}\n', ...
    num2str(Rs), num2str(Js));

for ji = 1:numel(Js)
    J_val = Js(ji);  % current basis dimension

    % --- Feature engineering: natural spline basis for this J ---
    % Knot vector: J_val interior + 2 boundary = J_val+2 total points
    knotsJ = linspace(0, P-1, J_val + 2);
    NbasisJ = OceanStats.NonParametric.getNSBasis(freq(:), knotsJ(:));
    BJ = [ones(P, 1), NbasisJ];   % (256 x J_dim)
    [QJ, ~] = qr(BJ, 0);          % thin QR: (256 x J_dim)

    % Project train/test into spline coefficient space
    Theta_tr = X_train * QJ;   % (N_train x J_dim)
    Theta_te = X_test  * QJ;   % (N_test  x J_dim)

    for ri = 1:numel(Rs)
        R_val = Rs(ri);  % current number of prototypes per class

        % --- K-means per class in coefficient space ---
        rng(RNG_SEED);
        protos      = [];
        protoLabels = [];

        for ci = 1:K
            classMask = (y_train == ci);
            ThetaC = Theta_tr(classMask, :);

            if R_val == 1
                proto = mean(ThetaC, 1);
            else
                km = OceanStats.Unsupervised.KMeans();
                km.fit(ThetaC, R_val);
                proto = km.Centroids;
            end

            protos      = [protos;      proto]; %#ok<AGROW>
            protoLabels = [protoLabels; ci * ones(size(proto,1), 1)]; %#ok<AGROW>
        end

        % --- Classify test set: nearest prototype ---
        D2 = sum(Theta_te.^2, 2) + sum(protos.^2, 2)' ...
            - 2 * (Theta_te * protos');
        D2 = max(D2, 0);
        [~, nn] = min(D2, [], 2);
        y_hat = protoLabels(nn);

        errTable(ri, ji) = mean(y_hat ~= y_test);
        fprintf('  R=%d, J=%2d  →  Test Error = %.1f%%\n', ...
            R_val, J_val, errTable(ri,ji)*100);
    end
end

%% === 7. Final Results Table ===
fprintf('\n[Step 7] Test Error Rate Table (rows=R prototypes/class, cols=J knots):\n');
fprintf('%8s', 'R \\ J');
for ji = 1:numel(Js), fprintf('%10d', Js(ji)); end
fprintf('\n');
for ri = 1:numel(Rs)
    fprintf('%8d', Rs(ri));
    for ji = 1:numel(Js)
        fprintf('%9.1f%%', errTable(ri,ji)*100);
    end
    fprintf('\n');
end

[bestErr, bestIdx] = min(errTable(:));
[bestRi, bestJi]   = ind2sub(size(errTable), bestIdx);
fprintf('\nBest config: R=%d, J=%d  →  %.1f%% error\n', ...
    Rs(bestRi), Js(bestJi), bestErr*100);

%% === 8. Visualization: Error Heatmap ===
fprintf('\n[Step 8] Visualization: K x J Error Heatmap\n');

fig_c = figure('Name', 'Grid Search Heatmap', 'Color', viz.Colors.Background);
set(fig_c, 'Position', [200, 200, 600, 400]);

ax_h = axes('Parent', fig_c);
imagesc(ax_h, errTable * 100);
colormap(ax_h, flipud(hot));          % low error = bright/warm
cb = colorbar(ax_h);
cb.Color = viz.Colors.Text;
cb.Label.String = 'Test Error (%)';
cb.Label.Color  = viz.Colors.Text;

set(ax_h, ...
    'XTick', 1:numel(Js), 'XTickLabel', Js, ...
    'YTick', 1:numel(Rs), 'YTickLabel', Rs, ...
    'Color', viz.Colors.Background, ...
    'XColor', viz.Colors.Text, 'YColor', viz.Colors.Text, ...
    'FontSize', 11);
xlabel(ax_h, 'J (spline knots)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax_h, 'R (prototypes / class)', 'Color', viz.Colors.Text, 'FontSize', 12);
title(ax_h, 'ESL 12.5(c): Test Error (%) — R (prototypes/class) × J (knots) Grid', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');

% Annotate each cell with the error value
for ri = 1:numel(Rs)
    for ji = 1:numel(Js)
        errVal = errTable(ri, ji) * 100;   % FIXED: was errTable(ki,ji)
        % White text on dark cells, black on bright
        txtColor = 'w';
        if errVal < 15, txtColor = 'k'; end
        text(ax_h, ji, ri, sprintf('%.1f%%', errVal), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 12, 'FontWeight', 'bold', 'Color', txtColor);
    end
end

% Mark the best cell with a rectangle
rectangle(ax_h, 'Position', [bestJi-0.5, bestRi-0.5, 1, 1], ...
    'EdgeColor', [0.2 1 0.4], 'LineWidth', 2.5);

saveas(fig_c, fullfile(resultsDir, 'ex12_5c_grid_heatmap.png'));
fprintf('[Step 8] Saved: Results/ex12_5c_grid_heatmap.png\n');

fprintf('\n=== Part (c) Done ===\n');

%% ========================================================================
%  Figure 4: Smooth Prototypes using Optimal (R*, J*) from Grid Search
%  ========================================================================
%  Re-runs training with the best configuration identified in Part (c)
%  and plots per-class smooth prototypes, analogous to Part (b) but
%  showing the best-performing model.
% =========================================================================

R_opt = Rs(bestRi);   % optimal prototypes per class
J_opt = Js(bestJi);   % optimal number of knots

fprintf('\n=== Figure 4: Optimal Config Prototypes (R*=%d, J*=%d) ===\n', R_opt, J_opt);

% Rebuild basis and projection for optimal J
knotsOpt  = linspace(0, P-1, J_opt + 2);
NbasisOpt = OceanStats.NonParametric.getNSBasis(freq(:), knotsOpt(:));
BOpt      = [ones(P, 1), NbasisOpt];
[QOpt, ~] = qr(BOpt, 0);

Theta_tr_opt = X_train * QOpt;
Theta_te_opt = X_test  * QOpt;

% K-means per class with optimal R
rng(RNG_SEED);
optProtos      = [];
optProtoLabels = [];

for ci = 1:K
    classMask = (y_train == ci);
    ThetaC    = Theta_tr_opt(classMask, :);
    if R_opt == 1
        proto = mean(ThetaC, 1);
    else
        km = OceanStats.Unsupervised.KMeans();
        km.fit(ThetaC, R_opt);
        proto = km.Centroids;
    end
    optProtos      = [optProtos;      proto]; %#ok<AGROW>
    optProtoLabels = [optProtoLabels; ci * ones(size(proto,1), 1)]; %#ok<AGROW>
end

% Evaluate on test set
D2opt = sum(Theta_te_opt.^2, 2) + sum(optProtos.^2, 2)' ...
    - 2 * (Theta_te_opt * optProtos');
D2opt = max(D2opt, 0);
[~, nn_opt] = min(D2opt, [], 2);
y_hat_opt   = optProtoLabels(nn_opt);
errOpt      = mean(y_hat_opt ~= y_test);
fprintf('  Confirmed Test Error: %.1f%%\n', errOpt * 100);

% Plot
fig_d = figure('Name', 'Optimal Smooth Prototypes', 'Color', viz.Colors.Background);
set(fig_d, 'Position', [50, 80, 1800, 500]);

protoPaletteOpt = viz.getColorSequence(R_opt);

for ci = 1:K
    ax = subplot(1, K, ci, 'Parent', fig_d);
    hold(ax, 'on'); viz.styleAxes(ax);
    classMask = (y_train == ci);

    % Faded raw curves
    Xc = X_train(classMask, :);
    nShow = min(100, size(Xc, 1));
    rng(ci);
    showIdx = randperm(size(Xc, 1), nShow);
    for si = 1:nShow
        plot(ax, freq, Xc(showIdx(si), :), ...
            'Color', [palette(ci,:), 0.06], 'LineWidth', 0.3, ...
            'HandleVisibility', 'off');
    end

    % Optimal smooth prototypes
    classProtoIdx = find(optProtoLabels == ci);
    for pi_idx = 1:numel(classProtoIdx)
        smoothProto = QOpt * optProtos(classProtoIdx(pi_idx), :)';
        pCol = protoPaletteOpt(pi_idx, :);
        plot(ax, freq, smoothProto, 'Color', [0 0 0], ...
            'LineWidth', 3.0, 'HandleVisibility', 'off');
        plot(ax, freq, smoothProto, 'Color', pCol, ...
            'LineWidth', 2.0, 'DisplayName', sprintf('Proto %d', pi_idx));
    end

    title(ax, sprintf('%s (R*=%d)', classNamesCell{ci}, R_opt), ...
        'Color', viz.Colors.Text, 'FontSize', 11);
    xlabel(ax, 'Frequency');
    if ci == 1
        ylabel(ax, 'Log-periodogram', 'Color', viz.Colors.Text);
    end
    legend(ax, 'Location', 'southwest', 'FontSize', 7, ...
        'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
        'Color', [viz.Colors.Background, 0.85]);
    hold(ax, 'off');
end

sgtitle(fig_d, sprintf('ESL 12.5: Optimal Smooth Prototypes (R*=%d, J*=%d) — Test Error: %.1f%%', ...
    R_opt, J_opt, errOpt * 100), ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig_d, fullfile(resultsDir, 'ex12_5d_optimal_prototypes.png'));
fprintf('[Step 8] Saved: Results/ex12_5d_optimal_prototypes.png\n');

fprintf('\n=== Done ===\n');

