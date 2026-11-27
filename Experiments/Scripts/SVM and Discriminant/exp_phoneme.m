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


%% Part b

fprintf('\n=== ESL Exercise 12.5(b): Smooth Prototype Classification ===\n\n');
uniqueSpeakers = unique(speakers);
nSpeakers = numel(uniqueSpeakers);
rng(RNG_SEED);
perm = randperm(nSpeakers);
halfSp = floor(nSpeakers / 2);
trainSpeakers = uniqueSpeakers(perm(1:halfSp));
testSpeakers = uniqueSpeakers(perm(halfSp+1:end));

trainIdx = ismember(speakers, trainSpeakers);
testIdx = ismember(speakers, testSpeakers);

X_train = X(trainIdx, :); y_train = y(trainIdx);
X_test = X(testIdx, :); y_test = y(testIdx);

fprintf('[Step 1.5] Speaker-stratified split:\n');
fprintf('  Train: %d samples (%d speakers)\n', sum(trainIdx), halfSp);
fprintf('  Test:  %d samples (%d speakers)\n', sum(testIdx), nSpeakers - halfSp);


%% === 2-4. DataLake / DataWarehouse / CachingWarehouse ===
% [SKIP] Not needed for this single-run prototype experiment.
fprintf('[Step 2-4] [SKIP] No persistence layers for prototype experiment.\n');

%5-6. Feature Engineering: Natural Spline Basis Projection
%   Transform 256-dim raw log-periodograms into J-dim spline coeff via Q-R
%   based projection 
J_BASIS = 12; %Number of spline knots ( including boundaries)
K_PROTO = 5;

%Build Natural spline basis over the frequency domain

interiorKnots = linspace(0, P-1, J_BASIS+2); 
knots = interiorKnots(:);
N_basis = OceanStats.NonParametric.getNSBasis(freq(:), knots);
B = [ones(P, 1), N_basis]; % 256 x J with intercept
J = size(B, 2); %Actual dimension after intercept

%QR factorization for fast, stable projection
[Q, ~] = qr(B, 0); 
%Project training and test data into spline coeff space
Theta_train = X_train * Q;
Theta_test = X_test * Q;
fprintf('[Step 5-6] Spline basis projection:\n');
fprintf('  Knots: %d (J=%d basis dim), K=%d prototypes/class\n', ...
    J_BASIS, J, K_PROTO);
fprintf('  Projection: 256-dim -> %d-dim via QR(B)\n', J);

%% === 6.5. Model Selection ===
% [SKIP for Part b] Fixed configuration K=5, J=12.
% Part (c) will do the full grid search.
fprintf('[Step 6.5] [SKIP] Fixed config for Part (b). Grid search in Part (c).\n');

%% === 7. Model Training & Evaluation ===
% Per-class K-means in coefficient space -> nearest-prototype classification.
fprintf('[Step 7] Training: K-means in spline coefficient space\n');

rng(RNG_SEED);
allProtos = [];
allProtoLabels = []; 

for ci = 1:K
    classMask = (y_train == ci);
    ThetaClass = Theta_train(classMask, :);
    if K_PROTO == 1
        proto = mean(ThetaClass, 1);
    else
        km = OceanStats.Unsupervised.KMeans();
        km.fit(ThetaClass, K_PROTO);
        proto = km.Centroids;
    end
    allProtos = [allProtos; proto];
    allProtoLabels = [allProtoLabels; ci * ones(size(proto, 1), 1)];
end

%Classify test set: nearest prototype in theta-space
D2 = sum(Theta_test.^2, 2) + sum(allProtos.^2, 2)' - 2*(Theta_test * allProtos');

D2 = max(D2, 0);
[~, nearest] = min(D2, [], 2);
y_pred = allProto(nearest);

testErr = mean(y_pred ~= y_test);
fprintf('  Test Error: %.1f%% (K=%d prototypes/class, J=%d basis)\n', ...
    testErr * 100, K_PROTO, J);

%% Visual
palette = viz.getColorSequence(K);
fig_b = figure('Name', 'Smooth Prototypes', 'Color', viz.Colors.Background);
set(fig_b, 'Position', [50, 80, 1800, 500]);

for ci = 1:K
    ax = subplot(1, K, ci, 'Parent', fig_b);
    hold(ax, 'on'); viz.styleAxes(ax);
    classMask = (y_train == ci);
    %Faded raw curves (sub sample for speed)
    Xc = X_train(classMask, :);
    nShow = min(100, size(Xc, 1));
    rng(ci);
    showIdx = randperm(size(Xc, 1), nShow);
    for si = 1:nShow
        plot(ax, freq, Xc(showIdx(si), :), ...
            'Color', [palette(ci,:), 0.06], 'LineWidth', 0.3, ...
            'HandleVisibility', 'off');
    end

    %Smooth prototypes (reconstructed from Q * theta)
    protoPalette = viz.getColorSequence(K_PROTO);
    classProtoIdx = find(allProtoLabels == ci);
    for pi_idx = 1:numel(classProtoIdx)
        smoothProto = Q * allProtos(classProtoIdx(pi_idx), :)';
        pCol = protoPalette(pi_idx, :);
        
    end


    %
end
