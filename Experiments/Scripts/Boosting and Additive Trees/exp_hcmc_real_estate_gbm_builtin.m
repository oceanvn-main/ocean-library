% EXP_HCMC_REAL_ESTATE_GBM_BUILTIN  MART baseline using MATLAB built-in fitrensemble.
%
%   PURPOSE:    Apply the California Housing GBM experiment (ESL §10.14.1)
%               to the Ho Chi Minh City real estate dataset (2025).
%               Produces the same ESL figures (10.13–10.16) for HCMC data:
%               Learning Curve, Feature Importance, 1D PD, 2D PD.
%
%   DATASET:    Ho Chi Minh Real Estate 2025
%               Source: Data/Ho Chi Minh Real Estate Data 2025/data_public.csv
%               N ~ 51,304 raw rows. After cleaning: see Section 2.
%
%   TARGET:     Price (million VND) — log-transformed for normality.
%               Y = log10(Price_mil) so predictions are on a log10(mil VND) scale.
%               e.g., log10(5000) = 3.70 -> 5 billion VND.
%
%   MODEL:      fitrensemble LSBoost (MATLAB built-in, MSE loss).
%               Same hyperparameters as the California baseline.
%
%   ASSUMPTIONS:
%     A1. Drop rows missing Price, Area, Lat, or Lon (required for analysis).
%     A2. Drop geographically corrupted rows: Lat outside [10.4, 11.2]
%         and Lon outside [106.3, 107.1] — known bounds of HCMC province.
%     A3. Drop extreme price outliers: Price outside [500, 100000] million VND
%         (roughly 0.5 to 100 billion VND — covers virtually all residential).
%     A4. Drop extreme area outliers: Area outside [10, 5000] sqm.
%     A5. Missing optional features (Width, Bedrooms, etc.) imputed with
%         column median — better than dropping rows for sparse columns.
%     A6. Price is log10-transformed (log10(Price_bil)) to reduce skewness.
%     A7. Same train/test split (80/20), rng(42) for reproducibility.
%
%   REPRODUCIBILITY: rng(42, 'twister')

clc; clear; close all;
rng(42, 'twister');

%% === 1. Data Loading ===

scriptDir = fileparts(mfilename('fullpath'));
lvl1      = fileparts(scriptDir);
lvl2      = fileparts(lvl1);
rootDir   = fileparts(lvl2);
dataPath  = fullfile(rootDir, 'Data', 'Ho Chi Minh Real Estate Data 2025', 'data_public.csv');

assert(isfile(dataPath), 'EXP:DataNotFound', ...
    'data_public.csv not found at:\n  %s', dataPath);

T = readtable(dataPath, 'Encoding', 'UTF-8');
fprintf('[1] Loaded %d rows, %d columns\n', height(T), width(T));

%% === 2. Data Cleaning & Quality Filters ===
% ASSUMPTION A1-A4: Drop rows with missing or implausible values.
% Rationale: Geographic and price sanity filters remove data entry errors.

Price_mil = double(T.Price);       % million VND (already in this unit)
Area    = double(T.Area);         % m²
Lat     = double(T.Latitude);
Lon     = double(T.Longitude);
Width   = double(T.Width);        % metres
Length  = double(T.Length);       % metres
Beds    = double(T.Bedrooms);
Baths   = double(T.Bathrooms);
Floors  = double(T.Floors);
Alley   = double(T.AlleyWidth);   % metres


% Geographic sanity for HCMC province (known bounding box)
LAT_MIN = 10.40;  LAT_MAX = 11.20;
LON_MIN = 106.30; LON_MAX = 107.10;

% Price sanity: 500 to 100,000 million VND (0.5 to 100 billion)
PRICE_MIL_MIN = 500;
PRICE_MIL_MAX = 100000;

% Area sanity: 10 – 5000 m²
AREA_MIN = 10;
AREA_MAX = 5000;

validMask = ...
    ~isnan(Price_mil) & ~isnan(Area) & ~isnan(Lat) & ~isnan(Lon) & ...
    (Lat  >= LAT_MIN  & Lat  <= LAT_MAX)  & ...
    (Lon  >= LON_MIN  & Lon  <= LON_MAX)  & ...
    (Price_mil >= PRICE_MIL_MIN & Price_mil <= PRICE_MIL_MAX) & ...
    (Area >= AREA_MIN  & Area <= AREA_MAX);

n_raw = height(T);
T_clean   = T(validMask, :);

% Re-extract cleaned columns
Price_mil = Price_mil(validMask);
Area      = Area(validMask);
Lat       = Lat(validMask);
Lon       = Lon(validMask);
Width     = Width(validMask);
Length    = Length(validMask);
Beds      = Beds(validMask);
Baths     = Baths(validMask);
Floors    = Floors(validMask);
Alley     = Alley(validMask);

N_clean = sum(validMask);
fprintf('[2] After cleaning: N=%d (dropped %d rows)\n', N_clean, n_raw - N_clean);
fprintf('    Price [p5,p95] = [%.0f, %.0f] mil VND\n', prctile(Price_mil,5), prctile(Price_mil,95));
fprintf('    Area  [p5,p95] = [%.1f, %.1f] m^2\n', prctile(Area,5), prctile(Area,95));

%% === 3. Feature Engineering ===
% ASSUMPTION A5: Sparse columns (Beds, Baths, Floors, Width, Length, Alley)
%   imputed with column median so we keep all N_clean rows.
% ASSUMPTION A6: Y = log10(Price_bil) — right-skewed price benefits from log.

imputeMedian = @(v) fillmissing(v, 'constant', median(v, 'omitnan'));

Width  = imputeMedian(Width);
Length = imputeMedian(Length);
Beds   = imputeMedian(Beds);
Baths  = imputeMedian(Baths);
Floors = imputeMedian(Floors);
Alley  = imputeMedian(Alley);

% Derived features
PricePerSqm = Price_mil ./ Area;   % mil VND / m² — price density

% Feature matrix: 8 predictors
X = [Area, Width, Length, Beds, Baths, Floors, Alley, PricePerSqm];

% Log10-transform the target
% Y=log10(Price_mil): log10(1000)=3.0 -> 1 bil, log10(10000)=4.0 -> 10 bil
Y = log10(Price_mil);

featureNames = {'Area','Width','Length','Bedrooms','Bathrooms','Floors','AlleyWidth','PricePerSqm'};

N = size(X, 1);
p = size(X, 2);
fprintf('[3] Feature matrix: N=%d, p=%d\n', N, p);
fprintf('    Y=log10(Price_mil): median=%.3f (=%d mil VND), std=%.3f\n', ...
    median(Y), round(10^median(Y)), std(Y));

%% === 4. Train / Test Split (80/20) ===

idx   = randperm(N);
N_tr  = round(0.80 * N);
trIdx = idx(1:N_tr);
teIdx = idx(N_tr+1:end);

X_train = X(trIdx, :);   Y_train = Y(trIdx);
X_test  = X(teIdx,  :);  Y_test  = Y(teIdx);

fprintf('[4] N_train=%d, N_test=%d\n', N_tr, N - N_tr);

%% === 5. Model Configuration ===

M   = 800;
nu  = 0.10;
mns = 7;    % depth-3 binary tree: 2^3-1 = 7 internal splits

t = templateTree('MaxNumSplits', mns, 'MinLeafSize', 5);

fprintf('[5] fitrensemble: LSBoost, M=%d, nu=%.2f, MaxNumSplits=%d\n', M, nu, mns);

%% === 6. Training (timed) ===

fprintf('\n[6] Training MATLAB built-in GBM...\n');
tStart = tic;

mdl = fitrensemble(X_train, Y_train, ...
    'Method',            'LSBoost', ...
    'NumLearningCycles', M,         ...
    'LearnRate',         nu,        ...
    'Learners',          t);

elapsed = toc(tStart);
fprintf('[6] Training complete in %.1f seconds (%.1f min)\n', elapsed, elapsed/60);

%% === 7. Evaluation — Per-Iteration AAE on Train and Test ===
% Custom loss: mean absolute error in the log10 domain.
% In log10 space: AAE=0.1 -> 26% relative price error (10^0.1 - 1 ≈ 26%).
% The MAE here is directly interpretable as "orders of magnitude" error.

fprintf('[7] Computing cumulative AAE curves...\n');

maeLoss = @(Y, Yhat, W) mean(abs(Y - Yhat));

aae_train = loss(mdl, X_train, Y_train, 'LossFun', maeLoss, 'Mode', 'cumulative');
aae_test  = loss(mdl, X_test,  Y_test,  'LossFun', maeLoss, 'Mode', 'cumulative');

fprintf('[7] AAE curves computed.\n');

%% === 8. Final Metrics ===

aae_train_final = aae_train(end);
aae_test_final  = aae_test(end);

ss_res  = sum((Y_test - predict(mdl, X_test)).^2);
ss_tot  = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - ss_res / ss_tot;

baseline_aae = mean(abs(Y_test - median(Y_train)));

%% === 9. Learning Curve Visualization (OceanViz) ===

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Ho Chi Minh Real Estate — MATLAB Built-in GBM Baseline');
fig.Position = [60 80 860 500];
ax  = axes('Parent', fig);
viz.styleAxes(ax);

lcData.X = (1:M)';
lcData.Lines = { ...
    struct('Y', aae_train, 'Name', 'Train Error  (LSBoost)'), ...
    struct('Y', aae_test,  'Name', 'Test Error   (LSBoost)') ...
};
lcData.RefLines = { ...
    struct('Y', baseline_aae, 'Name', sprintf('Baseline = %.3f', baseline_aae)) ...
};

viz.plot(ax, 'LearningCurve', lcData, ...
    'Title',      'AAE vs Boosting Iterations — HCMC Real Estate', ...
    'XLabel',     'Iterations M',                                   ...
    'YLabel',     'AAE in log_{10}(Price)',                         ...
    'ShowLegend', true);

ylim(ax, [0, max([aae_train(1), aae_test(1)]) * 1.05]);

%% === 10. Summary Printout ===

fprintf('\n=== EXPERIMENT SUMMARY (HCMC Real Estate — MATLAB Built-in Baseline) ===\n');
fprintf('  Dataset        : Ho Chi Minh Real Estate 2025 (N=%d, p=%d)\n', N, p);
fprintf('  Target         : log10(Price in million VND)\n');
fprintf('  Model          : fitrensemble LSBoost (MATLAB built-in)\n');
fprintf('  Boosting Rounds: %d  |  LearnRate=%.2f  |  MaxNumSplits=%d\n', M, nu, mns);
fprintf('  Training Time  : %.1f seconds (%.1f min)\n', elapsed, elapsed/60);
fprintf('  Train AAE      : %.4f  (log10 units)\n', aae_train_final);
fprintf('  Test  AAE      : %.4f  (log10 units)\n', aae_test_final);
fprintf('  Test  R2       : %.4f\n', r2_test);
fprintf('  Baseline AAE   : %.4f  (predict median of training set)\n', baseline_aae);
fprintf('  Test AAE in %%  : ~%.1f%% relative price error (10^AAE - 1)\n', (10^aae_test_final - 1)*100);
fprintf('======================================================================\n');

%% === 11. Feature Importance (ESL Figure 10.14 style) ===

fprintf('\n[11] Computing feature importance...\n');
imp = predictorImportance(mdl);
imp = 100 * imp / max(imp);

[sortedImp, sIdx] = sort(imp, 'ascend');
sortedNames = featureNames(sIdx);

viz11  = OceanViz.Core('Theme', 'Dark');
C11    = viz11.Colors;
fig11  = viz11.createFigure('Relative Importance of Predictors — HCMC Real Estate');
fig11.Position = [80 80 680 430];
ax11   = axes('Parent', fig11);
viz11.styleAxes(ax11);

barh(ax11, sortedImp, 'FaceColor', C11.Primary, 'EdgeColor', 'none');
ax11.YTick      = 1 : numel(featureNames);
ax11.YTickLabel = sortedNames;
ax11.YTickLabelRotation = 0;
xlim(ax11, [0, 115]);
xlabel(ax11, 'Relative Importance', 'Color', C11.Text, 'FontSize', 10);
title(ax11, 'Relative Importance of Predictors (ESL Fig 10.14 style)', ...
    'Color', C11.Text, 'FontSize', 11);
ax11.XGrid = 'on';
fprintf('[11] Feature importance plotted.\n');

%% === 12. 1D Partial Dependence (ESL Figure 10.15 style) ===
% Plot PD for the 4 most informative features based on importance ranking.
% ASSUMPTION: Top 4 = highest importance from Section 11.
% ASSUMPTION: N_ss=1000 subsample, grid [2nd-98th] percentile, 100 points.
% ASSUMPTION: PD centred at 0 (textbook convention).

fprintf('[12] Computing 1D partial dependence (top 4 features, N_ss=1000)...\n');
tPD1 = tic;

% Top 4 features by importance (descending order, last = highest)
[~, impOrder] = sort(imp, 'descend');
pdFeatIdx = impOrder(1:4);
pdFeatLbl = featureNames(pdFeatIdx);

nPDGrid  = 100;
nSubsamp = min(1000, N_tr);

rng(42, 'twister');
ssIdx = randperm(N_tr, nSubsamp);
X_ss  = X_train(ssIdx, :);

% PASS 1: Compute all 4 PD curves
pdGrids = cell(4, 1);
pdVals  = cell(4, 1);

for fi = 1:4
    featJ  = pdFeatIdx(fi);
    xmin   = prctile(X_train(:, featJ),  2);
    xmax   = prctile(X_train(:, featJ), 98);
    grid_j = linspace(xmin, xmax, nPDGrid)';

    X_batch           = repmat(X_ss, nPDGrid, 1);
    X_batch(:, featJ) = repelem(grid_j, nSubsamp, 1);
    Yhat_batch        = predict(mdl, X_batch);
    pv                = mean(reshape(Yhat_batch, nSubsamp, nPDGrid), 1)';
    pv                = pv - mean(pv);

    pdGrids{fi} = grid_j;
    pdVals{fi}  = pv;
end

allPD   = cell2mat(pdVals);
yGlobal = [min(allPD) - 0.1*range(allPD), max(allPD) + 0.1*range(allPD)];

% PASS 2: Plot with shared y-axis
viz12  = OceanViz.Core('Theme', 'Dark');
C12    = viz12.Colors;
fig12  = viz12.createFigure('Partial Dependence — Top 4 Features (ESL Fig 10.15 style)');
fig12.Position = [80 80 860 560];
figure(fig12);

for fi = 1:4
    featJ  = pdFeatIdx(fi);
    grid_j = pdGrids{fi};
    pv     = pdVals{fi};

    ax = subplot(2, 2, fi);
    viz12.styleAxes(ax);
    hold(ax, 'on');

    plot(ax, grid_j, pv, 'Color', C12.Primary, 'LineWidth', 1.8);
    ylim(ax, yGlobal);
    yline(ax, 0, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');

    xlabel(ax, pdFeatLbl{fi}, 'FontSize', 9);
    ylabel(ax, 'Partial Dep. [log_{10}]', 'FontSize', 9);
    title(ax, pdFeatLbl{fi}, 'FontSize', 10, 'FontWeight', 'bold');

    % Red decile ticks
    deciles = prctile(X_train(:, featJ), 10:10:90);
    yLims   = ylim(ax);
    tickH   = 0.04 * (yLims(2) - yLims(1));
    for dk = 1:numel(deciles)
        line(ax, [deciles(dk) deciles(dk)], [yLims(1), yLims(1)+tickH], ...
            'Color', [0.9 0.2 0.2], 'LineWidth', 1.5);
    end
    hold(ax, 'off');
end
fprintf('[12] 1D partial dependence plotted in %.1f sec.\n', toc(tPD1));

%% === 13. 2D Partial Dependence: Width x Floors (ESL Figure 10.16 style) ===
% SELECTION RATIONALE:
%   We do NOT use automatic top-1/top-2 importance features because
%   PricePerSqm = Price/Area — mathematically coupled, not independent.
%   Their joint PD would show a spurious inverse signal, not a real interaction.
%
%   Domain choice: Width x Floors
%   - "Mat tien" (frontage width) is one of the most valued attributes in HCMC.
%     Wider plots command a street-front premium independent of total area.
%   - Number of floors = vertical dimension, key in HCMC's dense townhouse stock.
%   - These two are statistically independent and capture a genuine joint effect:
%     wide multi-story townhouses command the highest price premiums in HCMC.
% ASSUMPTION: 20x20 grid, same N_ss=1000 subsample.

fprintf('[13] Computing 2D partial dependence: Width x Floors (20x20 grid)...\n');
tPD2 = tic;

% Hardcoded domain-appropriate pair (indices into featureNames):
%   featureNames = {'Area','Width','Length','Bedrooms','Bathrooms','Floors','AlleyWidth','PricePerSqm'}
%                     1       2        3          4           5         6         7              8
feat1 = 2;   % Width  [metres] - frontage
feat2 = 6;   % Floors [count]  - building height
nG    = 20;

g1 = linspace(prctile(X_train(:, feat1), 5), prctile(X_train(:, feat1), 95), nG)';
g2 = linspace(prctile(X_train(:, feat2), 5), prctile(X_train(:, feat2), 95), nG)';

[G1_grid, G2_grid] = meshgrid(g1, g2);
G1_flat = G1_grid(:);
G2_flat = G2_grid(:);

X_batch2d            = repmat(X_ss, nG^2, 1);
X_batch2d(:, feat1)  = repelem(G1_flat, nSubsamp, 1);
X_batch2d(:, feat2)  = repelem(G2_flat, nSubsamp, 1);

Yhat2d   = predict(mdl, X_batch2d);
pd2_flat = mean(reshape(Yhat2d, nSubsamp, nG^2), 1)';
pd2      = reshape(pd2_flat, nG, nG);
pd2      = pd2 - mean(pd2(:));

fprintf('[13] 2D partial dependence computed in %.1f sec.\n', toc(tPD2));

viz13  = OceanViz.Core('Theme', 'Scientific');
fig13  = viz13.createFigure(sprintf('Joint PD: %s × %s (ESL Fig 10.16 style)', ...
    featureNames{feat1}, featureNames{feat2}));
fig13.Position = [80 80 720 530];
ax13   = axes('Parent', fig13);
viz13.styleAxes(ax13);

pdData13.G1 = g1;
pdData13.G2 = g2;
pdData13.Z  = pd2;

viz13.plot(ax13, 'PartialDependence2D', pdData13, ...
    'Title',        sprintf('Joint PD: %s \\times %s  (ESL Fig 10.16 style)', ...
        featureNames{feat1}, featureNames{feat2}), ...
    'XLabel',       featureNames{feat2},  ...
    'YLabel',       featureNames{feat1},  ...
    'ZLabel',       'Partial Dep. [log_{10} Price]', ...
    'Colormap',     'jet',                ...
    'AzimuthDeg',   -50,                  ...
    'ElevationDeg',  28,                  ...
    'EdgeAlpha',     0.2,                 ...
    'ShowColorbar',  true);

fprintf('[13] 2D surface plotted. Done.\n');
