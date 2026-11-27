clc; clear; close all;
rng(42, 'twister');

%% === 1. Data Loading ===

scriptDir = fileparts(mfilename('fullpath'));
lvl1      = fileparts(scriptDir);
lvl2      = fileparts(lvl1);
rootDir   = fileparts(lvl2);
dataPath  = fullfile(rootDir, 'Data', 'California Housing Prices', 'housing.csv');

assert(isfile(dataPath), 'EXP:DataNotFound', ...
    'housing.csv not found at:\n  %s', dataPath);

T = readtable(dataPath);
fprintf('[1] Loaded %d rows, %d columns\n', height(T), width(T));

%% === 2. Feature Engineering (identical to custom experiment) ===

assert(all(T.households > 0), 'EXP:ZeroHouseholds');

AveRooms  = T.total_rooms    ./ T.households;
AveBedrms = T.total_bedrooms ./ T.households;
AveOccup  = T.population     ./ T.households;

X_raw = [T.median_income, T.households, AveOccup, T.latitude, T.longitude, ...
         AveRooms, AveBedrms, T.housing_median_age];
Y_raw = T.median_house_value ./ 1e5;

featureNames = {'MedInc','House','AveOccup','Lat','Lon','AveRooms','AveBedrms','MedAge'};

validMask = all(~isnan(X_raw), 2) & ~isnan(Y_raw);
X = X_raw(validMask, :);
Y = Y_raw(validMask);
N = size(X, 1);

assert(N > 20000, 'EXP:TooManyDropped', 'Only %d rows after NaN removal.', N);
fprintf('[2] N=%d, p=%d (dropped %d NaN rows)\n', N, size(X,2), sum(~validMask));

%% Train / Test Split (80/ 20, same seed = fair comparison)
idx = randperm(N);
N_tr = round(0.80*N);
trIdx = idx(1:N_tr);
teIdx = idx(N_tr+1:end);
X_train = X(trIdx, :); Y_train = Y(trIdx);
X_test = X(teIdx, :); Y_test = Y(teIdx);
fprintf('N_train = %d, N_test = %d\n', N_tr, N - N_tr);

%% Model Configuration
M = 800;
nu = 0.10; %Learnign rate
mns = 7; %MaxNumSplits: Depth-3 binary tree has at most 2^3 - 1 = 8 internal splits
t = templateTree('MaxNumSplits', mns, 'MinLeafSize', 5);
fprintf('[4] fitrensemble: LSBoost, M=%d, nu=%.2f, MaxNumSplits=%d\n', M, nu, mns);
%% Training
fprintf('Traning MAtlab build-in GBM...\n');
tStart = tic;
mdl = fitrensemble(X_train, Y_train, ...
    'Method',       'LSBoost', ...
    'NumLearningCycles', M,    ...
    'LearnRate',    nu,        ...
    'Learners',     t);

elapsed = toc(tStart);
fprintf('[5] Training complete in %.1f seconds (%.1f min)\n', elapsed, elapsed/60);

%% Evaluation - Per-iteration AAE on Train and Test


maeLoss = @(Y, Yhat, W) mean(abs(Y - Yhat));

aae_train = loss(mdl, X_train, Y_train, 'LossFun', maeLoss, 'Mode', 'cumulative');
aae_test  = loss(mdl, X_test,  Y_test,  'LossFun', maeLoss, 'Mode', 'cumulative');

fprintf('[6] AAE curves computed.\n');


%Final metrisc
aae_train_final = aae_train(end);
aae_test_final = aae_test(end);
ss_res = sum((Y_test - predict(mdl, X_test).^2));
ss_tot = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - ss_res / ss_tot;
baseline_aae = mean(abs(Y_test - median(Y_train)));

%% Visual
viz = OceanViz.Core('Theme', 'Scientific');
fig = viz.createFigure('California Housing — MATLAB Built-in GBM (LSBoost Baseline)');
fig.Position = [60 80 860 500];

ax = axes('Parent', fig);
viz.styleAxes(ax);

%% 
lcData = struct();
lcData.X = (1:M)';

lcData.Lines = {
    struct('Y', aae_train, 'Name', 'Train Error (LSBoost)'),
    struct('Y', aae_test, 'Name', 'Test Error (LSBoost)')};
lcData.RefLines = {
    struct('Y', baseline_aae, 'Name', sprintf('Baseline = %.3f', baseline_aae))};


viz.plot(ax, 'LearningCurve', lcData, ...
    'Title',      'Training and Test Absolute Error', ...
    'XLabel',     'Iterations M', ...
    'YLabel',     'Absolute Error', ...
    'ShowLegend', true);

% Y-axis from 0 to match textbook range
ylim(ax, [0, max([aae_train(1), aae_test(1)]) * 1.05]);


%% === 8. Summary Printout ===

fprintf('\n=== EXPERIMENT SUMMARY (MATLAB Built-in Baseline) ===\n');
fprintf('  Dataset        : California Housing (N=%d, p=%d)\n', N, size(X,2));
fprintf('  Model          : fitrensemble LSBoost (MATLAB built-in)\n');
fprintf('  Boosting Rounds: %d  |  LearnRate=%.2f  |  MaxNumSplits=%d\n', M, nu, mns);
fprintf('  Training Time  : %.1f seconds (%.1f min)\n', elapsed, elapsed/60);
fprintf('  Train AAE      : %.4f\n', aae_train_final);
fprintf('  Test  AAE      : %.4f\n', aae_test_final);
fprintf('  Test  R2       : %.4f\n', r2_test);
fprintf('  Baseline AAE   : %.4f  (constant = median(Y_train))\n', baseline_aae);
fprintf('  ---\n');
fprintf('  [Context] Textbook (Friedman 2001 / ESL §10.14.1):\n');
fprintf('            AAE ~ 0.31,  R2 ~ 0.84,  Baseline ~ 0.89\n');
fprintf('  [Note] LSBoost != Huber loss — results will differ from textbook\n');
fprintf('======================================================\n');

%% Feature Importance (10.42- 10.43 Eqs)

%% === 9. Feature Importance (ESL Figure 10.14) ===
% predictorImportance() sums the variance reduction (RSS decrease) at every
% split involving each feature across all M=800 trees, then normalizes so
% the maximum predictor scores 100. This is the "relative importance" measure
% Friedman (2001) uses for variable ranking.

fprintf('\n[9] Computing feature importance...\n');
imp = predictorImportance(mdl);          % 1×p raw importance
imp = 100 * imp / max(imp);             % normalize: best predictor = 100

% Sort ascending so most important is at the TOP of the horizontal bar chart
[sortedImp, sIdx] = sort(imp, 'ascend');
sortedNames = featureNames(sIdx);

viz9  = OceanViz.Core('Theme', 'Dark');
C9    = viz9.Colors;
fig9  = viz9.createFigure('Relative Importance of Predictors');
fig9.Position = [80 80 680 430];
ax9   = axes('Parent', fig9);
viz9.styleAxes(ax9);

barh(ax9, sortedImp, 'FaceColor', C9.Primary, 'EdgeColor', 'none');
ax9.YTick      = 1 : numel(featureNames);
ax9.YTickLabel = sortedNames;
ax9.YTickLabelRotation = 0;
xlim(ax9, [0, 115]);
xlabel(ax9, 'Relative Importance', 'Color', C9.Text, 'FontSize', 10);
title(ax9, 'Relative Importance of Predictors (ESL Fig 10.14)', ...
    'Color', C9.Text, 'FontSize', 11);
ax9.XGrid = 'on';

fprintf('[9] Feature importance plotted.\n');

%% === 10. 1-Variable Partial Dependence (ESL Figure 10.15) ===
% Partial dependence of response on ONE feature, averaging over all others:
%   PD(x_j) = (1/N_ss) * sum_i  f_hat(x_1i, ..., x_j_fixed, ..., x_pi)
%
% ASSUMPTION: Subsample N_ss=1000 rows from training set for marginalisation.
%   Rationale: using all N_tr=16k rows × 100 grid points × 800 trees is too slow.
%   1000 rows captures the joint distribution well (CLT: SE ≈ sigma/sqrt(1000)).
% ASSUMPTION: Grid spans [5th, 95th] percentile of each feature (100 points).
%   Rationale: avoids extrapolation artifacts at the tails.
% ASSUMPTION: PD values are centered at 0 (subtract mean) — matches textbook convention.

fprintf('[10] Computing 1D partial dependence (4 features, N_ss=1000)...\n');
tPD1 = tic;

pdFeatIdx = [1, 3, 8, 6];
pdFeatLbl = {'MedInc', 'AveOccup', 'HouseAge', 'AveRooms'};
nPDGrid   = 100;
nSubsamp  = min(1000, N_tr);

rng(42, 'twister');
ssIdx = randperm(N_tr, nSubsamp);
X_ss  = X_train(ssIdx, :);

% === PASS 1: Compute all PD curves before plotting ===
% This lets us compute a SHARED y-axis scale across all 4 panels.
pdGrids  = cell(4, 1);   % x-axis values per feature
pdVals   = cell(4, 1);   % y-axis values per feature

for fi = 1:4
    featJ  = pdFeatIdx(fi);
    % Use wider range (2nd-98th pct) so edge behavior is visible
    xmin   = prctile(X_train(:, featJ),  2);
    xmax   = prctile(X_train(:, featJ), 98);
    grid_j = linspace(xmin, xmax, nPDGrid)';

    % Vectorized batch predict
    X_batch                = repmat(X_ss, nPDGrid, 1);
    X_batch(:, featJ)      = repelem(grid_j, nSubsamp, 1);
    Yhat_batch             = predict(mdl, X_batch);
    pv                     = mean(reshape(Yhat_batch, nSubsamp, nPDGrid), 1)';
    pv                     = pv - mean(pv);   % centre at zero

    pdGrids{fi} = grid_j;
    pdVals{fi}  = pv;
end

% Shared Y-axis: span the global [min, max] of all 4 curves, with 10% padding
allPD   = cell2mat(pdVals);
yGlobal = [min(allPD) - 0.1*range(allPD),  max(allPD) + 0.1*range(allPD)];

% === PASS 2: Plot with consistent y-axis ===
viz10  = OceanViz.Core('Theme', 'Dark');
C10    = viz10.Colors;
fig10  = viz10.createFigure('Partial Dependence — Nonlocation Variables (ESL Fig 10.15)');
fig10.Position = [80 80 860 560];
figure(fig10);

for fi = 1:4
    featJ   = pdFeatIdx(fi);
    grid_j  = pdGrids{fi};
    pv      = pdVals{fi};

    ax = subplot(2, 2, fi);
    viz10.styleAxes(ax);
    hold(ax, 'on');

    plot(ax, grid_j, pv, 'Color', C10.Primary, 'LineWidth', 1.8);

    % Apply the SHARED y-axis
    ylim(ax, yGlobal);
    yline(ax, 0, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');   % zero reference

    xlabel(ax, pdFeatLbl{fi}, 'FontSize', 9);
    ylabel(ax, 'Partial Dependence', 'FontSize', 9);
    title(ax, pdFeatLbl{fi}, 'FontSize', 10, 'FontWeight', 'bold');

    % Red decile tick marks (from TRAINING marginal distribution)
    deciles = prctile(X_train(:, featJ), 10:10:90);
    yLims   = ylim(ax);
    tickH   = 0.04 * (yLims(2) - yLims(1));
    for dk = 1:numel(deciles)
        line(ax, [deciles(dk) deciles(dk)], [yLims(1), yLims(1)+tickH], ...
            'Color', [0.9 0.2 0.2], 'LineWidth', 1.5);
    end
    hold(ax, 'off');
end
fprintf('[10] 1D partial dependence plotted in %.1f sec.\n', toc(tPD1));

%% 2D Marginalization
fprintf('[11] Computing 2D partial dependence: HouseAge × AveOccup (%dx%d grid)...\n', 20, 20);
tPD2 = tic;

feat1 = 8;   % HouseAge
feat2 = 3;   % AveOccup
nG    = 20;

g1 = linspace(prctile(X_train(:, feat1),  5), prctile(X_train(:, feat1), 95), nG)';
g2 = linspace(prctile(X_train(:, feat2),  5), prctile(X_train(:, feat2), 95), nG)';

% --- VECTORIZED 2D MARGINALIZATION ---
% 1. Create all pairs of grid points (nG^2 = 400 pairs)
[G1_grid, G2_grid] = meshgrid(g1, g2); % 20x20
G1_flat = G1_grid(:); %400x1
G2_flat = G2_grid(:);

%X_ss; 1000x8
% 2. Replicate subsample for all pairs: [N_ss * 400] rows
X_batch = repmat(X_ss, nG^2, 1); % (1000*400x8) = > 400.000 x 8

% 3. Broadcast both features
X_batch(:, feat1) = repelem(G1_flat, nSubsamp, 1); %Repeat g1_flat nSubsamp times, 400x1 repeats 1000 times -> 400.000 x 8 
X_batch(:, feat2) = repelem(G2_flat, nSubsamp, 1);

% 4. Predict and average
Yhat_batch = predict(mdl, X_batch);
pd2_flat   = mean(reshape(Yhat_batch, nSubsamp, nG^2), 1)'; %400.000x1 -> 1000 x 400 -> mean(1000x400) -> 1 x 400 -> tranpose -> 400x1
pd2        = reshape(pd2_flat, nG, nG);
pd2        = pd2 - mean(pd2(:));
% -------------------------------------

fprintf('[11] 2D partial dependence computed in %.1f sec.\n', toc(tPD2));

viz11  = OceanViz.Core('Theme', 'Dark');
C11    = viz11.Colors;
fig11  = viz11.createFigure('Two-Variable Partial Dependence: HouseAge x AveOccup');
fig11.Position = [80 80 700 520];
ax11   = axes('Parent', fig11);

[G2, G1] = meshgrid(g2, g1);   % G2 = AveOccup (x-axis), G1 = HouseAge (y-axis)
surf(ax11, G2, G1, pd2, 'EdgeAlpha', 0.25, 'FaceAlpha', 0.95);
colormap(ax11, 'hot');
shading(ax11, 'interp');
view(ax11, [-50, 28]);

% Colorbar with label
cb = colorbar(ax11);
cb.Color            = C11.Text;
cb.Label.String     = 'Partial Dependence  (\times$100k USD)';
cb.Label.Color      = C11.Text;
cb.Label.FontSize   = 9;
cb.Label.Interpreter = 'latex';


% Style axes to match OceanViz Dark theme
ax11.Color           = C11.Background;
ax11.GridColor       = C11.Grid;
ax11.XColor          = C11.Text;
ax11.YColor          = C11.Text;
ax11.ZColor          = C11.Text;
ax11.LineWidth       = 0.8;
grid(ax11, 'on');

xlabel(ax11, 'AveOccup',           'Color', C11.Text, 'FontSize', 10);
ylabel(ax11, 'HouseAge',           'Color', C11.Text, 'FontSize', 10);
zlabel(ax11, 'Partial Dependence', 'Color', C11.Text, 'FontSize', 9);
title(ax11, 'Joint Partial Dependence: HouseAge \times AveOccup (ESL Fig 10.16)', ...
    'Color', C11.Text, 'FontSize', 11);

fprintf('[11] 2D surface plotted. Done.\n');
viz.styleAxes(ax11);

