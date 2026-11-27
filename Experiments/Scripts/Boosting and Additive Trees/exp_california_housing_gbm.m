% EXP_CALIFORNIA_HOUSING_GBM  MART on California Housing (ESL §10.14.1).
%
%   DATASET:        California Housing Prices (Pace & Barry, 1997)
%                   Source: Data/California Housing Prices/housing.csv
%                   N ~ 20,433 (after NaN removal), p = 8
%
%   MODEL:          GradientBoostingMachine, Algorithm 10.3 (Friedman 2001)
%
%   HYPERPARAMETERS:
%     NumTrees     = 800      (M — boosting rounds)
%     LearningRate = 0.10     (nu — shrinkage)
%     MaxDepth     = 3        (J ~ 6-8 terminal nodes; see ASSUMPTION A3)
%     MinLeafSize  = 5        (following library default)
%     ClassicHuberLoss delta  (see ASSUMPTION A4)
%
%   ASSUMPTIONS:
%     A1. Predictors: 8 numeric features derived from raw columns (see §2).
%         Rationale: textbook uses per-household ratios, not raw totals.
%     A2. Response: Y = median_house_value / 100,000 (units of $100k).
%         Rationale: textbook explicitly states this unit.
%     A3. MaxDepth=3 approximates J=6 terminal nodes.
%         Rationale: DecisionTree uses depth, not leaf count.
%         Depth-3 allows up to 2^3 = 8 leaves — closest to J=6.
%         This is an approximation; exact J=6 cannot be enforced directly.
%     A4. ClassicHuberLoss delta = 0.5 * IQR(Y_train - median(Y_train)).
%         Loss type: piecewise Huber (Friedman 2001, ESL §10.14) — matches textbook.
%         Delta computed from training data only (not copied from textbook).
%     A5. 80/20 random train/test split, seed=42 for reproducibility.
%     A6. Rows with NaN in any feature are dropped entirely (simple listwise
%         deletion). An assertion guards that N_clean > 20,000.
%
%   REPRODUCIBILITY: rng(42, 'twister') at top of script.
%
%   EXPERIMENT RULES (experiment.md):
%     - All metrics computed from Y_true and Y_pred (no hard-coded values).
%     - Textbook benchmarks printed in summary for contextual comparison only.
%     - No special-casing; no future/test data used during training.

clc; clear; close all;
rng(42, 'twister');

%% === 1. Data Loading ===

% Script lives at: <root>/Experiments/Scripts/Boosting and Additive Trees/
% 4 fileparts calls to reach project root.
scriptDir = fileparts(mfilename('fullpath'));        % .../Boosting and Additive Trees
lvl1      = fileparts(scriptDir);                   % .../Scripts
lvl2      = fileparts(lvl1);                        % .../Experiments
rootDir   = fileparts(lvl2);                        % project root
dataPath  = fullfile(rootDir, 'Data', 'California Housing Prices', 'housing.csv');

assert(isfile(dataPath), ...
    'EXP:DataNotFound', ...
    'housing.csv not found at expected path:\n  %s', dataPath);

T = readtable(dataPath);
fprintf('[1] Loaded %d rows, %d columns from housing.csv\n', height(T), width(T));

%% === 2. Feature Engineering ===
% Derive 8 numeric predictors per textbook description (A1).
% All derivations are explicit arithmetic on raw columns.

% Guard: ensure no division-by-zero from zero households
assert(all(T.households > 0), 'EXP:ZeroHouseholds', ...
    'Unexpected rows with households = 0.');

% Per-household ratios (derived, not raw)
AveRooms  = T.total_rooms    ./ T.households;      % avg rooms per household
AveBedrms = T.total_bedrooms ./ T.households;      % avg bedrooms per household
AveOccup  = T.population     ./ T.households;      % avg occupants per household

% Raw numeric predictors
MedInc  = T.median_income;
House   = T.households;
Lat     = T.latitude;
Lon     = T.longitude;
MedAge  = T.housing_median_age;

% Assemble feature matrix (N x 8), column order documented
%   Col 1: MedInc   Col 2: House    Col 3: AveOccup  Col 4: Lat
%   Col 5: Lon      Col 6: AveRooms Col 7: AveBedrms Col 8: MedAge
featureNames = {'MedInc','House','AveOccup','Lat','Lon', ...
                'AveRooms','AveBedrms','MedAge'};
X_raw = [MedInc, House, AveOccup, Lat, Lon, AveRooms, AveBedrms, MedAge];

% Response: units of $100k (A2)
Y_raw = T.median_house_value ./ 1e5;

% Listwise deletion for NaN — sourced from total_bedrooms missing values (A6)
validMask = all(~isnan(X_raw), 2) & ~isnan(Y_raw);
X = X_raw(validMask, :);
Y = Y_raw(validMask);
N = size(X, 1);
p = size(X, 2);

assert(N > 20000, 'EXP:TooManyDropped', ...
    'After NaN removal only %d rows remain (expected > 20,000). Check data.', N);

fprintf('[2] Feature engineering complete. N=%d, p=%d (dropped %d NaN rows)\n', ...
    N, p, sum(~validMask));
fprintf('    Features: %s\n', strjoin(featureNames, ', '));

%% === 3. Train / Test Split (80/20, A5) ===

idx      = randperm(N);
N_tr     = round(0.80 * N);
N_te     = N - N_tr;
trIdx    = idx(1:N_tr);
teIdx    = idx(N_tr+1:end);

X_train  = X(trIdx, :);   Y_train = Y(trIdx);
X_test   = X(teIdx,  :);  Y_test  = Y(teIdx);

fprintf('[3] Split: N_train=%d (%.0f%%), N_test=%d (%.0f%%)\n', ...
    N_tr, 100*N_tr/N, N_te, 100*N_te/N);

%% === 4. Model Configuration ===

% A4: Compute Huber delta from training residual spread (training data only)
F0_init = median(Y_train);
r0      = Y_train - F0_init;
delta   = 0.5 * iqr(r0);

fprintf('[4] Huber delta = %.4f (computed as 0.5 * IQR of initial training residuals)\n', delta);

loss = OceanStats.Loss.Regression.ClassicHuberLoss('Delta', delta);

M   = 800;     % boosting rounds
nu  = 0.10;    % learning rate (shrinkage)
dep = 3;       % tree depth -> up to 8 leaves ~ J=6 (A3)
mls = 5;       % min leaf size

% ASSUMPTION: QuantileCandidateGenerator(64) used instead of ExhaustiveCandidateGenerator.
% Rationale: At N=16,346, exhaustive search generates ~16k candidates per feature
% per node, making each tree O(N^2*p). Quantile with 64 bins reduces this to
% O(64*N*p) — ~250x faster with negligible loss in split quality at this N.
% This is dataset-scale engineering, not a statistical shortcut.
candGen = OceanStats.Supervised.Tree.QuantileCandidateGenerator(64);

gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine( ...
    'Loss',              loss,    ...
    'NumTrees',          M,       ...
    'LearningRate',      nu,      ...
    'MaxDepth',          dep,     ...
    'MinLeafSize',       mls,     ...
    'CandidateGenerator', candGen);


fprintf('[4] GBM configured: M=%d, nu=%.2f, MaxDepth=%d, MinLeafSize=%d\n', ...
    M, nu, dep, mls);

%% === 5. Training ===

fprintf('\n[5] Training GBM on %d samples...\n', N_tr);
gbm.fit(X_train, Y_train);

%% === 6. Evaluation — Per-Iteration AAE on Train and Test ===
% Reconstruct cumulative predictions by replaying Trees_ sequentially.
% No test information was used during training (experiment.md rule enforced
% by computing test predictions only AFTER fit() returns).

%Custom loss function
fprintf('Computing Cumulative AAE curves...\n');
maeLoss = @(Y, YHat, W) mean(abs(Y - Yhat));

aae_train = zeros(M, 1);
aae_test  = zeros(M, 1);

fprintf('[6] Computing per-iteration AAE (vectorized cumulative)...\n');
timeStats = tic;

% ---- Step 1: Score all M trees on train and test in one forward pass ----
% This mirrors fitrensemble loss(..., 'Mode', 'cumulative') which also
% scores each tree once and accumulates. No sequential dependency here:
% all predict() calls are independent and can be computed up front.
%
% Memory: N_tr×M doubles = ~%d MB
%         N_te×M doubles = ~%d MB
nu = gbm.LearningRate;
timeStats = tic;

all_tr = zeros(N_tr, M);   % N_tr × M — each column = one tree's contribution
all_te = zeros(N_te, M);   % N_te × M

for m = 1:M
    all_tr(:, m) = nu * gbm.Trees_{m}.predict(X_train);
    all_te(:, m) = nu * gbm.Trees_{m}.predict(X_test);
end
elapsed = toc(timeStats);
fprintf('[6] Scoring all trees complete in %.1f seconds (%.1f min)\n', elapsed, elapsed/60);


% ---- Step 2: Cumulative ensemble predictions (vectorized, no loop) ----
% Equivalent to MATLAB's 'cumulative' mode: cumsum along tree axis (dim 2).
% F_cumul_tr(n, m) = F0 + sum_{k=1..m} nu * tree_k.predict(x_n)
F_cumul_tr = gbm.F0_ + cumsum(all_tr, 2);   % N_tr × M
F_cumul_te = gbm.F0_ + cumsum(all_te, 2);   % N_te × M

% ---- Step 3: Vectorized AAE over all M iterations at once ----
% mean(|Y - F_cumul|, 1) computes column-wise mean → M×1 result.
aae_train = mean(abs(Y_train - F_cumul_tr), 1)';   % M×1
aae_test  = mean(abs(Y_test  - F_cumul_te), 1)';   % M×1

% Final scalars — all computed from data, not hardcoded
aae_train_final = aae_train(end);
aae_test_final  = aae_test(end);

F_te = F_cumul_te(:, end);   % final cumulative prediction — used for R2 below

% R² on test set — computed from scratch
ss_res  = sum((Y_test - F_te).^2);
ss_tot  = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - ss_res / ss_tot;

% Constant predictor baseline: predict median(Y_train) for all test points
baseline_aae = mean(abs(Y_test - median(Y_train)));

fprintf('[6] Done. Train AAE=%.4f | Test AAE=%.4f | R2=%.4f\n', ...
    aae_train_final, aae_test_final, r2_test);

%% === 7. Visualization (OceanViz.Core — LearningCurve strategy) ===

viz = OceanViz.Core('Theme', 'Dark');

fig = viz.createFigure('California Housing — GBM Convergence (MART)');
fig.Position = [60 80 860 500];

ax = axes('Parent', fig);
viz.styleAxes(ax);

lcData.X = (1:M)';
lcData.Lines = { ...
    struct('Y', aae_train, 'Name', 'Train Error'), ...
    struct('Y', aae_test,  'Name', 'Test Error')  ...
};
lcData.RefLines = { ...
    struct('Y', baseline_aae, 'Name', sprintf('Baseline = %.3f', baseline_aae)) ...
};

viz.plot(ax, 'LearningCurve', lcData, ...
    'Title',      'Training and Test Absolute Error', ...
    'XLabel',     'Iterations M', ...
    'YLabel',     'Absolute Error', ...
    'ShowLegend', true);

% Y-axis from 0 to match textbook range
ylim(ax, [0, max([aae_train(1), aae_test(1)]) * 1.05]);

%% === 8. Summary Printout ===

fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset        : California Housing (N=%d, p=%d)\n', N, p);
fprintf('  Model          : GBM, Loss=%s\n', loss.name());
fprintf('  Boosting Rounds: %d  |  LearningRate=%.2f  |  MaxDepth=%d\n', M, nu, dep);
fprintf('  Train AAE      : %.4f\n', aae_train_final);
fprintf('  Test  AAE      : %.4f\n', aae_test_final);
fprintf('  Test  R2       : %.4f\n', r2_test);
fprintf('  Baseline AAE   : %.4f  (constant = median(Y_train))\n', baseline_aae);
fprintf('  ---\n');
fprintf('  [Context] Textbook (Friedman 2001 / ESL §10.14.1):\n');
fprintf('            AAE ~ 0.31,  R2 ~ 0.84,  Baseline ~ 0.89\n');
fprintf('==========================\n');
