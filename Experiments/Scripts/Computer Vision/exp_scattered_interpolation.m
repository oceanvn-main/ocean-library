%% exp_scattered_interpolation.m
%  Exercise 4.1 — Scattered Data Interpolation (Szeliski §4.1)
%
%  Workflow:
%   Step 1 : Generate a smoothly-varying function (sinusoids + Gaussians)
%   Step 2 : Sample at random scatter locations (training set)
%   Step 3 : Fit scattered data interpolation techniques (RBF, KNN, Natural)
%   Step 4 : Measure fitting error on a regular evaluation grid
%   Step 5 : Tune hyper-parameters via K-fold cross-validation
%   Step 6 : Repeat experiment with new scatter samples (parameter stability)
%   Step 7 : (Optional) Piecewise-smooth test function — harder problem
%
%  Key classes used:
%   OceanData.Generators.SmoothFunctionGenerator
%   OceanMath.Interpolation.{IScatteredInterpolant, ScatteredRBF,
%       ScatteredNatural, ScatteredKNN, InterpolationFactory}
%   OceanStats.Interpolation.{ScatterFitter, InterpolationError, InterpolationCV}
%   OceanViz.Core → ScatterInterpStrategy
%
%  Author: Antigravity Code Assistant
%  Date:   2026-08-18
%  Ref:    Szeliski, Computer Vision: Algorithms and Applications (2e), Ex 4.1

clear; clc; close all;

%% ════════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%% ════════════════════════════════════════════════════════════════════════
DIM       = 2;          % 1 or 2 — input dimensionality
N_TRAIN   = 50;         % scatter training samples
N_TERMS   = 4;          % sinusoid/Gaussian components in ground truth
MODE      = 'mixed';    % 'sinusoid' | 'gaussian' | 'mixed'
NOISE_STD = 0.05;       % observation noise
SEED      = 42;         % reproducibility
DOMAIN    = [0, 1];     % [low, high] per dimension
N_GRID    = 80;         % evaluation grid density per dimension
K_FOLDS   = 5;          % CV folds for parameter tuning

fprintf('Ex 4.1 — Scattered Data Interpolation | Dim=%d | N=%d | Mode=%s\n', ...
    DIM, N_TRAIN, MODE);

%% ════════════════════════════════════════════════════════════════════════
%  STEP 1 — Generate ground-truth function + training samples
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Step 1] Generating smooth function (%s, D=%d) ...\n', MODE, DIM);

gen = OceanData.Generators.SmoothFunctionGenerator( ...
    'N',          N_TRAIN, ...
    'Dim',        DIM, ...
    'Mode',       MODE, ...
    'NumTerms',   N_TERMS, ...
    'NoiseStd',   NOISE_STD, ...
    'DomainLow',  DOMAIN(1), ...
    'DomainHigh', DOMAIN(2), ...
    'Seed',       SEED);

[X_train, Y_train] = gen.generate();
f_true  = gen.MetaData.f_true;

fprintf('  Training samples: %d x %d  |  noise std = %.3f\n', ...
    size(X_train,1), size(X_train,2), NOISE_STD);

%% ════════════════════════════════════════════════════════════════════════
%  STEP 2 — Sample sanity: compute noise-free ground truth at train sites
%  (Not used for fitting — just for validation reporting)
%% ════════════════════════════════════════════════════════════════════════
% NOTE: This step is for diagnostics only. We do not use f_true during
%       fitting. Training uses only the noisy observations Y_train.
Y_noisefree = f_true(X_train);
train_noise_rmse = sqrt(mean((Y_train - Y_noisefree).^2));
fprintf('[Step 2] Empirical noise RMSE at train sites: %.5f (expected ~%.3f)\n', ...
    train_noise_rmse, NOISE_STD);

%% ════════════════════════════════════════════════════════════════════════
%  STEP 3 — Fit interpolation techniques
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Step 3] Fitting interpolants ...\n');

import OceanMath.Interpolation.ScatteredInterpolant

fitter = OceanStats.Interpolation.ScatterFitter( ...
    'DomainLow',  DOMAIN(1), ...
    'DomainHigh', DOMAIN(2), ...
    'Dim',        DIM, ...
    'NGrid',      N_GRID);

fitter.addModel('RBF-ThinPlate',        ScatteredInterpolant('Method', 'rbf', 'Kernel', 'thinplate'));
fitter.addModel('RBF-Gaussian',         ScatteredInterpolant('Method', 'rbf', 'Kernel', 'gaussian', 'Epsilon', 0.2));
fitter.addModel('RBF-Multiquadric',     ScatteredInterpolant('Method', 'rbf', 'Kernel', 'multiquadric', 'Epsilon', 0.5));
fitter.addModel('RBF-InvMultiquadric',  ScatteredInterpolant('Method', 'rbf', 'Kernel', 'inversemultiquadric', 'Epsilon', 0.5));
fitter.addModel('KernelRegression',     ScatteredInterpolant('Method', 'kernel-regression', 'Kernel', 'gaussian', 'Epsilon', 0.2));
fitter.addModel('KNN-5-distance',       ScatteredInterpolant('Method', 'knn', 'K', 5, 'Weights', 'distance'));
fitter.addModel('Natural',              ScatteredInterpolant('Method', 'natural', 'SubMethod', 'natural'));

if DIM == 1
    fitter.addModel('SmoothSpline', ScatteredInterpolant('Method', 'spline', 'Lambda', 1e-4));
end

fitter.fitAll(X_train, Y_train);
fprintf('  Fitted %d interpolants.\n', numel(fitter.getNames()));

%% ════════════════════════════════════════════════════════════════════════
%  STEP 4 — Measure error on regular evaluation grid
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Step 4] Evaluating on %d-pt grid ...\n', N_GRID^DIM);

fitter.evaluateAll(f_true, 'Grid', true);
fitter.printSummary();

%% ════════════════════════════════════════════════════════════════════════
%  STEP 5 — Cross-validation: tune RBF Gaussian epsilon
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Step 5] Cross-validation: tuning RBF-Gaussian epsilon ...\n');

cv = OceanStats.Interpolation.InterpolationCV( ...
    'ModelType',  'rbf', ...
    'TuneParam',  'rbf-epsilon', ...
    'ParamGrid',  logspace(-2, 1, 20), ...
    'KFolds',     K_FOLDS, ...
    'KernelName', 'gaussian', ...
    'Verbose',    true);
cv.fit(X_train, Y_train);

fprintf('[Step 5] Best epsilon = %.4f\n', cv.BestParam);

% Evaluate the CV-tuned model
import OceanStats.Interpolation.InterpolationError
err_cv = InterpolationError.fromFunc(cv.BestModel, f_true, ...
    'Grid', true, 'NGrid', N_GRID, 'Dim', DIM, ...
    'DomainLow', DOMAIN(1), 'DomainHigh', DOMAIN(2));
err_cv.printSummary(sprintf('CV-RBF-Gaussian (eps=%.4f)', cv.BestParam));

%% ════════════════════════════════════════════════════════════════════════
%  STEP 6 — Repeat with NEW random scatter samples
%           Check parameter stability (does best epsilon change much?)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Step 6] Repeating with new scatter samples (Seed=%d→%d) ...\n', SEED, SEED+1);

gen2 = OceanData.Generators.SmoothFunctionGenerator( ...
    'N',          N_TRAIN, ...
    'Dim',        DIM, ...
    'Mode',       MODE, ...
    'NumTerms',   N_TERMS, ...
    'NoiseStd',   NOISE_STD, ...
    'DomainLow',  DOMAIN(1), ...
    'DomainHigh', DOMAIN(2), ...
    'Seed',       SEED + 1);

[X_train2, Y_train2] = gen2.generate();
% NOTE: Same f_true (same function, different scatter sites)

cv2 = OceanStats.Interpolation.InterpolationCV( ...
    'ModelType',  'rbf', ...
    'TuneParam',  'rbf-epsilon', ...
    'ParamGrid',  logspace(-2, 1, 20), ...
    'KFolds',     K_FOLDS, ...
    'KernelName', 'gaussian', ...
    'Verbose',    false);
cv2.fit(X_train2, Y_train2);

fprintf('[Step 6] Best epsilon (new samples) = %.4f\n', cv2.BestParam);
fprintf('[Step 6] Epsilon change: %.2f%%\n', ...
    100 * abs(cv2.BestParam - cv.BestParam) / cv.BestParam);

%% ════════════════════════════════════════════════════════════════════════
%  STEP 7 (Optional) — Piecewise-smooth test function
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Step 7] Piecewise-smooth function (harder interpolation) ...\n');

gen_pw = OceanData.Generators.SmoothFunctionGenerator( ...
    'N',          N_TRAIN, ...
    'Dim',        DIM, ...
    'Mode',       'piecewise', ...
    'NumTerms',   N_TERMS, ...
    'NumPieces',  3, ...
    'NoiseStd',   NOISE_STD, ...
    'DomainLow',  DOMAIN(1), ...
    'DomainHigh', DOMAIN(2), ...
    'Seed',       SEED);

[X_pw, Y_pw]    = gen_pw.generate();
f_pw            = gen_pw.MetaData.f_true;

mdl_pw = ScatteredInterpolant('Method', 'rbf', 'Kernel', 'thinplate');
mdl_pw.fit(X_pw, Y_pw);

err_pw = InterpolationError.fromFunc(mdl_pw, f_pw, ...
    'Grid', true, 'NGrid', N_GRID, 'Dim', DIM, ...
    'DomainLow', DOMAIN(1), 'DomainHigh', DOMAIN(2));
err_pw.printSummary('RBF-ThinPlate (piecewise-smooth)');

% Compare vs smooth case
err_smooth = fitter.getError('RBF-ThinPlate');
fprintf('[Step 7] RMSE ratio piecewise/smooth = %.2fx\n', ...
    err_pw.RMSE / err_smooth.RMSE);

%% ════════════════════════════════════════════════════════════════════════
%  VISUALIZATION — Figure 1: Smooth function (best model = CV-tuned Gaussian)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[Viz] Rendering Figure 1 (smooth function, CV-tuned model) ...\n');

fig1 = figure('Color', [0.09 0.09 0.12], 'Position', [50, 480, 1400, 450], ...
    'Name', 'Ex 4.1 — Smooth Function Interpolation');

data1 = struct();
data1.X_train  = X_train;
data1.Y_train  = Y_train;
data1.model    = cv.BestModel;
data1.f_true   = f_true;
data1.dim      = DIM;
data1.domain   = DOMAIN;
data1.ngrid    = N_GRID;
data1.CVResult = cv;

ax_ph = axes(fig1);    % placeholder — strategy will replace it
viz = OceanViz.Core('Theme', 'Dark');
viz.plot(ax_ph, 'ScatterInterp', data1, ...
    'Title', sprintf('Ex 4.1 (%s, D=%d) — CV-Tuned RBF-Gaussian', MODE, DIM));

%% ════════════════════════════════════════════════════════════════════════
%  VISUALIZATION — Figure 2: Piecewise-smooth (optional step 7)
%% ════════════════════════════════════════════════════════════════════════
fprintf('[Viz] Rendering Figure 2 (piecewise-smooth function) ...\n');

fig2 = figure('Color', [0.09 0.09 0.12], 'Position', [50, 40, 1050, 420], ...
    'Name', 'Ex 4.1 — Piecewise-Smooth Function Interpolation');

data2 = struct();
data2.X_train  = X_pw;
data2.Y_train  = Y_pw;
data2.model    = mdl_pw;
data2.f_true   = f_pw;
data2.dim      = DIM;
data2.domain   = DOMAIN;
data2.ngrid    = N_GRID;

ax_ph2 = axes(fig2);
viz.plot(ax_ph2, 'ScatterInterp', data2, ...
    'Title', 'Ex 4.1 (Piecewise-smooth) — RBF-ThinPlate');

fprintf('\n[Done] All steps complete.\n');
