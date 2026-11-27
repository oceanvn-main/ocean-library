% EXP_MANDELBROT_TRAIN  Approximate the Mandelbrot set using Fourier Feature MLPs.
%
%   Uses Fourier Feature Mapping (Tancik et al. 2020) to overcome spectral bias.
%   Instead of raw (a,b) coordinates, the NN receives a 128-dim Fourier projection.
%
%   Fourier mapping:
%     gamma(X) = [cos(2*pi*B*X); sin(2*pi*B*X)]  where B ~ N(0, sigma^2)
%
%   Network: 128 -> 256 -> 256 -> 1  (sigmoid output, MSE loss)
%
%   Run this first, then run exp_mandelbrot_replay.

clc; clear; close all;
rng(42);

% --- Ensure project root is on MATLAB path ---
scriptDir = fileparts(mfilename('fullpath'));
rootDir   = fileparts(fileparts(fileparts(scriptDir)));
addpath(rootDir);

fprintf('=== Mandelbrot — Fourier Feature MLP ===\n\n');

% =========================================================================
%% 1. Fourier Feature Config (Data-Driven Sigma Selection)
%
%   Instead of hardcoding sigma, we derive it from the grid itself.
%   Two methods available — choose one:
%
%   METHOD A (fast, ~1 second):
%     Analytical Nyquist bound.
%     sigma = f_N / 3 = 1 / (6 * delta_x)
%     Guarantees 99.7% of random frequencies stay below the aliasing limit.
%
%   METHOD B (slower, ~minutes):
%     Empirical U-Curve sweep. Trains probe networks at log-spaced sigma
%     values, picks the bottom of the validation MSE curve.
%     Set  FOURIER_SWEEP = true  to activate.
%
% =========================================================================
FOURIER_M     = 64;    % Number of frequency components -> 2M = 128-dim input
FOURIER_SWEEP = false; % true = run U-curve sweep (slower but more precise)

% =========================================================================
%% 2. Generate the TRUE Mandelbrot Set (Dense Grid)
% =========================================================================
GRID_RES = 150;
MAX_ITER = 100;
RE_RANGE = [-2.5, 1.0];
IM_RANGE = [-1.2, 1.2];

fprintf('[1] Computing true Mandelbrot set on %dx%d grid...\n', GRID_RES, GRID_RES);

a_vec = linspace(RE_RANGE(1), RE_RANGE(2), GRID_RES);
b_vec = linspace(IM_RANGE(1), IM_RANGE(2), GRID_RES);
[AA, BB] = meshgrid(a_vec, b_vec);

C = complex(AA, BB);
Z = zeros(size(C));
escapeIter = MAX_ITER * ones(size(C));
for iter = 1:MAX_ITER
    mask = abs(Z) <= 2;
    Z(mask) = Z(mask).^2 + C(mask);
    escaped = (abs(Z) > 2) & (escapeIter == MAX_ITER);
    escapeIter(escaped) = iter;
end
groundTruth = escapeIter / MAX_ITER;   % [GRID_RES x GRID_RES]

fprintf('   Inside set: %d pts (%.1f%%)\n\n', ...
    sum(groundTruth(:) == 1), 100*mean(groundTruth(:) == 1));

% =========================================================================
%% 3. Build Dataset with Fourier Feature Mapping
% =========================================================================
a_norm = (AA(:) - RE_RANGE(1)) / diff(RE_RANGE) * 2 - 1;
b_norm = (BB(:) - IM_RANGE(1)) / diff(IM_RANGE) * 2 - 1;
N      = GRID_RES^2;
X_raw  = [a_norm'; b_norm'];      % [2 x N]  raw coords in [-1,1]
Y      = groundTruth(:)';         % [1 x N]

fprintf('[2] Applying Fourier Feature Mapping  2 -> %d dims  (M=%d)...\n', 2*FOURIER_M, FOURIER_M);

% --- Select sigma (data-driven) ----------------------------------------
if FOURIER_SWEEP
    % METHOD B: Empirical U-Curve (slower, most principled)
    fprintf('  Running U-Curve sigma sweep...\n');
    [FOURIER_SIGMA, ~] = OceanStats.Preprocessing.FourierFeatureMap.sweepSigma(X_raw, Y, ...
        'SigmaMin', 0.5, 'SigmaMax', 40, 'NumSteps', 8, ...
        'M', 32, 'HiddenDim', 64, 'Epochs', 150, 'PlotResult', true);
else
    % METHOD A: Analytical Nyquist bound (fast, safe)
    FOURIER_SIGMA = OceanStats.Preprocessing.FourierFeatureMap.recommendSigma(X_raw);
end

ffm = OceanStats.Preprocessing.FourierFeatureMap();
X   = ffm.encode(X_raw, 'M', FOURIER_M, 'Sigma', FOURIER_SIGMA);  % [128 x N]

fprintf('   %s\n\n', ffm.describe());

% =========================================================================
%% 4. Network Architecture: 2M -> 256 -> 256 -> 1
% =========================================================================
INPUT_DIM = ffm.outputDim();   % 128
L1        = 256;
L2        = 256;
OUT_DIM   = 1;
numEpochs = 2000;
lr        = 0.001;
nSamples  = N;

% He init
W1 = randn(L1, INPUT_DIM) * sqrt(2/INPUT_DIM);  b1 = zeros(L1, 1);
W2 = randn(L2, L1)        * sqrt(2/L1);          b2 = zeros(L2, 1);
W3 = randn(OUT_DIM, L2)   * sqrt(2/L2);          b3 = zeros(OUT_DIM, 1);

layerSizes    = [INPUT_DIM, L1, L2, OUT_DIM];
activationFns = {'linear', 'tanh', 'tanh', 'sigmoid'};

HID_DIM = L1;  % For frame struct compatibility

fprintf('[3] Training %d epochs on %d->256->256->1 network...\n', numEpochs, INPUT_DIM);
fprintf('    Please wait.\n\n');

% =========================================================================
%% 5. Pre-select Demo Samples for Replay
% =========================================================================
rng(7);
demoSampIdx = randi(N, numEpochs, 1);
demoA1 = zeros(L1, numEpochs);
demoA2 = zeros(OUT_DIM, numEpochs);
demoX  = zeros(2, numEpochs);     % Store raw (a,b) for ComplexPointRenderer
demoY  = zeros(1, numEpochs);

% =========================================================================
%% 6. Training Loop (Mini-batch Adam)
% =========================================================================
weightHistory = cell(numEpochs, 1);
costHistory   = zeros(1, numEpochs);
accHistory    = zeros(1, numEpochs);

% One AdamOptimizer per parameter (6 params: W1,b1,W2,b2,W3,b3)
adamOpts = OceanStats.Optim.AdamOptimizer.forParams(6, 'LR', lr);
[oW1, ob1, oW2, ob2, oW3, ob3] = deal( ...
    adamOpts{1}, adamOpts{2}, adamOpts{3}, adamOpts{4}, adamOpts{5}, adamOpts{6});

fprintf('  Optimizer: %s\n\n', oW1.describe());
BATCH = 1024;

for ep = 1:numEpochs
    idx = randperm(N, BATCH);
    Xb  = X(:, idx);
    Yb  = Y(idx);

    % Forward
    A1 = tanh(W1 * Xb + b1);
    A2 = tanh(W2 * A1 + b2);
    Z3 = W3 * A2 + b3;
    A3 = 1 ./ (1 + exp(-Z3));

    err = A3 - Yb;
    costHistory(ep) = mean(err.^2);

    % R2 every 25 epochs
    if mod(ep, 25) == 0 || ep == 1
        A1f = tanh(W1*X + b1);
        A2f = tanh(W2*A1f + b2);
        A3f = 1./(1+exp(-(W3*A2f + b3)));
        ss_res = sum((Y - A3f).^2);
        ss_tot = sum((Y - mean(Y)).^2);
        accHistory(ep) = max(0, 1 - ss_res/ss_tot);
    else
        accHistory(ep) = accHistory(max(ep-1,1));
    end

    % Backprop
    dZ3 = err / BATCH;
    dW3 = dZ3 * A2';   db3 = sum(dZ3, 2);
    dA2 = W3' * dZ3;
    dZ2 = dA2 .* (1 - A2.^2);
    dW2 = dZ2 * A1';   db2_g = sum(dZ2, 2);
    dA1 = W2' * dZ2;
    dZ1 = dA1 .* (1 - A1.^2);
    dW1 = dZ1 * Xb';   db1_g = sum(dZ1, 2);

    % Adam updates — clean, no manual beta/m/v management
    W1 = oW1.step(W1, dW1);
    b1 = ob1.step(b1, db1_g);
    W2 = oW2.step(W2, dW2);
    b2 = ob2.step(b2, db2_g);
    W3 = oW3.step(W3, dW3);
    b3 = ob3.step(b3, db3);



    % Checkpoint
    weightHistory{ep} = [b1(:); W1(:); b2(:); W2(:); b3(:); W3(:)];

    % Demo frame — store raw coords for renderer
    xi_raw = X_raw(:, demoSampIdx(ep));
    xi_f   = ffm.encodeNew(xi_raw);
    a1i    = tanh(W1 * xi_f + b1);
    a2i    = tanh(W2 * a1i + b2);
    a3i    = 1 ./ (1 + exp(-(W3 * a2i + b3)));
    demoA1(:, ep) = a1i;
    demoA2(:, ep) = a3i;
    demoX(:, ep)  = xi_raw;         % raw (a_norm, b_norm) for ComplexPointRenderer
    demoY(ep)     = Y(demoSampIdx(ep));

    if mod(ep, 100) == 0 || ep == 1
        fprintf('  [ep %4d/%d]  MSE: %.5f  R2: %.4f\n', ...
            ep, numEpochs, costHistory(ep), accHistory(ep));
    end
end

fprintf('\nTraining complete. Final R2: %.4f\n\n', accHistory(end));

% =========================================================================
%% 7. Save Checkpoint  (split into small metadata + separate weights file)
%    Metadata file loads instantly (~MB).
%    Weights file is large but only read lazily per-epoch during replay.
% =========================================================================
outputDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'Outputs', 'Mandelbrot');
if ~isfolder(outputDir), mkdir(outputDir); end

% Pre-bake display grid in normalized coords
DISPLAY_RES = 150;
a_d = linspace(RE_RANGE(1), RE_RANGE(2), DISPLAY_RES);
b_d = linspace(IM_RANGE(1), IM_RANGE(2), DISPLAY_RES);
adn = linspace(-1, 1, DISPLAY_RES);
bdn = linspace(-1, 1, DISPLAY_RES);
[ADN, BDN] = meshgrid(adn, bdn);
gridNorm_raw = [ADN(:)'; BDN(:)'];
gridNorm     = ffm.encodeNew(gridNorm_raw);   % [128 x DISPLAY_RES^2]
FourierB     = ffm.B;

% --- Small metadata file (loads in < 1 second) ---
metaFile = fullfile(outputDir, 'mandelbrot_fourier_checkpoint.mat');
save(metaFile, ...
    'costHistory', 'accHistory', ...
    'INPUT_DIM', 'HID_DIM', 'OUT_DIM', 'numEpochs', 'nSamples', ...
    'demoA1', 'demoA2', 'demoX', 'demoY', ...
    'layerSizes', 'activationFns', ...
    'groundTruth', 'gridNorm', 'a_d', 'b_d', ...
    'RE_RANGE', 'IM_RANGE', 'MAX_ITER', 'GRID_RES', 'DISPLAY_RES', ...
    'FOURIER_M', 'FOURIER_SIGMA', 'FourierB');

% --- Separate weights file (large, read lazily by engine) ---
weightsFile = fullfile(outputDir, 'mandelbrot_fourier_checkpoint_weights.mat');
save(weightsFile, 'weightHistory', '-v7.3');

fprintf('\n[4] Checkpoint saved:\n    Meta:    %s\n    Weights: %s\n\n', ...
    metaFile, weightsFile);
fprintf('Run  exp_mandelbrot_replay  to animate.\n');

