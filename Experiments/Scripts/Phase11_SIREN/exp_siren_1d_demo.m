%% exp_siren_1d_demo.m — NN-5: SIREN vs Fourier+ReLU vs Plain MLP on 1-D signal
%
%  Illustrates the spectral bias problem and the two solutions:
%    1. Plain MLP (ReLU)         — learns low-frequencies only (blurry fit)
%    2. Fourier + ReLU           — overcomes spectral bias with random features
%    3. SIREN (Sine activation)  — overcomes spectral bias via network architecture
%
%  Signal:  f(x) = sin(5*pi*x) + 0.3*sin(30*pi*x)   (multi-scale)
%           The 30*pi component is inivisble to a plain MLP in < 5k epochs.
%
%  Author: Ocean Library — NN-5 Phase Experiment
%  Date:   2026-03-12

addpath('c:\Ocean Library\Matlab code practice');
import OceanDL.Graph.*
import OceanDL.Simulation.*
import OceanStats.Optim.AdamOptimizer
import OceanStats.Loss.Regression.MSELoss
import OceanStats.Preprocessing.FourierFeatureMap

rng(42);

% =========================================================================
%% DATASET — multi-scale 1-D signal
% =========================================================================
N_train  = 200;
N_dense  = 1000;   % dense grid for evaluation + plotting

x_train  = linspace(-1, 1, N_train)';     % (N x 1)
x_dense  = linspace(-1, 1, N_dense)';

target_fn = @(x) sin(5*pi*x) + 0.3*sin(30*pi*x);   % multi-scale ground truth

y_train = target_fn(x_train);
y_dense = target_fn(x_dense);

EPOCHS      = 1000;
BATCH       = 32;
LR          = 0.005;
HIDDEN      = 64;
N_HIDDEN    = 3;     % number of hidden layers

% =========================================================================
%% 1. PLAIN MLP (ReLU) — no special input encoding
% =========================================================================
fprintf('Training Plain MLP (ReLU)...\n');

sizes_plain = [1, repmat(HIDDEN, 1, N_HIDDEN), 1];
acts_plain  = [{'linear'}, repmat({'relu'}, 1, N_HIDDEN), {'linear'}];

specP  = NNTopologySpec.make(sizes_plain, acts_plain);
graphP = NNGraph.fromSpec(specP);
fwdEng = VectorizedEngine();

trainerP = Trainer(graphP, MSELoss(), @() AdamOptimizer('LR', LR));
histP    = trainerP.fit(x_train, y_train, 'Epochs', EPOCHS, 'BatchSize', BATCH);
[yP_dense, ~] = fwdEng.forward(graphP, x_dense, 'infer');

fprintf('  Final train loss: %.5f\n', histP.Loss(end));

% =========================================================================
%% 2. FOURIER + ReLU — Fourier features then standard MLP
% =========================================================================
fprintf('Training Fourier + ReLU MLP...\n');

M_fourier = 64;
sigma     = 5;       % reasonable for signal in [-1,1] x 30*pi frequencies

fourierEnc = FourierFeatureMap('M', M_fourier, 'Sigma', sigma);

% Encode training data (column-major: d_in x N)
X_train_col = x_train';        % (1 x N_train)
X_enc_train = fourierEnc.encode(X_train_col);   % (2*M x N_train)

X_dense_col = x_dense';
X_enc_dense = fourierEnc.encodeNew(X_dense_col);

% (2*M x N) → (N x 2*M) for Trainer (row-major)
X_enc_tr_row = X_enc_train';    % (N_train x 2*M)
X_enc_de_row = X_enc_dense';    % (N_dense x 2*M)

d_fourier    = fourierEnc.outputDim();   % 2*M
sizes_four   = [d_fourier, repmat(HIDDEN, 1, N_HIDDEN), 1];
acts_four    = [{'linear'}, repmat({'relu'}, 1, N_HIDDEN), {'linear'}];

specF  = NNTopologySpec.make(sizes_four, acts_four);
graphF = NNGraph.fromSpec(specF);

trainerF = Trainer(graphF, MSELoss(), @() AdamOptimizer('LR', LR));
histF    = trainerF.fit(X_enc_tr_row, y_train, 'Epochs', EPOCHS, 'BatchSize', BATCH);
[yF_dense, ~] = fwdEng.forward(graphF, X_enc_de_row, 'infer');

fprintf('  Final train loss: %.5f\n', histF.Loss(end));
fprintf('  FourierFeatureMap.outputDim() = %d  (should be %d)\n', d_fourier, 2*M_fourier);

% =========================================================================
%% 3. SIREN — sinusoidal hidden layers, direct coordinate input
% =========================================================================
fprintf('Training SIREN (Sine activation)...\n');

sizes_siren = [1, repmat(HIDDEN, 1, N_HIDDEN), 1];
acts_siren  = [{'linear'}, repmat({'sine'}, 1, N_HIDDEN), {'linear'}];

specS  = NNTopologySpec.make(sizes_siren, acts_siren);
graphS = NNGraph.fromSpec(specS);

% SIREN weight init: first hidden layer uniform(-1/d_in, 1/d_in)
% subsequent hidden uniform(-sqrt(6/d_in)/omega_0, sqrt(6/d_in)/omega_0)
omega0 = 30;
nL_s   = graphS.numLayers();
for L = 2:nL_s-1
    d_in_L = specS.LayerSizes(L-1);
    if L == 2
        bound = 1 / d_in_L;
    else
        bound = sqrt(6 / d_in_L) / omega0;
    end
    graphS.setWeightMatrix(L-1, (rand(specS.LayerSizes(L), d_in_L)*2 - 1) * bound);
end

trainerS = Trainer(graphS, MSELoss(), @() AdamOptimizer('LR', LR * 0.1));
histS    = trainerS.fit(x_train, y_train, 'Epochs', EPOCHS, 'BatchSize', BATCH);
[yS_dense, ~] = fwdEng.forward(graphS, x_dense, 'infer');

fprintf('  Final train loss: %.5f\n', histS.Loss(end));

% =========================================================================
%% VISUALISATION
% =========================================================================
fprintf('\nRendering comparison figure...\n');

fig = figure('Name', 'NN-5: SIREN vs Fourier+ReLU vs Plain MLP', ...
    'Color', [0.08 0.08 0.12], 'Position', [100 80 1280 760]);

% ---- Top row: predictions -------------------------------------------
ax1 = subplot(2, 3, [1 2 3]);
set(ax1, 'Color', [0.05 0.05 0.09], 'XColor', [0.6 0.6 0.65], ...
    'YColor', [0.6 0.6 0.65], 'GridColor', [0.25 0.25 0.3], ...
    'GridAlpha', 0.5); hold(ax1, 'on'); grid(ax1, 'on');

plot(ax1, x_train, y_train, 'o', 'Color', [0.9 0.9 0.9], ...
    'MarkerSize', 2, 'DisplayName', 'Training data');
plot(ax1, x_dense, y_dense, '-', 'Color', [0.5 0.5 0.55], ...
    'LineWidth', 1, 'DisplayName', 'Ground truth');
plot(ax1, x_dense, yP_dense, '--', 'Color', [1.0 0.45 0.3], ...
    'LineWidth', 2, 'DisplayName', sprintf('Plain ReLU  (loss=%.4f)', histP.Loss(end)));
plot(ax1, x_dense, yF_dense, '-', 'Color', [0.3 0.75 1.0], ...
    'LineWidth', 2, 'DisplayName', sprintf('Fourier+ReLU  (loss=%.4f)', histF.Loss(end)));
plot(ax1, x_dense, yS_dense, '-', 'Color', [0.4 1.0 0.5], ...
    'LineWidth', 2.5, 'DisplayName', sprintf('SIREN  (loss=%.4f)', histS.Loss(end)));

legend(ax1, 'Location', 'northeast', 'TextColor', [0.85 0.85 0.9], ...
    'Color', [0.08 0.08 0.13], 'FontSize', 9);
title(ax1, 'f(x) = sin(5\pi x) + 0.3\cdotsin(30\pi x)  —  1-D Spectral Bias Demo', ...
    'Color', [1 1 1], 'FontSize', 13, 'FontWeight', 'bold');
xlabel(ax1, 'x', 'Color', [0.85 0.85 0.9]);
ylabel(ax1, 'f(x)', 'Color', [0.85 0.85 0.9]);
xlim(ax1, [-1 1]);

% ---- Bottom row: loss curves ----------------------------------------
plotLoss_ = @(ax, hist, col, label) plot(ax, hist.Loss, '-', ...
    'Color', col, 'LineWidth', 1.8, 'DisplayName', label);

titles = {'Plain ReLU MLP', 'Fourier + ReLU', 'SIREN'};
hists  = {histP, histF, histS};
cols   = {[1.0 0.45 0.3], [0.3 0.75 1.0], [0.4 1.0 0.5]};

for k = 1:3
    ax = subplot(2, 3, 3+k);
    set(ax, 'Color', [0.05 0.05 0.09], 'XColor', [0.6 0.6 0.65], ...
        'YColor', [0.6 0.6 0.65], 'YScale', 'log', 'GridColor', [0.25 0.25 0.3], ...
        'GridAlpha', 0.5); hold(ax,'on'); grid(ax,'on');
    plotLoss_(ax, hists{k}, cols{k}, titles{k});
    title(ax, titles{k}, 'Color', [1 1 1], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax, 'Epoch', 'Color', [0.8 0.8 0.85]);
    ylabel(ax, 'Loss (log)', 'Color', [0.8 0.8 0.85]);
end

sgtitle(fig, 'NN-5: Neural Implicit Representations — Spectral Bias Demo', ...
    'Color', [1 1 1], 'FontSize', 14, 'FontWeight', 'bold');

fprintf('[Done] Compare the three panels for visual quality gap.\n');
fprintf('       Plain MLP misses the high-frequency 30pi component.\n');
fprintf('       Fourier+ReLU and SIREN both recover it.\n\n');

% =========================================================================
%% SUMMARY TABLE
% =========================================================================
fprintf('%-18s  %-10s\n', 'Model', 'Final Loss');
fprintf('%-18s  %-10s\n', repmat('-',1,18), repmat('-',1,10));
fprintf('%-18s  %.6f\n', 'Plain ReLU MLP',  histP.Loss(end));
fprintf('%-18s  %.6f\n', 'Fourier + ReLU', histF.Loss(end));
fprintf('%-18s  %.6f\n', 'SIREN',          histS.Loss(end));
fprintf('\nFourierFeatureMap.outputDim() = %d  (expected %d)  — PASS: %d\n', ...
    d_fourier, 2*M_fourier, d_fourier == 2*M_fourier);
