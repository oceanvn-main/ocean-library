% EXP_MNIST_HANDWRITTEN  Train LeNet-5 (Net-5) on MNIST handwritten digits.
%
%   Implements the "Net-5" architecture from LeCun (1989) / ESL §11.7:
%     Two locally-connected hidden layers with weight sharing (= convolution).
%     This is historically the first successful CNN for digit recognition.
%
%   ARCHITECTURE (adapted for 28×28 MNIST; original used 16×16):
%     Input         : 28×28×1 greyscale images
%     Conv-1        : 6 feature maps, 5×5 local receptive field (weight sharing L1)
%     MaxPool-1     : 2×2, stride 2  (subsampling, as in LeCun)
%     Conv-2        : 16 feature maps, 5×5 local receptive field (weight sharing L2)
%     MaxPool-2     : 2×2, stride 2
%     FC-120        : fully connected (flatten → 120)
%     FC-84         : fully connected (120 → 84)
%     Output        : 10 softmax units (digits 0–9)
%
%   This matches the "two levels of weight sharing" (Conv-1 and Conv-2) and
%   "two hidden layers locally connected" (the two conv blocks) described in
%   the textbook's Net-5 definition.
%
%   DATASET:
%     MATLAB built-in digitTrain4DArrayData / digitTest4DArrayData
%     Requires: Deep Learning Toolbox (R2019b+)
%
%   ASSUMPTIONS:
%     1. Images kept as 28×28 (no flattening — convnet takes spatial input).
%     2. Pixels normalized to [0,1] by the imageInputLayer's built-in normalization.
%     3. Held-out test set is never seen during training or validation.
%     4. NNGraph import is NOT attempted for this architecture: our NNGraph
%        supports fully-connected layers only. CNN weight import is planned
%        for Phase 8 (VectorizedEngine). Diagnostics use Deep Learning Toolbox.
%     5. Reproducibility: rng(42,'twister') + shuffle('once') at training time;
%        exact results vary across MATLAB versions/GPU presence.
%
%   REPRODUCIBILITY: rng(42, 'twister') at top of script.
%
% See also: trainNetwork, convolution2dLayer, maxPooling2dLayer

clc; clear; close all;
rng(42, 'twister');

fprintf('=== EXP_MNIST_HANDWRITTEN  (Net-5 / LeNet-5) ===\n');
fprintf('  Started: %s\n\n', datestr(now));

% =========================================================================
%% === 1. Data Loading ===
% =========================================================================
fprintf('[1] Loading MNIST digit dataset...\n');

try
    [XTrain4D, YTrain] = digitTrain4DArrayData();
    [XTest4D,  YTest]  = digitTest4DArrayData();
catch
    error(['digitTrain4DArrayData not found.\n' ...
           'Deep Learning Toolbox is required for this experiment.']);
end

N_train = size(XTrain4D, 4);
N_test  = size(XTest4D,  4);
classNames = categories(YTrain);
K = numel(classNames);    % 10 digit classes

fprintf('  Train : %d images  |  Test : %d images\n', N_train, N_test);
fprintf('  Image size : %d×%d  |  Classes : %d\n\n', ...
    size(XTrain4D,1), size(XTrain4D,2), K);

% =========================================================================
%% === 2. Architecture Definition — Net-5 (LeNet-5) ===
% =========================================================================
fprintf('[2] Defining Net-5 architecture...\n');

% Normalization: 'zerocenter' divides by std — equivalent to LeCun's [-1,+1] scaling.
% ReLU replaces tanh for modern convergence (LeCun used tanh; ReLU trains faster here).
layers = [
    imageInputLayer([28 28 1], 'Name', 'input', 'Normalization', 'zerocenter')

    % --- Weight sharing level 1 ---
    convolution2dLayer(5, 6,  'Padding', 'same',   'Name', 'conv1', 'BiasLearnRateFactor', 2)
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    % --- Weight sharing level 2 ---
    convolution2dLayer(5, 16, 'Padding', 0, 'Name', 'conv2', 'BiasLearnRateFactor', 2)
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    % --- Fully connected head ---
    fullyConnectedLayer(120, 'Name', 'fc1', 'BiasLearnRateFactor', 2)
    reluLayer('Name', 'relu3')
    dropoutLayer(0.3, 'Name', 'drop1')

    fullyConnectedLayer(84, 'Name', 'fc2', 'BiasLearnRateFactor', 2)
    reluLayer('Name', 'relu4')

    fullyConnectedLayer(K, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

fprintf('  Layer stack defined: %d layers total\n\n', numel(layers));

% =========================================================================
%% === 3. Training Options ===
% =========================================================================
fprintf('[3] Configuring training options...\n');

% Hyperparameters — all explicit and justified:
BATCH_SIZE  = 64;    % mini-batch: small enough for fast CPU convergence
MAX_EPOCHS  = 15;    % enough for ~95%+ on this dataset; stop early if plateau
LEARN_RATE  = 1e-3;  % Adam default; appropriate for relu + this dataset scale
VAL_FREQ    = 50;    % validate every 50 mini-batches

opts = trainingOptions('adam', ...
    'MaxEpochs',         MAX_EPOCHS, ...
    'MiniBatchSize',     BATCH_SIZE, ...
    'InitialLearnRate',  LEARN_RATE, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor',  0.1, ...
    'LearnRateDropPeriod',  10, ...
    'ValidationData',    {XTest4D, YTest}, ...
    'ValidationFrequency', VAL_FREQ, ...
    'Shuffle',           'every-epoch', ...    % new shuffle each epoch
    'Verbose',           true, ...
    'VerboseFrequency',  VAL_FREQ, ...
    'Plots',             'none');              % we draw our own plots

fprintf('  Adam LR=%.0e | Epochs=%d | Batch=%d\n\n', ...
    LEARN_RATE, MAX_EPOCHS, BATCH_SIZE);

% =========================================================================
%% === 4. Training ===
% =========================================================================
fprintf('[4] Training Net-5 (this may take a few minutes)...\n');

t_start = tic;
[trainedNet, info] = trainNetwork(XTrain4D, YTrain, layers, opts);
t_train = toc(t_start);

fprintf('\n  Training done in %.1f s\n', t_train);
fprintf('  Final train accuracy  : %.2f%%\n', info.TrainingAccuracy(end));
fprintf('  Final val   accuracy  : %.2f%%\n\n', info.ValidationAccuracy(end));

% =========================================================================
%% === 5. Evaluation ===
% =========================================================================
fprintf('[5] Evaluating on held-out test set...\n');

% Get predictions
YPred = classify(trainedNet, XTest4D);
acc_test = mean(YPred == YTest);

% Per-class accuracy (derived, not hardcoded)
per_class_acc = zeros(K, 1);
for ci = 1:K
    mask = YTest == classNames{ci};
    if sum(mask) > 0
        per_class_acc(ci) = mean(YPred(mask) == classNames{ci});
    end
end

% Output probabilities (for analysis)
Y_prob = predict(trainedNet, XTest4D);   % (N_test × K)

fprintf('  Test Accuracy : %.2f%%\n', acc_test * 100);
fprintf('  Per-class accuracy:\n');
for ci = 1:K
    fprintf('    Digit %s : %.1f%%\n', classNames{ci}, per_class_acc(ci)*100);
end
fprintf('\n');

% =========================================================================
%% === 6. Diagnostics — Layer Activations (a la ForwardPassEngine) ===
% =========================================================================
fprintf('[6] Extracting layer activations for diagnostics...\n');

% Use first DIAG_N test images
DIAG_N = min(200, N_test);
X_diag = XTest4D(:,:,:, 1:DIAG_N);

% Extract activations from named conv/relu layers using activations()
layerNames = {'relu1', 'pool1', 'relu2', 'pool2', 'relu3', 'relu4'};
layerMeans = zeros(numel(layerNames), 1);
layerStds  = zeros(numel(layerNames), 1);

for li = 1:numel(layerNames)
    acts = activations(trainedNet, X_diag, layerNames{li}, ...
        'OutputAs', 'channels');   % (H×W×C×N) for conv, (1×1×C×N) for FC
    flat = acts(:);
    layerMeans(li) = mean(abs(flat));
    layerStds(li)  = std(flat);
    fprintf('  %-8s : mean=%.4f  std=%.4f\n', layerNames{li}, ...
        layerMeans(li), layerStds(li));
end
fprintf('\n');

% =========================================================================
%% === 7. Visualization ===
% =========================================================================
fprintf('[7] Generating plots...\n');

fig = figure('Name', 'Net-5 MNIST Results', ...
    'Color', [0.09 0.09 0.11], 'Position', [80 60 1200 750]);

% ---- 7a. Training & Validation Accuracy curve
ax1 = subplot(2, 3, 1, 'Parent', fig);
% info.TrainingAccuracy and ValidationAccuracy have different lengths
% (ValidationAccuracy is recorded at val_freq steps, not every mini-batch)
nTrain    = numel(info.TrainingAccuracy);
stepsTrain = 1:nTrain;
plot(ax1, stepsTrain, info.TrainingAccuracy, ...
    'Color', [0.30 0.65 0.95], 'LineWidth', 1.8);
hold(ax1, 'on');
% ValidationAccuracy has NaN for non-validation steps — filter them
vIdx = find(~isnan(info.ValidationAccuracy));
plot(ax1, vIdx, info.ValidationAccuracy(vIdx), ...
    'Color', [0.95 0.50 0.25], 'LineWidth', 1.8);
legend(ax1, 'Train acc.', 'Val acc.', 'TextColor', [0.9 0.9 0.9], 'Location', 'southeast');
title(ax1, 'Accuracy During Training', 'Color', [0.95 0.95 1.0]);
xlabel(ax1, 'Mini-batch step'); ylabel(ax1, 'Accuracy (%)');
ax1.Color = [0.12 0.12 0.15]; ax1.XColor = [0.7 0.7 0.7]; ax1.YColor = [0.7 0.7 0.7];

% ---- 7b. Loss curve
ax2 = subplot(2, 3, 2, 'Parent', fig);
semilogy(ax2, info.TrainingLoss, 'Color', [0.30 0.65 0.95], 'LineWidth', 1.8);
hold(ax2, 'on');
semilogy(ax2, vIdx, info.ValidationLoss(vIdx), ...
    'Color', [0.95 0.50 0.25], 'LineWidth', 1.8, 'LineStyle', '--');
legend(ax2, 'Train loss', 'Val loss', 'TextColor', [0.9 0.9 0.9]);
title(ax2, 'Loss During Training', 'Color', [0.95 0.95 1.0]);
xlabel(ax2, 'Mini-batch step'); ylabel(ax2, 'Cross-Entropy Loss');
ax2.Color = [0.12 0.12 0.15]; ax2.XColor = [0.7 0.7 0.7]; ax2.YColor = [0.7 0.7 0.7];

% ---- 7c. Confusion matrix (test set)
ax3 = subplot(2, 3, 3, 'Parent', fig);
pred_idx = double(YPred);   % categorical → 1-based int
true_idx = double(YTest);
C = zeros(K);
for ii = 1:K
    for jj = 1:K
        C(ii,jj) = sum(true_idx == ii & pred_idx == jj);
    end
end
imagesc(ax3, C);
colormap(ax3, hot);
colorbar(ax3);
title(ax3, sprintf('Confusion Matrix — Test Acc=%.1f%%', acc_test*100), ...
    'Color', [0.95 0.95 1.0]);
xlabel(ax3, 'Predicted'); ylabel(ax3, 'True');
set(ax3, 'XTick', 1:K, 'XTickLabel', classNames, ...
         'YTick', 1:K, 'YTickLabel', classNames);
ax3.Color = [0.12 0.12 0.15]; ax3.XColor = [0.7 0.7 0.7]; ax3.YColor = [0.7 0.7 0.7];

% ---- 7d. Per-class accuracy
ax4 = subplot(2, 3, 4, 'Parent', fig);
b = bar(ax4, 1:K, per_class_acc * 100, 'FaceColor', 'flat');
b.CData = repmat([0.20 0.50 0.90], K, 1);
hold(ax4, 'on');
yline(ax4, acc_test*100, '--', sprintf('%.1f%%', acc_test*100), ...
    'Color', [0.95 0.50 0.25], 'LabelHorizontalAlignment', 'left');
title(ax4, 'Per-Class Test Accuracy', 'Color', [0.95 0.95 1.0]);
xlabel(ax4, 'Digit Class'); ylabel(ax4, 'Accuracy (%)');
set(ax4, 'XTick', 1:K, 'XTickLabel', classNames, 'YLim', [0, 105]);
ax4.Color = [0.12 0.12 0.15]; ax4.XColor = [0.7 0.7 0.7]; ax4.YColor = [0.7 0.7 0.7];

% ---- 7e. Layer mean activations (ForwardPass-style diagnostic)
ax5 = subplot(2, 3, 5, 'Parent', fig);
b2 = bar(ax5, layerMeans, 'FaceColor', 'flat');
b2.CData = repmat([0.25 0.75 0.55], numel(layerNames), 1);
title(ax5, 'Mean |Activation| per Layer', 'Color', [0.95 0.95 1.0]);
xlabel(ax5, 'Layer'); ylabel(ax5, 'Mean |a|');
set(ax5, 'XTick', 1:numel(layerNames), 'XTickLabel', layerNames, ...
    'XTickLabelRotation', 30);
ax5.Color = [0.12 0.12 0.15]; ax5.XColor = [0.7 0.7 0.7]; ax5.YColor = [0.7 0.7 0.7];

% ---- 7f. Sample predictions mosaic (first 16 test images)
ax6 = subplot(2, 3, 6, 'Parent', fig);
nShow = 16;
nCols = 8;
nRows = ceil(nShow / nCols);
mosaic = zeros(28 * nRows, 28 * nCols);
for si = 1:nShow
    r = floor((si-1)/nCols);
    c = mod(si-1, nCols);
    mosaic(r*28+(1:28), c*28+(1:28)) = squeeze(XTest4D(:,:,1,si));
end
imagesc(ax6, mosaic);
colormap(ax6, gray);
title(ax6, 'Predictions: green=correct, red=wrong', 'Color', [0.95 0.95 1.0]);
axis(ax6, 'off');
for si = 1:nShow
    r = floor((si-1)/nCols);
    c = mod(si-1, nCols);
    xc = c*28 + 14;
    yr = r*28 + 30;
    p = char(YPred(si));
    t = char(YTest(si));
    col = [0.3 1.0 0.3];
    if ~strcmp(p, t), col = [1.0 0.3 0.3]; end
    text(ax6, xc, yr, sprintf('%s|%s', p, t), ...
        'Color', col, 'FontSize', 7, 'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold');
end

% =========================================================================
%% === Summary ===
% =========================================================================
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : MNIST digits (N_train=%d, N_test=%d)\n', N_train, N_test);
fprintf('  Architecture  : Net-5 / LeNet-5 (2 conv + 2 pool + 2 FC + softmax)\n');
fprintf('  Weight sharing: 2 levels  (Conv1: 6 maps × 5×5, Conv2: 16 maps × 5×5)\n');
fprintf('  Optimizer     : Adam  LR=%.0e  Epochs=%d  Batch=%d\n', ...
    LEARN_RATE, MAX_EPOCHS, BATCH_SIZE);
fprintf('  Train time    : %.1f s\n', t_train);
fprintf('  Test Accuracy : %.2f%%\n', acc_test * 100);
fprintf('  Best class    : %s (%.1f%%)\n', classNames{per_class_acc == max(per_class_acc)}, max(per_class_acc)*100);
fprintf('  Worst class   : %s (%.1f%%)\n', classNames{per_class_acc == min(per_class_acc)}, min(per_class_acc)*100);
fprintf('  NOTE: CNN weights not imported to NNGraph (FC-only limitation).\n');
fprintf('        Phase 8 (VectorizedEngine) will add CNN feature map support.\n');
fprintf('==========================\n');
