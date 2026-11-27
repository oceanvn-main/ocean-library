% EXP_MNIST_TRAINING_PLAYBACK  Live Animation of MLP Training on MNIST
%
%   Trains a 100-16-10 MLP on downsampled MNIST digits instantly
%   in the background, captures the weight snapshots per epoch, and
%   then runs a fluid high-contrast animation of the
%   learning process, scaling up edge weights and organizing node activations.

clc; clear; close all;
rng(42, 'twister');

fprintf('=== LIVE TRAINING VISUALIZATION (MNIST) ===\n');

% =========================================================================
%% CONFIG — change these to adjust the experiment
% =========================================================================

% Input image resolution for the network: IMG_RES x IMG_RES pixels.
% Lower = fewer input nodes = cleaner visualization.
%   7  →  49 inputs  (very clean graph, less detail)
%   10 → 100 inputs  (recommended: good balance)
%   14 → 196 inputs  (denser graph)
IMG_RES   = 10;      
INPUT_DIM = IMG_RES * IMG_RES;  % network input layer size
HID_DIM   = 16;     % hidden layer nodes (increase for better accuracy)
OUT_DIM   = 10;     % 10 digit classes

%% 1. Load data  — use FULL dataset for best training
fprintf('Loading and downsampling MNIST digits...\n');
try
    [XTrain4D, YTrain] = digitTrain4DArrayData();
catch
    error('Deep Learning Toolbox required for digitTrain4DArrayData.');
end

nSamples = size(XTrain4D, 4);  % all available samples (usually 5000)
fprintf('  Total training samples: %d\n', nSamples);

X = zeros(INPUT_DIM, nSamples);
for i = 1:nSamples
    img = XTrain4D(:,:,1,i);
    img = imresize(double(img) / 255.0, [IMG_RES IMG_RES], 'bicubic');
    X(:,i) = reshape(img, INPUT_DIM, 1) * 2.0 - 1.0;  % scale to [-1, 1]
end
Y = dummyvar(grp2idx(YTrain(1:nSamples)))';  % OUT_DIM x nSamples one-hot

fprintf('  Data ready: %d x %d input matrix.\n\n', INPUT_DIM, nSamples);

%% 2. Custom Matrix SGD Pre-Training
% Instead of fighting MATLAB's native train() state resets on 1-epoch loops,
% we use our own explicit vectorized gradient descent. This runs instantly 
% and guarantees the exact parameter format needed for replay.
hiddenSize = HID_DIM;
numEpochs  = 500;  % More epochs for full-dataset convergence

W1 = randn(hiddenSize, INPUT_DIM) * sqrt(2 / INPUT_DIM);
b1 = zeros(hiddenSize, 1);
W2 = randn(OUT_DIM, hiddenSize) * sqrt(2 / (hiddenSize + OUT_DIM));
b2 = zeros(OUT_DIM, 1);

fprintf('Pre-training via custom Matrix-SGD for %d epochs...\n', numEpochs);
weightHistory = cell(numEpochs, 1);
costHistory = zeros(numEpochs, 1);
accHistory = zeros(numEpochs, 1);

% Adam Optimizer State
m_W1 = 0; v_W1 = 0; m_b1 = 0; v_b1 = 0;
m_W2 = 0; v_W2 = 0; m_b2 = 0; v_b2 = 0;
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
lr = 0.005;  % Tuned for 5000-sample batch: smaller LR for stable convergence

for ep = 1:numEpochs
    % --- Forward Pass ---
    Z1 = W1 * X + b1;
    A1 = tanh(Z1);
    
    Z2 = W2 * A1 + b2;
    % Softmax
    maxZ2 = max(Z2, [], 1);
    expZ2 = exp(Z2 - maxZ2);
    A2 = expZ2 ./ sum(expZ2, 1);
    
    % --- Metrics ---
    costHistory(ep) = -sum(sum(Y .* log(A2 + 1e-8))) / nSamples;
    
    [~, pIdx] = max(A2);
    [~, tIdx] = max(Y);
    accHistory(ep) = mean(pIdx == tIdx);
    
    % --- Backward Pass ---
    dZ2 = A2 - Y;
    dW2 = (dZ2 * A1') / nSamples;
    db2 = sum(dZ2, 2) / nSamples;
    
    dA1 = W2' * dZ2;
    dZ1 = dA1 .* (1 - A1.^2);
    dW1 = (dZ1 * X') / nSamples;
    db1 = sum(dZ1, 2) / nSamples;
    
    % --- Adam Update ---
    % W1
    m_W1 = beta1 * m_W1 + (1 - beta1) * dW1;
    v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1.^2);
    m_hat_W1 = m_W1 / (1 - beta1^ep);
    v_hat_W1 = v_W1 / (1 - beta2^ep);
    W1 = W1 - lr * m_hat_W1 ./ (sqrt(v_hat_W1) + epsilon);
    
    % b1
    m_b1 = beta1 * m_b1 + (1 - beta1) * db1;
    v_b1 = beta2 * v_b1 + (1 - beta2) * (db1.^2);
    m_hat_b1 = m_b1 / (1 - beta1^ep);
    v_hat_b1 = v_b1 / (1 - beta2^ep);
    b1 = b1 - lr * m_hat_b1 ./ (sqrt(v_hat_b1) + epsilon);
    
    % W2
    m_W2 = beta1 * m_W2 + (1 - beta1) * dW2;
    v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2.^2);
    m_hat_W2 = m_W2 / (1 - beta1^ep);
    v_hat_W2 = v_W2 / (1 - beta2^ep);
    W2 = W2 - lr * m_hat_W2 ./ (sqrt(v_hat_W2) + epsilon);
    
    % b2
    m_b2 = beta1 * m_b2 + (1 - beta1) * db2;
    v_b2 = beta2 * v_b2 + (1 - beta2) * (db2.^2);
    m_hat_b2 = m_b2 / (1 - beta1^ep);
    v_hat_b2 = v_b2 / (1 - beta2^ep);
    b2 = b2 - lr * m_hat_b2 ./ (sqrt(v_hat_b2) + epsilon);
    
    % --- Capture exact format for NNGraph ParameterStore ---
    % W1 is 16x100 (dOut x dIn), W2 is 10x16. flatten to column-major.
    weightHistory{ep} = [b1(:); W1(:); b2(:); W2(:)];
end
fprintf('Pre-training complete! Final Accuracy: %.1f%%\n\n', accHistory(end)*100);

%% 3. Set up OceanViz Custom Dashboard
themeCore = OceanViz.Core('Theme', 'Dark');
colors = themeCore.Colors;

fig = figure('Name', 'Live Training Visualizer', 'NumberTitle', 'off', ...
             'Color', colors.Background, 'Position', [100, 100, 1300, 750], ...
             'MenuBar', 'none', 'ToolBar', 'none');

% Left panels: 3 subplots stacked
axCost = subplot(3, 3, 1);
axAcc  = subplot(3, 3, 4);
axImg  = subplot(3, 3, 7);

% Right panel: Large network area
axNet  = subplot(3, 3, [2 3 5 6 8 9]);
axis(axNet, 'off');

set(axCost, 'Color', colors.Background, 'XColor', colors.Grid, 'YColor', colors.Grid);
title(axCost, sprintf('Training Cost (%d samples)', nSamples), 'Color', colors.Text);
xlim(axCost, [1 numEpochs]);
ylim(axCost, [0 max(costHistory)*1.1]);
lineCost = animatedline(axCost, 'Color', colors.Secondary, 'LineWidth', 2);
grid(axCost, 'on');

set(axAcc, 'Color', colors.Background, 'XColor', colors.Grid, 'YColor', colors.Grid);
title(axAcc, 'Training Accuracy', 'Color', colors.Text);
xlim(axAcc, [1 numEpochs]);
ylim(axAcc, [0 1.05]);
lineAcc = animatedline(axAcc, 'Color', colors.Primary, 'LineWidth', 2);
grid(axAcc, 'on');

set(axImg, 'Color', colors.Background, 'XColor', colors.Background, 'YColor', colors.Background);
colormap(axImg, 'gray');
title(axImg, sprintf('Input (%dx%d)', IMG_RES, IMG_RES), 'Color', colors.Text);

%% 4. Set up OceanStats Graph for Live Injection
spec = OceanDL.Graph.NNTopologySpec.make( ...
    [INPUT_DIM, hiddenSize, OUT_DIM], {'linear', 'tanh', 'softmax'});
graph = OceanDL.Graph.NNGraph.fromSpec(spec);
paramStore = OceanDL.Graph.ParameterStore();

layout = OceanDL.Viz.NNLayeredLayout();
strategy = OceanViz.Strategies.NetworkGraphStrategy();

%% 5. Start Replay Animation
fprintf('Starting Replay Animation...\n');
btn = uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
    'Position', [20 20 60 30], 'Callback', 'setappdata(gcf, ''stopAnim'', true)');

setappdata(gcf, 'stopAnim', false);

for ep = 1:numEpochs
    stopRequested = false;
    try
        stopRequested = getappdata(fig, 'stopAnim');
    catch
    end
    if stopRequested, break; end
    
    % 1. Inject Captured Weights
    paramStore.setFromFlat(graph, weightHistory{ep});
    
    % 2. Pick a random sample from the FULL training set
    sampIdx = randi(nSamples);
    x_samp = X(:, sampIdx);
    img_samp = reshape(x_samp, [IMG_RES, IMG_RES])';
    y_true = find(Y(:, sampIdx) == 1) - 1;  % digit class 0-9
    
    % 3. Forward Pass so nodes light up
    graph.forwardPass(x_samp');
    
    % 4. Update the small plots
    addpoints(lineCost, ep, costHistory(ep));
    addpoints(lineAcc,  ep, accHistory(ep));
    
    % Convert pixel values back from [-1,1] to [0,1] for proper gray display
    img_display = (img_samp + 1.0) / 2.0;
    imagesc(axImg, img_display);
    colormap(axImg, 'gray');
    clim(axImg, [0 1]);
    set(axImg, 'XTick', [], 'YTick', []);
    title(axImg, sprintf('Input Digit: %d  (epoch %d)', y_true, ep), 'Color', colors.Text);
    
    % 5. Redraw the network graph
    data = layout.toData(graph, struct('Colors', colors));
    cla(axNet);
    strategy.plot(axNet, data, struct('Colors', colors, 'NodeDisplay', 'Both'));
    title(axNet, sprintf('Epoch %d / %d  |  Accuracy: %.1f%%', ep, numEpochs, accHistory(ep)*100), ...
        'Color', colors.Text);
    
    drawnow limitrate;
    pause(0.12);
end

fprintf('Animation Complete.\n');
