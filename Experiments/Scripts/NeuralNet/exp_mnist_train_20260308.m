clc; clear;
rng(42, 'twister');
fprintf('=== EXP_MNIST_TRAIN ===\n');

%% Config
IMG_RES = 28;
INPUT_DIM = IMG_RES * IMG_RES;
HID_DIM = 64;
OUT_DIM = 10;
numEpochs = 500;

%Adam Params
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
lr = 0.003;

%Loading existing checkpoint
scriptDir = fileparts(mfilename('fullpath'));
outPutDir = fullfile(fileparts(scriptDir), 'Outputs', 'NeuralNet');
checkFile = fullfile(outPutDir, 'mnist_checkpoint.mat');


if isfile(checkFile)
    info = dir(checkFile);
    fprintf('[!] Existing checkpoint found:\n');
    fprintf('    %s\n', checkFile);
    fprintf('    Size: %.1f MB  |  Saved: %s\n', info.bytes/1e6, info.date);
    
    % Load metadata only to show summary
    ck = load(checkFile, 'accHistory', 'numEpochs', 'IMG_RES', 'HID_DIM');
    fprintf('    Network: [%d → %d → %d]  |  Epochs: %d  |  Final acc: %.1f%%\n\n', ...
        ck.IMG_RES * ck.IMG_RES, ck.HID_DIM, 10, ck.numEpochs, ck.accHistory(end)*100);
    
    answer = input('Retrain from scratch? [y/N]: ', 's');
    if ~strcmpi(strtrim(answer), 'y')
        fprintf('\nSkipping training. Run  exp_mnist_replay  to animate.\n');
        return;
    end
    fprintf('\nRetraining...\n\n');
end

%% Load and Preprocess Full Dataset
fprintf('[1] Loading MNIST (full resolution: %dx%d = %d inputs)...\n', ...
    IMG_RES, IMG_RES, INPUT_DIM);
try
    [XTrain4D, YTrain] = digitTrain4DArrayData(); % 28x28x1x5000
catch
    error('Deep Learning Toolbox required (digitTrain4DArrayData).');
end
nSamples = size(XTrain4D, 4);
fprintf('Samples: %d\n', nSamples);
X = zeros(INPUT_DIM, numEpochs); % 784 x 500

for i = 1:nSamples
    img = double(XTrain4D(:,:,1, i));
    if max(img(:)) > 1
        img = img / 255.0;
    end
    if IMG_RES ~= 28
        img = imresize(img, [IMG_RES IMG_RES], 'bicubic');
    end
    X(:, i) = reshape(img, INPUT_DIM, 1) * 2.0 - 1.0; % Convert the range to [-1, 1]
end
Y = dummyvar(grp2idx(YTrain(1:nSamples)))';
%% Initialize Weights 

%Break symmetry
W1 = randn(HID_DIM, INPUT_DIM) * sqrt(2 / INPUT_DIM);

%Avoid dead neurons ( Saruration due to activation func)
b1 = zeros(HID_DIM, 1);

W2 = randn(HID_DIM, INPUT_DIM) * sqrt(2 / INPUT_DIM);
b2 = zeros(HID_DIM, 1);

%Pick one demo sample index per epoch
rng(99);
demoSampIdx = randi(nSamples, numEpochs, 1);
rng(42, 'twister'); %Restore main seed

%Pre-allocate activation storage
demoA1 = zeros(HID_DIM, numEpochs); %Hidden activations per epoch
demoA2 = zeros(OUT_DIM, numEpochs); %Ouput activaton per epoch
demoX = zeros(INPUT_DIM, numEpochs); %the demo sample pixel values
demoY = zeros(1, numEpochs); % its true label (0- 9)


%% Adam optimizer + Training Loop
m_W1=0; v_W1=0; m_b1=0; v_b1=0;
m_W2=0; v_W2=0; m_b2=0; v_b2=0;

weightHistory = cell(numEpochs, 1);
costHistory   = zeros(numEpochs, 1);
accHistory    = zeros(numEpochs, 1);

fprintf('[2] Training for %d epochs (Adam, lr=%.4f)...\n', numEpochs, lr);

for ep = 1:numEpochs
    % --- Forward ---
    Z1 = W1 * X + b1;
    A1 = tanh(Z1);
    Z2 = W2 * A1 + b2;
    maxZ2  = max(Z2, [], 1);
    expZ2  = exp(Z2 - maxZ2);
    A2     = expZ2 ./ sum(expZ2, 1);  % softmax

    % --- Metrics ---
    costHistory(ep) = -sum(sum(Y .* log(A2 + 1e-8))) / nSamples;
    [~, pIdx] = max(A2);
    [~, tIdx] = max(Y);
    accHistory(ep)  = mean(pIdx == tIdx);

    % --- Backward ---
    dZ2 = A2 - Y;
    dW2 = (dZ2 * A1') / nSamples;
    db2 = sum(dZ2, 2) / nSamples;
    dA1 = W2' * dZ2;
    dZ1 = dA1 .* (1 - A1.^2);
    dW1 = (dZ1 * X') / nSamples;
    db1 = sum(dZ1, 2) / nSamples;

    % --- Adam Update ---
    m_W1 = beta1*m_W1 + (1-beta1)*dW1; v_W1 = beta2*v_W1 + (1-beta2)*(dW1.^2);
    m_b1 = beta1*m_b1 + (1-beta1)*db1; v_b1 = beta2*v_b1 + (1-beta2)*(db1.^2);
    m_W2 = beta1*m_W2 + (1-beta1)*dW2; v_W2 = beta2*v_W2 + (1-beta2)*(dW2.^2);
    m_b2 = beta1*m_b2 + (1-beta1)*db2; v_b2 = beta2*v_b2 + (1-beta2)*(db2.^2);
    
    mh_W1 = m_W1/(1-beta1^ep); vh_W1 = v_W1/(1-beta2^ep);
    mh_b1 = m_b1/(1-beta1^ep); vh_b1 = v_b1/(1-beta2^ep);
    mh_W2 = m_W2/(1-beta1^ep); vh_W2 = v_W2/(1-beta2^ep);
    mh_b2 = m_b2/(1-beta1^ep); vh_b2 = v_b2/(1-beta2^ep);
    
    W1 = W1 - lr * mh_W1 ./ (sqrt(vh_W1) + epsilon);
    b1 = b1 - lr * mh_b1 ./ (sqrt(vh_b1) + epsilon);
    W2 = W2 - lr * mh_W2 ./ (sqrt(vh_W2) + epsilon);
    b2 = b2 - lr * mh_b2 ./ (sqrt(vh_b2) + epsilon);

    % --- Save checkpoint (flat parameter vector) ---
    weightHistory{ep} = [b1(:); W1(:); b2(:); W2(:)];

    % --- Save pre-computed demo sample activations ---
    xi  = X(:, demoSampIdx(ep));
    a1i = tanh(W1 * xi + b1);
    ez  = exp(W2 * a1i + b2); a2i = ez / sum(ez);
    demoA1(:, ep) = a1i;
    demoA2(:, ep) = a2i;
    demoX(:,  ep) = xi;
    [~, lbl] = max(Y(:, demoSampIdx(ep)));
    demoY(ep) = lbl - 1;  % 0-9

    % Print progress every 50 epochs
    if mod(ep, 50) == 0 || ep == 1
        fprintf('  [ep %3d/%d]  Cost: %.4f  Accuracy: %.1f%%\n', ...
            ep, numEpochs, costHistory(ep), accHistory(ep)*100);
    end
end

fprintf('\nTraining complete. Final accuracy: %.1f%%\n\n', accHistory(end)*100);

%% Save Checkpoint
outputDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'Outputs', 'NeuralNet');
if ~isfolder(outputDir)
    mkdir(outputDir);
end
outFile = fullfile(outputDir, 'mnist_checkpoint.mat');

save(outFile, ...
    'weightHistory', 'costHistory', 'accHistory', ...
    'IMG_RES', 'INPUT_DIM', 'HID_DIM', 'OUT_DIM', 'numEpochs', 'nSamples', ...
    'X', 'Y', ...
    'demoA1', 'demoA2', 'demoX', 'demoY', 'demoSampIdx');  % pre-computed replay data

fprintf('[3] Checkpoint saved to:\n    %s\n\n', outFile);
fprintf('Run  exp_mnist_replay  to animate the training history.\n');
