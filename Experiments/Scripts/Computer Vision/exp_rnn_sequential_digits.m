% EXP_RNN_SEQUENTIAL_DIGITS  Recurrent networks on images (Goodfellow et al.
%   2016, Ch. 10) -- USPS handwritten digits read as SEQUENCES.
%
%   QUESTIONS (each answered by measurement, not assumed):
%     Q1  Reading a 16 x 16 digit row by row (T = 16 steps of 16 pixels),
%         how do the five recurrent cells of Ch. 10 compare -- Elman (10.2),
%         Leaky units (10.9), GRU / LSTM (10.10), Echo-state (10.8)?
%     Q2  Does seeing the future help (bidirectional GRU, Sec. 10.3)?
%     Q3  Does a 2-D recurrent sweep (ReNet2D, four directions) compete with
%         a small CNN on the same data?
%     Q4  Where does the training signal die? ||dL/dh_t|| over t for each
%         trained cell (Sec. 10.7).
%
%   DATA IMPORT (the repo's layout contract): the USPS file stores one
%   image per ROW with pixels in ROW-MAJOR order. UspsDigitsReader returns
%   it untouched; DimFormat.fromRowMajor restores the index meaning and
%   DimFormat.toCanonical converts to H x W x C x N, then validate() checks it.
%
%   MODELS: all recurrent encoders use 64 hidden units (ESN: a 400-unit fixed
%   reservoir -- it has no trainable recurrent weights, so it gets more units,
%   the usual practice). Trained with Adam (lr 3e-3), batch 64, global
%   gradient-norm clipping at 1 (Sec. 10.11.1). ESN readout: closed-form ridge.
%
%   REPRODUCIBILITY: all weights seeded; minibatch order from RandStream(1).
%
%   Author: Ocean Library  |  2026-09-23

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(project_root);
out_dir = fullfile(script_dir, 'Results', 'rnn_cv');
if ~isfolder(out_dir), mkdir(out_dir); end
if ~isempty(getenv('OCEAN_QUICK'))                      % quick/dry runs never overwrite full results
    out_dir = fullfile(out_dir, 'quick'); if ~isfolder(out_dir), mkdir(out_dir); end
end

import OceanDL.Layout.*
import OceanDL.Recurrent.*
import OceanDL.Autograd.*
cfg = struct('Epochs', 8, 'Batch', 64, 'LR', 3e-3, 'Clip', 1, 'Hidden', 64);

% === 1. Import: source rows -> canonical SSCB, validated ===================
[Xtr, ytr] = importDigits("train");
[Xte, yte] = importDigits("test");
ytr = ytr + 1; yte = yte + 1;                           % classes 1..10 (digit d -> d+1)
if ~isempty(getenv('OCEAN_QUICK'))                      % dry run: tiny subset, 1 epoch
    cfg.Epochs = 1; Xtr = Xtr(:, :, :, 1:256); ytr = ytr(1:256); Xte = Xte(:, :, :, 1:300); yte = yte(1:300);
end
fprintf('Imported USPS: train %s, test %s (H x W x C x N)\n', mat2str(size(Xtr)), mat2str(size(Xte)));

% === 2. Models =============================================================
H = cfg.Hidden;
models = {
  "Elman",   SequenceClassifier(RecurrentLayer(ElmanCell(16, H, 'Seed', 1)), H, 10, 'ImageMode', "rows", 'Seed', 11)
  "Leaky",   SequenceClassifier(RecurrentLayer(LeakyCell(16, H, 'AlphaMin', 0.05, 'Seed', 2)), H, 10, 'ImageMode', "rows", 'Seed', 12)
  "GRU",     SequenceClassifier(RecurrentLayer(GRUCell(16, H, 'Seed', 3)), H, 10, 'ImageMode', "rows", 'Seed', 13)
  "LSTM",    SequenceClassifier(RecurrentLayer(LSTMCell(16, H, 'Seed', 4)), H, 10, 'ImageMode', "rows", 'Seed', 14)
  "BiGRU",   SequenceClassifier(Bidirectional(@() GRUCell(16, H, 'Seed', 5)), 2*H, 10, 'ImageMode', "rows", 'Seed', 15)
  "ReNet2D", ReNetClassifier(1, 2, 32, 32, 10, 'Seed', 6)
  "CNN",     OceanDL.nn.Sequential( ...
                 OceanDL.nn.Conv2d(1, 16, 3, 'Padding', 1, 'Seed', 7), OceanDL.nn.Activation("relu"), ...
                 OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2])), ...
                 OceanDL.nn.Conv2d(16, 32, 3, 'Padding', 1, 'Seed', 8), OceanDL.nn.Activation("relu"), ...
                 OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2])), ...
                 OceanDL.nn.Legacy(OceanDL.Conv.Flatten()), ...
                 OceanDL.nn.Linear(4*4*32, 10, 'Seed', 9))
};
nM = size(models, 1);
names = [models{:, 1}, "ESN"];
res = struct('Name', cellstr(names), 'Params', 0, 'TestAcc', [], 'TrainLoss', [], 'Seconds', 0, 'GradOverTime', []);

% === 3. Train the gradient-based models ====================================
for m = 1:nM
    net = models{m, 2};
    params = net.parameters();
    res(m).Params = net.numParameters();
    opt = OceanDL.optim.ModuleOptimizer(params, @() OceanStats.Optim.AdamOptimizer('LR', cfg.LR));
    order = RandStream('mt19937ar', 'Seed', 1);
    tStart = tic;
    for ep = 1:cfg.Epochs
        net.train();
        perm = randperm(order, numel(ytr));
        losses = [];
        for b = 1:floor(numel(ytr) / cfg.Batch)
            idx = perm((b-1)*cfg.Batch + 1 : b*cfg.Batch);
            opt.zeroGrad();
            L = F.crossEntropy(net.forward(Xtr(:, :, :, idx)), ytr(idx));
            L.backward();
            OceanDL.nn.clipGradNorm(params, cfg.Clip);
            opt.step();
            losses(end+1) = L.Data; %#ok<SAGROW>
        end
        res(m).TrainLoss(ep) = mean(losses);
        res(m).TestAcc(ep) = accuracy(net, Xte, yte);
        fprintf('%-8s epoch %d  train loss %.4f  test acc %.2f%%  (%.0fs)\n', names(m), ep, ...
            res(m).TrainLoss(ep), 100 * res(m).TestAcc(ep), toc(tStart));
    end
    res(m).Seconds = toc(tStart);
end

% === 4. Echo-state network: fixed reservoir, closed-form readout ===========
esn = RecurrentLayer(EchoStateCell(16, 400, 'SpectralRadius', 0.95, 'Density', 0.1, 'InputScale', 0.5, 'Seed', 10));
tStart = tic;
finalState = @(X) finalHidden(esn, ImageSequence.toSequence(X, "rows"));
Htr = finalState(Xtr);
Ytr = full(sparse(ytr, 1:numel(ytr), 1, 10, numel(ytr)));            % one-hot targets
[V, c] = EchoStateCell.fitRidgeReadout(Htr, Ytr, 1e-2);
[~, pred] = max(V * finalState(Xte) + c, [], 1);
e = nM + 1;
res(e).Params = numel(V) + numel(c);                                % only the readout is trained
res(e).TestAcc = mean(pred(:) == yte(:));
res(e).Seconds = toc(tStart);
fprintf('%-8s ridge readout            test acc %.2f%%  (%.0fs)\n', "ESN", 100 * res(e).TestAcc, res(e).Seconds);

% === 5. Q4: gradient flow over time in the trained recurrent encoders ======
probe = Xte(:, :, :, 1:256);
for m = 1:4                                                          % Elman, Leaky, GRU, LSTM
    enc = models{m, 2}.Encoder;
    enc.TrackStates = true;
    logits = models{m, 2}.forward(probe);
    backward(F.crossEntropy(logits, yte(1:256)));
    g = Diagnostics.gradientNormsOverTime(enc);
    res(m).GradOverTime = g / g(end);
    enc.TrackStates = false;
    models{m, 2}.zeroGrad();
end

% === 6. Summary table ======================================================
fprintf('\n%-8s %9s %10s %9s\n', 'model', 'params', 'test acc', 'seconds');
for k = 1:numel(res)
    fprintf('%-8s %9d %9.2f%% %9.0f\n', res(k).Name, res(k).Params, 100 * res(k).TestAcc(end), res(k).Seconds);
end
fprintf('\n||dL/dh_t|| / ||dL/dh_T|| at t = 1 (signal reaching the first row):\n');
for m = 1:4
    fprintf('  %-6s %.3g\n', res(m).Name, res(m).GradOverTime(1));
end
save(fullfile(out_dir, 'rnn_sequential_digits_results.mat'), 'res', 'cfg');

% === 7. Figures ============================================================
viz = OceanViz.Core('Theme', 'Dark');
pal = turbo(numel(res) + 2); pal = pal(2:end-1, :);            % distinct colours (theme sequence repeats after 7)
fig = viz.createFigure('RNNs on USPS digits');
fig.Position = [60 60 1500 480];
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
for k = 1:nM
    plot(ax, 1:cfg.Epochs, 100 * res(k).TestAcc, '-o', 'Color', pal(k, :), 'LineWidth', 1.5, 'DisplayName', res(k).Name);
end
yline(ax, 100 * res(e).TestAcc, '--', 'Color', pal(e, :), 'LineWidth', 1.5, 'DisplayName', 'ESN (ridge)');
xlabel(ax, 'epoch'); ylabel(ax, 'test accuracy (%)');
title(ax, 'Q1-Q3: test accuracy', 'Color', viz.Colors.Text);
legend(ax, 'Location', 'southeast', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
finalAcc = arrayfun(@(h) 100 * h.TestAcc(end), res);
bh = bar(ax, categorical(names, names), finalAcc, 'FaceColor', 'flat');
bh.CData = pal;
ylim(ax, [min(finalAcc) - 5, 100]);
for k = 1:numel(res)
    text(ax, k, finalAcc(k) + 0.3, sprintf('%.1f%%\n%dk p', finalAcc(k), round(res(k).Params / 1000)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', viz.Colors.Text, 'FontSize', 8);
end
ylabel(ax, 'final test accuracy (%)');
title(ax, 'Final accuracy (label: params)', 'Color', viz.Colors.Text);

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
for m = 1:4
    semilogy(ax, 1:16, res(m).GradOverTime, '-o', 'Color', pal(m, :), 'LineWidth', 1.5, 'DisplayName', res(m).Name);
end
set(ax, 'YScale', 'log');
xlabel(ax, 'time step t (image row)'); ylabel(ax, '||dL/dh_t|| / ||dL/dh_T||');
title(ax, 'Q4: gradient reaching each row', 'Color', viz.Colors.Text);
legend(ax, 'Location', 'southeast', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
viz.export(fig, fullfile(out_dir, 'rnn_sequential_digits'));

% ------------------------------------------------------------------------ %
function [X, y] = importDigits(split)
import OceanDL.Layout.*
[rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
X = DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape);   % (N,16,16,1), source index meaning
X = DimFormat.toCanonical(X, fmt.Format);                          % "BSSC" -> "SSCB": 16 x 16 x 1 x N
DimFormat.validate(X, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(y));
end

function acc = accuracy(net, X, y)
net.eval();
guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
correct = 0;
for s = 1:500:numel(y)
    idx = s:min(numel(y), s + 499);
    [~, pred] = max(OceanDL.Autograd.Tensor.raw(net.forward(X(:, :, :, idx))), [], 1);
    correct = correct + sum(pred(:) == y(idx));
end
acc = correct / numel(y);
end

function Hs = finalHidden(layer, S)
guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
Hs = zeros(layer.Cell.HiddenSize, size(S, 2));
for s = 1:1000:size(S, 2)
    idx = s:min(size(S, 2), s + 999);
    [~, st] = layer.forward(S(:, idx, :));
    Hs(:, idx) = OceanDL.Autograd.Tensor.raw(st.h);
end
end
