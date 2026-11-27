% EXP_RNN_LONG_DEPENDENCIES  Goodfellow et al. (2016) Sec. 10.7-10.10 put to
%   the test: do gated / leaky cells really learn dependencies that a
%   vanilla RNN cannot, and does the gap GROW with sequence length?
%
%   PART A -- ADDING PROBLEM (Hochreiter & Schmidhuber 1997):
%     sum the two marked values of a length-T sequence; T = 10, 30, 60, 120.
%     Test MSE vs the no-memory baseline 1/6; "solved" = MSE < 0.01.
%     Cells: Elman, Leaky (alpha down to 0.01), GRU, LSTM (64 units), each
%     trained 500 Adam iterations (lr 3e-3, batch 64, clip 1); Echo-state
%     (400 units, ridge readout, no BPTT) as the no-gradient reference.
%     Diagnostic: g(1)/g(T) = ||dL/dh_1|| / ||dL/dh_T|| at INITIALISATION,
%     T = 120: how much training signal can reach the first step at all.
%
%   PART B -- PIXEL-BY-PIXEL DIGITS (Le et al. 2015 "sequential MNIST"):
%     USPS 16 x 16 read one pixel per step (T = 256 -- 16x longer than the
%     row-by-row experiment). 3 epochs per cell, same optimiser.
%
%   All numbers are measured; nothing is tuned per cell (same budget).
%   REPRODUCIBILITY: data streams RandStream(1..), weights seeded.
%
%   Author: Ocean Library  |  2026-09-23

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(project_root);
out_dir = fullfile(script_dir, 'Results', 'rnn_long');
if ~isfolder(out_dir), mkdir(out_dir); end
if ~isempty(getenv('OCEAN_QUICK'))                      % quick/dry runs never overwrite full results
    out_dir = fullfile(out_dir, 'quick'); if ~isfolder(out_dir), mkdir(out_dir); end
end
import OceanDL.Recurrent.*
import OceanDL.Autograd.*
import OceanDL.Layout.*
Add = OceanData.Generators.Sequences.AddingProblem;

quick = ~isempty(getenv('OCEAN_QUICK'));
cfg = struct('Ts', [10 30 60 120], 'Iters', 500, 'Batch', 64, 'LR', 3e-3, 'Clip', 1, 'Hidden', 64, ...
             'PixelEpochs', 3, 'PixelTrain', 7291);
if quick
    cfg.Ts = [10 30]; cfg.Iters = 20; cfg.PixelEpochs = 1; cfg.PixelTrain = 256;
end
cellNames = ["Elman", "Leaky", "GRU", "LSTM"];
makeCell = @(name, D, seed) makeRecurrentCell(name, D, cfg.Hidden, seed);

% === PART A: adding problem ================================================
nT = numel(cfg.Ts);
mse = nan(numel(cellNames) + 1, nT);                 % last row: ESN
curves = cell(numel(cellNames), nT);
for j = 1:nT
    T = cfg.Ts(j);
    [Xte, yte] = Add.generate(1000, T, RandStream('mt19937ar', 'Seed', 1000 + T));
    for k = 1:numel(cellNames)
        rnn = RecurrentLayer(makeCell(cellNames(k), 2, 10*k + j));
        head = OceanDL.nn.Linear(cfg.Hidden, 1, 'Seed', 100 + k);
        params = [rnn.parameters(), head.parameters()];
        opt = OceanDL.optim.ModuleOptimizer(params, @() OceanStats.Optim.AdamOptimizer('LR', cfg.LR));
        data = RandStream('mt19937ar', 'Seed', T);
        curve = zeros(1, cfg.Iters);
        tStart = tic;
        for it = 1:cfg.Iters
            [X, y] = Add.generate(cfg.Batch, T, data);
            opt.zeroGrad();
            [~, st] = rnn.forward(X);
            L = F.mseLoss(head.forward(st.h), y);
            L.backward();
            OceanDL.nn.clipGradNorm(params, cfg.Clip);
            opt.step();
            curve(it) = L.Data;
        end
        curves{k, j} = curve;
        [~, st] = GradMode.noGrad(@() rnn.forward(Xte));
        pred = Tensor.raw(GradMode.noGrad(@() head.forward(st.h)));
        mse(k, j) = mean((pred - yte).^2);
        fprintf('Adding T=%3d  %-6s test MSE %.4f  (baseline %.4f)  %.0fs\n', T, cellNames(k), mse(k, j), Add.BaselineMSE, toc(tStart));
    end
    % Echo-state reference: fixed reservoir, closed-form readout on the final state.
    esn = RecurrentLayer(EchoStateCell(2, 400, 'SpectralRadius', 0.95, 'LeakRate', 0.3, 'Seed', 7));
    [Xtr, ytr] = Add.generate(4000, T, RandStream('mt19937ar', 'Seed', 2000 + T));
    Htr = finalHidden(esn, Xtr); Hte = finalHidden(esn, Xte);
    [V, c] = EchoStateCell.fitRidgeReadout(Htr, ytr, 1e-4);
    mse(end, j) = mean((V * Hte + c - yte).^2);
    fprintf('Adding T=%3d  %-6s test MSE %.4f\n', T, "ESN", mse(end, j));
end

% Gradient reaching step 1 at initialisation (the Sec. 10.7 mechanism), T = 120.
Tg = cfg.Ts(end);
[Xg, yg] = Add.generate(256, Tg, RandStream('mt19937ar', 'Seed', 5));
g1 = zeros(1, numel(cellNames));
gCurves = zeros(numel(cellNames), Tg);
for k = 1:numel(cellNames)
    rnn = RecurrentLayer(makeCell(cellNames(k), 2, 500 + k), 'TrackStates', true);
    head = OceanDL.nn.Linear(cfg.Hidden, 1, 'Seed', 600 + k);
    [~, st] = rnn.forward(Xg);
    backward(F.mseLoss(head.forward(st.h), yg));
    g = Diagnostics.gradientNormsOverTime(rnn);
    gCurves(k, :) = g / g(end);
    g1(k) = gCurves(k, 1);
end

% === PART B: pixel-by-pixel digits (T = 256) ===============================
[Xtr, ytr] = importDigits("train"); [Xte, yte] = importDigits("test");
ytr = ytr + 1; yte = yte + 1;
Xtr = Xtr(:, :, :, 1:cfg.PixelTrain); ytr = ytr(1:cfg.PixelTrain);
if quick, Xte = Xte(:, :, :, 1:200); yte = yte(1:200); end
pixAcc = zeros(numel(cellNames), cfg.PixelEpochs);
pixSec = zeros(1, numel(cellNames));
for k = 1:numel(cellNames)
    net = SequenceClassifier(RecurrentLayer(makeCell(cellNames(k), 1, 700 + k)), cfg.Hidden, 10, ...
        'ImageMode', "pixels", 'Seed', 800 + k);
    params = net.parameters();
    opt = OceanDL.optim.ModuleOptimizer(params, @() OceanStats.Optim.AdamOptimizer('LR', cfg.LR));
    order = RandStream('mt19937ar', 'Seed', 1);
    tStart = tic;
    for ep = 1:cfg.PixelEpochs
        perm = randperm(order, numel(ytr));
        for b = 1:floor(numel(ytr) / cfg.Batch)
            idx = perm((b-1)*cfg.Batch + 1 : b*cfg.Batch);
            opt.zeroGrad();
            L = F.crossEntropy(net.forward(Xtr(:, :, :, idx)), ytr(idx));
            L.backward();
            OceanDL.nn.clipGradNorm(params, cfg.Clip);
            opt.step();
        end
        pixAcc(k, ep) = accuracy(net, Xte, yte);
        fprintf('Pixels T=256 %-6s epoch %d  test acc %.2f%%  (%.0fs)\n', cellNames(k), ep, 100 * pixAcc(k, ep), toc(tStart));
    end
    pixSec(k) = toc(tStart);
end

% === Summary ===============================================================
fprintf('\nAdding problem, test MSE (baseline %.4f, solved < 0.01):\n%-7s', Add.BaselineMSE, '');
fprintf('%s\n', strjoin(compose("%9s", "T=" + string(cfg.Ts)), ''));
rows = [cellNames, "ESN"];
for k = 1:numel(rows)
    fprintf('%-7s', rows(k)); fprintf('%9.4f', mse(k, :)); fprintf('\n');
end
fprintf('\nAt initialisation, T=%d: ||dL/dh_1|| / ||dL/dh_T||\n', Tg);
for k = 1:numel(cellNames), fprintf('  %-6s %.3g\n', cellNames(k), g1(k)); end
fprintf('\nPixel-by-pixel USPS (T=256), test accuracy after %d epochs:\n', cfg.PixelEpochs);
for k = 1:numel(cellNames), fprintf('  %-6s %.2f%%  (%.0fs)\n', cellNames(k), 100 * pixAcc(k, end), pixSec(k)); end
save(fullfile(out_dir, 'rnn_long_dependencies_results.mat'), 'cfg', 'mse', 'curves', 'gCurves', 'g1', 'pixAcc', 'pixSec', 'cellNames');

% === Figures ===============================================================
viz = OceanViz.Core('Theme', 'Dark');
pal = turbo(numel(rows) + 2); pal = pal(2:end-1, :);
fig = viz.createFigure('Long-term dependencies');
fig.Position = [60 60 1500 470];
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
for k = 1:numel(rows)
    style = '-o'; if rows(k) == "ESN", style = '--s'; end
    plot(ax, cfg.Ts, mse(k, :), style, 'Color', pal(k, :), 'LineWidth', 1.6, 'DisplayName', rows(k));
end
yline(ax, Add.BaselineMSE, ':', 'no memory (1/6)', 'Color', viz.Colors.Text, 'LineWidth', 1.2, 'HandleVisibility', 'off');
yline(ax, 0.01, ':', 'solved (0.01)', 'Color', viz.Colors.Text, 'LineWidth', 1.2, 'HandleVisibility', 'off');
set(ax, 'YScale', 'log');
xlabel(ax, 'sequence length T'); ylabel(ax, 'test MSE');
title(ax, 'Adding problem: error vs length', 'Color', viz.Colors.Text);
legend(ax, 'Location', 'northwest', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
for k = 1:numel(cellNames)
    plot(ax, 1:Tg, gCurves(k, :), '-', 'Color', pal(k, :), 'LineWidth', 1.6, 'DisplayName', cellNames(k));
end
set(ax, 'YScale', 'log');
xlabel(ax, 'time step t'); ylabel(ax, '||dL/dh_t|| / ||dL/dh_T||');
title(ax, sprintf('Gradient reaching step t at init (T=%d)', Tg), 'Color', viz.Colors.Text);
legend(ax, 'Location', 'southeast', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
for k = 1:numel(cellNames)
    plot(ax, 1:cfg.PixelEpochs, 100 * pixAcc(k, :), '-o', 'Color', pal(k, :), 'LineWidth', 1.6, 'DisplayName', cellNames(k));
end
xlabel(ax, 'epoch'); ylabel(ax, 'test accuracy (%)');
title(ax, 'Pixel-by-pixel USPS (T = 256)', 'Color', viz.Colors.Text);
legend(ax, 'Location', 'southeast', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
viz.export(fig, fullfile(out_dir, 'rnn_long_dependencies'));

% ------------------------------------------------------------------------ %
function c = makeRecurrentCell(name, D, H, seed)
switch name
    case "Elman", c = OceanDL.Recurrent.ElmanCell(D, H, 'Seed', seed);
    case "Leaky", c = OceanDL.Recurrent.LeakyCell(D, H, 'AlphaMin', 0.01, 'Seed', seed);
    case "GRU",   c = OceanDL.Recurrent.GRUCell(D, H, 'Seed', seed);
    case "LSTM",  c = OceanDL.Recurrent.LSTMCell(D, H, 'Seed', seed);
end
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

function [X, y] = importDigits(split)
[rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
X = OceanDL.Layout.DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape);
X = OceanDL.Layout.DimFormat.toCanonical(X, fmt.Format);
OceanDL.Layout.DimFormat.validate(X, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(y));
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
net.train();
end
