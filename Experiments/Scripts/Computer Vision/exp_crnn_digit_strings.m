% EXP_CRNN_DIGIT_STRINGS  Reading multi-digit strings with a CRNN trained by
%   CTC (Shi et al. 2017; Graves et al. 2006) -- the CV sequence-labelling
%   application of Goodfellow Ch. 10: variable-length OUTPUT sequences with
%   no alignment between image columns and labels.
%
%   DATA: strings of 3-5 USPS digits (16 x 96 images) composed with random
%   gaps; train strings use USPS train digits, test strings USPS test
%   digits (disjoint images). Imported through the layout contract.
%
%   MODEL: CRNN = CNN (2 x [conv3x3, ReLU, pool2]) -> 24 columns as time
%   steps -> GRU (bidirectional vs unidirectional, 96 units) -> 11 classes
%   per step (blank + 10 digits) -> CTC loss; greedy decoding.
%
%   MEASURED: character error rate (edit distance / length) and exact
%   string accuracy per epoch, for BiGRU vs one-directional GRU (does right
%   context help reading?), plus the learned CTC alignment (per-column class
%   posteriors) on test strings.
%
%   Author: Ocean Library  |  2026-09-23

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(project_root);
out_dir = fullfile(script_dir, 'Results', 'crnn');
if ~isfolder(out_dir), mkdir(out_dir); end
if ~isempty(getenv('OCEAN_QUICK'))                      % quick/dry runs never overwrite full results
    out_dir = fullfile(out_dir, 'quick'); if ~isfolder(out_dir), mkdir(out_dir); end
end
import OceanDL.Recurrent.*
import OceanDL.Autograd.*

quick = ~isempty(getenv('OCEAN_QUICK'));
cfg = struct('TrainStrings', 6000, 'TestStrings', 1000, 'Epochs', 8, 'Batch', 32, 'LR', 1e-3, 'Clip', 5, 'Width', 96);
if quick, cfg.TrainStrings = 128; cfg.TestStrings = 64; cfg.Epochs = 1; end

% === Data: import digits (layout contract), compose strings ================
[Dtr, ltr] = importDigits("train");
[Dte, lte] = importDigits("test");
compose = @(D, l, M, seed) OceanData.Generators.Sequences.DigitStrings.compose(D, l, M, ...
    'MinLength', 3, 'MaxLength', 5, 'Width', cfg.Width, 'Stream', RandStream('mt19937ar', 'Seed', seed));
[Xtr, ytr] = compose(Dtr, ltr, cfg.TrainStrings, 1);
[Xte, yte] = compose(Dte, lte, cfg.TestStrings, 2);
toClasses = @(labels) cellfun(@(d) d + 2, labels, 'UniformOutput', false);   % blank = 1, digit d -> d + 2
ttr = toClasses(ytr); tte = toClasses(yte);
fprintf('Strings: train %s, test %s\n', mat2str(size(Xtr)), mat2str(size(Xte)));

% === Train BiGRU and GRU CRNNs =============================================
variants = ["BiGRU", "GRU"];
res = struct('Name', cellstr(variants), 'CER', [], 'SeqAcc', [], 'Loss', [], 'Seconds', 0, 'Params', 0);
nets = cell(1, 2);
for v = 1:2
    net = CRNN(16, 11, 'Channels', [32 64], 'Hidden', 96, 'Bidirectional', variants(v) == "BiGRU", 'Seed', 10);
    params = net.parameters();
    res(v).Params = net.numParameters();
    opt = OceanDL.optim.ModuleOptimizer(params, @() OceanStats.Optim.AdamOptimizer('LR', cfg.LR));
    order = RandStream('mt19937ar', 'Seed', 3);
    tStart = tic;
    for ep = 1:cfg.Epochs
        perm = randperm(order, cfg.TrainStrings);
        losses = [];
        for b = 1:floor(cfg.TrainStrings / cfg.Batch)
            idx = perm((b-1)*cfg.Batch + 1 : b*cfg.Batch);
            opt.zeroGrad();
            L = F.ctcLoss(net.forward(Xtr(:, :, :, idx)), ttr(idx));
            L.backward();
            OceanDL.nn.clipGradNorm(params, cfg.Clip);
            opt.step();
            losses(end+1) = L.Data; %#ok<SAGROW>
        end
        r = evaluate(net, Xte, tte);
        res(v).CER(ep) = r.CER; res(v).SeqAcc(ep) = r.SequenceAccuracy; res(v).Loss(ep) = mean(losses);
        fprintf('%-6s epoch %d  CTC loss %.3f  test CER %.2f%%  exact strings %.2f%%  (%.0fs)\n', variants(v), ep, ...
            mean(losses), 100 * r.CER, 100 * r.SequenceAccuracy, toc(tStart));
    end
    res(v).Seconds = toc(tStart);
    nets{v} = net;
end
save(fullfile(out_dir, 'crnn_digit_strings_results.mat'), 'res', 'cfg');

% === Figures ===============================================================
viz = OceanViz.Core('Theme', 'Dark');
pal = [0.30 0.75 0.95; 0.98 0.60 0.20];
fig = viz.createFigure('CRNN + CTC on digit strings');
fig.Position = [60 60 1500 760];
tl = tiledlayout(fig, 3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl, [1 2]); viz.styleAxes(ax); hold(ax, 'on');
for v = 1:2
    plot(ax, 1:cfg.Epochs, 100 * res(v).CER, '-o', 'Color', pal(v, :), 'LineWidth', 1.6, 'DisplayName', res(v).Name + " CER");
    plot(ax, 1:cfg.Epochs, 100 * (1 - res(v).SeqAcc), '--s', 'Color', pal(v, :), 'LineWidth', 1.2, 'DisplayName', res(v).Name + " string error");
end
set(ax, 'YScale', 'log'); xlabel(ax, 'epoch'); ylabel(ax, 'test error (%)');
title(ax, 'Character / whole-string error on unseen writers', 'Color', viz.Colors.Text);
legend(ax, 'Location', 'northeast', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);

% Learned alignment on one test string: per-column posteriors (softmax over classes).
best = nets{1};
m = 1;
guard = GradMode.disable(); %#ok<NASGU>
Z = Tensor.raw(best.forward(Xte(:, :, :, m)));
P = exp(Z - max(Z, [], 1)); P = P ./ sum(P, 1); P = reshape(P, 11, []);
ax = nexttile(tl, [1 2]);
imagesc(ax, [0.5 size(P, 2) + 0.5], [0.5 16.5], Xte(:, :, 1, m)); colormap(ax, gray); axis(ax, 'image'); hold(ax, 'on');
title(ax, sprintf('test string "%s"', sprintf('%d', yte{m})), 'Color', viz.Colors.Text);
set(ax, 'XTick', [], 'YTick', []);
ax = nexttile(tl, [1 2]); viz.styleAxes(ax);
imagesc(ax, P); colormap(ax, turbo); axis(ax, 'tight');
set(ax, 'YTick', 1:11, 'YTickLabel', ["blank", string(0:9)], 'TickLabelInterpreter', 'none');
xlabel(ax, 'column (time step)'); title(ax, 'CTC per-column posteriors: spikes = emitted digits', 'Color', viz.Colors.Text);

decoded = CTCDecoder.greedy(best.forward(Xte(:, :, :, 1:6)), 1);
for k = 1:6
    ax = nexttile(tl);
    imagesc(ax, Xte(:, :, 1, k)); colormap(ax, gray); axis(ax, 'image'); set(ax, 'XTick', [], 'YTick', []);
    predStr = sprintf('%d', decoded{k} - 2); trueStr = sprintf('%d', yte{k});
    col = [0.4 0.9 0.4]; if ~strcmp(predStr, trueStr), col = [1 0.4 0.4]; end
    title(ax, sprintf('true %s  ->  read %s', trueStr, predStr), 'Color', col, 'FontSize', 9);
end
viz.export(fig, fullfile(out_dir, 'crnn_digit_strings'));

fprintf('\n%-6s %8s %10s %14s %8s\n', 'model', 'params', 'test CER', 'exact strings', 'seconds');
for v = 1:2
    fprintf('%-6s %8d %9.2f%% %13.2f%% %8.0f\n', res(v).Name, res(v).Params, 100 * res(v).CER(end), 100 * res(v).SeqAcc(end), res(v).Seconds);
end

% ------------------------------------------------------------------------ %
function r = evaluate(net, X, targets)
guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
pred = cell(1, numel(targets));
for s = 1:200:numel(targets)
    idx = s:min(numel(targets), s + 199);
    pred(idx) = OceanDL.Recurrent.CTCDecoder.greedy(net.forward(X(:, :, :, idx)), 1);
end
r = OceanDL.Recurrent.CTCDecoder.errorRates(pred, targets);
end

function [X, y] = importDigits(split)
[rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
X = OceanDL.Layout.DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape);
X = OceanDL.Layout.DimFormat.toCanonical(X, fmt.Format);
OceanDL.Layout.DimFormat.validate(X, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(y));
end
