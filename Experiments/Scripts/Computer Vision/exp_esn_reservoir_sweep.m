% EXP_ESN_RESERVOIR_SWEEP  A fair chance for echo-state networks
%   (Goodfellow et al. 2016, Sec. 10.8) on USPS digits read row by row.
%
%   The untuned ESN in exp_rnn_sequential_digits (final state only, one
%   setting) scored 67.8%. Reservoir computing lives or dies by its
%   hyper-parameters and readout, so here they are searched properly:
%     spectral radius rho(W)   {0.5, 0.8, 0.95, 1.1, 1.5}   memory vs chaos
%     input scale              {0.1, 0.3, 1}                 how hard inputs drive the tanh
%     leak rate a              {1, 0.5}                      time scale of the units
%     readout features         final state | time-mean of states | ALL 16 states
%     ridge penalty lambda     {1e-4, 1e-2, 1}
%   Selection uses a VALIDATION split (last 1291 training images); the test
%   set is touched once, for the selected configuration of each readout.
%   Then the best setting is re-run with a 1000-unit reservoir.
%   Nothing is trained by gradient descent: every readout is closed-form ridge.
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
import OceanDL.Recurrent.*

quick = ~isempty(getenv('OCEAN_QUICK'));
grid = struct('Rho', [0.5 0.8 0.95 1.1 1.5], 'InScale', [0.1 0.3 1], 'Leak', [1 0.5], 'Lambda', [1e-4 1e-2 1]);
H = 400;
if quick, grid.Rho = [0.8 1.1]; grid.InScale = 0.3; grid.Leak = 1; H = 100; end

[Xall, yall] = importDigits("train"); [Xte, yte] = importDigits("test");
yall = yall + 1; yte = yte + 1;
nVal = 1291;
if quick, Xall = Xall(:, :, :, 1:1500); yall = yall(1:1500); nVal = 300; end
Str = ImageSequence.toSequence(Xall, "rows");  Ste = ImageSequence.toSequence(Xte, "rows");
trainIdx = 1:numel(yall) - nVal;  valIdx = numel(yall) - nVal + 1 : numel(yall);
onehot = @(y) full(sparse(y, 1:numel(y), 1, 10, numel(y)));

readouts = ["final", "mean", "all"];
rows = [];
for rho = grid.Rho
    for sc = grid.InScale
        for lk = grid.Leak
            esn = RecurrentLayer(EchoStateCell(16, H, 'SpectralRadius', rho, 'InputScale', sc, 'LeakRate', lk, 'Seed', 1));
            Ftr = features(esn, Str); Fte = features(esn, Ste);
            for r = 1:numel(readouts)
                A = Ftr.(readouts(r));
                for lam = grid.Lambda
                    [V, c] = EchoStateCell.fitRidgeReadout(A(:, trainIdx), onehot(yall(trainIdx)), lam);
                    valAcc = acc(V, c, A(:, valIdx), yall(valIdx));
                    rows = [rows; struct('Rho', rho, 'InScale', sc, 'Leak', lk, 'Readout', readouts(r), ...
                        'Lambda', lam, 'ValAcc', valAcc)]; %#ok<AGROW>
                end
            end
            fprintf('rho %.2f  in %.1f  leak %.1f   best val acc: final %.1f%%  mean %.1f%%  all %.1f%%\n', rho, sc, lk, ...
                100 * max([rows([rows.Rho] == rho & [rows.InScale] == sc & [rows.Leak] == lk & [rows.Readout] == "final").ValAcc]), ...
                100 * max([rows([rows.Rho] == rho & [rows.InScale] == sc & [rows.Leak] == lk & [rows.Readout] == "mean").ValAcc]), ...
                100 * max([rows([rows.Rho] == rho & [rows.InScale] == sc & [rows.Leak] == lk & [rows.Readout] == "all").ValAcc]));
        end
    end
end

% Test accuracy of the validation-selected configuration, per readout (refit on all training data).
fprintf('\nSelected on validation, evaluated once on test (H = %d):\n', H);
best = struct('Readout', num2cell(readouts), 'Config', [], 'TestAcc', 0);
for r = 1:numel(readouts)
    cand = rows([rows.Readout] == readouts(r));
    [~, i] = max([cand.ValAcc]);
    b = cand(i);
    esn = RecurrentLayer(EchoStateCell(16, H, 'SpectralRadius', b.Rho, 'InputScale', b.InScale, 'LeakRate', b.Leak, 'Seed', 1));
    Ftr = features(esn, Str); Fte = features(esn, Ste);
    [V, c] = EchoStateCell.fitRidgeReadout(Ftr.(readouts(r)), onehot(yall), b.Lambda);
    best(r).Config = b; best(r).TestAcc = acc(V, c, Fte.(readouts(r)), yte);
    fprintf('  %-5s rho %.2f in %.1f leak %.1f lambda %g : val %.2f%%  TEST %.2f%%\n', readouts(r), b.Rho, b.InScale, ...
        b.Leak, b.Lambda, 100 * b.ValAcc, 100 * best(r).TestAcc);
end

% Bigger reservoir with the best final/mean settings (all-states readout would be 16k-dimensional).
Hbig = 1000; if quick, Hbig = 200; end
big = struct('Readout', {"final", "mean"}, 'TestAcc', 0);
for r = 1:2
    b = best(r).Config;
    esn = RecurrentLayer(EchoStateCell(16, Hbig, 'SpectralRadius', b.Rho, 'InputScale', b.InScale, 'LeakRate', b.Leak, 'Seed', 1));
    Ftr = features(esn, Str); Fte = features(esn, Ste);
    [V, c] = EchoStateCell.fitRidgeReadout(Ftr.(big(r).Readout), onehot(yall), b.Lambda);
    big(r).TestAcc = acc(V, c, Fte.(big(r).Readout), yte);
    fprintf('  H=%d %-5s TEST %.2f%%\n', Hbig, big(r).Readout, 100 * big(r).TestAcc);
end
save(fullfile(out_dir, 'esn_reservoir_sweep_results.mat'), 'rows', 'best', 'big', 'grid', 'H', 'Hbig');

% Figure: validation accuracy over (rho, input scale) for each readout, best leak & lambda.
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ESN reservoir sweep');
fig.Position = [60 60 1400 420];
tl = tiledlayout(fig, 1, numel(readouts), 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:numel(readouts)
    M = zeros(numel(grid.InScale), numel(grid.Rho));
    for i = 1:numel(grid.InScale)
        for j = 1:numel(grid.Rho)
            sel = rows([rows.Readout] == readouts(r) & [rows.Rho] == grid.Rho(j) & [rows.InScale] == grid.InScale(i));
            M(i, j) = 100 * max([sel.ValAcc]);
        end
    end
    ax = nexttile(tl); viz.styleAxes(ax);
    imagesc(ax, M); colormap(ax, turbo); cb = colorbar(ax); cb.Color = viz.Colors.Text;
    set(ax, 'XTick', 1:numel(grid.Rho), 'XTickLabel', string(grid.Rho), 'YTick', 1:numel(grid.InScale), ...
        'YTickLabel', string(grid.InScale), 'TickLabelInterpreter', 'none');
    for i = 1:size(M, 1), for j = 1:size(M, 2)
        text(ax, j, i, sprintf('%.1f', M(i, j)), 'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 8);
    end, end
    xlabel(ax, 'spectral radius \rho(W)'); ylabel(ax, 'input scale');
    title(ax, sprintf('readout "%s": val acc %% (test %.1f%%)', readouts(r), 100 * best(r).TestAcc), 'Color', viz.Colors.Text);
end
viz.export(fig, fullfile(out_dir, 'esn_reservoir_sweep'));

% ------------------------------------------------------------------------ %
function Fs = features(layer, S)
% Reservoir states for every image: final, time-mean, and all steps stacked.
guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
[~, N, T] = size(S, 1:3);
Hn = layer.Cell.HiddenSize;
Fs = struct('final', zeros(Hn, N), 'mean', zeros(Hn, N), 'all', zeros(Hn * T, N));
for s = 1:1000:N
    idx = s:min(N, s + 999);
    Y = OceanDL.Autograd.Tensor.raw(layer.forward(S(:, idx, :)));   % Hn x n x T
    Fs.final(:, idx) = Y(:, :, end);
    Fs.mean(:, idx) = mean(Y, 3);
    Fs.all(:, idx) = reshape(permute(Y, [1 3 2]), Hn * T, numel(idx));
end
end

function a = acc(V, c, Fx, y)
[~, pred] = max(V * Fx + c, [], 1);
a = mean(pred(:) == y(:));
end

function [X, y] = importDigits(split)
[rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
X = OceanDL.Layout.DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape);
X = OceanDL.Layout.DimFormat.toCanonical(X, fmt.Format);
OceanDL.Layout.DimFormat.validate(X, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(y));
end
