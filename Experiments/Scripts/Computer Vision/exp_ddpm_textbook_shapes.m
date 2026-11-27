% EXP_DDPM_TEXTBOOK_SHAPES  Train a textbook DDPM (Lai et al., "The Principles
%   of Diffusion Models", Sec. 2.2; Ho et al. 2020) on synthetic shape images
%   and save it as a ModelCard for the Model Lab GUI.
%
%   DATA      ComplexShapeImages 16 x 16 (spiral, star, ring with gap, crescent,
%             cross, heart; random pose / scale / stroke), 8000 train, scaled to [-1, 1].
%   FORWARD   LinearVP: L = 1000 steps, beta_i^2 linear 1e-4 .. 0.02, Eq. (2.2.1).
%   TRAINING  eps-prediction, L_simple of Eq. (2.2.10) with INTEGER steps
%             i ~ U{1..L} (TimeSampler 'Discrete'), U-Net with Ho's time injection
%             GN(h + emb) ('ScaleShift' false), dropout 0.1, weight EMA.
%             Class-conditional (6 shapes; label dropped 10% of the time, i.e.
%             classifier-free guidance training -- Sec. 8.3 -- so the GUI can
%             ask for a chosen shape later without retraining).
%   SAMPLING  DDPM ancestral, all L = 1000 steps, Eq. (2.2.4)-(2.2.5), clipping
%             x_phi to [-1, 1] (Ho et al.'s reference sampler).
%   DEVIATIONS from Ho et al. (compute budget, CPU): 16 x 16 images, a small
%   U-Net, 3000 steps at batch 32 with learning rate 1e-3 (warm-up + cosine)
%   instead of 2e-4 for 800k steps, EMA decay 0.999 instead of 0.9999.
%
%   OUTPUT (Results/ddpm_textbook): ddpm_shapes16.mat (ModelCard: recipe + EMA
%   weights + metadata), samples per class and guidance, training curve,
%   per-class "does the requested shape come out" accuracy.
%   RUN: resumable (checkpointed); OCEAN_QUICK=1 for a smoke run.
%
%   Author: Ocean Library | 2026-09-26

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
root = fullfile(script_dir, '..', '..', '..');
addpath(root); addpath(script_dir);
quick = ~isempty(getenv('OCEAN_QUICK'));
out_dir = fullfile(script_dir, 'Results', 'ddpm_textbook'); if quick, out_dir = fullfile(out_dir, 'quick'); end
if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('Size', 16, 'NTrain', 8000, 'NTest', 1200, 'L', 1000, 'BetaSqStart', 1e-4, 'BetaSqEnd', 0.02, ...
    'Widths', [32 64], 'Dropout', 0.1, 'Steps', 3000, 'Batch', 32, 'LR', 1e-3, 'EMADecay', 0.999, ...
    'CondDropout', 0.1, 'SamplesPerClass', 10, 'Guidance', [1 3], 'EvalSteps', 1500, 'Seed', 1, 'Verbose', 250);
if quick
    cfg.NTrain = 400; cfg.NTest = 120; cfg.Steps = 20; cfg.SamplesPerClass = 2; cfg.EvalSteps = 40; cfg.L = 50; cfg.Verbose = 0;
end
ck = fullfile(out_dir, 'checkpoints');
G = OceanData.Generators.Shapes2D.ComplexShapeImages;
K = numel(G.ClassNames);

% ---------------- data ----------------
data = OceanDL.Training.Checkpoint.run(ck, sprintf('shapes%d_%d_%d', cfg.Size, cfg.NTrain, cfg.NTest), @() makeData(G, cfg));
X = single(2 * data.Xtr - 1);                                   % [0, 1] -> [-1, 1]
y = data.ytr;

% ---------------- model (as a ModelCard recipe) ----------------
R = @OceanDL.Training.ModelCard.recipe;
netR = R("OceanDL.Vision.UNet2D", {}, {'InChannels', 1, 'Widths', cfg.Widths, 'ResPerLevel', 1, 'NumClasses', K, ...
    'Groups', 8, 'Dropout', cfg.Dropout, 'ScaleShift', false, 'Seed', cfg.Seed});
schR = R("OceanDL.Generative.Diffusion.Schedules.LinearVP", {}, {'L', cfg.L, 'BetaSqStart', cfg.BetaSqStart, 'BetaSqEnd', cfg.BetaSqEnd});
timeR = R("OceanDL.Generative.Diffusion.TimeSampler", {"uniform"}, {'Discrete', cfg.L});
dmR = R("OceanDL.Generative.Diffusion.DiffusionModel", {netR, schR}, {'Target', "eps", 'TimeSampler', timeR, ...
    'Weighting', "simple", 'NumClasses', K, 'CondDropout', cfg.CondDropout, 'Seed', cfg.Seed + 1});
key = sprintf('ddpm_shapes%d_%s', cfg.Size, gen_hash(struct('cfg', cfg, 'recipe', dmR)));
fprintf('Textbook DDPM: train, or load its checkpoint (%s)\n', key);
trained = OceanDL.Training.Checkpoint.run(ck, key, @() train(dmR, X, y, cfg));

dm = OceanDL.Training.ModelCard.build(dmR);
dm.to('DataType', "single");
dm.loadStateDict(trained.EMA);                                  % sample with the EMA weights
dm.eval();
card = fullfile(out_dir, sprintf('ddpm_shapes%d.mat', cfg.Size));
meta = struct('Dataset', sprintf("shapes%d", cfg.Size), 'ClassNames', G.ClassNames, 'DataRange', [-1 1], ...
    'ImageSize', [cfg.Size cfg.Size 1], 'Book', "Lai et al. Sec. 2.2 (DDPM), Sec. 8.3 (CFG)", ...
    'TrainSteps', cfg.Steps, 'SecondsPerStep', trained.StepSeconds, 'FinalLoss', mean(trained.Curve(max(1, end-99):end)));
OceanDL.Training.ModelCard.save(card, dm, dmR, 'Meta', meta, 'State', trained.EMA);
fprintf('Saved model card %s\n', card);

% ---------------- samples: does the requested shape come out? ----------------
ev = OceanDL.Generative.SampleEvaluator(double(data.Xtr), data.ytr, 'NumClasses', K, 'Steps', cfg.EvalSteps, 'Seed', 0);
fprintf('Shape classifier (for checking samples): test accuracy %.1f%%\n', 100 * ev.testAccuracy(double(data.Xte), data.yte));
labels = repelem((1:K).', cfg.SamplesPerClass);
smp = OceanDL.Generative.Diffusion.Samplers.DDPMAncestral('Steps', cfg.L);
rows = {}; grids = {};
for w = cfg.Guidance
    t0 = tic;
    S = dm.generate(numel(labels), [cfg.Size cfg.Size 1], smp, 'Labels', labels, 'Guidance', w, ...
        'Stream', RandStream('mt19937ar', 'Seed', 7));
    sec = toc(t0);
    S = min(1, max(0, (double(S) + 1) / 2));
    [~, pred] = max(ev.probabilities(S), [], 1);
    perClass = arrayfun(@(k) mean(pred(labels == k) == k), 1:K);
    rows(end+1, :) = [{w, mean(pred(:) == labels(:)), sec / numel(labels)}, num2cell(perClass)]; %#ok<SAGROW>
    grids{end+1} = gen_tile(S, cfg.SamplesPerClass); %#ok<SAGROW>
end
T = cell2table(rows, 'VariableNames', [{'Guidance', 'RequestedShapeAccuracy', 'SecPerSample'}, cellstr(G.ClassNames)]);
fprintf('\n=== Textbook DDPM on %dx%d shapes: are samples the requested shape? ===\n', cfg.Size, cfg.Size);
disp(T);
save(fullfile(out_dir, 'ddpm_shapes_results.mat'), 'T', 'cfg', 'trained');
plotAll(grids, trained, T, G.ClassNames, cfg, out_dir);

% ======================================================================== %
function s = makeData(G, cfg)
[Xtr, ytr] = G.generate(cfg.NTrain, 'Size', cfg.Size, 'Seed', 1);
[Xte, yte] = G.generate(cfg.NTest, 'Size', cfg.Size, 'Seed', 2);
s = struct('Xtr', single(Xtr), 'ytr', ytr, 'Xte', single(Xte), 'yte', yte);
end

function out = train(dmR, X, y, cfg)
dm = OceanDL.Training.ModelCard.build(dmR);
dm.to('DataType', "single");
ema = OceanDL.optim.ModelEMA(dm, 'Decay', cfg.EMADecay, 'Warmup', true);
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 2);
n = size(X, 4);
lossAt = @(s) batchLoss(dm, X, y, randi(rs, n, 1, cfg.Batch));
r = OceanDL.Training.fit(dm, lossAt, cfg.Steps, 'LR', cfg.LR, 'EMA', ema, 'Verbose', cfg.Verbose);
out = struct('EMA', ema.Shadow, 'State', dm.stateDict(), 'Curve', r.Curve, 'Seconds', r.Seconds, 'StepSeconds', r.StepSeconds);
end

function L = batchLoss(dm, X, y, idx)
L = dm.forward(X(:, :, :, idx), y(idx));
end

function plotAll(grids, trained, T, classNames, cfg, out_dir)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Textbook DDPM samples'); fig.Position = [60 60 1300 520];
tl = tiledlayout(fig, 1, numel(grids) + 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for g = 1:numel(grids)
    ax = nexttile(tl); image(ax, grids{g} * 255); colormap(ax, gray(256)); axis(ax, 'image', 'off');
    title(ax, sprintf('guidance w = %g: %.0f%% requested shape', T.Guidance(g), 100 * T.RequestedShapeAccuracy(g)), ...
        'Color', viz.Colors.Text);
    ylabel(ax, strjoin(classNames, ' / '));
end
ax = nexttile(tl); viz.styleAxes(ax);
plot(ax, movmean(trained.Curve, max(1, round(numel(trained.Curve) / 50))), 'LineWidth', 1.6);
xlabel(ax, 'step'); ylabel(ax, 'L_{simple} (Eq. 2.2.10)'); title(ax, 'Training loss', 'Color', viz.Colors.Text);
viz.export(fig, fullfile(out_dir, sprintf('ddpm_shapes%d_samples', cfg.Size)));
end
