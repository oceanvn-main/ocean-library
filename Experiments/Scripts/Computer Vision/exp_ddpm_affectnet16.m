% EXP_DDPM_AFFECTNET16  Train a textbook DDPM (Lai et al., "The Principles of
%   Diffusion Models", Sec. 2.2; Ho et al. 2020) on REAL faces and save it as a
%   ModelCard for the Model Lab GUI. Same model and training code as
%   exp_ddpm_textbook_shapes.m; only the data (and the device) differ.
%
%   DATA      AffectNet, the clean 6-class protocol (AffectNetReader.cleanProtocol:
%             de-duplicated re-split, AffectNet JPEGs only, folder == csv label;
%             anger, contempt, disgust, fear, sad, surprise), 16 x 16 GRAYSCALE,
%             6000 training faces picked as evenly over the 6 expressions as the
%             class sizes allow, scaled to [-1, 1].
%   FORWARD   LinearVP: L = 1000 steps, beta_i^2 linear 1e-4 .. 0.02, Eq. (2.2.1).
%   TRAINING  eps-prediction, L_simple of Eq. (2.2.10), integer steps i ~ U{1..L},
%             U-Net (Ho's GN(h + emb)), dropout 0.1, weight EMA. Conditioned on the
%             expression; label dropped 10% of the time (classifier-free guidance,
%             Sec. 8.3), so the GUI can ask for an expression.
%   SAMPLING  DDPM ancestral, all L = 1000 steps, Eq. (2.2.4)-(2.2.5), x_phi clipped.
%   DEVICE    GPU when MATLAB can use one (FP32), else CPU.
%
%   WHAT TO EXPECT at 16 x 16: face-like images (oval, eye / mouth blobs,
%   shading). Expressions are barely resolvable at this size, so the
%   "requested expression" check is judged against the expression classifier's
%   OWN test accuracy on real faces (its ceiling), not against 100%.
%   Memorisation check: nearest-training-face distance of samples vs that of
%   real test faces (samples much closer than test faces = copying).
%
%   OUTPUT (Results/ddpm_affectnet): ddpm_affectnet16.mat (ModelCard), sample
%   grids per guidance, training curve, results table.
%   RUN: the trained weights are checkpointed; OCEAN_QUICK=1 for a smoke run.
%
%   Author: Ocean Library | 2026-09-26

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
root = fullfile(script_dir, '..', '..', '..');
addpath(root); addpath(script_dir);
quick = ~isempty(getenv('OCEAN_QUICK'));
out_dir = fullfile(script_dir, 'Results', 'ddpm_affectnet'); if quick, out_dir = fullfile(out_dir, 'quick'); end
if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('Size', 16, 'NTrain', 6000, 'L', 1000, 'BetaSqStart', 1e-4, 'BetaSqEnd', 0.02, ...
    'Widths', [64 128], 'Dropout', 0.1, 'Steps', 2000, 'Batch', 64, 'LR', 1e-3, 'EMADecay', 0.999, ...
    'CondDropout', 0.1, 'SamplesPerClass', 10, 'Guidance', [1 3], 'EvalSteps', 2000, 'Seed', 1, 'Verbose', 250);
if quick
    cfg.NTrain = 600; cfg.Steps = 20; cfg.SamplesPerClass = 2; cfg.EvalSteps = 40; cfg.L = 50; cfg.Verbose = 5;
end
device = "cpu"; if OceanDL.Device.gpuAvailable(), device = "gpu"; end
fprintf('Device: %s\n', device);
ck = fullfile(out_dir, 'checkpoints');

% ---------------- data ----------------
D = OceanViz.App.ModelLab.Datasets.affectnet(cfg.Size);         % same data + split the Model Lab shows
K = numel(D.ClassNames);
pick = balancedPick(D.y, cfg.NTrain, K, cfg.Seed);
X = single(2 * D.X(:, :, :, pick) - 1);                         % [0, 1] -> [-1, 1]
y = D.y(pick);
fprintf('Faces: %d training (of %d available), %d test; per expression:\n', numel(y), numel(D.y), numel(D.yte));
disp(array2table(accumarray(y, 1, [K 1]).', 'VariableNames', cellstr(D.ClassNames)));

% ---------------- model (as a ModelCard recipe) ----------------
R = @OceanDL.Training.ModelCard.recipe;
netR = R("OceanDL.Vision.UNet2D", {}, {'InChannels', 1, 'Widths', cfg.Widths, 'ResPerLevel', 1, 'NumClasses', K, ...
    'Groups', 8, 'Dropout', cfg.Dropout, 'ScaleShift', false, 'Seed', cfg.Seed});
schR = R("OceanDL.Generative.Diffusion.Schedules.LinearVP", {}, {'L', cfg.L, 'BetaSqStart', cfg.BetaSqStart, 'BetaSqEnd', cfg.BetaSqEnd});
timeR = R("OceanDL.Generative.Diffusion.TimeSampler", {"uniform"}, {'Discrete', cfg.L});
dmR = R("OceanDL.Generative.Diffusion.DiffusionModel", {netR, schR}, {'Target', "eps", 'TimeSampler', timeR, ...
    'Weighting', "simple", 'NumClasses', K, 'CondDropout', cfg.CondDropout, 'Seed', cfg.Seed + 1});
key = sprintf('ddpm_affectnet%d_%s', cfg.Size, gen_hash(struct('cfg', cfg, 'recipe', dmR, 'pick', pick(:).')));
fprintf('Textbook DDPM on faces: train, or load its checkpoint (%s)\n', key);
trained = OceanDL.Training.Checkpoint.run(ck, key, @() train(dmR, X, y, cfg, device));
fprintf('Trained: %.3f s/step, %.1f min total\n', trained.StepSeconds, trained.Seconds / 60);

dm = OceanDL.Training.ModelCard.build(dmR);
dm.to('DataType', "single");
dm.loadStateDict(trained.EMA);                                  % sample with the EMA weights
dm.eval();
card = fullfile(out_dir, sprintf('ddpm_affectnet%d.mat', cfg.Size));
meta = struct('Dataset', sprintf("affectnet%d", cfg.Size), 'ClassNames', D.ClassNames, 'DataRange', [-1 1], ...
    'ImageSize', [cfg.Size cfg.Size 1], 'Book', "Lai et al. Sec. 2.2 (DDPM), Sec. 8.3 (CFG)", ...
    'Protocol', "AffectNetReader.cleanProtocol, grayscale", 'NTrain', numel(y), 'TrainSteps', cfg.Steps, ...
    'SecondsPerStep', trained.StepSeconds, 'Device', device, 'FinalLoss', mean(trained.Curve(max(1, end-99):end)));
OceanDL.Training.ModelCard.save(card, dm, dmR, 'Meta', meta, 'State', trained.EMA);
fprintf('Saved model card %s\n', card);

% ---------------- samples: faces? the requested expression? copies? ----------------
ev = OceanDL.Generative.SampleEvaluator(double(D.X), D.y, 'NumClasses', K, 'Steps', cfg.EvalSteps, 'Seed', 0);
ceiling = ev.testAccuracy(double(D.Xte), D.yte);
fprintf('Expression classifier on REAL 16x16 test faces: %.1f%% (chance %.1f%%) = ceiling for the check below\n', ...
    100 * ceiling, 100 / K);
if device == "gpu", dm.to('Device', "gpu"); end
labels = repelem((1:K).', cfg.SamplesPerClass);
smp = OceanDL.Generative.Diffusion.Samplers.DDPMAncestral('Steps', cfg.L);
Xref = double(D.Xte(:, :, :, 1:min(numel(labels), end)));
rows = {}; grids = {};
for w = cfg.Guidance
    t0 = tic;
    S = dm.generate(numel(labels), [cfg.Size cfg.Size 1], smp, 'Labels', labels, 'Guidance', w, ...
        'Stream', RandStream('mt19937ar', 'Seed', 7));
    sec = toc(t0);
    S = min(1, max(0, (double(gather(S)) + 1) / 2));
    [~, pred] = max(ev.probabilities(S), [], 1);
    q = ev.evaluate(S, Xref, double(X + 1) / 2);
    perClass = arrayfun(@(k) mean(pred(labels == k) == k), 1:K);
    rows(end+1, :) = [{w, mean(pred(:) == labels(:)), q.FrechetDist, q.NNDistance, q.RefNNDistance, ...
        sec / numel(labels)}, num2cell(perClass)]; %#ok<SAGROW>
    grids{end+1} = gen_tile(S, cfg.SamplesPerClass); %#ok<SAGROW>
end
T = cell2table(rows, 'VariableNames', [{'Guidance', 'RequestedExpression', 'FrechetDist', 'NNDistSamples', ...
    'NNDistRealTest', 'SecPerSample'}, cellstr(D.ClassNames)]);
fprintf('\n=== Textbook DDPM on %dx%d faces (classifier ceiling %.1f%%) ===\n', cfg.Size, cfg.Size, 100 * ceiling);
disp(T);
curve = trained.Curve;                                         % weights live in the card, not here
save(fullfile(out_dir, 'ddpm_affectnet_results.mat'), 'T', 'cfg', 'curve', 'ceiling');
plotAll(grids, trained, T, D.ClassNames, cfg, out_dir);

% ======================================================================== %
function idx = balancedPick(y, n, K, seed)
% n indices, as equal per class as the class sizes allow (small classes give
% all they have; the rest is shared by the others), random within a class.
cnt = accumarray(y(:), 1, [K 1]);
take = zeros(K, 1);
while sum(take) < min(n, sum(cnt))
    open = find(take < cnt);
    q = max(1, floor((min(n, sum(cnt)) - sum(take)) / numel(open)));
    for k = open.'
        take(k) = min(cnt(k), take(k) + q);
        if sum(take) >= n, break; end
    end
end
rs = RandStream('mt19937ar', 'Seed', seed);
idx = [];
for k = 1:K
    members = find(y(:) == k);
    idx = [idx; members(randperm(rs, numel(members), take(k)))]; %#ok<AGROW>
end
idx = sort(idx);
end

function out = train(dmR, X, y, cfg, device)
dm = OceanDL.Training.ModelCard.build(dmR);
dm.to('DataType', "single", 'Device', device);
if device == "gpu", X = gpuArray(X); end
ema = OceanDL.optim.ModelEMA(dm, 'Decay', cfg.EMADecay, 'Warmup', true);
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 2);
n = size(X, 4);
lossAt = @(s) batchLoss(dm, X, y, randi(rs, n, 1, cfg.Batch));
r = OceanDL.Training.fit(dm, lossAt, cfg.Steps, 'LR', cfg.LR, 'EMA', ema, 'Verbose', cfg.Verbose);
H = @OceanDL.Training.ModelCard.toHost;                        % checkpoint must load without a GPU
out = struct('EMA', H(ema.Shadow), 'State', H(dm.stateDict()), 'Curve', gather(r.Curve), ...
    'Seconds', r.Seconds, 'StepSeconds', r.StepSeconds);
end

function L = batchLoss(dm, X, y, idx)
L = dm.forward(X(:, :, :, idx), y(idx));
end

function plotAll(grids, trained, T, classNames, cfg, out_dir)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Textbook DDPM on faces'); fig.Position = [60 60 1300 520];
tl = tiledlayout(fig, 1, numel(grids) + 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for g = 1:numel(grids)
    ax = nexttile(tl); image(ax, grids{g} * 255); colormap(ax, gray(256)); axis(ax, 'image', 'off');
    title(ax, sprintf('guidance w = %g: %.0f%% requested expression', T.Guidance(g), 100 * T.RequestedExpression(g)), ...
        'Color', viz.Colors.Text);
    ylabel(ax, strjoin(classNames, ' / '));
end
ax = nexttile(tl); viz.styleAxes(ax);
plot(ax, movmean(trained.Curve, max(1, round(numel(trained.Curve) / 50))), 'LineWidth', 1.6);
xlabel(ax, 'step'); ylabel(ax, 'L_{simple} (Eq. 2.2.10)'); title(ax, 'Training loss', 'Color', viz.Colors.Text);
viz.export(fig, fullfile(out_dir, sprintf('ddpm_affectnet%d_samples', cfg.Size)));
end
