function R = gen_compare_vae_hvae(D, cfg)
% GEN_COMPARE_VAE_HVAE  VAE vs 2-layer hierarchical VAE on dataset D.
%
%   CONTROLLED COMPARISON: both models are OceanDL.Generative.HierarchicalVAE
%   with IDENTICAL encoder / decoder networks; the only difference is one
%   extra latent layer at the lowest resolution:
%     VAE   LatentsPerLevel = cfg.VAELatents   (one latent, prior N(0, I)
%                                               -- provably the plain VAE, see
%                                               HierarchicalVAETest)
%     HVAE  LatentsPerLevel = cfg.HVAELatents  (top latent N(0, I) + a learned
%                                               prior p(z1 | z2) for the lower one)
%   Same data, steps, batch, optimiser, KL warm-up, free bits and seeds.
%
%   Measured (test set): negative ELBO (nats / image, bits / dim), its split
%   into reconstruction and KL per layer, active units per layer, posterior-
%   mean reconstruction MSE; samples scored by gen_evaluator's classifier
%   (confidence, class coverage, classifier-feature Frechet distance,
%   nearest-neighbour novelty, sharpness) at temperatures 1 and 0.8; wall-clock
%   per training step and per sampled image; parameter counts.
%   Every trained model is checkpointed (resumable after a crash) and saved as
%   an OceanDL.Training.ModelCard (cfg.OutDir/cards/<dataset>_<model>.mat) for
%   the Model Lab.
%
%   OPTIONAL cfg fields (defaults in brackets):
%     Binarize [false]  Bernoulli likelihood needs 0/1 pixels: training batches are
%                       DYNAMICALLY binarised (pixel ~ Bernoulli(intensity), fresh
%                       each time), the test set once with a fixed seed -> the
%                       negative ELBO is a true bound on -log p(x) (no grey-pixel floor).
%     FreeBitsPer ["layer"]  "layer" | "channel" | "unit" (HierarchicalVAE)
%     Device ["cpu"]   "cpu" | "gpu" | "auto" (training only; evaluation on CPU)

cfg = withDefaults(cfg);
device = cfg.Device;
if device == "auto", device = "cpu"; if OceanDL.Device.gpuAvailable(), device = "gpu"; end, end
fprintf('Training device: %s | binarised Bernoulli: %d | free bits %.3g per %s\n', device, cfg.Binarize, cfg.FreeBits, cfg.FreeBitsPer);
Xte = D.Xte;                                                    % the test data the bound is measured on
if cfg.Binarize, Xte = single(rand(RandStream('mt19937ar', 'Seed', cfg.Seed + 11), size(Xte)) < Xte); end
models = {"VAE", cfg.VAELatents; "HVAE", cfg.HVAELatents};
recipes = cell(1, 2);
ev = gen_evaluator(D, cfg.EvalSteps, cfg.CkptDir);
nets = cell(1, 2); train = cell(1, 2);
for m = 1:2
    [name, lat] = models{m, :};
    recipes{m} = OceanDL.Training.ModelCard.recipe("OceanDL.Generative.HierarchicalVAE", {}, ...
        {'Widths', cfg.Widths, 'LatentsPerLevel', lat, 'LatentChannels', cfg.LatentChannels, 'Likelihood', "bernoulli", ...
         'FreeBits', cfg.FreeBits, 'FreeBitsPer', cfg.FreeBitsPer, 'Groups', 8, 'Seed', cfg.Seed}); %#ok<AGROW>
    make = @() OceanDL.Training.ModelCard.build(recipes{m}).to('DataType', "single");
    settings = struct('Recipe', recipes{m}, 'Steps', cfg.Steps, 'Batch', cfg.Batch, 'LR', cfg.LR, ...
        'KLWarmup', cfg.KLWarmup, 'Binarize', cfg.Binarize, 'Seed', cfg.Seed, 'NTrain', numel(D.ytr), 'Code', 2);
    key = sprintf('%s_%s_s%d_b%d_%s', D.Name, name, cfg.Steps, cfg.Batch, gen_hash(settings));   % every setting in the name
    fprintf('%s  (LatentsPerLevel %s): train, or load its checkpoint\n', name, mat2str(lat));
    train{m} = OceanDL.Training.Checkpoint.run(cfg.CkptDir, key, @() fitVae(make, D, cfg, device));
    nets{m} = make(); nets{m}.loadStateDict(train{m}.State);
end

% ---------------- test-set evaluation ----------------
rows = {};
for m = 1:2
    net = nets{m}; net.eval(); net.Beta = 1;
    [negElbo, recon, klLayer] = testElbo(net, Xte);
    dims = D.Size ^ 2;
    au = net.activeUnits(Xte(:, :, :, 1:min(1000, end)));
    Xr = net.reconstruct(Xte);
    mse = mean((double(Xr) - double(Xte)) .^ 2, 'all');
    for T = [1 0.8]
        t0 = tic;
        S = net.sample(cfg.Samples, [D.Size D.Size], 'Temperature', T, 'Stream', RandStream('mt19937ar', 'Seed', cfg.Seed + 7));
        sec = toc(t0) / cfg.Samples;
        q = ev.evaluate(S, D.Xte(:, :, :, 1:min(cfg.Samples, end)), D.Xtr);
        rows(end+1, :) = {models{m, 1}, T, negElbo, negElbo / (dims * log(2)), recon, sum(klLayer), mat2str(round(klLayer, 1)), ...
            mat2str(au), mse, q.Confidence, q.ClassEntropy, q.ClassesHit, q.FrechetDist, q.NNDistance, q.RefNNDistance, ...
            q.Sharpness, train{m}.StepSeconds, 1e3 * sec, train{m}.Params}; %#ok<AGROW>
        if T == 1, samples{m} = S; end %#ok<AGROW>
    end
    recons{m} = Xr(:, :, :, 1:12); %#ok<AGROW>
    card = fullfile(cfg.OutDir, 'cards', sprintf('%s_%s.mat', lower(D.Name), lower(models{m, 1})));
    meta = struct('Dataset', string(D.Name), 'ImageSize', [D.Size D.Size 1], 'ClassNames', D.ClassNames, ...
        'DataRange', [0 1], 'Binarized', cfg.Binarize, 'TrainSteps', cfg.Steps, 'Batch', cfg.Batch, ...
        'TestNegELBO', negElbo, 'KLPerLayer', klLayer, 'ActiveUnits', au, 'About', models{m, 1} + " " + mat2str(models{m, 2}));
    OceanDL.Training.ModelCard.save(card, net, recipes{m}, 'Meta', meta);
    fprintf('Saved model card %s\n', card);
end
T = cell2table(rows, 'VariableNames', {'Model', 'Temperature', 'TestNegELBO', 'BitsPerDim', 'Recon', 'KL', 'KLPerLayer', ...
    'ActiveUnits', 'ReconMSE', 'Confidence', 'ClassEntropy', 'ClassesHit', 'FrechetDist', 'NNDist', 'RefNNDist', ...
    'Sharpness', 'SecPerStep', 'MsPerSample', 'Params'});
fprintf('\n=== %s: VAE vs 2-layer HVAE (evaluator accuracy %.1f%%) ===\n', upper(D.Name), 100 * ev.Accuracy);
disp(T);
R = struct('Table', T, 'Train', {train}, 'EvaluatorAccuracy', ev.Accuracy, 'Config', cfg);
save(fullfile(cfg.OutDir, sprintf('vae_vs_hvae_%s.mat', D.Name)), 'R');
plotAll(D, samples, recons, train, T, cfg, Xte);
end

% ======================================================================== %
function out = fitVae(make, D, cfg, device)
net = make();
net.to('Device', device);
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 1);
rb = RandStream('mt19937ar', 'Seed', cfg.Seed + 3);             % binarisation noise
X = D.Xtr; if device == "gpu", X = gpuArray(X); end
n = size(X, 4);
warm = max(1, round(cfg.KLWarmup * cfg.Steps));
r = OceanDL.Training.fit(net, @(s) net.forward(batchOf(X, randi(rs, n, 1, cfg.Batch), cfg.Binarize, rb)), cfg.Steps, ...
    'LR', cfg.LR, 'OnStep', @(s) setBeta(net, min(1, s / warm)), ...
    'Record', @() [net.LastStats.NegELBO, net.LastStats.Recon, net.LastStats.KL], 'Verbose', cfg.Verbose);
out = struct('State', OceanDL.Training.ModelCard.toHost(net.stateDict()), 'Curve', r.Curve, 'Records', r.Records, ...
    'Seconds', r.Seconds, 'StepSeconds', r.StepSeconds, 'Params', net.numParameters(), 'Device', device);
end

function B = batchOf(X, idx, binarize, rb)
% Dynamic binarisation: each pixel ~ Bernoulli(intensity), fresh on every use.
B = X(:, :, :, idx);
if binarize, B = cast(cast(rand(rb, size(B)), 'like', B) < B, 'like', B); end
end

function cfg = withDefaults(cfg)
d = struct('Binarize', false, 'FreeBitsPer', "layer", 'Device', "cpu");
f = fieldnames(d);
for k = 1:numel(f), if ~isfield(cfg, f{k}), cfg.(f{k}) = d.(f{k}); end, end
end

function setBeta(net, b)
net.Beta = b;
end

function [negElbo, recon, klLayer] = testElbo(net, X)
% Single-sample negative ELBO averaged over the whole test set (Beta = 1);
% posterior samples come from the model's own seeded stream (reproducible).
n = size(X, 4); tot = 0; rec = 0; kl = 0;
for b = 1:256:n
    k = b:min(n, b + 255);
    OceanDL.Autograd.GradMode.noGrad(@() net.forward(X(:, :, :, k)));
    s = net.LastStats;
    tot = tot + s.NegELBO * numel(k); rec = rec + s.Recon * numel(k); kl = kl + s.KLPerLayer * numel(k);
end
negElbo = tot / n; recon = rec / n; klLayer = kl / n;
end

function plotAll(D, samples, recons, train, T, cfg, Xte)
viz = OceanViz.Core('Theme', 'Dark');
tile = @(X, c) gen_tile(X, c);
% 1. samples
fig = viz.createFigure('VAE vs HVAE samples'); fig.Position = [60 60 1300 480];
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
panels = {D.Xte(:, :, :, 1:36), 'real (test)'; samples{1}(:, :, :, 1:36), 'VAE samples'; samples{2}(:, :, :, 1:36), 'HVAE (2 layers) samples'};
for p = 1:3
    ax = nexttile(tl); image(ax, tile(panels{p, 1}, 6) * 255); colormap(ax, gray(256)); axis(ax, 'image', 'off');
    title(ax, panels{p, 2}, 'Color', viz.Colors.Text);
end
viz.export(fig, fullfile(cfg.OutDir, sprintf('vae_vs_hvae_%s_samples', D.Name)));
% 2. reconstructions
fig = viz.createFigure('Reconstructions'); fig.Position = [60 60 1100 330];
ax = axes(fig);
grid = [gen_tile(Xte(:, :, :, 1:12), 12); gen_tile(recons{1}, 12); gen_tile(recons{2}, 12)];
image(ax, grid * 255); colormap(ax, gray(256)); axis(ax, 'image', 'off');
title(ax, 'rows: test images | VAE reconstruction | HVAE reconstruction', 'Color', viz.Colors.Text);
viz.export(fig, fullfile(cfg.OutDir, sprintf('vae_vs_hvae_%s_recon', D.Name)));
% 3. training curves + KL per layer + key metrics
fig = viz.createFigure('Training and metrics'); fig.Position = [60 60 1400 420];
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
pal = [0.30 0.75 0.95; 0.98 0.60 0.20];
ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
names = ["VAE", "HVAE"];
for m = 1:2
    w = max(1, round(numel(train{m}.Curve) / 50));
    plot(ax, movmean(train{m}.Records(:, 1), w), 'Color', pal(m, :), 'LineWidth', 1.6, 'DisplayName', names(m));
end
xlabel(ax, 'step'); ylabel(ax, 'train negative ELBO (nats / image)');
legend(ax, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background); title(ax, 'Training', 'Color', viz.Colors.Text);
ax = nexttile(tl); viz.styleAxes(ax);
t1 = T(T.Temperature == 1, :);
b = bar(ax, categorical(t1.Model), [t1.Recon, t1.KL], 'stacked');
b(1).FaceColor = [0.45 0.45 0.55]; b(2).FaceColor = [0.98 0.60 0.20];
ylabel(ax, 'test negative ELBO (nats / image)');
legend(ax, ["reconstruction", "KL (all layers)"], 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'Location', 'southoutside');
title(ax, sprintf('Test bound (lower = better): %s', strjoin(compose('%s %.1f', t1.Model, t1.TestNegELBO), ', ')), ...
    'Color', viz.Colors.Text, 'Interpreter', 'none');
ax = nexttile(tl); viz.styleAxes(ax);
metrics = [t1.Confidence, t1.ClassEntropy, t1.Sharpness];
bar(ax, categorical(["confidence", "class entropy", "sharpness"]), metrics.');
legend(ax, t1.Model, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
title(ax, sprintf('Samples (T = 1); Frechet dist: %s', strjoin(compose('%s %.2f', t1.Model, t1.FrechetDist), ', ')), ...
    'Color', viz.Colors.Text, 'Interpreter', 'none');
viz.export(fig, fullfile(cfg.OutDir, sprintf('vae_vs_hvae_%s_metrics', D.Name)));
end
