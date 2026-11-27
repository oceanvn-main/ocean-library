function R = gen_compare_vae_diffusion(D, cfg)
% GEN_COMPARE_VAE_DIFFUSION  A simple VAE vs a simple diffusion model on D.
%
%   VAE        OceanDL.Generative.VAE: ConvEncoder / ConvDecoder (cfg.VAEWidths,
%              latent (H/f) x (W/f) x cfg.LatentChannels), Bernoulli likelihood,
%              KL warm-up; ONE decoder pass per sample.
%   Diffusion  OceanDL.Generative.Diffusion.DiffusionModel: small UNet2D
%              (cfg.UNetWidths), rectified flow (velocity target, logit-normal
%              times, the SD3 / FLUX recipe), weight EMA; images scaled to
%              [-1, 1]; sampled with Euler at several step counts and Heun --
%              quality as a function of network evaluations (NFE).
%   Both unconditional; same data, training steps and batch size.
%
%   Measured: samples scored by gen_evaluator's classifier (confidence, class
%   coverage, classifier-feature Frechet distance, nearest-neighbour novelty,
%   sharpness); NFE and wall-clock per sample; seconds per training step;
%   parameters; the VAE's test negative ELBO (the diffusion model's
%   likelihood would need an ODE trace estimate and is not computed here).

ev = gen_evaluator(D, cfg.EvalSteps, cfg.CkptDir);
H = D.Size; Xref = D.Xte(:, :, :, 1:min(cfg.Samples, end));

% ---------------- VAE ----------------
f = 2 ^ (numel(cfg.VAEWidths) - 1);
makeVae = @() OceanDL.Generative.VAE( ...
    OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', cfg.VAEWidths, 'OutChannels', 2 * cfg.LatentChannels, ...
        'Groups', 8, 'MidAttention', false, 'Seed', cfg.Seed), ...
    OceanDL.Vision.ConvDecoder('LatentChannels', cfg.LatentChannels, 'Widths', cfg.VAEWidths, 'OutChannels', 1, ...
        'Groups', 8, 'MidAttention', false, 'Seed', cfg.Seed + 1), ...
    'Likelihood', "bernoulli", 'FreeBits', cfg.FreeBits, 'Seed', cfg.Seed + 2).to('DataType', "single");
fprintf('VAE: train, or load its checkpoint\n');
vs = struct('Widths', cfg.VAEWidths, 'LatentChannels', cfg.LatentChannels, 'Steps', cfg.Steps, 'Batch', cfg.Batch, ...
    'LR', cfg.LR, 'KLWarmup', cfg.KLWarmup, 'FreeBits', cfg.FreeBits, 'Seed', cfg.Seed, 'NTrain', numel(D.ytr));
tv = OceanDL.Training.Checkpoint.run(cfg.CkptDir, sprintf('%s_vae_s%d_b%d_%s', D.Name, cfg.Steps, cfg.Batch, gen_hash(vs)), ...
    @() fitVae(makeVae, D, cfg));
vae = makeVae(); vae.loadStateDict(tv.State); vae.eval(); vae.Beta = 1;

% ---------------- diffusion ----------------
makeNet = @() OceanDL.Vision.UNet2D('InChannels', 1, 'Widths', cfg.UNetWidths, 'ResPerLevel', 1, 'Groups', 8, 'Seed', cfg.Seed);
makeDm = @(net) OceanDL.Generative.Diffusion.DiffusionModel(net, OceanDL.Generative.Diffusion.Schedules.RectifiedFlow(), ...
    'Target', "velocity", 'Seed', cfg.Seed + 3);
fprintf('Diffusion (rectified flow, U-Net %s): train, or load its checkpoint\n', mat2str(cfg.UNetWidths));
ds = struct('Widths', cfg.UNetWidths, 'Steps', cfg.DiffusionSteps, 'Batch', cfg.Batch, 'LR', cfg.LR, ...
    'Seed', cfg.Seed, 'NTrain', numel(D.ytr));
td = OceanDL.Training.Checkpoint.run(cfg.CkptDir, sprintf('%s_flow_s%d_b%d_%s', D.Name, cfg.DiffusionSteps, cfg.Batch, gen_hash(ds)), ...
    @() fitDiffusion(makeNet, makeDm, D, cfg));
unet = makeNet(); unet.to('DataType', "single"); unet.loadStateDict(td.EMA);            % sample with EMA weights
dm = makeDm(unet);

% ---------------- evaluation ----------------
rows = {}; S = struct();
stream = @() RandStream('mt19937ar', 'Seed', cfg.Seed + 7);
t0 = tic; Sv = vae.sample(cfg.Samples, [H / f, H / f, cfg.LatentChannels], stream()); sv = toc(t0) / cfg.Samples;
q = ev.evaluate(Sv, Xref, D.Xtr);
negElbo = vaeTestElbo(vae, D.Xte);
rows(end+1, :) = row("VAE", "decoder", 1, sv, q, tv.StepSeconds, tv.Params, negElbo);
S.VAE = Sv;
samplers = {"Euler", OceanDL.Generative.Diffusion.Samplers.EulerODE('Steps', 4)
            "Euler", OceanDL.Generative.Diffusion.Samplers.EulerODE('Steps', 8)
            "Euler", OceanDL.Generative.Diffusion.Samplers.EulerODE('Steps', 16)
            "Euler", OceanDL.Generative.Diffusion.Samplers.EulerODE('Steps', 32)
            "Heun",  OceanDL.Generative.Diffusion.Samplers.HeunODE('Steps', 16)};
for k = 1:size(samplers, 1)
    [sname, smp] = samplers{k, :};
    t0 = tic; [Sd, info] = dm.generate(cfg.Samples, [H H 1], smp, 'Stream', stream()); sd = toc(t0) / cfg.Samples;
    Sd = (Sd + 1) / 2;                                                                   % back to [0, 1]
    q = ev.evaluate(Sd, Xref, D.Xtr);
    rows(end+1, :) = row("Diffusion", sprintf('%s %d', sname, smp.Steps), info.NFE, sd, q, td.StepSeconds, td.Params, NaN); %#ok<AGROW>
    S.(sprintf('%s%d', sname, smp.Steps)) = Sd;
end
T = cell2table(rows, 'VariableNames', {'Model', 'Sampler', 'NFE', 'MsPerSample', 'Confidence', 'ClassEntropy', ...
    'ClassesHit', 'FrechetDist', 'NNDist', 'RefNNDist', 'Sharpness', 'SecPerTrainStep', 'Params', 'TestNegELBO'});
fprintf('\n=== %s: VAE vs diffusion (evaluator accuracy %.1f%%) ===\n', upper(D.Name), 100 * ev.Accuracy);
disp(T);
R = struct('Table', T, 'VAETrain', rmfield(tv, 'State'), 'DiffusionTrain', rmfield(td, {'State', 'EMA'}), ...
    'EvaluatorAccuracy', ev.Accuracy, 'Config', cfg);
save(fullfile(cfg.OutDir, sprintf('vae_vs_diffusion_%s.mat', D.Name)), 'R');
plotAll(D, S, T, tv, td, cfg);
end

% ======================================================================== %
function r = row(model, sampler, nfe, sec, q, stepSec, params, negElbo)
r = {model, string(sampler), nfe, 1e3 * sec, q.Confidence, q.ClassEntropy, q.ClassesHit, q.FrechetDist, ...
    q.NNDistance, q.RefNNDistance, q.Sharpness, stepSec, params, negElbo};
end

function out = fitVae(make, D, cfg)
net = make();
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 1);
n = size(D.Xtr, 4); warm = max(1, round(cfg.KLWarmup * cfg.Steps));
r = OceanDL.Training.fit(net, @(s) net.forward(D.Xtr(:, :, :, randi(rs, n, 1, cfg.Batch))), cfg.Steps, ...
    'LR', cfg.LR, 'OnStep', @(s) setBeta(net, min(1, s / warm)), 'Record', @() net.LastStats.NegELBO, 'Verbose', cfg.Verbose);
out = struct('State', net.stateDict(), 'Curve', r.Curve, 'Records', r.Records, 'Seconds', r.Seconds, ...
    'StepSeconds', r.StepSeconds, 'Params', net.numParameters());
end

function setBeta(net, b)
net.Beta = b;
end

function out = fitDiffusion(makeNet, makeDm, D, cfg)
net = makeNet(); dm = makeDm(net);
dm.to('DataType', "single");
ema = OceanDL.optim.ModelEMA(net, 'Decay', 0.999, 'Warmup', true);
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 1);
X = 2 * D.Xtr - 1; n = size(X, 4);
r = OceanDL.Training.fit(dm, @(s) dm.forward(X(:, :, :, randi(rs, n, 1, cfg.Batch))), cfg.DiffusionSteps, ...
    'LR', cfg.LR, 'EMA', ema, 'Verbose', cfg.Verbose);
out = struct('State', net.stateDict(), 'EMA', ema.Shadow, 'Curve', r.Curve, 'Seconds', r.Seconds, ...
    'StepSeconds', r.StepSeconds, 'Params', net.numParameters());
end

function negElbo = vaeTestElbo(vae, X)
n = size(X, 4); tot = 0;
for b = 1:256:n
    k = b:min(n, b + 255);
    OceanDL.Autograd.GradMode.noGrad(@() vae.forward(X(:, :, :, k)));
    tot = tot + vae.LastStats.NegELBO * numel(k);
end
negElbo = tot / n;
end

function plotAll(D, S, T, tv, td, cfg)
viz = OceanViz.Core('Theme', 'Dark');
% 1. samples: real | VAE | diffusion Euler 32 | diffusion Euler 4
fig = viz.createFigure('VAE vs diffusion samples'); fig.Position = [40 60 1500 420];
tl = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
panels = {D.Xte(:, :, :, 1:36), 'real (test)'; S.VAE(:, :, :, 1:36), 'VAE (1 NFE)'; ...
          S.Euler32(:, :, :, 1:36), 'flow, Euler 32 NFE'; S.Euler4(:, :, :, 1:36), 'flow, Euler 4 NFE'};
for p = 1:4
    ax = nexttile(tl); image(ax, gen_tile(panels{p, 1}, 6) * 255); colormap(ax, gray(256)); axis(ax, 'image', 'off');
    title(ax, panels{p, 2}, 'Color', viz.Colors.Text);
end
viz.export(fig, fullfile(cfg.OutDir, sprintf('vae_vs_diffusion_%s_samples', D.Name)));
% 2. quality vs NFE + training curves
fig = viz.createFigure('Quality vs compute'); fig.Position = [40 60 1400 420];
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
d = T(T.Model == "Diffusion" & startsWith(T.Sampler, "Euler"), :); v = T(T.Model == "VAE", :); h = T(startsWith(T.Sampler, "Heun"), :);
for spec = {{'FrechetDist', 'classifier-feature Frechet distance (lower = better)'}, {'Confidence', 'classifier confidence (higher = better)'}}
    ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
    fld = spec{1}{1};
    plot(ax, d.NFE, d.(fld), '-o', 'Color', [0.30 0.75 0.95], 'LineWidth', 1.8, 'DisplayName', 'flow, Euler');
    plot(ax, h.NFE, h.(fld), 's', 'Color', [0.55 0.85 0.35], 'MarkerSize', 9, 'LineWidth', 1.8, 'DisplayName', 'flow, Heun');
    plot(ax, v.NFE, v.(fld), 'd', 'Color', [0.98 0.60 0.20], 'MarkerSize', 10, 'LineWidth', 1.8, 'DisplayName', 'VAE');
    set(ax, 'XScale', 'log'); xlabel(ax, 'network evaluations per sample (NFE)'); ylabel(ax, spec{1}{2});
    legend(ax, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
end
ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
w = @(c) movmean(c, max(1, round(numel(c) / 50)));
yyaxis(ax, 'left'); plot(ax, w(tv.Records(:, 1)), 'Color', [0.98 0.60 0.20]); ylabel(ax, 'VAE negative ELBO (nats)');
yyaxis(ax, 'right'); plot(ax, w(td.Curve), 'Color', [0.30 0.75 0.95]); ylabel(ax, 'flow-matching loss');
xlabel(ax, 'step'); title(ax, 'Training', 'Color', viz.Colors.Text);
viz.export(fig, fullfile(cfg.OutDir, sprintf('vae_vs_diffusion_%s_quality', D.Name)));
end
