% EXP_VIT_CV  Vision Transformers vs CNNs (Szeliski, Computer Vision 2nd ed.,
%   Ch. 5, Sec. 5.5.3; Dosovitskiy et al. 2021) -- claims MEASURED on real data.
%
%   E1  DATA EFFICIENCY (Szeliski takeaway 1: CNNs exploit locality / translation
%       equivariance; Transformers model global context but need more data).
%       ViT vs a CNN of similar size at 10 / 25 / 50 / 100 % of the training
%       set, SAME number of optimiser steps per run (equal compute), on
%         USPS digits (16 x 16 grey)  and  AffectNet faces (RGB, ImageSize option).
%   E2  ViT ABLATIONS on USPS: positions learned / sincos / none, pooling
%       cls / mean, Pre-LN vs Post-LN each with and without warmup
%       (takeaway 3: residual + LayerNorm placement), patch size 2 / 4 / 8.
%   E3  SWIN vs ViT COMPUTE SCALING (Liu et al. 2021): wall-clock time per
%       training step (forward + backward) at image sizes 32 .. 128 on real
%       face images -- global attention is O(T^2) in the number of tokens T,
%       window attention O(T M^2); log-log slopes are fitted and reported.
%       E3_swin_affectnet: Swin accuracy on the clean faces (same budget as E1).
%   E4  MAE PRE-TRAINING (He et al. 2022; takeaway 2): masked-patch
%       reconstruction on the face training images WITHOUT labels, then
%       fine-tune the backbone with 10% / 100% of the labels vs the same ViT
%       from scratch (same steps).
%   E5  ATTENTION ROLLOUT (Abnar & Zuidema 2020) of the AffectNet ViT on test
%       faces: which regions reach the [CLS] decision?
%
%   DATA IMPORT: USPS via UspsDigitsReader + DimFormat (row-major pixels ->
%   SSCB). AffectNet via AffectNetReader with the CLEAN 6-CLASS PROTOCOL
%   (user decision, 2026-09-23): the provided split is unusable (1,688 test
%   files are copies of train files -- every "sad" test image; happy/neutral
%   are FFHQ images, a source confound), so: pool Train+Test, keep one copy
%   of each distinct image (MD5), AffectNet JPEGs only, labels where folder ==
%   labels.csv, classes anger/contempt/disgust/fear/sad/surprise, seeded
%   stratified 80/20 split. Resized to cfg.AffectSize.
%   Normalisation: per-channel mean/std of the (subsampled) training set.
%
%   TRAINING (identical for every model): AdamW, weight decay 0.05 (not on
%   norms/biases/positions/[CLS]), warmup 5% + cosine, label smoothing 0.1,
%   batch 64, grad-norm clip 1, augmentation (random crop; + flip and
%   photometric jitter for faces only -- a mirrored digit is another digit).
%
%   RUN: all parts (hours on CPU) or a subset via cfg.Parts; OCEAN_QUICK=1 for a
%   smoke run (writes to Results/vit_cv/quick, never over full results).
%   RESUMABLE: every finished training run is saved (atomically) to
%   <out_dir>/checkpoints/<run key>.mat; rerunning the script reloads finished
%   runs, so a crash / power cut loses only the run in progress. Delete the
%   checkpoint folder to retrain from scratch. Each run resets the seeded
%   augmentation stream, so a resumed run equals an uninterrupted one.
%
%   Author: Ocean Library  |  2026-09-23

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(project_root);
out_dir = fullfile(script_dir, 'Results', 'vit_cv');
if ~isfolder(out_dir), mkdir(out_dir); end
quick = ~isempty(getenv('OCEAN_QUICK'));
if quick
    out_dir = fullfile(out_dir, 'quick'); if ~isfolder(out_dir), mkdir(out_dir); end
end

cfg = struct();
cfg.Parts        = ["E1_usps", "E2_usps", "E1_affectnet", "E5_rollout"];
cfg.Fractions    = [0.10 0.25 0.50 1.00];
cfg.UspsSteps    = 1500;         % optimiser steps per USPS run
cfg.AffectSteps  = 2000;         % optimiser steps per AffectNet run
cfg.AffectSize   = [48 48];      % resolution option ([96 96] = native)
cfg.AffectLabels = "agreeing";   % "agreeing" | "folder" | "csv"
cfg.AffectClasses = ["anger" "contempt" "disgust" "fear" "sad" "surprise"];   % clean protocol: no FFHQ-confounded classes
cfg.AffectTrainCap = Inf;        % cap training images (compute budget)
cfg.ScaleSizes   = [32 64 96 128];   % E3 resolutions (multiples of PatchSize*WindowSize*2)
cfg.ScaleBatch   = 4; cfg.ScaleReps = 5;
cfg.MaeSteps     = 3000;         % E4 pre-training steps (labels unused)
cfg.MaeFractions = [0.10 1.00];  % E4 label fractions for fine-tuning
cfg.Batch = 64; cfg.LR = 1e-3; cfg.WD = 0.05; cfg.Smoothing = 0.1; cfg.Clip = 1; cfg.Seed = 1;
if quick
    cfg.Fractions = [0.25 1.00]; cfg.UspsSteps = 30; cfg.AffectSteps = 20; cfg.AffectTrainCap = 600;
    cfg.ScaleSizes = [32 64]; cfg.ScaleReps = 2; cfg.MaeSteps = 20;
end
cfg.CkptDir = fullfile(out_dir, 'checkpoints');
if ~isfolder(cfg.CkptDir), mkdir(cfg.CkptDir); end
env = getenv('OCEAN_VIT_PARTS');                      % e.g. OCEAN_VIT_PARTS="E1_usps,E2_usps"
if ~isempty(env), cfg.Parts = string(strsplit(env, ',')); end
results = struct();

% ============================== USPS =======================================
if any(startsWith(cfg.Parts, "E1_usps") | startsWith(cfg.Parts, "E2_usps"))
    [Xu, yu] = importUsps("train"); [Xut, yut] = importUsps("test");
    if quick, Xut = Xut(:, :, :, 1:300); yut = yut(1:300); end
    uspsAug = OceanDL.Vision.ImageAugment('Padding', 1, 'Seed', cfg.Seed);
    uspsViT = @(varargin) OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 64, ...
        'Depth', 4, 'Heads', 4, 'NumClasses', 10, 'Seed', cfg.Seed, varargin{:});
    uspsCNN = @() cnn(1, [32 64], 16, 10, cfg.Seed);
end

if any(cfg.Parts == "E1_usps")
    results.E1_usps = dataEfficiency("USPS", Xu, yu, Xut, yut, {@() uspsViT(), uspsCNN}, ["ViT", "CNN"], ...
        cfg.Fractions, cfg.UspsSteps, uspsAug, cfg);
end

if any(cfg.Parts == "E2_usps")
    variants = {
        "ViT (learned pos, cls, pre-LN)",   @() uspsViT(),                           true
        "positions: sincos (fixed)",        @() uspsViT('Positions', "sincos"),      true
        "positions: none",                  @() uspsViT('Positions', "none"),        true
        "pooling: mean",                    @() uspsViT('Pooling', "mean"),          true
        "pre-LN, no warmup",                @() uspsViT(),                           false
        "post-LN, warmup",                  @() uspsViT('Norm', "post"),             true
        "post-LN, no warmup",               @() uspsViT('Norm', "post"),             false
        "patch 2 (64 tokens)",              @() uspsViT('PatchSize', 2),             true
        "patch 8 (4 tokens)",               @() uspsViT('PatchSize', 8),             true };
    E2 = struct('Name', cellstr([variants{:, 1}]), 'TestAcc', 0, 'Seconds', 0, 'Params', 0, 'Curve', []);
    for v = 1:size(variants, 1)
        key = sprintf('E2_%s_s%d', variants{v, 1}, cfg.UspsSteps);
        r = checkpointed(cfg, key, @() runOnce(variants{v, 2}, Xu, yu, Xut, yut, cfg.UspsSteps, uspsAug, cfg, variants{v, 3}));
        E2(v).TestAcc = r.Acc; E2(v).Seconds = r.Sec; E2(v).Params = r.Params; E2(v).Curve = r.Curve;
        fprintf('E2 %-32s test acc %.2f%%  (%d params, %.0fs)\n', variants{v, 1}, 100 * r.Acc, r.Params, r.Sec);
    end
    results.E2_usps = E2;
end

% ============================== AffectNet ==================================
if any(ismember(cfg.Parts, ["E1_affectnet", "E3_swin_affectnet", "E4_mae", "E5_rollout"]))
    cacheDir = fullfile(out_dir, 'cache');
    protocol = {'Split', "resplit", 'Sources', "affectnet", 'Labels', cfg.AffectLabels, 'Classes', cfg.AffectClasses, ...
        'TestFraction', 0.2, 'SplitSeed', 0, 'ImageSize', cfg.AffectSize, 'CacheDir', cacheDir};
    [Xa, ya, infoA] = OceanData.Adapters.AffectNetReader.read("train", protocol{:}, 'MaxImages', cfg.AffectTrainCap, 'Seed', cfg.Seed);
    [Xat, yat, infoT] = OceanData.Adapters.AffectNetReader.read("test", protocol{:}, 'MaxImages', ifelse(quick, 300, Inf), 'Seed', cfg.Seed);
    fprintf('AffectNet clean protocol (%d classes, %s labels, %s): train %d, test %d; %d duplicate files removed from the pool\n', ...
        numel(infoA.Classes), cfg.AffectLabels, mat2str(cfg.AffectSize), numel(ya), numel(yat), infoA.Report.DuplicatesRemoved);
    disp(infoA.Report.PerClassSourceEligible); disp(infoT.Report.PerClassSourceEligible);
    mu = mean(Xa, [1 2 4]); sd = std(Xa, 0, [1 2 4]);                 % per-channel statistics (train only)
    Xa = double((Xa - mu) ./ sd); Xat = double((Xat - mu) ./ sd);
    OceanDL.Layout.DimFormat.validate(Xa, "SSCB", 'FeatureDim', 3, 'BatchSize', numel(ya));
    faceAug = OceanDL.Vision.ImageAugment('Padding', 3, 'FlipProb', 0.5, 'Brightness', 0.2, 'Contrast', 0.2, 'Seed', cfg.Seed);
    p = cfg.AffectSize(1) / 6;                                          % 6 x 6 patch grid at any resolution
    faceViT = @() OceanDL.Vision.ViT('ImageSize', cfg.AffectSize, 'Channels', 3, 'PatchSize', p, 'Dim', 96, ...
        'Depth', 4, 'Heads', 4, 'NumClasses', numel(infoA.Classes), 'DropPath', 0.1, 'Seed', cfg.Seed);
    faceCNN = @() cnn(3, [32 64 128], cfg.AffectSize(1), numel(infoA.Classes), cfg.Seed);
end

if any(cfg.Parts == "E1_affectnet")
    results.E1_affectnet = dataEfficiency("AffectNet", Xa, ya, Xat, yat, {faceViT, faceCNN}, ["ViT", "CNN"], ...
        cfg.Fractions, cfg.AffectSteps, faceAug, cfg);
    results.AffectReport = struct('Train', infoA.Report, 'Test', infoT.Report);
end

if any(cfg.Parts == "E3_swin_affectnet")
    % 48 / 4 = 12 x 12 grid, window 3 -> 16 windows; merged 6 x 6 -> 4 windows.
    faceSwin = @() OceanDL.Vision.Swin('Channels', 3, 'PatchSize', cfg.AffectSize(1) / 12, 'Dim', 64, ...
        'Depths', [2 2], 'Heads', [2 4], 'WindowSize', 3, 'NumClasses', numel(infoA.Classes), 'DropPath', 0.1, 'Seed', cfg.Seed);
    results.E3_swin_affectnet = dataEfficiency("AffectNet", Xa, ya, Xat, yat, {faceSwin}, "Swin", ...
        cfg.Fractions(end), cfg.AffectSteps, faceAug, cfg);
end

if any(cfg.Parts == "E4_mae")
    maeViT = @() OceanDL.Vision.ViT('ImageSize', cfg.AffectSize, 'Channels', 3, 'PatchSize', p, 'Dim', 96, ...
        'Depth', 4, 'Heads', 4, 'NumClasses', numel(infoA.Classes), 'DropPath', 0.1, ...
        'Positions', "sincos", 'Pooling', "mean", 'Seed', cfg.Seed);     % MAE backbone: no [CLS], fixed positions
    backbone = maeViT();
    mae = OceanDL.Vision.MaskedAutoencoder(backbone, 'MaskRatio', 0.75, 'DecoderDim', 64, 'DecoderDepth', 2, ...
        'DecoderHeads', 4, 'NormPixLoss', true, 'Seed', cfg.Seed);     % faces: natural images -> normalised targets
    pre = checkpointed(cfg, sprintf('E4_mae_pretrain_s%d', cfg.MaeSteps), @() pretrainMae(mae, Xa, cfg.MaeSteps, faceAug, cfg));
    mae.loadStateDict(pre.State);                                       % no-op on a fresh run; restores on resume
    maeCurve = pre.Curve; maeSec = pre.Sec;
    fprintf('E4 MAE pre-training: %d steps, loss %.3f -> %.3f (%.0fs)\n', cfg.MaeSteps, ...
        mean(maeCurve(1:min(50, end))), mean(maeCurve(max(1, end-49):end)), maeSec);
    sd0 = backbone.stateDict();
    E4 = struct('Init', {}, 'Fraction', {}, 'NTrain', {}, 'TestAcc', {}, 'BalancedAcc', {}, 'Seconds', {});
    order = randperm(RandStream('mt19937ar', 'Seed', cfg.Seed), numel(ya));
    for f = cfg.MaeFractions
        idx = order(1:round(f * numel(ya)));
        for init = ["scratch", "MAE"]
            if init == "MAE", maker = @() withState(maeViT(), sd0);     % every backbone weight incl. the (unused) head
            else, maker = maeViT; end
            key = sprintf('E4_%s_f%03d_s%d_pre%d', init, round(100 * f), cfg.AffectSteps, cfg.MaeSteps);
            r = checkpointed(cfg, key, @() runOnce(maker, Xa(:, :, :, idx), ya(idx), Xat, yat, cfg.AffectSteps, faceAug, cfg, true));
            E4(end+1) = struct('Init', init, 'Fraction', f, 'NTrain', numel(idx), 'TestAcc', r.Acc, ...
                'BalancedAcc', r.Bal, 'Seconds', r.Sec); %#ok<SAGROW>
            fprintf('E4 %-7s %5.0f%% labels (%5d imgs)  test acc %.2f%%  balanced %.2f%%  (%.0fs)\n', ...
                init, 100 * f, numel(idx), 100 * r.Acc, 100 * r.Bal, r.Sec);
        end
    end
    results.E4_mae = struct('Runs', E4, 'PretrainCurve', maeCurve, 'PretrainSeconds', maeSec);
    plotMae(mae, Xat(:, :, :, 1:6), sd, mu, maeCurve, E4, out_dir);
end

if any(cfg.Parts == "E3_scaling")
    % Real face images at each resolution; identical width/depth for both models.
    E3 = struct('Model', {}, 'Size', {}, 'Tokens', {}, 'SecPerStep', {}, 'Params', {});
    for H = cfg.ScaleSizes
        [Xs, ys] = OceanData.Adapters.AffectNetReader.read("train", 'Split', "resplit", 'Sources', "affectnet", ...
            'Labels', cfg.AffectLabels, 'Classes', cfg.AffectClasses, 'TestFraction', 0.2, 'SplitSeed', 0, ...
            'ImageSize', [H H], 'MaxImages', cfg.ScaleBatch, 'Seed', cfg.Seed, 'CacheDir', fullfile(out_dir, 'cache'));
        Xs = double(Xs);
        models = {"ViT",  @() OceanDL.Vision.ViT('ImageSize', [H H], 'Channels', 3, 'PatchSize', 4, 'Dim', 64, ...
                              'Depth', 4, 'Heads', 4, 'NumClasses', numel(cfg.AffectClasses), 'Seed', cfg.Seed)
                  "Swin", @() OceanDL.Vision.Swin('Channels', 3, 'PatchSize', 4, 'Dim', 64, 'Depths', [2 2], ...
                              'Heads', [4 4], 'WindowSize', 4, 'NumClasses', numel(cfg.AffectClasses), 'Seed', cfg.Seed)};
        for m = 1:size(models, 1)
            net = models{m, 2}();
            sec = stepTime(net, Xs, ys, cfg.ScaleReps);
            E3(end+1) = struct('Model', models{m, 1}, 'Size', H, 'Tokens', (H / 4)^2, 'SecPerStep', sec, ...
                'Params', net.numParameters()); %#ok<SAGROW>
            fprintf('E3 %-4s %3d x %-3d (%4d tokens)  %.3f s/step\n', models{m, 1}, H, H, (H / 4)^2, sec);
        end
    end
    for mdl = ["ViT", "Swin"]
        sel = E3([E3.Model] == mdl);
        c = polyfit(log([sel.Tokens]), log([sel.SecPerStep]), 1);
        fprintf('E3 %-4s time ~ tokens^%.2f\n', mdl, c(1));
    end
    results.E3_scaling = E3;
    plotScaling(E3, out_dir);
end

if any(cfg.Parts == "E5_rollout")
    % The E1 ViT trained on 100% of the data (same checkpoint key: reused, or trained once here).
    [~, st] = dataEfficiency("AffectNet", Xa, ya, Xat, yat, {faceViT}, "ViT", 1.00, cfg.AffectSteps, faceAug, cfg);
    vit = withState(faceViT(), st{1});
    vit.eval();
    show = 1:8;
    maps = OceanDL.Vision.AttentionRollout.compute(vit, Xat(:, :, :, show));
    [~, pred] = max(OceanDL.Autograd.GradMode.noGrad(@() vit.forward(Xat(:, :, :, show))).Data, [], 1);
    results.E5 = struct('Maps', maps, 'Pred', pred, 'True', yat(show).');
    plotRollout(Xat(:, :, :, show) .* sd + mu, maps, pred, yat(show), infoT.Classes, out_dir);
end

tag = strjoin(cfg.Parts, '+');                          % per-part files: separate runs never overwrite each other
save(fullfile(out_dir, ['vit_cv_results_' char(tag) '.mat']), 'results', 'cfg');
plotSummary(results, cfg, out_dir, tag);

% ======================================================================== %
function [T, states] = dataEfficiency(name, X, y, Xt, yt, makers, modelNames, fractions, steps, aug, cfg)
% states{m}: stateDict of model m trained on the last fraction.
T = struct('Model', {}, 'Fraction', {}, 'NTrain', {}, 'TestAcc', {}, 'BalancedAcc', {}, 'Seconds', {}, 'Params', {});
states = cell(1, numel(makers));
order = randperm(RandStream('mt19937ar', 'Seed', cfg.Seed), numel(y));
for f = fractions
    idx = order(1:round(f * numel(y)));
    for m = 1:numel(makers)
        key = sprintf('E1_%s_%s_f%03d_s%d', name, modelNames(m), round(100 * f), steps);
        r = checkpointed(cfg, key, @() runOnce(makers{m}, X(:, :, :, idx), y(idx), Xt, yt, steps, aug, cfg, true));
        T(end+1) = struct('Model', modelNames(m), 'Fraction', f, 'NTrain', numel(idx), 'TestAcc', r.Acc, ...
            'BalancedAcc', r.Bal, 'Seconds', r.Sec, 'Params', r.Params); %#ok<AGROW>
        fprintf('E1 %-9s %-4s %5.0f%% (%5d imgs)  test acc %.2f%%  balanced %.2f%%  (%.0fs)\n', name, modelNames(m), ...
            100 * f, numel(idx), 100 * r.Acc, 100 * r.Bal, r.Sec);
        if f == fractions(end), states{m} = r.State; end
    end
end
end

function out = checkpointed(cfg, key, fn)
% Resume support: a finished run is stored once and reloaded on the next
% execution. Written to a temp file then renamed, so a power cut mid-save
% never leaves a truncated checkpoint that would be trusted.
file = fullfile(cfg.CkptDir, [matlab.lang.makeValidName(key) '.mat']);
if isfile(file)
    s = load(file, 'out'); out = s.out;
    fprintf('  [checkpoint] reusing %s\n', key);
    return;
end
out = fn();
tmp = [file '.tmp'];
save(tmp, 'out', '-v7');
movefile(tmp, file, 'f');
end

function r = runOnce(maker, X, y, Xt, yt, steps, aug, cfg, warmup)
% One training run from a fresh model; the augmentation stream restarts at its
% seed so the result does not depend on which runs happened before it.
if aug.Stream ~= RandStream.getGlobalStream(), reset(aug.Stream); end
net = maker();
[acc, sec, curve, bal] = trainEval(net, X, y, Xt, yt, steps, aug, cfg, warmup);
r = struct('Acc', acc, 'Sec', sec, 'Curve', curve, 'Bal', bal, 'Params', net.numParameters(), 'State', net.stateDict());
end

function net = withState(net, sd)
net.loadStateDict(sd);
end

function [acc, sec, curve, bal] = trainEval(net, X, y, Xt, yt, steps, aug, cfg, warmup)
params = net.parameters();
opt = OceanDL.optim.ModuleOptimizer.forModule(net, ...
    @(wd) OceanStats.Optim.AdamWOptimizer('LR', cfg.LR, 'WeightDecay', wd), 'WeightDecay', cfg.WD);
sched = OceanDL.optim.WarmupCosine(cfg.LR, 'WarmupSteps', round(0.05 * steps * warmup), 'TotalSteps', steps, 'MinLR', cfg.LR / 100);
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 7);
n = numel(y); curve = zeros(1, steps);
t0 = tic;
net.train();
for s = 1:steps
    idx = randi(rs, n, 1, min(cfg.Batch, n));
    opt.setLR(sched.at(s));
    opt.zeroGrad();
    L = OceanDL.Autograd.F.crossEntropy(net.forward(aug.apply(X(:, :, :, idx))), y(idx), 'LabelSmoothing', cfg.Smoothing);
    L.backward();
    OceanDL.nn.clipGradNorm(params, cfg.Clip);
    opt.step();
    curve(s) = L.Data;
end
sec = toc(t0);
net.eval();
guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
pred = zeros(numel(yt), 1);
for b = 1:250:numel(yt)
    k = b:min(numel(yt), b + 249);
    [~, p] = max(OceanDL.Autograd.Tensor.raw(net.forward(Xt(:, :, :, k))), [], 1);
    pred(k) = p;
end
acc = mean(pred == yt(:));
cls = unique(yt);
bal = mean(arrayfun(@(c) mean(pred(yt == c) == c), cls));               % mean per-class recall
end

function r = pretrainMae(mae, X, steps, aug, cfg)
% Self-supervised: images only (no labels); same optimiser recipe as trainEval.
if aug.Stream ~= RandStream.getGlobalStream(), reset(aug.Stream); end
opt = OceanDL.optim.ModuleOptimizer.forModule(mae, ...
    @(wd) OceanStats.Optim.AdamWOptimizer('LR', cfg.LR, 'WeightDecay', wd), 'WeightDecay', cfg.WD);
sched = OceanDL.optim.WarmupCosine(cfg.LR, 'WarmupSteps', round(0.05 * steps), 'TotalSteps', steps, 'MinLR', cfg.LR / 100);
params = mae.parameters();
rs = RandStream('mt19937ar', 'Seed', cfg.Seed + 11);
curve = zeros(1, steps); n = size(X, 4);
t0 = tic;
mae.train();
for s = 1:steps
    idx = randi(rs, n, 1, min(cfg.Batch, n));
    opt.setLR(sched.at(s));
    opt.zeroGrad();
    L = mae.forward(aug.apply(X(:, :, :, idx)));
    L.backward();
    OceanDL.nn.clipGradNorm(params, cfg.Clip);
    opt.step();
    curve(s) = L.Data;
end
sec = toc(t0);
r = struct('Curve', curve, 'Sec', sec, 'State', mae.stateDict());
end

function sec = stepTime(net, X, y, reps)
% Median wall-clock of one full training step (forward, loss, backward); rep 0 warms up.
net.train();
t = zeros(1, reps);
for r = 0:reps
    net.zeroGrad();
    t0 = tic;
    L = OceanDL.Autograd.F.crossEntropy(net.forward(X), y);
    L.backward();
    if r > 0, t(r) = toc(t0); end
end
sec = median(t);
end

function net = cnn(inC, widths, H, K, seed)
% Plain CNN baseline: [conv3x3 -> ReLU -> maxpool2] per width, then linear.
layers = {};
c = inC;
for w = widths
    layers = [layers, {OceanDL.nn.Conv2d(c, w, 3, 'Padding', 1, 'Seed', seed + w), OceanDL.nn.Activation("relu"), ...
        OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2]))}]; %#ok<AGROW>
    c = w;
end
side = H / 2^numel(widths);
layers = [layers, {OceanDL.nn.Legacy(OceanDL.Conv.Flatten()), OceanDL.nn.Linear(side^2 * c, K, 'Seed', seed)}];
net = OceanDL.nn.Sequential(layers{:});
end

function [X, y] = importUsps(split)
[rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
X = OceanDL.Layout.DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape);
X = OceanDL.Layout.DimFormat.toCanonical(X, fmt.Format);
OceanDL.Layout.DimFormat.validate(X, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(y));
y = y + 1;
end

function v = ifelse(c, a, b)
if c, v = a; else, v = b; end
end

function plotRollout(imgs, maps, pred, truth, classes, out_dir)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ViT attention rollout');
fig.Position = [80 80 1400 380];
tl = tiledlayout(fig, 2, size(imgs, 4), 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:size(imgs, 4)
    I = min(1, max(0, imgs(:, :, :, k)));
    ax = nexttile(tl, k); image(ax, I); axis(ax, 'image', 'off');
    col = [0.4 0.9 0.4]; if pred(k) ~= truth(k), col = [1 0.45 0.45]; end
    title(ax, sprintf('%s -> %s', classes(truth(k)), classes(pred(k))), 'Color', col, 'FontSize', 8);
    ax = nexttile(tl, size(imgs, 4) + k);
    M = kron(maps(:, :, k), ones(size(I, 1) / size(maps, 1)));
    image(ax, I .* (0.25 + 0.75 * M / max(M(:)))); axis(ax, 'image', 'off');
end
title(tl, 'Attention rollout: brightness = share of the [CLS] decision drawn from each patch', 'Color', viz.Colors.Text);
viz.export(fig, fullfile(out_dir, 'vit_attention_rollout'));
end

function plotSummary(results, cfg, out_dir, tag)
viz = OceanViz.Core('Theme', 'Dark');
fields = intersect(["E1_usps", "E1_affectnet", "E2_usps"], string(fieldnames(results)).', 'stable');
if isempty(fields), return; end
fig = viz.createFigure('ViT vs CNN');
fig.Position = [60 60 520 * numel(fields) 440];
tl = tiledlayout(fig, 1, numel(fields), 'TileSpacing', 'compact', 'Padding', 'compact');
pal = [0.30 0.75 0.95; 0.98 0.60 0.20];
for f = fields
    ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on');
    if startsWith(f, "E1")
        T = results.(f);
        models = unique([T.Model], 'stable');
        for m = 1:numel(models)
            sel = T([T.Model] == models(m));
            plot(ax, 100 * [sel.Fraction], 100 * [sel.TestAcc], '-o', 'Color', pal(m, :), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s (%dk params)', models(m), round(sel(1).Params / 1000)));
        end
        set(ax, 'XScale', 'log'); xticks(ax, 100 * cfg.Fractions);
        xlabel(ax, 'training data used (%)'); ylabel(ax, 'test accuracy (%)');
        title(ax, strrep(f, "E1_", "Data efficiency: ") + " (equal steps)", 'Color', viz.Colors.Text, 'Interpreter', 'none');
        legend(ax, 'Location', 'southeast', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
    else
        E2 = results.(f);
        acc = 100 * [E2.TestAcc];
        barh(ax, categorical(string({E2.Name}), string({E2.Name})), acc, 'FaceColor', pal(1, :));
        xlim(ax, [min(acc) - 3, 100]);
        xlabel(ax, 'test accuracy (%)');
        title(ax, 'ViT ablations (USPS)', 'Color', viz.Colors.Text);
    end
end
viz.export(fig, fullfile(out_dir, ['vit_vs_cnn_' char(tag)]));
end

function plotScaling(E3, out_dir)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Swin vs ViT scaling');
fig.Position = [80 80 620 460];
ax = axes(fig); viz.styleAxes(ax); hold(ax, 'on');
pal = [0.30 0.75 0.95; 0.55 0.85 0.35];
models = ["ViT", "Swin"];
for m = 1:2
    sel = E3([E3.Model] == models(m));
    c = polyfit(log([sel.Tokens]), log([sel.SecPerStep]), 1);
    plot(ax, [sel.Tokens], [sel.SecPerStep], '-o', 'Color', pal(m, :), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('%s: time ~ T^{%.2f}', models(m), c(1)));
end
set(ax, 'XScale', 'log', 'YScale', 'log');
sel = E3([E3.Model] == "ViT");
xticks(ax, [sel.Tokens]);
xticklabels(ax, compose("%d (%dpx)", [sel.Tokens].', [sel.Size].'));
xlabel(ax, 'tokens T = (image side / 4)^2'); ylabel(ax, 'seconds per training step');
title(ax, 'Global (ViT) vs windowed (Swin) attention', 'Color', viz.Colors.Text);
legend(ax, 'Location', 'northwest', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
viz.export(fig, fullfile(out_dir, 'swin_vs_vit_scaling'));
end

function plotMae(mae, X, sd, mu, curve, E4, out_dir)
% Rows: input with the hidden 75% greyed / reconstruction (visible patches
% pasted back) / original. Right: pre-training curve and fine-tuning accuracy.
viz = OceanViz.Core('Theme', 'Dark');
mae.eval();
OceanDL.Autograd.GradMode.noGrad(@() mae.forward(X));
p = mae.PatchSize; [H, W, C, N] = size(X, 1:4);
target = OceanDL.Vision.MaskedAutoencoder.patchify(X, p);
pred = mae.LastPrediction;
if mae.NormPixLoss                                            % undo the per-patch normalisation for display
    pred = pred .* sqrt(var(target, 1, 1) + 1e-6) + mean(target, 1);
end
hid = reshape(mae.LastMask, 1, N, []);
toImg = @(P) OceanDL.Vision.MaskedAutoencoder.unpatchify(P, p, H / p(1), W / p(2));
recon = toImg(target .* ~hid + pred .* hid);
hiddenPix = toImg(repmat(double(hid), size(target, 1), 1, 1)) > 0;
show = @(Z) min(1, max(0, Z .* sd + mu));
masked = show(X); masked(hiddenPix) = 0.35;
rows = {masked, show(recon), show(X)}; names = ["input (75% hidden)", "reconstruction", "original"];
fig = viz.createFigure('MAE');
fig.Position = [60 60 1500 520];
cols = N + 3;
tl = tiledlayout(fig, 3, cols, 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:3
    for k = 1:N
        ax = nexttile(tl, (r - 1) * cols + k);
        image(ax, rows{r}(:, :, :, k)); axis(ax, 'image', 'off');
        if k == 1, title(ax, names(r), 'Color', viz.Colors.Text, 'FontSize', 8); end
    end
end
ax = nexttile(tl, N + 1, [3 1]); viz.styleAxes(ax);
plot(ax, movmean(curve, min(50, numel(curve))), 'Color', [0.98 0.60 0.20], 'LineWidth', 1.5);
xlabel(ax, 'pre-training step'); ylabel(ax, 'masked-patch MSE');
title(ax, 'MAE loss (no labels)', 'Color', viz.Colors.Text);
ax = nexttile(tl, N + 2, [3 2]); viz.styleAxes(ax); hold(ax, 'on');
fr = unique([E4.Fraction]);
acc = zeros(numel(fr), 2); inits = ["scratch", "MAE"];
for i = 1:numel(fr)
    for j = 1:2
        acc(i, j) = 100 * E4([E4.Fraction] == fr(i) & [E4.Init] == inits(j)).TestAcc;
    end
end
b = bar(ax, 1:numel(fr), acc);
b(1).FaceColor = [0.30 0.75 0.95]; b(2).FaceColor = [0.98 0.60 0.20];
set(ax, "XTick", 1:numel(fr), "XTickLabel", cellstr(string(round(100 * fr)) + "% labels"), "TickLabelInterpreter", "none");
ylabel(ax, 'test accuracy (%)'); ylim(ax, [0, max(acc(:)) + 10]);
legend(ax, ["from scratch", "MAE pre-trained"], 'Location', 'northwest', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
title(ax, 'Fine-tuning (same steps)', 'Color', viz.Colors.Text);
viz.export(fig, fullfile(out_dir, 'mae_faces'));
end
