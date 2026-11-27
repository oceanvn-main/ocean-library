% EXP_DIFFUSION_WALKTHROUGH  One real image through a diffusion model --
%   forward (noising) and reverse (denoising) -- with every intermediate
%   state recorded, viewed in OceanViz.App.ExperimentStudio.
%
%   MODEL: the EXACT ideal denoiser of a training set of real face photos
%   (OceanDL.Generative.Diffusion.EmpiricalDenoiser over the clean AffectNet
%   faces, 48 x 48 RGB) -- no network, no training. It is the Bayes-optimal
%   answer to "which clean image produced this noisy one?", so the reverse
%   process is shown exactly; the price is memorisation (it can only return
%   training faces), which the walkthrough makes visible on purpose.
%   Schedule: rectified flow  x_t = (1 - t) x0 + t eps  (SD3 / FLUX), images
%   scaled to [-1, 1], Euler sampler.
%
%   CHANNELS (open in ExperimentStudio; also saved as PNG in the output folder):
%   1  Forward: the photo at 256 px and 48 px and a face, t = 0 .. 1 -- noise
%      is added in ONE formula, no model; big images keep structure longer.
%      Schedule curves alpha(t), sigma(t), logSNR (and the SD3 shift).
%   2  One reverse step, opened up: the model's INPUT (x_t, t) and OUTPUTS
%      (guess of the clean image x0hat, guess of the noise epshat, velocity)
%      next to the true x0, eps, velocity and the error.
%   3  Round trips: noise the face / the photo to t0 = 0.3, 0.6, 0.9 and
%      denoise back. Rows: x_t (sampler state), x0hat (model's current guess
%      of the final image), epshat (its guess of the noise); columns: steps.
%      First columns: original, noised start; last: result, nearest training face.
%   4  Generation from pure noise.
%   5  "How many training faces is the model still considering?" --
%      effective number of candidates (exp entropy of the posterior over the
%      training set) against t, for every run.
%
%   RUN: a few minutes on CPU (reads the face cache written by exp_vit_cv
%   when present). Author: Ocean Library | 2026-09-25

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
root = fullfile(script_dir, '..', '..', '..');
addpath(root); addpath(script_dir);
out_dir = fullfile(script_dir, 'Results', 'diffusion_walkthrough');
if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('FaceSize', 48, 'Steps', 50, 'NoiseLevels', [0.3 0.6 0.9], 'Photo', 'corgi.jpg', ...
    'PhotoMaxSide', 256, 'TestFace', 1, 'Seed', 1, 'SmoothStd', 0, 'Upscale', 3);

% ---------------- data: the "training set" and the inputs ----------------
protocol = {'Split', "resplit", 'Sources', "affectnet", 'Labels', "agreeing", ...
    'Classes', ["anger" "contempt" "disgust" "fear" "sad" "surprise"], 'TestFraction', 0.2, 'SplitSeed', 0, ...
    'ImageSize', [cfg.FaceSize cfg.FaceSize], 'CacheDir', fullfile(script_dir, 'Results', 'vit_cv', 'cache'), 'Seed', 1};
Xtr = OceanData.Adapters.AffectNetReader.read("train", protocol{:}, 'MaxImages', Inf);
Xte = OceanData.Adapters.AffectNetReader.read("test", protocol{:}, 'MaxImages', Inf);
pm = @(X) 2 * double(X) - 1;                                   % [0, 1] -> [-1, 1] (diffusion convention)
show = @(X) min(1, max(0, (X + 1) / 2));                       % back to [0, 1] for display
face = pm(Xte(:, :, :, cfg.TestFace));                         % a face the model has NEVER seen
photo = double(imread(fullfile(root, 'Data', 'Resources', 'images', cfg.Photo))) / 255;
if size(photo, 3) == 1, photo = repmat(photo, 1, 1, 3); end
k = cfg.PhotoMaxSide / max(size(photo, 1:2));
resize = @(I, hw) OceanMath.Image.Resampling.ImageResampler('TargetSize', hw, 'FilterType', "Lanczos").apply(I);
photoBig = pm(min(1, max(0, resize(photo, round(k * size(photo, 1:2))))));
side = min(size(photo, 1:2));                                  % centre square crop, then 48 x 48
r0 = floor((size(photo, 1) - side) / 2); c0 = floor((size(photo, 2) - side) / 2);
photoSmall = pm(min(1, max(0, resize(photo(r0 + (1:side), c0 + (1:side), :), [cfg.FaceSize cfg.FaceSize]))));

sch = OceanDL.Generative.Diffusion.Schedules.RectifiedFlow();
den = OceanDL.Generative.Diffusion.EmpiricalDenoiser(pm(Xtr), sch, 'SmoothStd', cfg.SmoothStd);
K = size(Xtr, 4);
fprintf('Ideal denoiser over %d real face photos (%dx%d RGB), schedule: rectified flow\n', K, cfg.FaceSize, cfg.FaceSize);

res = OceanData.ExperimentResult("Diffusion walkthrough: one real image, forward and reverse", ...
    'Description', sprintf('Exact ideal denoiser over %d AffectNet faces; rectified flow; Euler %d steps', K, cfg.Steps));
for f = fieldnames(cfg).', res.addParam(f{1}, cfg.(f{1})); end
up = @(I) repelem(I, cfg.Upscale, cfg.Upscale, 1);            % enlarge small images for viewing
channel = @(name, img, desc) addChannel(res, out_dir, name, img, desc);

% ---------------- 1. forward process ----------------
ts = 0:0.1:1;
fwd = @(x, seed) forwardStrip(x, ts, seed, show);
channel("1a Forward - photo 256 px", fwd(photoBig, 11), ...
    "x_t = (1 - t) x0 + t eps for t = 0, 0.1, ..., 1 (left to right). No model is involved.");
channel("1b Forward - photo 48 px", up(fwd(photoSmall, 11)), ...
    "Same noise levels at 48 px: the image is gone earlier than at 256 px (fewer pixels to average out the noise).");
channel("1c Forward - unseen face 48 px", up(fwd(face, 12)), "The input face (from the test split) under the forward process.");
channel("1d Schedule curves", scheduleCurves(), "alpha(t), sigma(t) and logSNR(t) for rectified flow, plus the SD3 shift (s = 3).");

% ---------------- 2. one reverse step, opened up ----------------
t = 0.6;
e = randn(RandStream('mt19937ar', 'Seed', 13), size(face));
xt = (1 - t) * face + t * e;
[x0hat, epshat] = den.predict(xt, t);
v = e - face; vhat = epshat - x0hat;                           % flow velocity (target and prediction)
parts = {show(face), eshow(e), show(xt), show(x0hat), eshow(epshat), eshow(v / 2), eshow(vhat / 2), eshow(vhat - v)};
channel("2 One reverse step (t = 0.6) - inputs and outputs", up(row(parts)), ...
    "Columns: TRUE x0 | TRUE eps | INPUT x_t (with t = 0.6) | OUTPUT x0hat | OUTPUT epshat | target velocity eps - x0 | predicted velocity | velocity error. Noise shown with grey = 0.");
res.addMetric('step_t06_effectiveCandidates', den.LastEffectiveN);
res.addMetric('step_t06_topWeight', den.LastTopWeight);
fprintf('One step at t = 0.6: the model spreads its belief over %.1f of %d faces (top weight %.2f)\n', ...
    den.LastEffectiveN, K, den.LastTopWeight);

% ---------------- 3. round trips ----------------
inputs = {"face", face, 21; "photo", photoSmall, 22};
curves = struct('Name', {}, 'T', {}, 'N', {});
for i = 1:size(inputs, 1)
    [nm, x, seed] = inputs{i, :};
    for t0 = cfg.NoiseLevels
        e = randn(RandStream('mt19937ar', 'Seed', seed), size(x));
        xt0 = (1 - t0) * x + t0 * e;
        [out, rec] = reverse(den, sch, xt0, t0, max(2, round(cfg.Steps * t0)));
        nn = rec.Extra{end}(3);
        grid = trajectoryGrid(rec, show(x), show(xt0), show(out), show(pm(Xtr(:, :, :, nn))), show);
        channel(sprintf("3 Round trip - %s, t0 = %.1f", nm, t0), up(grid), ...
            "Rows: x_t (sampler state) | x0hat (model's guess of the final image) | epshat (guess of the noise). Columns: original, noised start, steps (t decreasing), result, nearest training face.");
        tag = sprintf('%s_t%02d', nm, round(10 * t0));
        res.addMetric([tag '_rmseToInput'], sqrt(mean((out - x) .^ 2, 'all')));
        res.addMetric([tag '_nearestTrainFace'], nn);
        curves(end+1) = struct('Name', sprintf('%s, t0 = %.1f', nm, t0), 'T', rec.T, ...
            'N', cellfun(@(q) q(1), rec.Extra)); %#ok<SAGROW>
        if nm == "face" && t0 == cfg.NoiseLevels(end), printSteps(rec, x, K); end
    end
end

% ---------------- 4. generation from pure noise ----------------
[out, rec] = reverse(den, sch, randn(RandStream('mt19937ar', 'Seed', 31), size(face)), 1, cfg.Steps);
nn = rec.Extra{end}(3);
grid = trajectoryGrid(rec, zeros(size(face)), show(rec.X{1}), show(out), show(pm(Xtr(:, :, :, nn))), show);
channel("4 Generation from pure noise", up(grid), ...
    "Rows: x_t | x0hat | epshat over the reverse steps from t = 1. Last columns: generated image, nearest training face (identical: the ideal model memorises).");
curves(end+1) = struct('Name', 'generation from noise', 'T', rec.T, 'N', cellfun(@(q) q(1), rec.Extra));
res.addMetric('generation_distanceToNearestTrainFace', sqrt(mean((out - pm(Xtr(:, :, :, nn))) .^ 2, 'all')));

% ---------------- 5. how many candidates are left ----------------
channel("5 Candidates left during denoising", candidateCurves(curves, K), ...
    "Effective number of training faces the model still considers plausible (exp entropy of its posterior) as t goes from the start to 0.");

res.save(fullfile(out_dir, 'diffusion_walkthrough.mat'));
fprintf('Saved %s (+ PNG per channel)\n', fullfile(out_dir, 'diffusion_walkthrough.mat'));
if usejava('desktop')
    OceanViz.App.ExperimentStudio.inspect(res);
end

% ======================================================================== %
function addChannel(res, out_dir, name, img, desc)
res.add(name, img, 'Description', desc);
imwrite(img, fullfile(out_dir, [regexprep(char(name), '[^\w\- =.,]', '') '.png']));
end

function I = eshow(e)
% Noise / velocity for display: 0 -> mid grey, +-2 -> white / black.
I = min(1, max(0, e / 4 + 0.5));
end

function S = forwardStrip(x, ts, seed, show)
e = randn(RandStream('mt19937ar', 'Seed', seed), size(x));
S = row(arrayfun(@(t) show((1 - t) * x + t * e), ts, 'UniformOutput', false));
end

function R = row(imgs)
% Place images side by side with a thin dark separator.
[H, ~, C] = size(imgs{1});
sep = 0.15 * ones(H, 2, C);
R = imgs{1};
for k = 2:numel(imgs), R = [R, sep, imgs{k}]; end %#ok<AGROW>
end

function [out, rec] = reverse(den, sch, x1, t0, steps)
rec = OceanDL.Generative.Diffusion.TrajectoryRecorder(den.handle(), ...
    'OnCall', @() [den.LastEffectiveN, den.LastTopWeight, den.LastTopIndex]);
out = OceanDL.Generative.Diffusion.Samplers.EulerODE('Steps', steps).sample(rec.handle(), sch, size(x1), ...
    'X1', x1, 'TStart', t0);
end

function G = trajectoryGrid(rec, orig, start, result, nearest, show)
% 3 rows (x_t, x0hat, epshat) x (2 + 8 steps + 2) columns.
n = rec.numCalls();
pick = unique(round(linspace(1, n, min(8, n))));
[H, W, C] = size(orig);
blank = 0.15 * ones(H, W, C);
rows = cell(3, 1);
for r = 1:3
    cells = cell(1, numel(pick) + 4);
    if r == 1, cells{1} = orig; cells{2} = start; else, cells{1} = blank; cells{2} = blank; end
    for j = 1:numel(pick)
        k = pick(j);
        switch r
            case 1, cells{2 + j} = show(rec.X{k});
            case 2, cells{2 + j} = show(rec.X0{k});
            case 3, cells{2 + j} = eshow(rec.Eps{k});
        end
    end
    if r == 1, cells{end-1} = result; cells{end} = nearest; else, cells{end-1} = blank; cells{end} = blank; end
    rows{r} = row(cells);
end
sepH = 0.15 * ones(2, size(rows{1}, 2), C);
G = [rows{1}; sepH; rows{2}; sepH; rows{3}];
end

function printSteps(rec, x, K)
fprintf('\nReverse process for the face from t0 = %.1f (every 5th step)\n', rec.T(1));
fprintf('   t      candidates (of %d)   top weight   top training face   |x0hat - input| (RMSE)\n', K);
for k = [1:5:rec.numCalls(), rec.numCalls()]
    q = rec.Extra{k};
    fprintf('  %.3f   %12.1f          %.3f       %6d              %.3f\n', rec.T(k), q(1), q(2), q(3), ...
        sqrt(mean((rec.X0{k} - x) .^ 2, 'all')));
end
end

function I = renderFigure(fig)
I = double(print(fig, '-RGBImage', '-r110')) / 255;
close(fig);
end

function I = scheduleCurves()
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Schedule'); fig.Visible = 'off'; fig.Position = [50 50 1000 360];
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
t = linspace(0, 1, 201);
a = nexttile(tl); viz.styleAxes(a); hold(a, 'on');
f1 = OceanDL.Generative.Diffusion.Schedules.RectifiedFlow(); f3 = OceanDL.Generative.Diffusion.Schedules.RectifiedFlow('Shift', 3);
plot(a, t, f1.alpha(t), 'LineWidth', 2, 'DisplayName', '\alpha(t) signal');
plot(a, t, f1.sigma(t), 'LineWidth', 2, 'DisplayName', '\sigma(t) noise');
plot(a, t, f3.sigma(t), '--', 'LineWidth', 1.5, 'DisplayName', '\sigma(t), SD3 shift 3');
xlabel(a, 't'); legend(a, 'Location', 'east', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
title(a, 'x_t = \alpha(t) x_0 + \sigma(t) \epsilon', 'Color', viz.Colors.Text);
b = nexttile(tl); viz.styleAxes(b); hold(b, 'on');
tt = linspace(0.01, 0.99, 199);
plot(b, tt, f1.logSNR(tt), 'LineWidth', 2, 'DisplayName', 'rectified flow');
plot(b, tt, f3.logSNR(tt), '--', 'LineWidth', 1.5, 'DisplayName', 'shift 3');
yline(b, 0, ':', 'Color', viz.Colors.Text, 'HandleVisibility', 'off');
xlabel(b, 't'); ylabel(b, 'log SNR = log(\alpha^2 / \sigma^2)');
legend(b, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
title(b, 'Signal-to-noise ratio (0 = equal parts)', 'Color', viz.Colors.Text);
I = renderFigure(fig);
end

function I = candidateCurves(curves, K)
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Candidates'); fig.Visible = 'off'; fig.Position = [50 50 900 420];
ax = axes(fig); viz.styleAxes(ax); hold(ax, 'on');
cols = turbo(numel(curves) + 1);
for k = 1:numel(curves)
    plot(ax, curves(k).T, curves(k).N, '-', 'LineWidth', 1.8, 'Color', cols(k, :), 'DisplayName', curves(k).Name);
end
yline(ax, K, ':', sprintf('all %d training faces', K), 'Color', viz.Colors.Text, 'HandleVisibility', 'off');
set(ax, 'YScale', 'log', 'XDir', 'reverse'); ylim(ax, [0.8, 1.5 * K]);
xlabel(ax, 't  (reverse process runs right to left)'); ylabel(ax, 'training faces still plausible');
legend(ax, 'Location', 'southwest', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
title(ax, 'The model narrows down which image it is making', 'Color', viz.Colors.Text);
I = renderFigure(fig);
end
