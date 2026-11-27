% EXP_REPLAY_RECORD_DIFFUSION  Record the full reverse process of the trained
%   16 x 16 AffectNet DDPM (exp_ddpm_affectnet16.m; Lai et al. Sec. 2.2) and
%   open it in Replay Studio -- scrub all 1000 steps without re-running.
%
%   RECORDED  (the spec below, all parameters, no hard-coded results)
%     level 0  numbers at EVERY step: t_i, alphaBar_i, sigma, sigma^2(i),
%              rms(eps_phi), the identity residual
%     level 1  images every Every steps: x_i, eps_phi, x_phi, mu; the result x_0
%     level 2  full images in a detail window (Center +- Half)
%     view     occlusion of eps_phi at OcclusionSteps (Zeiler & Fergus; NOT in Lai et al.)
%   OUTPUT     Results/ddpm_affectnet16_replay.h5, then Replay Studio (Diffusion layout).
%
%   Author: Ocean Library | 2026-09-27

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
root = fullfile(script_dir, '..', '..', '..');
addpath(root);
out_dir = fullfile(script_dir, 'Results'); if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('Card', fullfile(root, 'Experiments', 'Scripts', 'Computer Vision', 'Results', 'ddpm_affectnet', 'ddpm_affectnet16.mat'), ...
    'Class', "surprise", 'Guidance', 1, 'Seed', 7, 'Every', 20, 'Center', 50, 'Half', 1, ...
    'OcclusionSteps', [900 500 100 50], 'OcclusionSize', 4, 'OcclusionStride', 2);

% --- the trained model (ModelCard) and its predictor ---
[model, card] = OceanDL.Training.ModelCard.load(cfg.Card);
names = string(card.Meta.ClassNames);
label = find(names == cfg.Class);
if isempty(label), error('Class "%s" not in the card (%s).', cfg.Class, strjoin(names, ', ')); end
predict = model.predictor('Labels', label, 'Guidance', cfg.Guidance);
sch = model.Schedule;
smp = OceanDL.Generative.Diffusion.Samplers.DDPMAncestral('Steps', sch.L);      % textbook: all L steps
range = [-1 1]; if isfield(card.Meta, 'DataRange'), range = card.Meta.DataRange; end

% --- record ---
S = OceanReplay.Core.Sampling;
spec = OceanReplay.Core.Spec() ...
    .addGroup("numbers", 0, ["tLevel" "tNext" "alphaBar" "alphaBarNext" "sigma" "varNext" "epsRms" "identityResidual"]) ...
    .addGroup("images", 1, ["x" "eps" "x0hat" "mu"], struct('step', S.every(cfg.Every))) ...
    .addGroup("detail", 2, ["x" "eps" "x0hat" "mu"], struct('step', S.window(cfg.Center, cfg.Half))) ...
    .addGroup("final", 1, "result") ...
    .addView("occlusion", "occlusion", struct('size', cfg.OcclusionSize, 'stride', cfg.OcclusionStride), ...
             struct('step', S.list(cfg.OcclusionSteps)));
d = OceanDL.Generative.Inspect.DiffusionRecordable(predict, sch, smp, card.Meta.ImageSize, 'Seed', cfg.Seed, ...
    'ClipX0', range, 'DataRange', range, 'ModelName', "ddpm_affectnet16 (" + cfg.Class + ", w = " + cfg.Guidance + ")");
file = fullfile(out_dir, 'ddpm_affectnet16_replay.h5');
rec = OceanReplay.Record.Recorder(d, spec, file, 'Overwrite', true, 'Title', "AffectNet 16x16 DDPM -- reverse process", ...
    'Seeds', struct('sampler', cfg.Seed));
fprintf('Estimated size: %.1f MB\n', rec.EstimatedBytes / 1e6);
tic; rec.record(); fprintf('Recorded in %.0f s -> %s\n', toc, file);
issues = OceanReplay.IO.Schema.validate(file);
if ~isempty(issues), warning('%s', strjoin(issues, newline)); end

% --- replay (no computation from here on) ---
if ~batchStartupOptionUsed, studio = OceanViz.App.Replay.ReplayStudio(file); end   % no window under matlab -batch
