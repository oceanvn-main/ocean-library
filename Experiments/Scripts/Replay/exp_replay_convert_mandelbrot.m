% EXP_REPLAY_CONVERT_MANDELBROT  Convert the legacy Mandelbrot training
%   checkpoint (exp_mandelbrot_train.m, replayed by exp_mandelbrot_replay.m)
%   into an OceanReplay recording and open it in Replay Studio.
%
%   The network's output map is computed ONCE here, with the old renderer's
%   own code (DecisionBoundaryRenderer.predictMap), at the epochs the spec
%   selects; cost and R^2 are copied for every epoch. Weights are read from the
%   1.5 GB sidecar one epoch at a time.
%   OUTPUT  Results/mandelbrot_replay.h5, then Replay Studio (Default layout).
%
%   Author: Ocean Library | 2026-09-27

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
root = fullfile(script_dir, '..', '..', '..');
addpath(root);
out_dir = fullfile(script_dir, 'Results'); if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('Checkpoint', fullfile(root, 'Experiments', 'Outputs', 'Mandelbrot', 'mandelbrot_fourier_checkpoint.mat'), ...
    'MapEvery', 20, 'MapEarly', 1:10);          % maps every MapEvery epochs + every early epoch

d = OceanViz.Visualizers.NNCheckpointRecordable(cfg.Checkpoint);
S = OceanReplay.Core.Sampling;
spec = OceanReplay.Core.Spec() ...
    .addGroup("curves", 0, ["cost" "rSquared"]) ...
    .addGroup("maps", 1, "prediction", struct('epoch', S.union(S.every(cfg.MapEvery), S.list(cfg.MapEarly)))) ...
    .addGroup("target", 0, "groundTruth");
file = fullfile(out_dir, 'mandelbrot_replay.h5');
rec = OceanReplay.Record.Recorder(d, spec, file, 'Overwrite', true, 'Title', "Mandelbrot -- Fourier-feature MLP training");
fprintf('Estimated size: %.1f MB\n', rec.EstimatedBytes / 1e6);
tic; rec.record(); fprintf('Converted in %.0f s -> %s\n', toc, file);

if ~batchStartupOptionUsed, studio = OceanViz.App.Replay.ReplayStudio(file); end   % no window under matlab -batch
