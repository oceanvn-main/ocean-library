% EXP_REPLAY_RECORD_KMEANS  Record Lloyd's k-means iterations (ESL 2nd ed.
%   Sec. 14.3.6, Algorithm 14.1) and open them in Replay Studio -- no
%   re-running needed afterwards (OceanReplay, Planning/System/Replay).
%
%   DATA      a Gaussian mixture from OceanData.Generators.ClassificationGenerator
%             (N, classes, separation and seed below; all stored in the recording).
%   RECORDED  every iteration: centroids, assignment C(i), sse, #changed; the data once.
%   OUTPUT    Results/kmeans_replay.h5, then Replay Studio (Clustering layout).
%
%   Author: Ocean Library | 2026-09-27

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '..', '..'));
out_dir = fullfile(script_dir, 'Results'); if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('N', 300, 'Classes', 3, 'Separation', 2.0, 'DataSeed', 7, 'K', 3, 'InitSeed', 4, 'MaxIter', 50);

% --- data (seeded generator; parameters recorded in the file's meta) ---
gen = OceanData.Generators.ClassificationGenerator('N', cfg.N, 'Classes', cfg.Classes, ...
    'Dimensions', 2, 'Separation', cfg.Separation);
gen.Seed = cfg.DataSeed;
[X, ~] = gen.generate();

% --- record ---
spec = OceanReplay.Core.Spec().addGroup("all", 0, ["data" "centroids" "assignment" "sse" "changed"]);   % every point
file = fullfile(out_dir, 'kmeans_replay.h5');
d = OceanStats.Unsupervised.KMeansRecordable(X, cfg.K, 'Seed', cfg.InitSeed, 'MaxIter', cfg.MaxIter, ...
    'DataName', sprintf("ClassificationGenerator N=%d, classes=%d, separation=%g, seed=%d", ...
                        cfg.N, cfg.Classes, cfg.Separation, cfg.DataSeed));
out = OceanReplay.Record.Recorder.run(d, spec, file, 'Overwrite', true, 'Title', "k-means (Lloyd) on a Gaussian mixture", ...
    'Seeds', struct('data', cfg.DataSeed, 'init', cfg.InitSeed));
r = OceanReplay.Core.Recording(file);
fprintf('Recorded %d iterations to %s (%s).\n', numel(r.axis("iteration").Values), file, ...
    strjoin(OceanReplay.IO.Schema.validate(file), '; '));

% --- replay (no computation from here on) ---
if ~batchStartupOptionUsed, studio = OceanViz.App.Replay.ReplayStudio(file); end   % no window under matlab -batch
