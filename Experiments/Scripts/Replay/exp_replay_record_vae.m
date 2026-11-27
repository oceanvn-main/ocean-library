% EXP_REPLAY_RECORD_VAE  Record the trained USPS VAE and hierarchical VAE on test
%   images and open them in Replay Studio (OceanReplay, Planning/System/Replay).
%
%   MODELS    Results/generative/cards/usps_vae.mat (1 latent layer) and
%             usps_hvae.mat (2 latent layers), trained by exp_vae_vs_hvae_usps.m.
%   DATA      USPS test set; cfg.PerClass images of every class (first ones in
%             test-set order). Binarised exactly as in training / Model Lab:
%             the whole test set with RandStream('mt19937ar', Seed = cfg.BinarizeSeed).
%   RECORDED  per image (Kingma & Welling 2019, Ch. 2): x, label, mu_i, sigma_i,
%             z_i, KL per unit, reconstruction, decoder mean at a sampled z,
%             -log p(x|z), KL_i, -ELBO, bits/dim, and every traced activation.
%             eps for image k comes from substream k of cfg.SampleSeed.
%   OUTPUT    Results/usps_vae_replay.h5, Results/usps_hvae_replay.h5, then Replay Studio.
%
%   Author: Ocean Library | 2026-09-27

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '..', '..'));
out_dir = fullfile(script_dir, 'Results'); if ~isfolder(out_dir), mkdir(out_dir); end
card_dir = fullfile(script_dir, '..', 'Computer Vision', 'Results', 'generative', 'cards');

cfg = struct('Cards', ["usps_vae" "usps_hvae"], 'PerClass', 5, 'BinarizeSeed', 12, 'SampleSeed', 1);

% --- test data (as the models saw it) ---
D = OceanViz.App.ModelLab.Datasets.get('usps');
Xall = double(D.Xte);
Xall = double(rand(RandStream('mt19937ar', 'Seed', cfg.BinarizeSeed), size(Xall)) < Xall);
y = double(D.yte(:));
classes = unique(y).';
idx = cell2mat(arrayfun(@(c) reshape(find(y == c, cfg.PerClass), 1, []), classes, 'UniformOutput', false));
X = Xall(:, :, :, idx);

files = strings(1, 0);
for name = cfg.Cards
    [model, card] = OceanDL.Training.ModelCard.load(fullfile(card_dir, name + ".mat"));
    X = cast(X, 'like', OceanDL.Interpret.ITraceable.likeOf(model));   % the model's type / device
    r = OceanDL.Generative.Inspect.VAERecordable(model, X, 'Ids', idx, 'Labels', y(idx), 'ClassNames', D.ClassNames, ...
        'Seed', cfg.SampleSeed, 'ModelName', name, 'DatasetName', "USPS test (binarised, seed " + cfg.BinarizeSeed + ")");
    [c, ~] = r.catalog();
    spec = OceanReplay.Core.Spec().addGroup("all", 0, [c.Name]);          % every channel, every image
    file = fullfile(out_dir, name + "_replay.h5");
    OceanReplay.Record.Recorder.run(r, spec, file, 'Overwrite', true, ...
        'Title', sprintf("%s on USPS test images (%d latent layer(s))", name, r.NumLatent), ...
        'Seeds', struct('binarize', cfg.BinarizeSeed, 'sample', cfg.SampleSeed));
    rec = OceanReplay.Core.Recording(file);
    fprintf('Recorded %d images from %s to %s (%s).\n', numel(rec.axis("image").Values), name, file, ...
        strjoin(OceanReplay.IO.Schema.validate(file), '; '));
    files(end+1) = file; %#ok<SAGROW>
end

% --- replay (no computation from here on) ---
if ~batchStartupOptionUsed, studio = OceanViz.App.Replay.ReplayStudio(files(end)); end   % Open/Recent switches to the other
