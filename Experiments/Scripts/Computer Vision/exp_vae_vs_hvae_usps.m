% EXP_VAE_VS_HVAE_USPS  VAE vs 2-layer hierarchical VAE on REAL images
%   (USPS handwritten digits, 16 x 16).
%
%   Question: does a second, lower-resolution latent layer with a LEARNED
%   prior p(z1 | z2) improve the likelihood bound and the samples, and at what
%   compute cost? Controlled: identical networks, only the extra layer differs
%   (see gen_compare_vae_hvae). Hierarchy:
%     levels 16x16 / 8x8 / 4x4, widths [32 64 64]
%     VAE   latents [0 1 0]  -> z1: 8x8x4   (the unused 4x4 level is not built)
%     HVAE  latents [0 1 1]  -> z1: 8x8x4, z2: 4x4x4 (top, N(0, I))
%
%   LIKELIHOOD: Bernoulli on DYNAMICALLY BINARISED digits (test set binarised once,
%   fixed seed), so the reported negative ELBO is a true bound on -log p(x).
%   (Revised 2026-09-26: the first run used grey pixels, which gave the Bernoulli
%   loss a 33-nat floor; free bits are now per latent UNIT so they actually bind;
%   log-variances soft-clamped.) Models are also saved as ModelCards in
%   Results/generative/cards (usps_vae.mat, usps_hvae.mat) for the Model Lab.
%
%   RUN: resumable (every model checkpointed). OCEAN_QUICK=1 for a smoke run
%   (writes to Results/generative/quick). Trains on the GPU when MATLAB can use one.
%
%   Author: Ocean Library  |  2026-09-25

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '..', '..')); addpath(script_dir);
quick = ~isempty(getenv('OCEAN_QUICK'));
out_dir = fullfile(script_dir, 'Results', 'generative'); if quick, out_dir = fullfile(out_dir, 'quick'); end
if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('Widths', [32 64 64], 'VAELatents', [0 1 0], 'HVAELatents', [0 1 1], 'LatentChannels', 4, ...
    'Steps', 4000, 'Batch', 32, 'LR', 1e-3, 'KLWarmup', 0.1, 'Samples', 1000, ...
    'Binarize', true, 'FreeBits', 0.05, 'FreeBitsPer', "unit", 'Device', "auto", ...   % nats per latent unit
    'EvalSteps', 2000, 'Seed', 1, 'Verbose', 250, 'OutDir', out_dir, 'CkptDir', fullfile(out_dir, 'checkpoints'));
if quick
    cfg.Steps = 30; cfg.Samples = 64; cfg.EvalSteps = 60; cfg.Verbose = 0;
end
D = gen_load_data("usps", quick, cfg.CkptDir);
R = gen_compare_vae_hvae(D, cfg);
