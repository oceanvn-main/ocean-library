% EXP_VAE_VS_HVAE_SHAPES  VAE vs 2-layer hierarchical VAE on a SYNTHETIC
%   CHALLENGE set: ComplexShapeImages 32 x 32 -- thin spirals, sharp stars,
%   rings with a gap, crescents, crosses and hearts under random rotation,
%   scale, translation and stroke width (8000 train / 2000 test).
%
%   Why a challenge: single-latent VAEs tend to blur thin strokes and sharp
%   corners and to mix classes; a hierarchy can put GLOBAL structure (class,
%   pose) in the top latent and LOCAL detail in the lower one.
%   Hierarchy (controlled; see gen_compare_vae_hvae):
%     levels 32 / 16 / 8 / 4, widths [16 32 64 64]
%     VAE   latents [0 0 1 0]  -> z1: 8x8x4
%     HVAE  latents [0 0 1 1]  -> z1: 8x8x4, z2: 4x4x4 (top, N(0, I))
%
%   RUN: resumable. OCEAN_QUICK=1 for a smoke run (Results/generative/quick).
%
%   Author: Ocean Library  |  2026-09-25

clc; clear; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '..', '..')); addpath(script_dir);
quick = ~isempty(getenv('OCEAN_QUICK'));
out_dir = fullfile(script_dir, 'Results', 'generative'); if quick, out_dir = fullfile(out_dir, 'quick'); end
if ~isfolder(out_dir), mkdir(out_dir); end

cfg = struct('Widths', [16 32 64 64], 'VAELatents', [0 0 1 0], 'HVAELatents', [0 0 1 1], 'LatentChannels', 4, ...
    'Steps', 4000, 'Batch', 64, 'LR', 1e-3, 'KLWarmup', 0.1, 'FreeBits', 0.25, 'Samples', 1000, ...
    'EvalSteps', 2000, 'Seed', 1, 'Verbose', 250, 'OutDir', out_dir, 'CkptDir', fullfile(out_dir, 'checkpoints'));
if quick
    cfg.Steps = 30; cfg.Samples = 64; cfg.EvalSteps = 60; cfg.Verbose = 0;
end
D = gen_load_data("shapes", quick, cfg.CkptDir);
R = gen_compare_vae_hvae(D, cfg);
