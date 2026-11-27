% EXP_VAE_VS_DIFFUSION_SHAPES  A simple VAE vs a simple diffusion model on a
%   SYNTHETIC CHALLENGE set: ComplexShapeImages 32 x 32 (thin spirals, sharp
%   stars, rings with a gap, crescents, crosses, hearts; random pose, scale,
%   stroke width), both unconditional.
%
%   VAE        ConvEncoder / ConvDecoder widths [16 32 64], latent 8x8x4, Bernoulli
%   Diffusion  UNet2D widths [16 32 64] (attention at 8x8), rectified flow,
%              weight EMA; Euler 4/8/16/32 steps and Heun 16
%   Why a challenge: thin, sharp, topologically varied shapes expose the
%   VAE's averaging (blur) and test whether the flow model keeps every class
%   (mode coverage) at few sampling steps.
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

cfg = struct('VAEWidths', [16 32 64], 'LatentChannels', 4, 'UNetWidths', [16 32 64], ...
    'Steps', 4000, 'DiffusionSteps', 6000, 'Batch', 64, 'LR', 1e-3, 'KLWarmup', 0.1, 'FreeBits', 0.25, ...
    'Samples', 1000, 'EvalSteps', 2000, 'Seed', 1, 'Verbose', 250, 'OutDir', out_dir, ...
    'CkptDir', fullfile(out_dir, 'checkpoints'));
if quick
    cfg.Steps = 30; cfg.DiffusionSteps = 30; cfg.Samples = 64; cfg.EvalSteps = 60; cfg.Verbose = 0;
end
D = gen_load_data("shapes", quick, cfg.CkptDir);
R = gen_compare_vae_diffusion(D, cfg);
