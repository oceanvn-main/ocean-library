% EXP_VAE_VS_DIFFUSION_USPS  A simple VAE vs a simple diffusion model on REAL
%   images (USPS handwritten digits, 16 x 16), both unconditional.
%
%   VAE        ConvEncoder / ConvDecoder widths [32 64], latent 8x8x4, Bernoulli
%   Diffusion  UNet2D widths [32 64], rectified flow (velocity target,
%              logit-normal times), weight EMA; Euler 4/8/16/32 steps and Heun 16
%   Question: how much sample quality does iterative denoising buy over one
%   decoder pass, and how many network evaluations does it need? (see
%   gen_compare_vae_diffusion for the metrics).
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

cfg = struct('VAEWidths', [32 64], 'LatentChannels', 4, 'UNetWidths', [32 64], ...
    'Steps', 4000, 'DiffusionSteps', 6000, 'Batch', 64, 'LR', 1e-3, 'KLWarmup', 0.1, 'FreeBits', 0.25, ...
    'Samples', 1000, 'EvalSteps', 2000, 'Seed', 1, 'Verbose', 250, 'OutDir', out_dir, ...
    'CkptDir', fullfile(out_dir, 'checkpoints'));
if quick
    cfg.Steps = 30; cfg.DiffusionSteps = 30; cfg.Samples = 64; cfg.EvalSteps = 60; cfg.Verbose = 0;
end
D = gen_load_data("usps", quick, cfg.CkptDir);
R = gen_compare_vae_diffusion(D, cfg);
