% demo_sensitivity.m
% Demonstrates the "Failure Modes" of Bootstrap vs Bayesian Inference
% using the BootstrapSensitivityExplorer class.

clear; clc;
import OceanStats.Data.BootstrapSensitivityExplorer
import OceanViz.Core

fprintf('Initializing Sensitivity Explorer...\n');
explorer = BootstrapSensitivityExplorer();

%% 1. Gaussian Prior Sensitivity (The "Tau" Nudge)
% Demonstrates that when prior is strong (tau -> 0), Bayes ignores data,
% while Bootstrap follows data.
fprintf('\n[1] Running Gaussian Prior Sensitivity...\n');
data_gauss = randn(20, 1) + 2; % Data centered at 2
tau_values = [10, 1, 0.1];     % Broad -> Moderate -> Strong Prior
explorer.exploreGaussianPrior(data_gauss, tau_values);
fprintf('    Check Figure 1: Note convergence at Tau=10, divergence at Tau=0.1\n');

%% 2. Sparse Multinomial (The "Zero Count" Nudge)
% Demonstrates that Bootstrap fails to quantify risk for unobserved events.
fprintf('\n[2] Running Sparse Multinomial Sensitivity...\n');
counts_sparse = [50, 20, 0];   % 3rd category has ZERO counts
alpha_values  = [0.1, 1, 5];   % Varying Bayes smoothing strength
explorer.exploreSparseMultinomial(counts_sparse, alpha_values);
fprintf('    Check Figure 2: Bootstrap is stuck at 0. Bayes shows uncertainty.\n');

%% 3. Dirichlet-Bootstrap Equivalence (Simplex Plot)
% Demonstrates visually that Resampling Raw Data ~ Dirichlet Posterior
fprintf('\n[3] Running Dirichlet Equivalence (Simplex)...\n');
counts_simplex = [8, 4, 3];    % Small dataset
explorer.exploreDirichletEquivalence(counts_simplex);
fprintf('    Check Figure 3: Bootstrap scatter should match Dirichlet "blob".\n');

fprintf('\nDone! Review the 3 generated figures.\n');
