% EXP_LIKELIHOOD_SURFACE_OPTIMIZATION Navigating the Likelihood Surface.
%
%   DATASET:        Dynamic ARMA(1,1) time series (T=500, phi=0.7, theta=-0.4, sigma2=1.0)
%   MODELS:         Steepest Ascent, Exact Newton-Raphson, DFP, BFGS
%   ASSUMPTIONS:    Conditional Gaussian innovations e(t) ~ N(0, sigma2), Box-Jenkins filter
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   Author: Ocean Library System
%   Date:   2026-09-03

clc; clear; close all;

% Dynamic path resolution: ensure Ocean Library root is on MATLAB search path
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(project_root));

rng(42, 'twister');

fprintf('=== EXPERIMENT: NAVIGATING THE LIKELIHOOD SURFACE ===\n\n');

% === 1. Dynamic Data Generation (Module 1) ===
phi_true = 0.7;
theta_true = -0.4;

sigma2 = 1.0;
T = 5000;
seed = 42;

y = OceanStats.Time.ARMAProcess.simulate(phi_true, theta_true, sigma2, T, seed);
fprintf('  [Module 1] Dynamic ARMA(1,1) realization generated: T=%d, phi_true=%.2f, theta_true=%.2f\n', T, phi_true, theta_true);

% === 2. Log-Likelihood Function & Grid Sweep (Module 2) ===
likeFun = @(p) OceanStats.Time.ARMALikelihood.logLikelihood(p, y, sigma2);

fprintf('  [Module 2] Sweeping 2D log-likelihood grid terrain...\n');
phi_range = [-0.9, 0.9];
theta_range = [-0.9, 0.9];
res = 100;
[PhiGrid, ThetaGrid, LGrid] = OceanStats.Time.ARMALikelihood.gridSweep(y, sigma2, phi_range, theta_range, res);

% === 3. Optimization Engines Execution (Module 3 & 4) ===
start_param = [-0.6, 0.2]; % Initial starting coordinate far from true peak
fprintf('  [Module 3] Initial starting coordinate: theta^(0) = [%.2f, %.2f]\n\n', start_param(1), start_param(2));

wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();

% Engine A: Steepest Ascent
fprintf('  Running Engine A: Steepest Ascent...\n');
[p_steep, f_steep, hist_steep] = wrapper.solve(likeFun, start_param, 'Method', 'steepdesc', 'IsMaximization', true);

% Engine B: Newton-Raphson
fprintf('  Running Engine B: Newton-Raphson...\n');
[p_newton, f_newton, hist_newton] = wrapper.solve(likeFun, start_param, 'Method', 'newton', 'IsMaximization', true);

% Engine C: DFP Quasi-Newton
fprintf('  Running Engine C: DFP Quasi-Newton...\n');
[p_dfp, f_dfp, hist_dfp] = wrapper.solve(likeFun, start_param, 'Method', 'dfp', 'IsMaximization', true);

% Engine D: BFGS Quasi-Newton
fprintf('  Running Engine D: BFGS Quasi-Newton...\n');
[p_bfgs, f_bfgs, hist_bfgs] = wrapper.solve(likeFun, start_param, 'Method', 'bfgs', 'IsMaximization', true);

% === 4. Telemetry Logging & Summary Printout ===
histories = struct();
histories.Steepest = hist_steep;
histories.Newton   = hist_newton;
histories.DFP      = hist_dfp;
histories.BFGS     = hist_bfgs;

fprintf('\n================== OPTIMIZATION ENGINES SUMMARY ==================\n');
fprintf('%-18s | %-6s | %-12s | %-12s | %-12s\n', 'Engine', 'Iters', 'Max Log-L', 'Final Phi', 'Final Theta');
fprintf('-------------------------------------------------------------------\n');
fprintf('%-18s | %-6d | %-12.4f | %-12.4f | %-12.4f\n', 'Steepest Ascent', hist_steep.iterations, f_steep, p_steep(1), p_steep(2));
fprintf('%-18s | %-6d | %-12.4f | %-12.4f | %-12.4f\n', 'Newton-Raphson', hist_newton.iterations, f_newton, p_newton(1), p_newton(2));
fprintf('%-18s | %-6d | %-12.4f | %-12.4f | %-12.4f\n', 'DFP Quasi-Newton', hist_dfp.iterations, f_dfp, p_dfp(1), p_dfp(2));
fprintf('%-18s | %-6d | %-12.4f | %-12.4f | %-12.4f\n', 'BFGS Quasi-Newton', hist_bfgs.iterations, f_bfgs, p_bfgs(1), p_bfgs(2));
fprintf('-------------------------------------------------------------------\n');
fprintf('  True Parameters   : phi = %.4f, theta = %.4f\n', phi_true, theta_true);
fprintf('===================================================================\n\n');

% === 5. OceanViz Visualization & Rendering Engine (Module 5) ===
vizData = struct();
vizData.PhiGrid    = PhiGrid;
vizData.ThetaGrid  = ThetaGrid;
vizData.LGrid      = LGrid;
vizData.TrueParam  = [phi_true, theta_true];
vizData.StartParam = start_param;
vizData.Histories  = histories;

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Navigating the Likelihood Surface');
ax_dummy = axes('Parent', fig);

% Visualization options: Set Animate = true to view step-by-step trajectory movie
viz.plot(ax_dummy, "LikelihoodOptimization", vizData, ...
    "Animate", true, ...   % Set to true for live frame-by-frame animation movie
    "PauseTime", 0.05, ...  % Delay in seconds between frames
    "SaveGif", "");        % Set to e.g. "trajectory.gif" to export movie to GIF
