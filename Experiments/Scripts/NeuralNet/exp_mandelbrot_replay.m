% EXP_MANDELBROT_REPLAY  Animate how a Fourier Feature MLP approximates
%   the Mandelbrot set. Uses mandelbrot_fourier_checkpoint.mat.
%
%   Left panel : True Mandelbrot escape-time map (static reference)
%   Right panel: NN output at each training epoch (Fourier Feature MLP)
%
%   Requires: run  exp_mandelbrot_train  first.

clc; close all;

% --- Ensure project root is on MATLAB path ---
scriptDir = fileparts(mfilename('fullpath'));
rootDir   = fileparts(fileparts(fileparts(scriptDir)));
addpath(rootDir);

% --- Checkpoint path ---
scriptDir = fileparts(mfilename('fullpath'));
checkFile = fullfile(fileparts(fileparts(scriptDir)), ...
    'Outputs', 'Mandelbrot', 'mandelbrot_fourier_checkpoint.mat');

% --- Load ONLY the small fields needed by MandelbrotReferenceRenderer ---
% Do NOT do a full ck = load(checkFile) here — that loads 1.5 GB for nothing.
% The engine.run() handles the full load internally.
ck_display = load(checkFile, 'groundTruth', 'RE_RANGE', 'IM_RANGE', 'a_d', 'b_d');

% --- Configure engine ---
engine = OceanViz.Visualizers.NNReplayEngine(); %Instantiate the Orchestrator

engine.SampleRenderer = OceanViz.Adapters.MandelbrotReferenceRenderer(); %True Mandelbrot (ground truth)
engine.SampleRendererConfig = struct('Checkpoint', ck_display);

engine.PredictionRenderer = OceanViz.Strategies.DecisionBoundaryRenderer(); %NN approximation
engine.PredictionRenderer.Colormap = 'hot';

engine.Layout      = OceanViz.Visualizers.NNGraphLayout(); %Standard layout
engine.FigureTitle = 'Mandelbrot — Fourier Feature MLP Approximation';
engine.FPS         = inf;
engine.EpochStep   = 5;

% Detailed explanation shown at bottom of replay window:
engine.Description = { ...
    'LEFT (Ground Truth): Re-evaluates exact Z(k+1)=Z(k)^2+C math at 400x400 resolution per zoom gesture (sharp at any scale).', ...
    'RIGHT (Neural Net) : Scales pre-baked 150x150 grid predictions. Blurry on zoom due to 1/150 training sampling limit, Nyquist-bounded Fourier bandwidth, and smooth MLP activations.'};

% --- Run (blocks until animation finishes or user clicks Stop) ---
engine.run(checkFile);

% --- Post-replay: attach HD zoom to the True Mandelbrot panel ---
% Scroll-wheel to zoom in. Double-click to reset view.
% Each zoom gesture recomputes the fractal at 400x400 resolution.
if isvalid(engine.Axes.axSample)
    ctrl = OceanViz.Controllers.MandelbrotZoomController();
    ctrl.attach(engine.Axes.axSample, 'MaxIter', 200, 'GridRes', 400);
    fprintf('Zoom mode active on True Mandelbrot panel.\n');
end
