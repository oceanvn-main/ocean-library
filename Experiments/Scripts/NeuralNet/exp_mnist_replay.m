% EXP_MNIST_REPLAY  Animate a saved MNIST training checkpoint in OceanViz.
%
%   Loads the checkpoint from exp_mnist_train and plays it back using the
%   generic NNReplayEngine with pluggable renderers.
%
%   Requires: run  exp_mnist_train  first.

clc; close all;

% --- Checkpoint path ---
scriptDir = fileparts(mfilename('fullpath'));
checkFile = fullfile(fileparts(fileparts(scriptDir)), ...
    'Outputs', 'NeuralNet', 'mnist_checkpoint.mat');

% --- Configure engine ---
engine = OceanViz.Visualizers.NNReplayEngine(); %Instantiate the Oschestrator

engine.SampleRenderer = OceanViz.Adapters.ImageSampleRenderer(); %Hire the sample renderer
engine.SampleRendererConfig = struct('IMG_RES', 28);

engine.PredictionRenderer = OceanViz.Strategies.NetworkGraphRenderer(); %Hire the prediction renderer

engine.Layout     = OceanViz.Visualizers.NNGraphLayout(); %Hire the layout
engine.FigureTitle = 'MNIST Training Replay — [784 → 64 → 10]';
engine.FPS        = 10;

% --- Run ---
engine.run(checkFile);
