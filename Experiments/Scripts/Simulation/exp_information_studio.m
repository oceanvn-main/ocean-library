%% exp_information_studio.m
% Launches the InformationStudio GUI dashboard to visualize a Markov chain
% simulation as an animated particle on a 2D grid.
%
% Pipeline:
%   MarkovChain → MarkovSimulator → SimulationHistory
%       → GridMapper2D → ParticleVisualizer → InformationStudio
%
% Author: Ocean Library
% Date:   2026-02-27

clc; clear; close all;

%% 0. Add library to path
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), '..')));

%% 1. Define the graph topology
%
% RandomWalkGraph knows about the grid structure and derives the transition
% matrix from it. MarkovChain knows nothing about geometry.
%
% --- TOPOLOGY OPTIONS ---
% Change numCols here to resize the entire simulation:
numCols   = 3;         % 3x3 = 9 states
%
% Change selfLoops to switch between Simulation-only and Theory-ready:
%   0.0 -> Physics mode (bipartite, oscillates, NOT ergodic) 
%   1.0 -> Theory mode (aperiodic, strictly ergodic)
selfLoops = 1.0; 

%% 2. Build the simulation pipeline
rwg  = OceanMath.Information.Processes.RandomWalkGraph.fromGrid2D(numCols, ...
            'SelfLoops', selfLoops);
sim  = OceanMath.Simulation.MarkovSimulator(rwg);
hist = OceanMath.Simulation.SimulationHistory(sim);

fprintf('RandomWalkGraph: %d nodes, ergodic=%s\n', ...
    rwg.getNumStates(), mat2str(rwg.isErgodic()));


%% 3. Pre-run 30 steps so there is history to scrub through immediately
sim.runFor(30);
fprintf('Pre-run complete. %d steps recorded.\n', hist.getLength());

%% 4. Create the spatial adapter — derived from numCols, never hardcoded
mapper = OceanViz.Adapters.GridMapper2D(numCols, numCols, 'CellSize', 1.0);

%% 5. Launch InformationStudio
studio = OceanViz.App.InformationStudio(sim, hist, mapper, 'PlaybackFPS', 6);
studio.show();

fprintf('\nInformationStudio launched.\n');
fprintf('Press Play (⏯) to animate, or step through the history manually.\n');
fprintf('Step Forward (⏩) at end-of-history generates new sim steps live.\n');

