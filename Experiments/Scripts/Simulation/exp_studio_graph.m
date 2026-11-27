%% exp_studio_graph.m
% Launches InformationStudio with a custom graph topology using GraphMapper.
%
% GraphMapper maps each Markov state to an arbitrary 2D node position.
% Node layout is computed automatically from the adjacency matrix — no
% positions need to be hardcoded.
%
% Pipeline:
%   RandomWalkGraph (adjacency W) → MarkovSimulator → SimulationHistory
%       → GraphMapper (spring/circular/spectral layout) → InformationStudio
%
% Try changing LAYOUT below to see how the same graph looks under
% different auto-positioning algorithms.
%
% Author: Ocean Library
% Date:   2026-02-27

clc; clear; close all;

%% 0. Add library to path
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), '..')));

%% 1. Define the graph topology (adjacency matrix)
%
% Choose a graph to walk on.
% This example uses a "barbell" graph: two cliques connected by a bridge.
% It is NOT ergodic in pure form — the bridge dominates the mixing.
%
% --- GRAPH OPTIONS ---
%   'barbell'   - two triangles linked by a long path
%   'star'      - one hub connected to all leaves
%   'cycle'     - states arranged in a ring
%   'smallworld'- Watts-Strogatz style locally connected + shortcuts

graphType = 'barbell';

switch graphType
    case 'barbell'
        % Two triangles (nodes 1-3, 4-6), joined by bridge 3-4
        W = zeros(6);
        W(1,2)=0.5; W(2,3)=2.0; W(1,3)=1.5;   % triangle A (asymmetric weights)
        W(4,5)=1.2; W(5,6)=0.8; W(4,6)=1.0;   % triangle B
        W(3,4)=2.5;                           % strong bridge
        W = W + W';                           % make symmetric

    case 'star'
        K = 8;   % 1 hub + 7 leaves
        W = zeros(K);
        W(1, 2:end) = 1;
        W = W + W';

    case 'cycle'
        K = 9;
        W = diag(ones(K-1,1), 1) + diag(ones(K-1,1), -1);
        W(1,K) = 1; W(K,1) = 1;

    case 'smallworld'
        K   = 10;
        W   = diag(ones(K-1,1), 1) + diag(ones(K-1,1), -1);   % ring baseline
        W(1,K) = 1; W(K,1) = 1;
        % Add 3 random shortcuts
        rng(42);
        for j = 1:3
            i1 = randi(K); i2 = randi(K);
            W(i1,i2) = 1; W(i2,i1) = 1;
        end
end

% Add self-loops for ergodicity (makes the graph aperiodic)
selfLoopWeight = 1.0;
W = W + selfLoopWeight * eye(size(W,1));

%% 2. Build the simulation pipeline
rwg  = OceanMath.Information.Processes.RandomWalkGraph.fromAdjacency(W, 'Symmetric', true);
sim  = OceanMath.Simulation.MarkovSimulator(rwg);
hist = OceanMath.Simulation.SimulationHistory(sim);

fprintf('RandomWalkGraph [%s]: %d nodes, ergodic=%s\n', ...
    graphType, rwg.getNumStates(), mat2str(rwg.isErgodic()));

%% 3. Pre-run 30 steps so there is history to view immediately
sim.runFor(30);
fprintf('Pre-run complete. %d steps recorded.\n', hist.getLength());

%% 4. Build the spatial mapper — auto-layout from the adjacency matrix
%
% Layout options:
%   'circular'  — evenly spaced on a circle (best for small dense graphs)
%   'spring'    — force-directed Fruchterman-Reingold (best for sparse graphs)
%   'spectral'  — Laplacian eigenvectors (best for community detection)

LAYOUT = 'spring';

mapper = OceanViz.Adapters.GraphMapper.fromAdjacency(W, 'Layout', LAYOUT);
fprintf('Mapper: %s layout, %d nodes.\n', LAYOUT, mapper.getNumStates());

%% 5. Launch InformationStudio
studio = OceanViz.App.InformationStudio(sim, hist, mapper, 'PlaybackFPS', 5);
studio.show();

fprintf('\nInformationStudio launched (Graph mode: %s, %s layout).\n', graphType, LAYOUT);
fprintf('Press Play to animate. Step Fwd at end of history generates new steps.\n');
