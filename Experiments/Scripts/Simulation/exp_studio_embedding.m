%% exp_studio_embedding.m
% Launches InformationStudio with EmbeddingMapper2D geometry.
%
% In this setup, each Markov state is associated with a high-dimensional
% feature vector (e.g., sensor reading, probability row, neural activation).
% EmbeddingMapper2D uses PCA to project those features into 2D for
% visualization. The particle walks the Markov chain but is displayed in
% the PCA embedding space — states that are "similar" in feature space
% will appear close together on the canvas.
%
% Pipeline:
%   MarkovChain (from features) → MarkovSimulator → SimulationHistory
%       → EmbeddingMapper2D (PCA) → InformationStudio
%
% Author: Ocean Library
% Date:   2026-02-27

clc; clear; close all;

%% 0. Add library to path
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), '..')));

%% 1. Define state feature vectors
%
% Each row = one Markov state's feature vector.
% Here we use 3 synthetic clusters of 4 states each (12 total),
% so the embedding should reveal 3 well-separated groups in 2D.
%
% In a real use case this could be:
%   - Rows of a trained HMM emission matrix
%   - Row-probability vectors of a large Markov chain
%   - Sensor readings associated with each discrete state

rng(42);
K        = 12;     % total number of states
F        = 20;     % feature dimensionality

% 3 clusters, 4 states each
centers = [randn(1,F)*0; randn(1,F)+5; randn(1,F)-5];
features = zeros(K, F);
for k = 1:K
    cluster = mod(k-1, 3) + 1;       % cycle through 3 clusters
    features(k,:) = centers(cluster,:) + 0.5*randn(1,F);
end

%% 2. Build a transition matrix biased by cluster membership
%
% States in the same cluster transition to each other with high probability,
% and occasionally jump to a different cluster (rare inter-cluster transitions).

W = zeros(K);
for i = 1:K
    for j = 1:K
        ci = mod(i-1, 3);
        cj = mod(j-1, 3);
        if ci == cj
            W(i,j) = 3.0;   % intra-cluster: high weight
        else
            W(i,j) = 0.2;   % inter-cluster: low weight
        end
    end
end
W = W + eye(K);   % self-loops for ergodicity

%% 3. Build the simulation pipeline
% Use RandomWalkGraph because it automatically normalizes weights into probabilities
rwg  = OceanMath.Information.Processes.RandomWalkGraph.fromAdjacency(W);
sim  = OceanMath.Simulation.MarkovSimulator(rwg);
hist = OceanMath.Simulation.SimulationHistory(sim);

fprintf('MarkovChain: %d states, %d feature dimensions\n', K, F);

%% 4. Pre-run 30 steps
sim.runFor(30);
fprintf('Pre-run complete. %d steps recorded.\n', hist.getLength());

%% 5. Build the embedding mapper
%
% EmbeddingMapper2D projects high-dim features to 2D using PCA.
% States that look similar in feature space will cluster together on screen.
%
% METHOD options:
%   'pca'   - Principal Component Analysis (fast, exact, always available)
%   'tsne'  - t-SNE  (nonlinear, requires Statistics Toolbox)
%   'umap'  - UMAP   (nonlinear, requires run_umap on the path)

METHOD = 'pca';
mapper = OceanViz.Adapters.EmbeddingMapper2D(features, 'Method', METHOD);
disp(mapper);  % shows variance explained if PCA

%% 6. Launch InformationStudio
studio = OceanViz.App.InformationStudio(sim, hist, mapper, 'PlaybackFPS', 5);
studio.show();

fprintf('\nInformationStudio launched (Embedding mode: %s).\n', METHOD);
fprintf('States with similar features appear close in 2D.\n');
fprintf('Watch the particle jump between 3 clusters rarely, stay within clusters often.\n');
