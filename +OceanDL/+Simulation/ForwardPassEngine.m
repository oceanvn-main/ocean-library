classdef ForwardPassEngine
    % FORWARDPASSENGINE  Diagnostic wrapper around NNGraph.forwardPass().
    %
    %   Runs a forward pass on any NNGraph and returns a ForwardPassResult
    %   struct with per-layer statistics useful for diagnosing training
    %   pathologies BEFORE training begins.
    %
    %   All computation is pure matrix math — no Deep Learning Toolbox needed.
    %
    %   DIAGNOSTICS COMPUTED:
    %     LayerActivations — (N×d_L) activation matrix per layer
    %     LayerMeans       — Mean absolute activation per layer
    %     LayerStd         — Std of activations per layer
    %     SaturationRate   — Fraction of neurons saturated (sigmoid/tanh)
    %     DeadNeuronMask   — True for ReLU neurons with mean |a| < threshold
    %
    %   WHAT EACH DIAGNOSTIC DETECTS:
    %     LayerMeans    → low value = "lazy" layer, not contributing much
    %     LayerStd      → low value = collapsed representation (redundant neurons)
    %     SaturationRate→ high value = vanishing gradient risk for sigmoid/tanh
    %     DeadNeuronMask→ permanently-off ReLU neurons (gradient = 0 always)
    %
    %   Usage:
    %     engine = OceanDL.Simulation.ForwardPassEngine();
    %     result = engine.run(graph, X);
    %     result = engine.run(graph, X, 'DeadThreshold', 0.05);
    %
    %   See also: OceanDL.Graph.NNGraph
    %             OceanViz.Strategies.ActivationFlowStrategy

    properties
        % Threshold below which a ReLU neuron is "dead" (mean |activation|)
        DeadThreshold (1,1) double {mustBeNonnegative} = 0.05
    end

    methods
        function obj = ForwardPassEngine(options)
            % FORWARDPASSENGINE  Construct with optional configuration.
            %
            % Name-Value:
            %   DeadThreshold (double): ReLU dead neuron threshold (default: 0.05)
            if nargin >= 1 && isfield(options, 'DeadThreshold')
                obj.DeadThreshold = options.DeadThreshold;
            end
        end

        function result = run(obj, graph, X)
            % RUN  Execute forward pass and return diagnostic result struct.
            %
            % INPUTS:
            %   graph - NNGraph instance
            %   X     - (N x d_in) input matrix (N samples, d_in features)
            %
            % OUTPUT:
            %   result - ForwardPassResult struct with fields:
            %     .Output          - (N x d_out) network output
            %     .NumLayers       - total number of layers
            %     .LayerNames      - cell of layer name strings
            %     .LayerActivations- cell{L} of (N x d_L) matrices
            %     .LayerMeans      - (L x 1) mean absolute activation per layer
            %     .LayerStd        - (L x 1) std of activations per layer
            %     .SaturationRate  - (L x 1) fraction saturated (sigmoid/tanh)
            %     .DeadNeuronMask  - cell{L} of (1 x d_L) logical vectors
            %     .Activations     - cell of activation function names per layer

            nL  = graph.numLayers();
            spec = graph.toSpec();

            % Run the forward pass — this updates NNNode.PreActivation_ + .Value_
            A_out = graph.forwardPass(X);

            % Collect per-layer activations using public API only
            [layerActs, A_out] = obj.collectActivations_(graph, X);

            % Compute diagnostics
            L           = nL;
            means       = zeros(L, 1);
            stds        = zeros(L, 1);
            satRates    = zeros(L, 1);
            deadMasks   = cell(L, 1);

            for li = 1:L
                A = layerActs{li};  % (N x d_L)

                if isempty(A)
                    means(li)     = 0;
                    stds(li)      = 0;
                    satRates(li)  = 0;
                    deadMasks{li} = [];
                    continue;
                end

                means(li) = mean(abs(A(:)));
                stds(li)  = std(A(:));

                act = lower(spec.Activations{li});
                switch act
                    case {'sigmoid', 'tanh'}
                        % Saturated = output very close to limits
                        if strcmp(act, 'sigmoid')
                            sat = A > 0.9 | A < 0.1;
                        else % tanh: limits ±1
                            sat = abs(A) > 0.9;
                        end
                        satRates(li) = mean(sat(:));
                    otherwise
                        satRates(li) = 0;
                end

                % Dead neuron: per column (neuron), check mean |a| < threshold
                meanPerNeuron = mean(abs(A), 1);  % (1 x d_L)
                if strcmp(act, 'relu')
                    deadMasks{li} = meanPerNeuron < obj.DeadThreshold;
                else
                    deadMasks{li} = false(1, size(A, 2));
                end
            end

            % Assemble result struct
            result.Output           = A_out;
            result.NumLayers        = nL;
            result.LayerNames       = spec.LayerNames;
            result.Activations      = spec.Activations;
            result.LayerActivations = layerActs;
            result.LayerMeans       = means;
            result.LayerStd         = stds;
            result.SaturationRate   = satRates;
            result.DeadNeuronMask   = deadMasks;
            result.DeadThreshold    = obj.DeadThreshold;
        end
    end

    methods (Access = private)

        function [layerActs, A_out] = collectActivations_(~, graph, X, ~) %#ok<INUSD>
            % COLLECTACTIVATIONS_  Collect full (N x d_L) activations per layer
            % using only the public NNGraph API: run forwardPass one sample at
            % a time and read each node's Value_ after each call.
            % This avoids touching any private method inside NNGraph.

            nL = graph.numLayers();
            N  = size(X, 1);

            % Pre-discover how many active nodes each layer has
            nActive = zeros(nL, 1);
            for L = 1:nL
                nActive(L) = numel(graph.getLayer(L).getActiveNodes());
            end

            % Pre-allocate storage
            layerActs = cell(nL, 1);
            for L = 1:nL
                layerActs{L} = zeros(N, nActive(L));
            end

            % Run forward pass one sample at a time
            for s = 1:N
                x_s = X(s, :);           % (1 x d_in)
                graph.forwardPass(x_s);   % updates node .Value_; return discarded

                % Input layer — read directly from X
                layerActs{1}(s, :) = x_s(1:nActive(1));

                % Hidden + output layers — read from node.Value_
                for L = 2:nL
                    activeNodes = graph.getLayer(L).getActiveNodes();
                    for ni = 1:numel(activeNodes)
                        layerActs{L}(s, ni) = activeNodes{ni}.Value_;
                    end
                end
            end

            % Run one final full-X pass so node displays show the mean
            A_out = graph.forwardPass(X);
        end
    end
end
