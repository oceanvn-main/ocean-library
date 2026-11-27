classdef NNGraphMemento
    % NNGRAPHMEMENTO  Immutable snapshot of an NNGraph's full parameter state.
    %
    %   Captures all edge weights, node biases, and node enable/disable masks.
    %   Topology is NOT stored — only the parameter values. The corresponding
    %   NNGraph must still exist and have the same topology to restore.
    %
    %   Usage:
    %     snap  = graph.saveSnapshot();      % capture current state
    %     graph.setWeightMatrix(1, newW);     % modify the graph
    %     graph.loadSnapshot(snap);           % restore to captured state
    %
    %   See also: OceanDL.Graph.NNGraph

    properties (SetAccess = private)
        % Number of layers in the graph (read-only)
        NumLayers   (1,1) int32

        % Cell{L} of (d_{L+1} x d_L) weight matrices, for L = 1..NumLayers-1
        Weights     cell

        % Cell{L} of (d_L x 1) bias vectors, for L = 1..NumLayers
        Biases      cell

        % Cell{L} of (d_L x 1) logical vectors — true = enabled node
        EnableMasks cell

        % Timestamp of when this snapshot was taken
        Timestamp   (1,:) char
    end

    methods
        function obj = NNGraphMemento(graph)
            % NNGRAPHMEMENTO  Create a snapshot from a live NNGraph.
            %
            % INPUT:
            %   graph - OceanDL.Graph.NNGraph instance
            if nargin == 0, return; end

            nL = graph.numLayers();
            obj.NumLayers   = int32(nL);
            obj.Weights     = cell(nL - 1, 1);
            obj.Biases      = cell(nL, 1);
            obj.EnableMasks = cell(nL, 1);
            obj.Timestamp   = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<TNOW1,DATST>

            % Capture edge weights layer by layer
            for L = 1:nL-1
                obj.Weights{L} = graph.getWeightMatrix(L);
            end

            % Capture biases + enable masks from nodes
            for L = 1:nL
                layer = graph.getLayer(L);
                nodes = layer.getNodes();
                nN    = numel(nodes);

                biasVec   = zeros(nN, 1);
                enableVec = true(nN, 1);
                for ni = 1:nN
                    biasVec(ni)   = nodes{ni}.Bias_;
                    enableVec(ni) = strcmp(nodes{ni}.getState(), 'Active');
                end
                obj.Biases{L}      = biasVec;
                obj.EnableMasks{L} = enableVec;
            end
        end

        function tf = isCompatible(obj, graph)
            % ISCOMPATIBLE  True iff this snapshot is loadable into graph.
            %   Checks that NumLayers and per-layer node counts match.
            tf = false;
            if graph.numLayers() ~= double(obj.NumLayers), return; end
            for L = 1:double(obj.NumLayers)
                if graph.numNodes(L) ~= numel(obj.Biases{L}), return; end
            end
            tf = true;
        end
    end
end
