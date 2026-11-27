classdef NNLayer < handle
    % NNLAYER  Composite node — one layer of neurons in the NNGraph.
    %
    %   COMPOSITE PATTERN:
    %     NNLayer acts as a container for NNNode objects.
    %     Operations (disable, enable, highlight) are delegated to all children.
    %     This lets NNGraph do "disable entire layer" in one call.
    %
    %   PARAMETER ACCOUNTING:
    %     Input layer (Type = 'input'): paramCount = 0 (no biases on input nodes).
    %     Hidden/Output layers: paramCount = numNodes (one bias per node).
    %     Edge weights are NOT counted here — they belong to NNEdge and are
    %     counted separately by ParameterStore.
    %
    %   See also: OceanDL.Graph.NNNode
    %             OceanDL.Graph.NNGraph

    % ------------------------------------------------------------------ %
    properties (Access = public)
        Index  (1,1) int32    % 1-indexed layer number (1 = input, L = output)
        Name   (1,1) string   % human-readable, e.g. "Hidden_1"
        Type   (1,1) string   % 'input' | 'hidden' | 'output'
    end

    properties (Access = private)
        Nodes_  % cell array of NNNode (ordered by NodeIndex)
    end

    % ================================================================== %
    methods
        function obj = NNLayer(index, name, type)
            % NNLAYER  Construct an empty layer (nodes added via addNode).
            arguments
                index (1,1) int32
                name  (1,1) string
                type  (1,1) string
            end
            obj.Index  = index;
            obj.Name   = name;
            obj.Type   = lower(type);
            obj.Nodes_ = {};
        end

        % ---- Node management ---------------------------------------- %

        function addNode(obj, node)
            % ADDNODE  Append a NNNode to this layer.
            obj.Nodes_{end+1} = node;
        end

        function node = getNodeAt(obj, nodeIdx)
            % GETNODEAT  Return the node at 1-indexed position nodeIdx.
            node = obj.Nodes_{nodeIdx};
        end

        function nodes = getNodes(obj)
            % GETNODES  Return all nodes as a cell array (incl. disabled).
            nodes = obj.Nodes_;
        end

        function nodes = getActiveNodes(obj)
            % GETACTIVENODES  Return only enabled (Active + Highlighted) nodes.
            mask  = cellfun(@(n) n.isEnabled(), obj.Nodes_);
            nodes = obj.Nodes_(mask);
        end

        % ---- Counts -------------------------------------------------- %

        function n = numNodes(obj)
            n = numel(obj.Nodes_);
        end

        function n = numActiveNodes(obj)
            n = sum(cellfun(@(n) n.isEnabled(), obj.Nodes_));
        end

        function n = paramCount(obj)
            % PARAMCOUNT  Number of BIAS parameters in this layer.
            %   Input layer has no bias — it simply passes through X.
            if strcmp(obj.Type, 'input')
                n = 0;
            else
                n = obj.numNodes();  % one bias per node
            end
        end

        % ---- Composite delegation ------------------------------------ %

        function disableAll(obj)
            cellfun(@(n) n.disable(), obj.Nodes_);
        end

        function enableAll(obj)
            cellfun(@(n) n.enable(), obj.Nodes_);
        end

        function highlightAll(obj)
            cellfun(@(n) n.highlight(), obj.Nodes_);
        end

        function deselectAll(obj)
            cellfun(@(n) n.deselect(), obj.Nodes_);
        end

        % ---- Internal helpers --------------------------------------- %

        function swapNodeOrder(obj, idxA, idxB)
            % SWAPNODEORDER  Swap cell positions of two nodes (for display).
            tmp               = obj.Nodes_{idxA};
            obj.Nodes_{idxA}  = obj.Nodes_{idxB};
            obj.Nodes_{idxB}  = tmp;
        end

        % ---- Display ------------------------------------------------- %

        function disp(obj)
            fprintf('NNLayer %d  [%s]  (%s)  %d nodes, %d active\n', ...
                obj.Index, obj.Name, obj.Type, ...
                obj.numNodes(), obj.numActiveNodes());
            for ni = 1:numel(obj.Nodes_)
                fprintf('  '); obj.Nodes_{ni}.disp();
            end
        end
    end
end
