classdef ParameterStore < handle
    % PARAMETERSTORE  Flat parameter vector manager for NNGraph.
    %
    %   INSPIRED BY:
    %     PyTorch's pytree convention + Flax NNX nnx.split().
    %     Provides a single, ordered double[] that contains ALL trainable
    %     parameters of a network — enabling gradient clipping, serialization,
    %     hyperparameter search, and future optimizer integration.
    %
    %   FLAT VECTOR LAYOUT (deterministic, same every call):
    %     For layers L = 2, 3, ..., numLayers (1-indexed):
    %       [biases_of_layer_L | weights_of_edge_L-1→L (column-major W)]
    %
    %     Example — network [2, 3, 1]:
    %       [ b_21, b_22, b_23,          ← 3 biases for layer 2
    %         w_11, w_21, w_31,          ← col 1 of W(1→2):  W(:,1)
    %         w_12, w_22, w_32,          ← col 2 of W(1→2):  W(:,2)
    %         b_31,                      ← 1 bias  for layer 3
    %         w_11, w_12, w_13 ]         ← col-major W(2→3)  (1×3 → 3 elems)
    %
    %     Total params = sum_{L=2}^{L_max} (d_L + d_{L-1} * d_L)
    %
    %   USAGE:
    %     ps = OceanDL.Graph.ParameterStore();
    %     v  = ps.flatten(graph);           % extract all params
    %     ps.setFromFlat(graph, v * 0.1);   % scale and write back
    %     n  = ps.paramCount(graph);        % total number of scalars
    %
    %   See also: OceanDL.Graph.NNGraph

    % ================================================================== %
    methods
        function v = flatten(obj, graph)
            % FLATTEN  Extract all trainable parameters into a double column vector.
            %
            %   v = ps.flatten(graph)   →   double[P, 1]
            %   where P = ps.paramCount(graph)
            nL = graph.numLayers();
            v  = [];

            for L = 2:nL
                % 1. Biases for layer L (one per node, in node order)
                nodes  = graph.getLayer(L).getNodes();
                biases = cellfun(@(n) n.Bias_, nodes);
                v      = [v; biases(:)];                   %#ok<AGROW>

                % 2. Weight matrix W_{L-1→L}  (d_L × d_{L-1}),  column-major
                W = graph.getWeightMatrix(L-1);
                v = [v; W(:)];                             %#ok<AGROW>
            end
        end

        function setFromFlat(obj, graph, v)
            % SETFROMFLAT  Scatter a flat parameter vector back into graph.
            %
            %   After this call every NNNode.Bias_ and NNEdge.Weight_ is
            %   updated to match the values in v.
            %
            %   v must have exactly paramCount(graph) elements.
            nL  = graph.numLayers();
            idx = 1;

            for L = 2:nL
                layer = graph.getLayer(L);
                nodes = layer.getNodes();
                dL    = numel(nodes);
                dPrev = graph.getLayer(L-1).numNodes();

                % Write biases
                if idx + dL - 1 > numel(v)
                    error('OceanDL:Graph:ParameterStore:LengthMismatch', ...
                        'v is too short. Expected %d params, got %d.', ...
                        obj.paramCount(graph), numel(v));
                end
                for ni = 1:dL
                    nodes{ni}.Bias_ = v(idx);
                    idx = idx + 1;
                end

                % Write weight matrix  (column-major, shape dL × dPrev)
                nWeights = dL * dPrev;
                W = reshape(v(idx : idx + nWeights - 1), dL, dPrev);
                graph.setWeightMatrix(L-1, W);
                idx = idx + nWeights;
            end

            % Guard: all of v should be consumed
            if idx - 1 ~= numel(v)
                error('OceanDL:Graph:ParameterStore:LengthMismatch', ...
                    'v has %d elements but graph expects %d params.', ...
                    numel(v), obj.paramCount(graph));
            end
        end

        function n = paramCount(obj, graph)
            % PARAMCOUNT  Total number of trainable scalar parameters.
            %
            %   n = biases + weights
            %     = sum_{L=2..L_max} [ d_L  +  d_{L-1} * d_L ]
            nL = graph.numLayers();
            n  = 0;
            for L = 2:nL
                dL    = graph.getLayer(L).numNodes();
                dPrev = graph.getLayer(L-1).numNodes();
                n = n + dL + dL * dPrev;
            end
        end

        function [s, e] = layerParamRange(obj, graph, L)
            % LAYERPARAMRANGE  [start, end] indices in flat vector for layer L.
            %
            %   Returns the 1-indexed range [s, e] covering both the biases
            %   and incoming weights for layer L (L >= 2).
            if L < 2
                error('OceanDL:Graph:ParameterStore:InvalidLayer', ...
                    'Layer index must be >= 2 (input layer has no params).');
            end
            nL = graph.numLayers();
            if L > nL
                error('OceanDL:Graph:ParameterStore:InvalidLayer', ...
                    'Layer %d out of range (network has %d layers).', L, nL);
            end

            % Accumulate offset from layers 2..(L-1)
            s = 1;
            for l = 2:L-1
                dl    = graph.getLayer(l).numNodes();
                dPrev = graph.getLayer(l-1).numNodes();
                s = s + dl + dl * dPrev;
            end

            dL    = graph.getLayer(L).numNodes();
            dPrev = graph.getLayer(L-1).numNodes();
            e     = s + dL + dL * dPrev - 1;
        end
    end
end
