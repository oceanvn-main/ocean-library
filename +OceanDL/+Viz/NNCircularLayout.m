classdef NNCircularLayout < OceanDL.Viz.ILayoutEngine
    % NNCIRCULARLAYOUT  Concentric rings layout for neural networks.
    %
    %   Places nodes in concentric circles:
    %     Input Layer  - Inner circle
    %     Hidden 1     - Ring 2
    %     ...
    %     Output Layer - Outermost ring
    %
    %   This layout is useful for visualizing architectural symmetry 
    %   and global connectivity patterns that are harder to see in 
    %   a horizontal grid.
    %
    % See also: OceanDL.Viz.NNLayeredLayout
    %           OceanDL.Viz.ILayoutEngine

    properties (Constant, Access = private)
        BASE_RADIUS    = 1.5
        RADIUS_STEP    = 1.5
        NODE_BASE_SIZE = 120
    end

    methods
        function data = toData(obj, graph, options)
            if nargin < 3, options = struct(); end

            % 1. Use base layered engine to obtain complete visual encoding (colors, weights, states, alpha)
            baseEng = OceanDL.Viz.NNLayeredLayout();
            data    = baseEng.toData(graph, options);

            % 2. Compute circular positions specifically for the visual nodes in data
            positions = obj.computeCircularPositions(graph, data);
            data.Positions = positions;

            % 3. Update OutputPos for output class labels
            if isfield(data, 'OutputLabels') && ~isempty(data.OutputLabels) && isfield(data, 'OutputPos')
                outRows = (data.NumNodes - numel(data.OutputLabels) + 1) : data.NumNodes;
                data.OutputPos = positions(outRows, :);
            end
        end
    end

    methods (Static)
        function pos = computeCircularPositions(graph, data)
            % COMPUTECIRCULARPOSITIONS  Concentric ring XY for visual nodes.
            nL    = data.NumLayers;
            Kv    = data.NumNodes;
            pos   = zeros(Kv, 2);

            % Count visual nodes per layer from data.NodeRealIDs
            layerVisualCounts = zeros(1, nL);
            for k = 1:Kv
                tok = regexp(data.NodeRealIDs{k}, '^L(\d+)_N\d+$', 'tokens');
                if ~isempty(tok)
                    L = str2double(tok{1}{1});
                    layerVisualCounts(L) = layerVisualCounts(L) + 1;
                end
            end

            % Assign circular positions layer by layer
            k = 1;
            for L = 1:nL
                dv = layerVisualCounts(L);
                if dv == 0, continue; end

                r = OceanDL.Viz.NNCircularLayout.BASE_RADIUS + ...
                    (L - 1) * OceanDL.Viz.NNCircularLayout.RADIUS_STEP;

                angles = linspace(0, 2*pi, dv + 1);
                angles = angles(1:end-1);   % avoid overlap at 0 / 2pi
                offset = (L - 1) * (pi / 8); % slight rotation per ring

                for ni = 1:dv
                    theta = angles(ni) + offset;
                    pos(k, :) = [r * cos(theta), r * sin(theta)];
                    k = k + 1;
                end
            end
        end

        function pos = computePositions(graph)
            % COMPUTEPOSITIONS  Concentric ring XY for all nodes (uncompressed fallback).
            nL    = graph.numLayers();
            spec  = graph.toSpec();
            sizes = spec.LayerSizes;
            K     = sum(sizes);
            pos   = zeros(K, 2);

            k = 1;
            for L = 1:nL
                d = sizes(L);
                r = OceanDL.Viz.NNCircularLayout.BASE_RADIUS + ...
                    (L - 1) * OceanDL.Viz.NNCircularLayout.RADIUS_STEP;

                angles = linspace(0, 2*pi, d + 1);
                angles = angles(1:end-1);
                offset = (L - 1) * (pi / 8);

                for ni = 1:d
                    theta = angles(ni) + offset;
                    pos(k, :) = [r * cos(theta), r * sin(theta)];
                    k = k + 1;
                end
            end
        end
    end
end
