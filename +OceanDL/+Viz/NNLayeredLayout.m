classdef NNLayeredLayout < OceanDL.Viz.ILayoutEngine
    % NNLAYEREDLAYOUT  Custom layered-grid layout backend for NNGraph.
    %
    %   Places nodes in a regular layered grid:
    %     X axis = layer index (1, 2, ..., L), equally spaced.
    %     Y axis = node rank within its layer, centered around 0.
    %
    %   LARGE-LAYER COMPRESSION:
    %     When a layer has more than VIZ_MAX_PER_LAYER nodes, only a
    %     representative subset of VIZ_MAX_PER_LAYER nodes are drawn.
    %     The layer label still shows the true node count.
    %     Edges are drawn between visual representatives only, using the
    %     actual weight value (mean of the corresponding real-weight block).
    %
    %   This is the DEFAULT layout engine for feedforward neural networks.
    %
    % See also: OceanDL.Viz.ILayoutEngine
    %           OceanViz.Strategies.NetworkGraphStrategy

    properties (Constant, Access = private)
        NODE_BASE_SIZE       = 120     % scatter marker area units
        NODE_HIGHLIGHT_BOOST = 1.2    % 20% larger if Highlighted
        ACTIVE_ALPHA         = 0.85   % edge opacity when both endpoints active
        INACTIVE_ALPHA       = 0.12   % edge dimmed when an endpoint is Disabled
        VIZ_MAX_PER_LAYER    = 16     % max nodes drawn per layer (visual cap)
        MAX_HEIGHT           = 12     % max vertical canvas span per layer
    end

    methods
        function data = toData(obj, graph, options)
            % Convert the backend NNGraph directly into an NNGraphData struct
            % mapping each node to a layer-based 2D position.
            
            if nargin < 3, options = struct(); end
            
            nL    = graph.numLayers();
            spec  = graph.toSpec();
            sizes = spec.LayerSizes;   % [d1, d2, ..., dL]

            % ---- 1. Determine representative node indices per layer --- %
            % For large layers, pick VIZ_MAX_PER_LAYER evenly-spaced nodes.
            vizIdx = cell(1, nL);   % real node indices shown per layer
            for L = 1:nL
                d = sizes(L);
                if d <= obj.VIZ_MAX_PER_LAYER
                    vizIdx{L} = 1:d;
                else
                    vizIdx{L} = unique(round(linspace(1, d, obj.VIZ_MAX_PER_LAYER)));
                end
            end

            % Total visual nodes
            vizCounts = cellfun(@numel, vizIdx);
            Kv = sum(vizCounts);

            % ---- 2. Build visual positions ---------------------------- %
            % X per layer: 0, 2, 4, ...   Y: centered, height capped.
            positions = zeros(Kv, 2);
            layerOffset = [0, cumsum(vizCounts(1:end-1))];  % start index per layer

            for L = 1:nL
                dv   = vizCounts(L);
                xVal = (L - 1) * 2;
                if dv == 1
                    yVals = 0;
                else
                    halfSpan = min((dv-1)/2, obj.MAX_HEIGHT/2);
                    yVals = linspace(-halfSpan, halfSpan, dv);
                end
                rows = layerOffset(L) + (1:dv);
                positions(rows, 1) = xVal;
                positions(rows, 2) = yVals;
            end

            % ---- 3. Build visual node appearance --------------------- %
            nodeColors  = repmat([0.08, 0.08, 0.12], Kv, 1);  % default: near-black
            nodeSizes   = ones(Kv, 1) * obj.NODE_BASE_SIZE;
            nodeStates  = repmat("Active", Kv, 1);
            nodeDead    = false(Kv, 1);
            nodeValues  = zeros(Kv, 1);   % raw activation for numeric overlay

            cBase = [0.18, 0.18, 0.22];
            cMax  = [0.20, 0.55, 1.00];
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                cBase = options.Colors.Background * 0.75 + options.Colors.Text * 0.25;
                cMax  = options.Colors.Primary;
            end

            % Find max output node to highlight among active nodes
            maxOutVal = -inf;
            maxOutIdx = -1;
            outNodes  = graph.getLayer(nL).getNodes();
            for vi = 1:numel(vizIdx{nL})
                val = outNodes{vizIdx{nL}(vi)}.Value_;
                if val > maxOutVal
                    maxOutVal = val;
                    maxOutIdx = layerOffset(nL) + vi;
                end
            end

            for L = 1:nL
                nodes  = graph.getLayer(L).getNodes();
                vIdx   = vizIdx{L};
                offset = layerOffset(L);

                for vi = 1:numel(vIdx)
                    ni  = vIdx(vi);
                    nd  = nodes{ni};
                    k   = offset + vi;
                    st  = nd.getState();
                    nodeStates(k)  = st;
                    nodeValues(k)  = nd.Value_;

                    switch st
                        case 'Disabled'
                            nodeColors(k,:) = [0.18 0.18 0.20];
                        case 'Highlighted'
                            nodeColors(k,:) = [0.00 1.00 0.90];
                            nodeSizes(k)    = nodeSizes(k) * obj.NODE_HIGHLIGHT_BOOST;
                        otherwise
                            % Uniform color for all nodes, except max output
                            if k == maxOutIdx && maxOutVal > -inf
                                nodeColors(k,:) = cMax;
                                nodeSizes(k)    = obj.NODE_BASE_SIZE * 1.5;
                            else
                                nodeColors(k,:) = cBase;
                            end
                    end

                    % Dead neuron: ReLU stuck at zero
                    if strcmp(nd.Activation, 'relu') && abs(nd.Value_) < 1e-4
                        nodeDead(k) = true;
                    end
                end
            end

            % ---- 4. Build visual edges from weight matrices ----------- %
            % Rather than iterating actual edge objects (expensive for
            % large nets), read the weight matrix for each adjacent pair
            % and build edges between visual representatives.
            totalVizEdges = sum(vizCounts(1:end-1) .* vizCounts(2:end));
            edgeFrom    = zeros(totalVizEdges, 1);
            edgeTo      = zeros(totalVizEdges, 1);
            edgeWeights = zeros(totalVizEdges, 1);
            edgeAlpha   = zeros(totalVizEdges, 1);

            ei = 1;
            for L = 1:nL-1
                W = graph.getWeightMatrix(L);  % (d_{L+1} × d_L)
                fromViz = vizIdx{L};           % real indices of visual FROM nodes
                toViz   = vizIdx{L+1};         % real indices of visual TO nodes
                fromOff = layerOffset(L);
                toOff   = layerOffset(L+1);

                for fi = 1:numel(fromViz)
                    for ti = 1:numel(toViz)
                        ri = toViz(ti);   % real row in W (destination neuron)
                        ci = fromViz(fi); % real col in W (source neuron)
                        w  = W(ri, ci);

                        edgeFrom(ei)    = fromOff + fi;
                        edgeTo(ei)      = toOff   + ti;
                        edgeWeights(ei) = w;
                        edgeAlpha(ei)   = obj.ACTIVE_ALPHA;
                        ei = ei + 1;
                    end
                end
            end

            % ---- 5. Build layer labels (show true size) -------------- %
            activations = spec.Activations;
            layerNames  = cell(1, nL);
            for L = 1:nL
                d    = sizes(L);
                act  = activations{L};
                comp = '';
                if d > obj.VIZ_MAX_PER_LAYER
                    comp = sprintf(' [%d→%d]', d, vizCounts(L));
                end
                if L == 1
                    prefix = 'Input';
                elseif L == nL
                    prefix = 'Output';
                else
                    prefix = 'Hidden';
                end
                layerNames{L} = sprintf('%s%s\n(%s)', prefix, comp, act);
            end

            % ---- 6. Ellipsis markers for compressed layers ------------ %
            % For each layer that was compressed, record the X midpoint
            % and a Y midpoint for the '⋮' annotation.
            ellipsisPos = zeros(0, 2);   % Nx2: [x, y] per compressed layer
            for L = 1:nL
                if sizes(L) > obj.VIZ_MAX_PER_LAYER
                    xVal = (L - 1) * 2;
                    % Midpoint between top and bottom visible nodes
                    rows = layerOffset(L) + (1:vizCounts(L));
                    yMid = mean(positions(rows, 2));
                    ellipsisPos(end+1, :) = [xVal, yMid]; %#ok<AGROW>
                end
            end

            % ---- 7. Output node labels (class index next to each node) - %
            % Label each output node with its class index (0-based).
            dOut    = sizes(nL);
            outRows = layerOffset(nL) + (1:vizCounts(nL));
            outPos  = positions(outRows, :);           % visible output node positions
            outLabels = cell(vizCounts(nL), 1);
            % Map visible node index back to real class index
            outVizIdx = vizIdx{nL};                    % real indices of visible output nodes
            for vi = 1:numel(outVizIdx)
                outLabels{vi} = num2str(outVizIdx(vi) - 1);  % 0-based class label
            end

            % ---- 8. Assemble struct ----------------------------------- %
            % NodeRealIDs: maps each visual node (in draw order) to its
            % actual node ID string ('L#_N#') in the live NNGraph.
            nodeRealIDs  = cell(Kv, 1);
            isInteractive = true;   % false when any layer was compressed
            k = 1;
            for L = 1:nL
                vIdx = vizIdx{L};
                if numel(vIdx) < sizes(L)
                    isInteractive = false;   % this layer is compressed
                end
                for vi = 1:numel(vIdx)
                    nodeRealIDs{k} = sprintf('L%d_N%d', L, vIdx(vi));
                    k = k + 1;
                end
            end

            data.Positions      = positions;
            data.NodeColors     = nodeColors;
            data.NodeSizes      = nodeSizes;
            data.NodeStates     = nodeStates;
            data.NodeValues     = nodeValues;     % raw activation (for numeric overlay)
            data.EdgeFrom       = edgeFrom;
            data.EdgeTo         = edgeTo;
            data.EdgeWeights    = edgeWeights;
            data.EdgeAlpha      = edgeAlpha;
            data.LayerNames     = layerNames;
            data.NodeDead       = nodeDead;
            data.NumLayers      = nL;
            data.NumNodes       = Kv;
            data.EllipsisPos    = ellipsisPos;
            data.OutputLabels   = outLabels;
            data.OutputPos      = outPos;
            data.NodeRealIDs    = nodeRealIDs;
            data.IsInteractive  = isInteractive;
        end
    end

    methods (Static)
        function pos = computePositions(graph)
            % COMPUTEPOSITIONS  Layered-grid XY for all nodes.
            %
            %   X = layer index (1..L), spaced by 2 units.
            %   Y = node rank within layer, centered (mean = 0),
            %       capped to MAX_HEIGHT so large layers don't collapse the canvas.
            %
            % Returns (K×2) double.

            MAX_HEIGHT = 12;   % max vertical span per layer (units)
            VIZ_MAX    = 16;   % max visible nodes per layer

            nL    = graph.numLayers();
            spec  = graph.toSpec();
            sizes = spec.LayerSizes;

            % Count visual nodes per layer
            vizCounts = zeros(1, nL);
            for L = 1:nL
                d = sizes(L);
                if d <= VIZ_MAX
                    vizCounts(L) = d;
                else
                    vizCounts(L) = VIZ_MAX;
                end
            end
            Kv  = sum(vizCounts);
            pos = zeros(Kv, 2);

            k = 1;
            for L = 1:nL
                dv   = vizCounts(L);
                xVal = (L - 1) * 2;
                if dv == 1
                    yVals = 0;
                else
                    halfSpan = min((dv-1)/2, MAX_HEIGHT/2);
                    yVals = linspace(-halfSpan, halfSpan, dv);
                end
                for ni = 1:dv
                    pos(k, :) = [xVal, yVals(ni)];
                    k = k + 1;
                end
            end
        end
    end
end
