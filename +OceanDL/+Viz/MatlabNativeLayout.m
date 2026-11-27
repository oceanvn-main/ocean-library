classdef MatlabNativeLayout < OceanDL.Viz.ILayoutEngine
    % MATLABNATIVELAYOUT  Layout backend wrapping MATLAB's digraph engine.
    %
    %   Builds a MATLAB `digraph` from the NNGraph edges, runs one of
    %   MATLAB's built-in layout algorithms to compute node positions, then
    %   returns those positions as an NNGraphData struct so that
    %   NetworkGraphStrategy can render with its own visual encoding on top.
    %
    %   Also provides a `render()` method that delegates entirely to MATLAB's
    %   native graph rendering pipeline — bypassing NetworkGraphStrategy.
    %
    %   Supported MATLAB layouts:
    %     'layered'   — left-to-right flow (ideal for feedforward NNs)
    %     'force'     — Fruchterman-Reingold force-directed
    %     'circle'    — all nodes on a circle
    %     'subspace'  — dimensionality-reduction embedding
    %     'auto'      — MATLAB chooses automatically
    %
    %   Usage:
    %     engine = OceanDL.Viz.MatlabNativeLayout('layered');
    %     data   = engine.toData(graph);          % → our strategy renders
    %     strat.plot(ax, data, options);
    %
    %     engine.render(ax, graph, options);       % → MATLAB renders natively
    %
    % See also: OceanDL.Viz.ILayoutEngine
    %           OceanDL.Viz.NNLayeredLayout

    properties
        MatlabLayout (1,:) char = 'layered'   % MATLAB layout string
    end

    properties (Constant, Access = private)
        ACTIVE_ALPHA   = 1.00
        INACTIVE_ALPHA = 0.15
        NODE_BASE_SIZE = 120
    end

    methods
        function obj = MatlabNativeLayout(matlabLayout)
            % MATLABNATIVELAYOUT  Construct with a MATLAB layout name.
            %
            % INPUTS:
            %   matlabLayout - (optional) char, default: 'layered'
            if nargin >= 1
                obj.MatlabLayout = matlabLayout;
            end
        end

        % ----------------------------------------------------------------
        function data = toData(obj, graph, options)
            % TODATA  Run MATLAB digraph layout → positions → NNGraphData.
            %
            %   Use this when you want MATLAB's layout algorithm for node
            %   placement but still want our visual encoding (activation
            %   colors, weight colors, disabled dimming).
            if nargin < 3, options = struct(); end

            [G, flatMap] = obj.buildDigraph_(graph);

            % Ask MATLAB to compute node positions using the chosen layout
            fig_hidden = figure('Visible', 'off');
            try
                ax_tmp = axes(fig_hidden);
                h = plot(ax_tmp, G, 'Layout', obj.MatlabLayout, ...
                    'NodeLabel', {}, 'EdgeLabel', {});
                xCoords = h.XData';   % (K×1)
                yCoords = h.YData';   % (K×1)
            catch ex
                close(fig_hidden);
                rethrow(ex);
            end
            close(fig_hidden);

            positions = [xCoords, yCoords];  % K×2

            % Delegate visual encoding to NNLayeredLayout (same logic)
            baseEngine = OceanDL.Viz.NNLayeredLayout();
            data       = baseEngine.toData(graph, options);

            % Override positions with MATLAB's layout
            data.Positions = positions;
        end

        % ----------------------------------------------------------------
        function render(obj, ax, graph, options)
            % RENDER  Let MATLAB draw the graph natively — full native pipeline.
            %
            %   This bypasses NetworkGraphStrategy entirely and uses MATLAB's
            %   digraph plot() directly. Edge widths proportional to |weight|.
            %
            % INPUTS:
            %   ax      - axes handle
            %   graph   - NNGraph
            %   options - (optional) struct:
            %     .Title       (string)
            %     .EdgeScale   (double) multiplier for edge width (default: 3)
            if nargin < 4, options = struct(); end

            scale = 3;
            if isfield(options, 'EdgeScale'), scale = options.EdgeScale; end

            [G, ~] = obj.buildDigraph_(graph);

            % Edge widths from absolute weight values
            weights = G.Edges.Weight;
            widths  = max(0.5, min(4, abs(weights) * scale));

            h = plot(ax, G, 'Layout', obj.MatlabLayout, ...
                'NodeLabel', {}, ...
                'EdgeLabel', {}, ...
                'LineWidth', 1.0, ...
                'ArrowSize', 8, ...
                'NodeColor', [0.3 0.6 1.0], ...
                'EdgeColor', [0.5 0.5 0.5]);

            % Apply per-edge widths
            h.LineWidth = mean(widths);   % scalar fallback
            for ei = 1:numedges(G)
                highlight(h, G.Edges.EndNodes(ei,1), G.Edges.EndNodes(ei,2), ...
                    'LineWidth', widths(ei));
            end

            ax.Color   = [0.10 0.10 0.12];
            ax.XColor  = 'none';
            ax.YColor  = 'none';
            ax.Box     = 'off';

            if isfield(options, 'Title')
                title(ax, options.Title, 'Color', [0.9 0.9 0.9]);
            end
        end
    end

    % ====================================================================
    methods (Access = private)
        function [G, flatMap] = buildDigraph_(obj, graph)
            % BUILDDIGRAPH_  Convert NNGraph edges to a MATLAB digraph.
            %   Returns a weighted digraph G and a map nodeID→flatIndex.

            nL    = graph.numLayers();
            spec  = graph.toSpec();
            K     = sum(spec.LayerSizes);

            % Create node name list (flat order)
            nodeNames = cell(K, 1);
            flatMap   = containers.Map('KeyType','char','ValueType','double');
            k = 1;
            for L = 1:nL
                nodes = graph.getLayer(L).getNodes();
                for ni = 1:numel(nodes)
                    nd = nodes{ni};
                    nodeNames{k} = char(nd.ID);  % node name
                    flatMap(char(nd.ID)) = k;   % node name to flat index
                    k = k + 1;
                end
            end

            % Build edge table
            allEdges = graph.getEdges();
            E        = numel(allEdges);
            srcIdx   = zeros(E, 1);
            dstIdx   = zeros(E, 1);
            wts      = zeros(E, 1);
            for ei = 1:E
                e         = allEdges{ei};
                srcIdx(ei) = flatMap(char(e.FromNode.ID));
                dstIdx(ei) = flatMap(char(e.ToNode.ID));
                wts(ei)    = e.Weight_;
            end

            G = digraph(srcIdx, dstIdx, wts, nodeNames);
        end
    end
end
