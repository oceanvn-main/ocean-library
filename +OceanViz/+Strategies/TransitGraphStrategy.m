classdef TransitGraphStrategy < handle
    % TRANSITGRAPHSTRATEGY  OceanViz renderer for transit/graph networks.
    %
    % Renders a transit network using spline curves for routes, themed
    % station markers, labels, path highlights, and annotations.
    %
    % Data contract (struct from TransitNetwork.toVizData()):
    %   .Positions   - [V x 2] station positions
    %   .NodeNames   - {1 x V} station names
    %   .Splines     - {1 x E} IParametricCurve objects (or empty)
    %   .EdgePairs   - [E x 2] [source, target] node IDs
    %   .EdgeLabels  - {1 x E} line names (e.g., 'NSL')
    %   .Style       - (optional) OceanViz.Adapters.GraphStyle
    %
    % Usage:
    %   viz  = OceanViz.Core('Theme', 'Dark');
    %   data = tn.toVizData();
    %   ax   = subplot(1,1,1);
    %   strat = OceanViz.Strategies.TransitGraphStrategy();
    %   strat.render(ax, data, viz.Colors);
    %
    % See also: OceanMath.Transit.TransitNetwork
    %           OceanViz.Adapters.GraphStyle

    properties (Constant)
        CURVE_SAMPLES  = 60     % points per spline evaluation
        NODE_SIZE_DEFAULT = 5   % default marker size (small dots)
        NODE_SIZE_INTERCHANGE = 10 % interchange marker size (prominent)
        NODE_SIZE_TERMINAL = 8  % terminal station marker size
        EDGE_WIDTH_DEFAULT = 2.5
        LABEL_OFFSET_RATIO = 0.015   % label offset as fraction of Y range
        PARALLEL_GAP_RATIO = 0.004   % corridor offset as fraction of data diagonal
    end

    methods
        function render(obj, ax, data, colors)
            % RENDER  Draw the full transit network.
            %
            %   strat.render(ax, data, colors)
            %
            %   Inputs:
            %     ax     - axes handle
            %     data   - struct from TransitNetwork.toVizData()
            %     colors - OceanViz.Core.Colors struct
            arguments
                obj
                ax
                data   (1,1) struct
                colors (1,1) struct
            end

            hold(ax, 'on');

            hasStyle = isfield(data, 'Style') && ~isempty(data.Style);

            % 1. Draw routes (splines or straight lines)
            obj.drawRoutes_(ax, data, colors, hasStyle);

            % 2. Draw highlighted paths (overlay)
            if hasStyle
                obj.drawHighlights_(ax, data, colors);
            end

            % 3. Draw station markers
            obj.drawStations_(ax, data, colors, hasStyle);

            % 4. Draw station labels
            obj.drawLabels_(ax, data, colors, hasStyle);

            % 5. Draw annotations
            if hasStyle
                obj.drawAnnotations_(ax, data);
            end

            % 6. Axes styling
            axis(ax, 'equal');
            ax.Color  = colors.Background;
            ax.XColor = colors.Text;
            ax.YColor = colors.Text;

            % Adaptive padding (relative)
            pos = data.Positions;
            xRange = range(pos(:,1)); if xRange == 0, xRange = 1; end
            yRange = range(pos(:,2)); if yRange == 0, yRange = 1; end
            xPad = xRange * 0.10;
            yPad = yRange * 0.10;
            xlim(ax, [min(pos(:,1)) - xPad, max(pos(:,1)) + xPad]);
            ylim(ax, [min(pos(:,2)) - yPad, max(pos(:,2)) + yPad]);
            % Note: hold state is left as-is for callers to manage
        end
    end

    % ================================================================== %
    methods (Access = private)

        function drawRoutes_(obj, ax, data, colors, hasStyle)
            % DRAWROUTES_ Extracts raw geometric Splines from the Math Network 
            % and visualizes them on the UI. Gracefully detects overlapping track sections 
            % (Shared Corridors) and offsets them dynamically to prevent visual collision.
            nEdges = size(data.EdgePairs, 1);
            t_dense = linspace(0, 1, obj.CURVE_SAMPLES)';

            % --- Group edges by corridor (unordered station pair) ---
            corridorMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for ei = 1:nEdges
                src = data.EdgePairs(ei, 1);
                tgt = data.EdgePairs(ei, 2);
                key = sprintf('%d-%d', min(src,tgt), max(src,tgt));
                if corridorMap.isKey(key)
                    corridorMap(key) = [corridorMap(key), ei];
                else
                    corridorMap(key) = ei;
                end
            end

            % --- Draw each corridor ---
            keys = corridorMap.keys();
            for ki = 1:numel(keys)
                edgeIndices = corridorMap(keys{ki});
                k = numel(edgeIndices);  % how many lines share this corridor

                % Compute offsets: centered around 0
                % Gap is proportional to map diagonal
                dataDiag = sqrt(range(data.Positions(:,1))^2 + range(data.Positions(:,2))^2);
                if dataDiag == 0, dataDiag = 1; end
                gap = dataDiag * obj.PARALLEL_GAP_RATIO;
                if k == 1
                    offsets = 0;
                else
                    offsets = linspace(-(k-1)/2, (k-1)/2, k) * gap;
                end

                for li = 1:k %Number of line sharing this corridor
                    ei = edgeIndices(li);
                    src = data.EdgePairs(ei, 1);
                    tgt = data.EdgePairs(ei, 2);

                    % Get base curve points
                    if ei <= numel(data.Splines) && ~isempty(data.Splines{ei})
                        pts = data.Splines{ei}.evaluate(t_dense);
                    else
                        pts = [data.Positions(src, :); data.Positions(tgt, :)]; %fallback to straigthline
                    end

                    % Apply perpendicular offset if sharing corridor
                    if k > 1
                        pts = obj.offsetCurve_(pts, offsets(li));
                    end

                    % Determine visual style
                    edgeColor = colors.Scatter;
                    edgeWidth = obj.EDGE_WIDTH_DEFAULT;
                    edgeDash  = '-';

                    if hasStyle
                        style = data.Style;
                        if style.hasEdgeColor(ei)
                            edgeColor = style.getEdgeColor(ei);
                        end
                        if style.hasEdgeWidth(ei)
                            edgeWidth = style.getEdgeWidth(ei);
                        end
                        edgeDash = style.getEdgeDash(ei);
                    end

                    plot(ax, pts(:,1), pts(:,2), edgeDash, ...
                        'Color', edgeColor, ...
                        'LineWidth', edgeWidth);
                end
            end
        end

        function drawHighlights_(obj, ax, data, colors)
            % DRAWHIGHLIGHTS_ Used for routing overlays. Iterates node sequence targets,
            % mapping them to their corresponding spline edge indices. Traces over the 
            % existing math with an augmented thick transparent brush to simulate a neon glow.
            style = data.Style;
            paths = style.getPaths(); %Get records saved {NodeSequence, Color, Width}
            t_dense = linspace(0, 1, obj.CURVE_SAMPLES)'; %200 points for smooth curve

            for pi = 1:numel(paths)
                p = paths{pi};
                nodeSeq = p.NodeSequence;

                for si = 1:numel(nodeSeq) - 1
                    src = nodeSeq(si);
                    tgt = nodeSeq(si + 1);

                    % Find matching edge index. Which edge connects src and tgt?
                    edgeIdx = obj.findEdgeIndex_(data, src, tgt);
                    if edgeIdx == 0, continue; end

                    % Get curve points
                    if edgeIdx <= numel(data.Splines) && ~isempty(data.Splines{edgeIdx})
                        pts = data.Splines{edgeIdx}.evaluate(t_dense);
                    else
                        pts = [data.Positions(src,:); data.Positions(tgt,:)]; %Falllback: 2 endpoints only
                    end

                    % Glow effect: wide semi-transparent line beneath
                    plot(ax, pts(:,1), pts(:,2), '-', ...
                        'Color', [p.Color, 0.25], ...
                        'LineWidth', p.Width * 2.5);

                    plot(ax, pts(:,1), pts(:,2), '-', ...
                        'Color', p.Color, ...
                        'LineWidth', p.Width);
                end
            end
        end

        function drawStations_(obj, ax, data, colors, hasStyle)
            % DRAWSTATIONS_ Scans data arrays for Interchange or Terminal flags, dynamically  
            % replacing default DOT markers with specialized icons (bullseyes or squares) 
            % so that complex topological features stand out distinctly to the user.
            V = size(data.Positions, 1);
            
            % Detect interchange stations from data metadata
            isInterchange = false(V, 1);
            if isfield(data, 'InterchangeIds') && ~isempty(data.InterchangeIds)
                isInterchange(data.InterchangeIds) = true;
            end
            
            % Detect terminal stations from data metadata
            isTerminal = false(V, 1);
            if isfield(data, 'TerminalIds') && ~isempty(data.TerminalIds)
                isTerminal(data.TerminalIds) = true;
            end

            for ni = 1:V
                x = data.Positions(ni, 1);
                y = data.Positions(ni, 2);

                nodeColor = colors.Text;
                icon      = 'circle';

                if hasStyle
                    style = data.Style;
                    if style.hasNodeColor(ni)
                        nodeColor = style.getNodeColor(ni);
                    end
                    if style.hasNodeIcon(ni)
                        icon = style.getNodeIcon(ni);
                    end
                end
                
                % Auto-promote from data metadata (interchange takes precedence)
                if isInterchange(ni) && strcmp(icon, 'circle')
                    icon = 'interchange';
                elseif isTerminal(ni) && strcmp(icon, 'circle')
                    icon = 'terminal';
                end

                switch icon
                    case 'circle'
                        % Small dot for regular stations
                        plot(ax, x, y, 'o', ...
                            'MarkerSize', obj.NODE_SIZE_DEFAULT, ...
                            'MarkerFaceColor', nodeColor, ...
                            'MarkerEdgeColor', colors.Background, ...
                            'LineWidth', 1.2, ...
                            'HandleVisibility', 'off');
                    case 'terminal'
                        % Terminal: larger diamond with white fill + dark border
                        plot(ax, x, y, 'd', ...
                            'MarkerSize', obj.NODE_SIZE_TERMINAL, ...
                            'MarkerFaceColor', [1 1 1], ...
                            'MarkerEdgeColor', [0.3 0.3 0.35], ...
                            'LineWidth', 1.8, ...
                            'HandleVisibility', 'off');
                    case 'square'
                        plot(ax, x, y, 's', ...
                            'MarkerSize', obj.NODE_SIZE_DEFAULT, ...
                            'MarkerFaceColor', nodeColor, ...
                            'MarkerEdgeColor', colors.Background, ...
                            'LineWidth', 1.2, ...
                            'HandleVisibility', 'off');
                    case 'diamond'
                        plot(ax, x, y, 'd', ...
                            'MarkerSize', obj.NODE_SIZE_DEFAULT, ...
                            'MarkerFaceColor', nodeColor, ...
                            'MarkerEdgeColor', colors.Background, ...
                            'LineWidth', 1.2, ...
                            'HandleVisibility', 'off');
                    case 'interchange'
                        % Prominent double-ring interchange marker
                        plot(ax, x, y, 'o', ...
                            'MarkerSize', obj.NODE_SIZE_INTERCHANGE, ...
                            'MarkerFaceColor', [1 1 1], ...
                            'MarkerEdgeColor', [0.15 0.15 0.15], ...
                            'LineWidth', 2.0, ...
                            'HandleVisibility', 'off');
                        plot(ax, x, y, 'o', ...
                            'MarkerSize', obj.NODE_SIZE_INTERCHANGE * 0.45, ...
                            'MarkerFaceColor', [0.15 0.15 0.15], ...
                            'MarkerEdgeColor', [0.15 0.15 0.15], ...
                            'LineWidth', 1.0, ...
                            'HandleVisibility', 'off');
                    otherwise
                        plot(ax, x, y, 'o', ...
                            'MarkerSize', obj.NODE_SIZE_DEFAULT, ...
                            'MarkerFaceColor', nodeColor, ...
                            'MarkerEdgeColor', colors.Background, ...
                            'LineWidth', 1.2, ...
                            'HandleVisibility', 'off');
                end
            end
        end

        function drawLabels_(obj, ax, data, colors, hasStyle)
            % DRAWLABELS_ Renders station names algorithmically preventing collision.
            % Checks metadata to parse whether labels should be clamped to 'left', 'right',
            % or centered offsets depending on the geometric layout. Colors labels distinctly 
            % (e.g. Gold for interchanges).
            V = size(data.Positions, 1);

            % Compute default label offset relative to data range
            ySpan = range(data.Positions(:,2));
            if ySpan == 0, ySpan = 1; end
            defaultOffset = [0, ySpan * obj.LABEL_OFFSET_RATIO];
            
            % Check if per-station label offsets are provided
            hasLabelOffsets = isfield(data, 'LabelOffsets') && ~isempty(data.LabelOffsets);

            % Detect interchange stations
            isInterchange = false(V, 1);
            if isfield(data, 'InterchangeIds') && ~isempty(data.InterchangeIds)
                isInterchange(data.InterchangeIds) = true;
            end
            
            % Detect terminal stations
            isTerminal = false(V, 1);
            if isfield(data, 'TerminalIds') && ~isempty(data.TerminalIds)
                isTerminal(data.TerminalIds) = true;
            end

            for ni = 1:V
                % Check visibility
                if hasStyle && ~data.Style.getNodeLabelVisible(ni)
                    continue;
                end

                % Per-station offset from metadata, or fallback
                if hasLabelOffsets && ni <= numel(data.LabelOffsets) && ~isempty(data.LabelOffsets{ni})
                    offset = data.LabelOffsets{ni};
                else
                    offset = defaultOffset;
                end
                
                x = data.Positions(ni, 1) + offset(1);
                y = data.Positions(ni, 2) + offset(2);
                
                % Determine text alignment from offset direction
                hAlign = 'center';
                if offset(1) > 0.05, hAlign = 'left'; elseif offset(1) < -0.05, hAlign = 'right'; end
                vAlign = 'middle';
                if offset(2) > 0.05, vAlign = 'bottom'; elseif offset(2) < -0.05, vAlign = 'top'; end

                % Distinct styling per station role
                if isInterchange(ni)
                    fontSize  = 7;
                    fontColor = [1.0, 0.85, 0.3];   % gold
                elseif isTerminal(ni)
                    fontSize  = 6.5;
                    fontColor = [0.6, 0.9, 1.0];    % light cyan
                else
                    fontSize  = 5.5;
                    fontColor = colors.Text;
                end

                text(ax, x, y, data.NodeNames{ni}, ...
                    'Color', fontColor, ...
                    'FontSize', fontSize, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', hAlign, ...
                    'VerticalAlignment', vAlign, ...
                    'Interpreter', 'none');
            end
        end

        function drawAnnotations_(~, ax, data)
            % DRAWANNOTATIONS_ Renders arbitrary free-text elements embedded in the 
            % JSON map file (e.g., "Airport", "City Center") onto the axes.
            style = data.Style;
            anns = style.getAnnotations();

            for ai = 1:numel(anns)
                a = anns{ai};
                text(ax, a.Position(1), a.Position(2), a.Text, ...
                    'Color', a.Color, ...
                    'FontSize', a.FontSize, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'Interpreter', 'none');
            end
        end

        function idx = findEdgeIndex_(~, data, src, tgt)
            % FINDEDGEINDEX_ Lookup table utility. Given two station IDs, discovers
            % which abstract edge connects them regardless of edge directionality.
            idx = 0;
            for i = 1:size(data.EdgePairs, 1)
                if (data.EdgePairs(i,1) == src && data.EdgePairs(i,2) == tgt) || ...
                   (data.EdgePairs(i,1) == tgt && data.EdgePairs(i,2) == src)
                    idx = i;
                    return;
                end
            end
        end

        function pts = offsetCurve_(~, pts, d)
            % OFFSETCURVE_  Calculates orthogonal displacement strictly for graphic layout.
            %
            % Computes finite differences to find the geometric tangent of the spline,
            % rotates 90 degrees strictly CCW to generate a unit normal vector, then
            % shifts the points sideways so overlapping lines render perfectly parallel side-by-side.
            N = size(pts, 1);
            if N < 2, return; end

            % Tangent via central differences (forward/backward at ends)
            dx = zeros(N, 1);
            dy = zeros(N, 1);
            %Where is point 1 heading
            dx(1)     = pts(2,1) - pts(1,1);
            dy(1)     = pts(2,2) - pts(1,2);
            %Where is point N heading
            dx(N)     = pts(N,1) - pts(N-1,1);
            dy(N)     = pts(N,2) - pts(N-1,2);
            %Where are the middle points heading?
            dx(2:N-1) = (pts(3:N,1) - pts(1:N-2,1)) / 2;
            dy(2:N-1) = (pts(3:N,2) - pts(1:N-2,2)) / 2;

            % Normalize tangent
            len = sqrt(dx.^2 + dy.^2);
            len(len < 1e-12) = 1;  % avoid division by zero
            dx = dx ./ len;
            dy = dy ./ len;

            % Normal = 90 deg CCW rotation of tangent: n = (-dy, dx)
            nx = -dy;
            ny =  dx;

            % Offset
            pts(:,1) = pts(:,1) + nx * d;
            pts(:,2) = pts(:,2) + ny * d;
        end
    end
end
