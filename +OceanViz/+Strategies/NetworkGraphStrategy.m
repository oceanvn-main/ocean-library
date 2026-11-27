classdef NetworkGraphStrategy < OceanViz.Strategies.BaseStrategy
    % NETWORKGRAPHSTRATEGY  OceanViz rendering strategy for NNGraph.
    %
    %   Renders a neural network from a pre-computed NNGraphData struct
    %   produced by any ILayoutEngine.
    %   How they should look to the user?
    %
    %   VISUAL ENCODING:
    %     Nodes  : scatter circles colored by activation value (blue-white),
    %              gray if Disabled, large yellow if Highlighted.
    %     Edges  : lines; blue = positive weight, red = negative weight.
    %              Width ∝ abs(weight), alpha = 0.15 if either endpoint Disabled.
    %     Labels : layer name + activation function above the top node.
    %
    %   Usage:
    %     import OceanDL.Viz.*;
    %     engine = NNLayeredLayout();
    %     data   = engine.toData(graph);
    %     strat  = OceanViz.Strategies.NetworkGraphStrategy();
    %     strat.plot(ax, data, struct('Title', 'My Network'));
    %
    %   Options struct fields (all optional):
    %     .Title       (string)  Title string for the axes
    %     .ShowWeights (logical) Show edge colorbar (default: true)
    %     .ShowLabels  (logical) Show layer name labels (default: true)
    %     .ShowDisabled(logical) Show disabled nodes dimmed (default: true)
    %
    % See also: OceanDL.Viz.ILayoutEngine
    %           OceanDL.Viz.NNLayeredLayout
    %           OceanDL.Viz.MatlabNativeLayout

    properties (Constant)
        % ---- Edge visual constants ----------------------------------- %
        EDGE_WIDTH_SCALE = 3.0                  % pt per unit weight magnitude
        MAX_EDGE_WIDTH   = 4.0                  % cap in pt

        % ---- Node visual constants ----------------------------------- %
        NODE_BASE_SIZE   = 120                  % scatter marker area units
    end

    properties
        % Pluggable node renderer — default is CircleNodeRenderer.
        % Swap with any INodeRenderer subclass to change node appearance.
        %   Example:
        %     strat = NetworkGraphStrategy('NodeRenderer', MyIconRenderer());
        % Type: OceanViz.Strategies.INodeRenderer (enforced in constructor)
        NodeRenderer  % INodeRenderer — type annotation removed to avoid load-time error
    end

    % ==================================================================
    methods
        function obj = NetworkGraphStrategy(options)
            % NETWORKGRAPHSTRATEGY  Construct with optional NodeRenderer.
            %
            % Name-Value:
            %   NodeRenderer - INodeRenderer instance (default: CircleNodeRenderer)
            if nargin < 1, options = struct(); end
            if isfield(options, 'NodeRenderer')
                obj.NodeRenderer = options.NodeRenderer;
            else
                obj.NodeRenderer = OceanViz.Strategies.CircleNodeRenderer();
            end
        end

        function plot(obj, ax, data, options)
            % PLOT  Render a NNGraphData struct into an axes object.
            %
            % INPUTS:
            %   ax      - MATLAB axes handle
            %   data    - NNGraphData struct from any ILayoutEngine.toData()
            %   options - (optional) plain struct with option fields

            if nargin < 4, options = struct(); end

            showWeights  = obj.getOption_(options, 'ShowWeights',  true);
            showLabels   = obj.getOption_(options, 'ShowLabels',   true);
            titleStr     = obj.getOption_(options, 'Title',        '');

            hold(ax, 'on');

            % ---- 1. Draw edges --------------------------------------- %
            obj.drawEdges_(ax, data, options);

            % ---- 2. Draw nodes --------------------------------------- %
            obj.drawNodes_(ax, data, options);

            % ---- 3. Layer labels ------------------------------------- %
            if showLabels
                obj.drawLabels_(ax, data, options);
            end

            % ---- 4. Edge colorbar ------------------------------------ %
            cbAx = obj.getOption_(options, 'ColorbarAxes', []);
            if showWeights && ~isempty(data.EdgeWeights)
                if ~isempty(cbAx) && isvalid(cbAx)
                    obj.drawColorbar_(cbAx, data, options);
                else
                    obj.drawColorbar_(ax, data, options);
                end
            end

            % ---- 5. Axes styling (adaptive tight fit) --------------- %
            cBg = [0.08 0.08 0.10];
            if isfield(options, 'Colors') && ~isempty(options.Colors), cBg = options.Colors.Background; end
            ax.Color  = cBg;
            ax.XColor = 'none';
            ax.YColor = 'none';
            ax.Box    = 'off';

            % Compute adaptive bounds from actual positions
            xMin = min(data.Positions(:,1));  xMax = max(data.Positions(:,1));
            yMin = min(data.Positions(:,2));  yMax = max(data.Positions(:,2));
            xRange = max(xMax - xMin, 0.1);
            yRange = max(yMax - yMin, 0.1);

            % Right side: extra room for output labels (0.25 offset + text width)
            xPadL = max(0.4, xRange * 0.04);
            xPadR = max(0.9, xRange * 0.08);
            yPadB = max(0.5, yRange * 0.05);
            yPadT = max(1.2, yRange * 0.12);  % headroom for layer-name labels

            ax.XLim = [xMin - xPadL,  xMax + xPadR];
            ax.YLim = [yMin - yPadB,  yMax + yPadT];

            % ---- 6. Ellipsis markers (compressed layers) ------------ %
            if isfield(data, 'EllipsisPos') && ~isempty(data.EllipsisPos)
                obj.drawEllipsis_(ax, data, options);
            end

            % ---- 7. Output class labels ----------------------------- %
            if isfield(data, 'OutputLabels') && ~isempty(data.OutputLabels)
                obj.drawOutputLabels_(ax, data, options);
            end

            % ---- 8. Numeric activation overlay ----------------------- %
            displayMode = obj.getOption_(options, 'NodeDisplay', 'Heatmap');
            if ismember(displayMode, {'Numbers', 'Both'}) && isfield(data, 'NodeValues')
                obj.drawNodeValues_(ax, data, options);
            end

            if ~isempty(titleStr)
                title(ax, titleStr, 'Color', obj.TITLE_COLOR, ...
                    'FontSize', 13, 'FontWeight', 'bold');
            end

            hold(ax, 'off');
        end
    end

    % ==================================================================
    methods (Access = private)

        function drawEdges_(obj, ax, data, options)
            % Draw each edge as a line, colored by weight sign.
            if isempty(data.EdgeFrom), return; end
            
            cPos = [0.25 0.55 1.00]; cNeg = [1.00 0.30 0.25];
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                cPos = options.Colors.Primary; cNeg = options.Colors.Secondary;
            end

            maxW = max(abs(data.EdgeWeights));
            if maxW < 1e-12, maxW = 1; end   % avoid divide-by-zero

            for ei = 1:numel(data.EdgeFrom)
                fromIdx = data.EdgeFrom(ei);
                toIdx   = data.EdgeTo(ei);
                w       = data.EdgeWeights(ei);
                alpha   = data.EdgeAlpha(ei);

                x1 = data.Positions(fromIdx, 1);
                y1 = data.Positions(fromIdx, 2);
                x2 = data.Positions(toIdx,   1);
                y2 = data.Positions(toIdx,   2);

                if w >= 0
                    col = cPos;
                else
                    col = cNeg;
                end

                lw = min(abs(w) / maxW * obj.MAX_EDGE_WIDTH, obj.MAX_EDGE_WIDTH);
                lw = max(lw, 0.3);   % minimum visible

                line(ax, [x1, x2], [y1, y2], ...
                    'Color',     [col(:)', alpha], ...  % RGBA (alpha in 4th element)
                    'LineWidth', lw, ...
                    'HitTest',   'off');
            end
        end

        function drawNodes_(obj, ax, data, options)
            % Call inner strategy or circle renderer
            if isfield(options, 'NodeDisplay') && strcmp(options.NodeDisplay, 'Numbers')
                return; % Heatmap hidden — handled in drawNodeValues_
            end
            
            isDeadArr = false(data.NumNodes, 1);
            if isfield(data, 'NodeDead')
                isDeadArr = data.NodeDead;
            end
            
            for k = 1:data.NumNodes
                obj.NodeRenderer.draw(ax, ...
                    data.Positions(k, 1), ...
                    data.Positions(k, 2), ...
                    data.NodeSizes(k), ...
                    data.NodeColors(k,:), ...
                    data.NodeStates(k), ...
                    isDeadArr(k));
            end
        end

        function drawLabels_(obj, ax, data, options)
            % Draw layer name labels above the top node of each layer.
            cLabel = [0.85 0.85 0.90];
            if isfield(options, 'Colors') && ~isempty(options.Colors), cLabel = options.Colors.Text; end
            
            nL = data.NumLayers;
            % Compute X per layer (group nodes by X coordinate)
            xVals = unique(data.Positions(:, 1));
            if numel(xVals) ~= nL, return; end   % safety check

            maxY  = max(data.Positions(:, 2));
            for L = 1:nL
                x = xVals(L);
                % Find topmost node in this layer
                mask = data.Positions(:,1) == x;
                topY = max(data.Positions(mask, 2));
                text(ax, x, topY + 0.55, data.LayerNames{L}, ...
                    'Color',                cLabel, ...
                    'FontSize',             8, ...
                    'HorizontalAlignment',  'center', ...
                    'VerticalAlignment',    'bottom', ...
                    'Interpreter',          'none');
            end
        end

        function drawColorbar_(obj, ax, data, options)
            % Add a shared colorbar for both edge weights and node activations.
            cPos = [0.25 0.55 1.00]; cNeg = [1.00 0.30 0.25]; cMid = [0.22 0.18 0.35]; cLabel = [0.85 0.85 0.90];
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                cPos = options.Colors.Primary; cNeg = options.Colors.Secondary; cMid = options.Colors.Background; cLabel = options.Colors.Text;
            end
            maxW = max(abs(data.EdgeWeights));
            if maxW < 1e-12, return; end

            nColors = 64;
            half    = nColors / 2;
            cmap    = zeros(nColors, 3);
            for ci = 1:half
                t          = (ci - 1) / (half - 1);
                cmap(ci,:) = (1-t) * cNeg + t * cMid;
            end
            for ci = half+1:nColors
                t          = (ci - half - 1) / (half - 1);
                cmap(ci,:) = (1-t) * cMid + t * cPos;
            end
            colormap(ax, cmap);
            clim(ax, [-maxW, maxW]);
            cb = colorbar(ax, 'eastoutside');
            cb.Color          = cLabel;
            cb.Label.String   = 'Edge weight / Node activation';
            cb.Label.Color    = cLabel;
            cb.FontSize       = 8;
        end

        function drawEllipsis_(~, ax, data, options)
            % Draw a vertical ellipsis at the midpoint of each compressed layer.
            % Uses the actual Unicode character ⋮ (U+22EE) — no TeX needed.
            cDim = [0.60, 0.60, 0.65];
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                if isfield(options.Colors, 'DimText')
                    cDim = options.Colors.DimText;
                else
                    cDim = options.Colors.Text * 0.5 + options.Colors.Background * 0.5;
                end
            end

            vertDots = char(8942);  % ⋮
            for ei = 1:size(data.EllipsisPos, 1)
                x = data.EllipsisPos(ei, 1);
                y = data.EllipsisPos(ei, 2);
                text(ax, x, y, vertDots, ...
                    'Color',               cDim, ...
                    'FontSize',            18, ...
                    'FontWeight',          'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment',   'middle', ...
                    'HitTest',             'off', ...
                    'Interpreter',         'none');
            end
        end

        function drawOutputLabels_(~, ax, data, options)
            % Draw class index labels to the right of each output node.
            cLabel = [0.85 0.88 0.92]; cAccent = [0.30 0.65 0.95];
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                cLabel = options.Colors.Text;
                cAccent = options.Colors.Primary;
            end

            xOffset = 0.25;   % shift label slightly right of node

            % Find max output value to highlight the predicted class
            outRows = (data.NumNodes - numel(data.OutputLabels) + 1) : data.NumNodes;
            outVals = data.NodeValues(outRows);
            maxIdx = -1;
            if numel(outVals) > 1 && max(outVals) > min(outVals)  % Not all equal
                [~, maxIdx] = max(outVals);
            end

            for vi = 1:numel(data.OutputLabels)
                x = data.OutputPos(vi, 1) + xOffset;
                y = data.OutputPos(vi, 2);

                tColor = cLabel;
                fWeight = 'normal';
                if vi == maxIdx
                    tColor = cAccent;
                    fWeight = 'bold';
                end

                text(ax, x, y, data.OutputLabels{vi}, ...
                    'Color',               tColor, ...
                    'FontSize',            9, ...
                    'FontWeight',          fWeight, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment',   'middle', ...
                    'HitTest',             'off', ...
                    'Interpreter',         'none');
            end
        end

        function drawNodeValues_(~, ax, data, options)
            % Render the raw activation value as text centred inside each node.
            % Shows '---' before first forward pass (Value = 0 but idle).
            
            cTxtLight = [1 1 1]; cTxtDark = [0.08 0.08 0.10];
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                cTxtDark = options.Colors.Background;
            end

            for k = 1:data.NumNodes
                x   = data.Positions(k, 1);
                y   = data.Positions(k, 2);
                val = data.NodeValues(k);
                if abs(val) < 1e-8
                    label = '0';
                else
                    label = sprintf('%.2f', val);
                end
                % Choose text color: white on dark nodes, black on bright
                brightness = mean(data.NodeColors(k,:));
                txtColor   = cTxtLight;
                if brightness > 0.65
                    txtColor = cTxtDark;  % dark on bright nodes
                end
                text(ax, x, y, label, ...
                    'Color',               txtColor, ...
                    'FontSize',            7, ...
                    'FontWeight',          'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment',   'middle', ...
                    'HitTest',             'off', ...
                    'PickableParts',       'none', ...
                    'Interpreter',         'none');
            end
        end

        function v = getOption_(~, options, field, default)
            if isfield(options, field)
                v = options.(field);
            else
                v = default;
            end
        end
    end
end
