classdef LegendManager < handle
    % LEGENDMANAGER  Collects legend entries and renders a legend panel.
    %
    % Reusable legend builder for any OceanViz visualization. Supports
    % line entries (colored lines), icon entries (marker symbols),
    % and custom symbol entries (dashed paths, highlights, etc.).
    %
    % Usage:
    %   lm = OceanViz.Adapters.LegendManager();
    %   lm.addLine('NSL', 'North-South Line', [0.85 0.15 0.15]);
    %   lm.addIcon('interchange', 'Interchange Station');
    %   lm.addSymbol('highlight', 'Shortest Path', [0.2 1 0.8], '--');
    %   lm.render(ax, 'Position', 'northeast');
    %
    %   % Or auto-populate from TransitNetwork:
    %   lm.fromTransitNetwork(tn);
    %   lm.render(ax);
    %
    % See also: OceanViz.Core, OceanViz.Adapters.GraphStyle

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Entries_  cell = {}   % cell array of entry structs
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = LegendManager()
            % LEGENDMANAGER  Create an empty legend manager.
            obj.Entries_ = {};
        end

        % ============================================================== %
        %  Add Entries
        % ============================================================== %

        function addLine(obj, code, label, color, options)
            % ADDLINE  Add a colored line entry.
            %
            %   lm.addLine('NSL', 'North-South Line', [0.85 0.15 0.15])
            arguments
                obj
                code   (1,:) char
                label  (1,:) char
                color  (1,3) double
                options.LineWidth (1,1) double = 3
                options.LineStyle (1,:) char = '-'
            end
            entry.Type      = 'line';
            entry.Code      = code;
            entry.Label     = label;
            entry.Color     = color;
            entry.LineWidth = options.LineWidth;
            entry.LineStyle = options.LineStyle;
            obj.Entries_{end+1} = entry;
        end

        function addIcon(obj, iconType, label, options)
            % ADDICON  Add an icon/marker explanation entry.
            %
            %   lm.addIcon('interchange', 'Interchange Station')
            %   lm.addIcon('terminal', 'Terminal', 'MarkerColor', [1 1 1])
            arguments
                obj
                iconType    (1,:) char
                label       (1,:) char
                options.MarkerColor (1,3) double = [1 1 1]
                options.MarkerSize  (1,1) double = 8
            end
            entry.Type        = 'icon';
            entry.IconType    = iconType;
            entry.Label       = label;
            entry.MarkerColor = options.MarkerColor;
            entry.MarkerSize  = options.MarkerSize;
            obj.Entries_{end+1} = entry;
        end

        function addSymbol(obj, code, label, color, lineStyle, options)
            % ADDSYMBOL  Add a custom symbol/style entry.
            %
            %   lm.addSymbol('path', 'Shortest Path', [0.2 1 0.8], '--')
            arguments
                obj
                code      (1,:) char
                label     (1,:) char
                color     (1,3) double
                lineStyle (1,:) char = '-'
                options.LineWidth (1,1) double = 2
            end
            entry.Type      = 'symbol';
            entry.Code      = code;
            entry.Label     = label;
            entry.Color     = color;
            entry.LineStyle = lineStyle;
            entry.LineWidth = options.LineWidth;
            obj.Entries_{end+1} = entry;
        end

        % ============================================================== %
        %  Auto-populate
        % ============================================================== %

        function fromTransitNetwork(obj, tn)
            % FROMTRANSITNETWORK  Auto-populate from a TransitNetwork.
            %
            %   Adds all registered lines and interchange icon.
            %
            %   lm.fromTransitNetwork(tn)
            arguments
                obj
                tn   % OceanMath.Transit.TransitNetwork
            end

            % Add all lines
            codes = tn.getLineNames();
            for i = 1:numel(codes)
                line = tn.getLine(codes{i});
                obj.addLine(line.Code, line.Name, line.Color);
            end

            % Add interchange icon if any exist
            ixIds = tn.getInterchanges();
            if ~isempty(ixIds)
                obj.addIcon('interchange', 'Interchange');
            end
        end

        % ============================================================== %
        %  Render
        % ============================================================== %

        function render(obj, ax, options)
            % RENDER  Draw the legend panel on the given axes.
            %
            %   lm.render(ax)
            %   lm.render(ax, 'Position', 'northeast')
            %   lm.render(ax, 'BackgroundColor', [0.1 0.1 0.1])
            arguments
                obj
                ax
                options.Position        (1,:) char = 'northeast'
                options.BackgroundColor (1,3) double = [0.12, 0.12, 0.14]
                options.BackgroundAlpha (1,1) double = 0.85
                options.TextColor       (1,3) double = [0.9, 0.9, 0.9]
                options.FontSize        (1,1) double = 9
                options.Padding         (1,1) double = 0.02
            end

            if isempty(obj.Entries_), return; end

            nEntries = numel(obj.Entries_);

            % Compute position in normalized axes coordinates
            [xBase, yBase, xDir] = obj.computePosition_( ...
                options.Position, nEntries, options.Padding);

            % Draw background box
            boxH = nEntries * 0.045 + 0.025;
            boxW = 0.22;
            if xDir < 0
                boxX = xBase - boxW;
            else
                boxX = xBase;
            end
            boxY = yBase - boxH;

            % Convert normalized to data coordinates
            xlims = get(ax, 'XLim');
            ylims = get(ax, 'YLim');
            dx = xlims(2) - xlims(1);
            dy = ylims(2) - ylims(1);

            bx = xlims(1) + boxX * dx;
            by = ylims(1) + boxY * dy;
            bw = boxW * dx;
            bh = boxH * dy;

            % Background rectangle
            rectangle(ax, 'Position', [bx, by, bw, bh], ...
                'FaceColor', options.BackgroundColor, ...
                'EdgeColor', [0.3 0.3 0.35], ...
                'LineWidth', 1, ...
                'Curvature', [0.05, 0.05]);

            % Draw entries top to bottom
            for i = 1:nEntries
                entry = obj.Entries_{i};
                rowY = yBase - 0.015 - (i - 1) * 0.045;

                % Convert to data coords
                ey = ylims(1) + rowY * dy;

                if xDir < 0
                    sampleX1 = xlims(1) + (xBase - boxW + 0.015) * dx;
                    sampleX2 = xlims(1) + (xBase - boxW + 0.06) * dx;
                    labelX   = xlims(1) + (xBase - boxW + 0.075) * dx;
                else
                    sampleX1 = xlims(1) + (xBase + 0.015) * dx;
                    sampleX2 = xlims(1) + (xBase + 0.06) * dx;
                    labelX   = xlims(1) + (xBase + 0.075) * dx;
                end

                switch entry.Type
                    case 'line'
                        line(ax, [sampleX1 sampleX2], [ey ey], ...
                            'Color', entry.Color, ...
                            'LineWidth', entry.LineWidth, ...
                            'LineStyle', entry.LineStyle);
                        text(ax, labelX, ey, entry.Label, ...
                            'Color', options.TextColor, ...
                            'FontSize', options.FontSize, ...
                            'VerticalAlignment', 'middle');

                    case 'icon'
                        midX = (sampleX1 + sampleX2) / 2;
                        obj.drawIconSample_(ax, entry.IconType, ...
                            midX, ey, entry.MarkerColor, entry.MarkerSize);
                        text(ax, labelX, ey, entry.Label, ...
                            'Color', options.TextColor, ...
                            'FontSize', options.FontSize, ...
                            'VerticalAlignment', 'middle');

                    case 'symbol'
                        line(ax, [sampleX1 sampleX2], [ey ey], ...
                            'Color', entry.Color, ...
                            'LineWidth', entry.LineWidth, ...
                            'LineStyle', entry.LineStyle);
                        text(ax, labelX, ey, entry.Label, ...
                            'Color', options.TextColor, ...
                            'FontSize', options.FontSize, ...
                            'VerticalAlignment', 'middle');
                end
            end
        end

        % ============================================================== %
        %  Utilities
        % ============================================================== %

        function clear(obj)
            % CLEAR  Remove all entries.
            obj.Entries_ = {};
        end

        function N = getEntryCount(obj)
            N = numel(obj.Entries_);
        end

        function describe(obj)
            fprintf('LegendManager: %d entries\n', obj.getEntryCount());
            for i = 1:numel(obj.Entries_)
                e = obj.Entries_{i};
                switch e.Type
                    case 'line'
                        fprintf('  [line]   %s: %s\n', e.Code, e.Label);
                    case 'icon'
                        fprintf('  [icon]   %s: %s\n', e.IconType, e.Label);
                    case 'symbol'
                        fprintf('  [symbol] %s: %s\n', e.Code, e.Label);
                end
            end
        end
    end

    % ================================================================== %
    methods (Access = private)

        function [xBase, yBase, xDir] = computePosition_(~, position, nEntries, padding)
            % COMPUTEPOSITION_  Map position string to normalized coords.
            switch lower(position)
                case 'northeast'
                    xBase = 1 - padding; yBase = 1 - padding; xDir = -1;
                case 'northwest'
                    xBase = padding; yBase = 1 - padding; xDir = 1;
                case 'southeast'
                    xBase = 1 - padding;
                    yBase = nEntries * 0.045 + 0.025 + padding;
                    xDir = -1;
                case 'southwest'
                    xBase = padding;
                    yBase = nEntries * 0.045 + 0.025 + padding;
                    xDir = 1;
                otherwise
                    xBase = 1 - padding; yBase = 1 - padding; xDir = -1;
            end
        end

        function drawIconSample_(~, ax, iconType, x, y, color, markerSize)
            % DRAWICONSAMPLE_  Draw a small icon sample for the legend.
            switch lower(iconType)
                case 'interchange'
                    plot(ax, x, y, 'o', ...
                        'MarkerSize', markerSize, ...
                        'MarkerFaceColor', [0.15 0.15 0.18], ...
                        'MarkerEdgeColor', color, ...
                        'LineWidth', 1.5);
                    plot(ax, x, y, '.', ...
                        'MarkerSize', markerSize * 0.5, ...
                        'Color', color);
                case 'terminal'
                    plot(ax, x, y, 's', ...
                        'MarkerSize', markerSize, ...
                        'MarkerFaceColor', color, ...
                        'MarkerEdgeColor', color * 0.7);
                otherwise
                    plot(ax, x, y, 'o', ...
                        'MarkerSize', markerSize, ...
                        'MarkerFaceColor', color, ...
                        'MarkerEdgeColor', color);
            end
        end
    end
end
