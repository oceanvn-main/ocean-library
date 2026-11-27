classdef GeographicScatter < handle
    % OCEANVIZ.STRATEGIES.GEOGRAPHICSCATTER
    % Generic strategy for rendering a scalar-colored scatter map on a
    % geographic (Longitude × Latitude) plane.
    %
    % OVERVIEW:
    %   Plots data points at their real geographic positions (PointX, PointY)
    %   and colors each point by a scalar value.  The scalar can be supplied
    %   in two ways:
    %
    %   DIRECT mode — caller pre-computes the color value per point:
    %     data.PointX     — N×1 X-axis (e.g. Longitude)
    %     data.PointY     — N×1 Y-axis (e.g. Latitude)
    %     data.PointColor — N×1 scalar value for each point (any unit)
    %
    %   GRID mode — caller supplies a pre-computed 2D scalar field and this
    %   strategy interpolates it to each point's position:
    %     data.PointX  — N×1 X-axis positions
    %     data.PointY  — N×1 Y-axis positions
    %     data.GridX   — 1×nG grid X values  (COLUMN direction of ZMap)
    %     data.GridY   — 1×nG grid Y values  (ROW direction of ZMap)
    %     data.ZMap    — nG×nG scalar field matrix
    %                    ROWS index GridY, COLUMNS index GridX (meshgrid order)
    %
    %   OVERLAY (optional, either or both):
    %     options.OverlayX / OverlayY — raw longitude/latitude vectors for any
    %         polygon overlay (convex hull, country boundary, custom region, …)
    %     options.BoundaryState — convenience: U.S. state name to draw via
    %         Mapping Toolbox 'usastatehi'. Falls back gracefully if toolbox
    %         is unavailable. Set '' to skip.
    %
    % OPTIONS:
    %   Title          — figure title string
    %   XLabel         — X-axis label.  Default: 'X'
    %   YLabel         — Y-axis label.  Default: 'Y'
    %   ColorLabel     — colorbar label. Default: 'Value'
    %   Colormap       — MATLAB colormap name. Default: 'cool'
    %   DotSize        — scatter marker size. Default: 12
    %   DotAlpha       — scatter marker face alpha [0..1]. Default: 0.75
    %   SymmetricColor — if true, clim is centred at 0 (e.g. for centred PD).
    %                    Default: true
    %   AspectRatio    — 'equal' or 'auto'. Default: 'equal'
    %   OverlayX       — vector of X coords for custom overlay polygon. Default: []
    %   OverlayY       — vector of Y coords for custom overlay polygon. Default: []
    %   BoundaryState  — U.S. state name (Mapping Toolbox). Default: ''
    %   BoundaryColor  — RGB triplet for overlay line. Default: [1, 0.55, 0]
    %   BoundaryWidth  — overlay line width. Default: 1.8
    %
    % USAGE (DIRECT mode — e.g. coloring by residuals):
    %   data.PointX     = X_test(:, lonCol);
    %   data.PointY     = X_test(:, latCol);
    %   data.PointColor = residuals;
    %   viz.plot(ax, 'GeographicScatter', data, ...
    %       'Title',       'Residual Map', ...
    %       'ColorLabel',  'Residual', ...
    %       'Colormap',    'RdBu', ...
    %       'AspectRatio', 'equal');
    %
    % USAGE (GRID mode — e.g. 2D Partial Dependence on location):
    %   data.PointX = X_train(:, lonCol);
    %   data.PointY = X_train(:, latCol);
    %   data.GridX  = g_lon';     % 1×nG
    %   data.GridY  = g_lat';     % 1×nG
    %   data.ZMap   = pd_geo;     % nG×nG, centred PD
    %   viz.plot(ax, 'GeographicScatter', data, ...
    %       'Title',          'Partial Dependence on Location (ESL Fig 10.17)', ...
    %       'ColorLabel',     'Partial Dependence  (\times$100k USD)', ...
    %       'BoundaryState',  'California', ...
    %       'Colormap',       'cool');
    %
    % USAGE (custom polygon overlay):
    %   data.PointX     = lons;
    %   data.PointY     = lats;
    %   data.PointColor = density;
    %   viz.plot(ax, 'GeographicScatter', data, ...
    %       'OverlayX', border_lon, 'OverlayY', border_lat, ...
    %       'BoundaryColor', [0 1 0]);
    %
    % See also: OceanViz.Core, OceanViz.Strategies.PartialDependence2D

    properties
        Parent      % Injected OceanViz.Core instance (set by Core.plot)
    end

    methods
        function plot(obj, ax, data, options)
            % PLOT  Render the geographic scatter map onto the given axes.
            arguments
                obj
                ax   (1,1) matlab.graphics.axis.Axes
                data (1,1) struct
                options struct
            end

            C = obj.Parent.Colors;

            % ── 1. Resolve options with defaults ───────────────────────────
            cmap       = obj.resolveOpt(options, 'Colormap',       'cool');
            dotSz      = obj.resolveOpt(options, 'DotSize',         12);
            dotAlpha   = obj.resolveOpt(options, 'DotAlpha',        0.75);
            cbLabel    = obj.resolveOpt(options, 'ColorLabel',      'Value');
            symColor   = obj.resolveOpt(options, 'SymmetricColor',  true);
            aspect     = obj.resolveOpt(options, 'AspectRatio',     'equal');
            overlayX   = obj.resolveOpt(options, 'OverlayX',        []);
            overlayY   = obj.resolveOpt(options, 'OverlayY',        []);
            stName     = obj.resolveOpt(options, 'BoundaryState',   '');
            bndCol     = obj.resolveOpt(options, 'BoundaryColor',   [1, 0.55, 0]);
            bndWidth   = obj.resolveOpt(options, 'BoundaryWidth',   1.8);

            % ── 2. Resolve scalar color per point ─────────────────────────
            % DIRECT mode: caller hands in PointColor directly.
            % GRID mode:   caller hands in GridX / GridY / ZMap → interpolate.
            if isfield(data, 'PointColor')
                % ---- DIRECT path ----------------------------------------
                colorVals = data.PointColor(:);

            elseif isfield(data, 'ZMap') && isfield(data, 'GridX') && isfield(data, 'GridY')
                % ---- GRID path ------------------------------------------
                % Build meshgrid in (GridX=cols, GridY=rows) order, matching
                % the ZMap storage convention (rows=GridY, cols=GridX).
                [G_x_mesh, G_y_mesh] = meshgrid(data.GridX, data.GridY);

                % Clip query points to grid extent to avoid NaN extrapolation
                x_q = max(min(data.PointX, max(data.GridX)), min(data.GridX));
                y_q = max(min(data.PointY, max(data.GridY)), min(data.GridY));

                colorVals = interp2(G_x_mesh, G_y_mesh, data.ZMap, ...
                    x_q, y_q, 'linear', NaN);
            else
                error('GeographicScatter:BadData', ...
                    ['data must contain either:\n' ...
                     '  DIRECT: data.PointColor (N×1)\n' ...
                     '  GRID:   data.GridX, data.GridY, data.ZMap']);
            end

            % Remove NaN (e.g. outside grid convex hull)
            validMask = ~isnan(colorVals);

            % ── 3. Scatter plot colored by scalar value ───────────────────
            hold(ax, 'on');

            scatter(ax, data.PointX(validMask), data.PointY(validMask), ...
                dotSz, colorVals(validMask), 'filled', ...
                'MarkerFaceAlpha', dotAlpha, ...
                'HandleVisibility', 'off');

            colormap(ax, cmap);

            % Color limits
            if symColor
                % Centre at zero: useful for centred PD or signed residuals
                clim_max = max(abs(colorVals(validMask)));
                if clim_max > 0
                    clim(ax, [-clim_max, clim_max]);
                end
            else
                % Use data range
                clim(ax, [min(colorVals(validMask)), max(colorVals(validMask))]);
            end

            cb              = colorbar(ax);
            cb.Color        = C.Text;
            cb.Label.String = cbLabel;
            cb.Label.Color  = C.Text;
            cb.Label.FontSize = 9;

            % ── 4. Overlay polygons ───────────────────────────────────────
            % 4a. Raw polygon supplied by caller (any geographic boundary)
            if ~isempty(overlayX) && ~isempty(overlayY)
                obj.drawPolygon(ax, overlayX, overlayY, bndCol, bndWidth);
            end

            % 4b. Convenience: U.S. state boundary via Mapping Toolbox
            if strtrim(stName) ~= ""
                obj.drawStateBoundary(ax, char(stName), bndCol, bndWidth);
            end

            % ── 5. Axis styling ────────────────────────────────────────────
            ax.Color     = C.Background;
            ax.GridColor = C.Grid;
            ax.XColor    = C.Text;
            ax.YColor    = C.Text;
            ax.LineWidth = 0.8;
            grid(ax, 'on');
            axis(ax, aspect);

            xlabel(ax, char(obj.resolveOpt(options, 'XLabel', 'X')), ...
                'Color', C.Text, 'FontSize', 10);
            ylabel(ax, char(obj.resolveOpt(options, 'YLabel', 'Y')), ...
                'Color', C.Text, 'FontSize', 10);
            if isfield(options, 'Title')
                title(ax, char(options.Title), 'Color', C.Text, 'FontSize', 11);
            end

            hold(ax, 'off');
        end
    end

    methods (Access = private)

        function drawPolygon(~, ax, x, y, color, lineWidth)
            % DRAWPOLYGON  Overlay an arbitrary polygon (any region boundary).
            plot(ax, x, y, ...
                'Color',           color, ...
                'LineWidth',       lineWidth, ...
                'HandleVisibility','off');
        end

        function drawStateBoundary(~, ax, stateName, bndColor, lineWidth)
            % DRAWSTATEBOUNDARY  Overlay a U.S. state boundary via Mapping Toolbox.
            % Falls back gracefully if the toolbox is unavailable.
            if ~license('test', 'map_toolbox')
                warning('GeographicScatter:NoMappingToolbox', ...
                    'Mapping Toolbox not available — skipping state boundary.');
                return;
            end

            try
                states = shaperead('usastatehi', 'UseGeoCoords', true, ...
                    'Selector', {@(name) strcmpi(name, stateName), 'Name'});

                if isempty(states)
                    warning('GeographicScatter:StateNotFound', ...
                        'State "%s" not found in usastatehi shapefile.', stateName);
                    return;
                end

                plot(ax, states.Lon, states.Lat, ...
                    'Color',           bndColor, ...
                    'LineWidth',       lineWidth, ...
                    'HandleVisibility','off');

            catch ME
                warning('GeographicScatter:BoundaryError', ...
                    'Could not draw state boundary: %s', ME.message);
            end
        end

        function val = resolveOpt(~, options, name, default)
            % RESOLVEOPT  Return options.(name) if present, else default.
            if isfield(options, name)
                val = options.(name);
            else
                val = default;
            end
        end
    end
end
