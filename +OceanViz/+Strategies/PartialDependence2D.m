classdef PartialDependence2D < handle
    % OCEANVIZ.STRATEGIES.PARTIALDEPENDENCE2D
    % Strategy for rendering a pre-computed 2D Partial Dependence surface.
    %
    % OVERVIEW:
    %   Unlike SurfaceManifoldStrategy (which calls predict() internally),
    %   this strategy is a "Dumb Renderer" — it accepts a matrix of values
    %   that were already computed externally and simply renders them as
    %   a styled 3D surface consistent with the OceanViz Dark theme.
    %
    %   This is the correct pattern for Partial Dependence plots, where the
    %   caller already owns the PD computation (vectorized batch predict).
    %
    % DATA INTERFACE:
    %   data.G1     — nG×1 vector of X-axis grid values (e.g., HouseAge)
    %   data.G2     — nG×1 vector of Y-axis grid values (e.g., AveOccup)
    %   data.Z      — nG×nG matrix of pre-computed Z values (e.g., PD)
    %                 ASSUMPTION: Z is already centered (Z = Z - mean(Z(:)))
    %
    % OPTIONS (passed via viz.plot name-value pairs):
    %   Title       — figure title string
    %   XLabel      — X-axis label (Feature 1 name)
    %   YLabel      — Y-axis label (Feature 2 name)
    %   ZLabel      — Z-axis label. Default: 'Partial Dependence'
    %   Colormap    — MATLAB colormap name. Default: 'hot'
    %   AzimuthDeg  — Camera azimuth in degrees. Default: -50
    %   ElevationDeg— Camera elevation in degrees. Default: 28
    %   EdgeAlpha   — Surface mesh transparency (0=solid, 1=wireframe). Default: 0.2
    %   ShowColorbar— Logical, whether to show colorbar. Default: true
    %
    % USAGE:
    %   viz = OceanViz.Core('Theme', 'Dark');
    %   data.G1 = g1;   % nG×1 vector
    %   data.G2 = g2;   % nG×1 vector
    %   data.Z  = pd2;  % nG×nG matrix
    %   fig = viz.createFigure('Joint PD: Age x Occupancy');
    %   ax  = axes('Parent', fig);
    %   viz.plot(ax, 'PartialDependence2D', data, ...
    %       'Title',    'Joint PD: HouseAge x AveOccup', ...
    %       'XLabel',   'AveOccup', ...
    %       'YLabel',   'HouseAge', ...
    %       'Colormap', 'hot');
    %
    % See also: OceanViz.Core, OceanViz.Strategies.LearningCurve

    properties
        Parent      % Injected OceanViz.Core instance (set by Core.plot)
    end

    methods
        function plot(obj, ax, data, options)
            % PLOT  Render the 2D PD surface onto the given axes.
            arguments
                obj
                ax   (1,1) matlab.graphics.axis.Axes
                data (1,1) struct
                options struct
            end

            C = obj.Parent.Colors;

            % ── 1. Resolve options with defaults ───────────────────────────
            zlbl   = obj.resolveOpt(options, 'ZLabel',       'Partial Dependence');
            cmap   = obj.resolveOpt(options, 'Colormap',     'hot');
            az     = obj.resolveOpt(options, 'AzimuthDeg',   -50);
            el     = obj.resolveOpt(options, 'ElevationDeg',  28);
            ea     = obj.resolveOpt(options, 'EdgeAlpha',     0.2);
            showCB = obj.resolveOpt(options, 'ShowColorbar',  true);

            % ── 2. Build meshgrid from G1 / G2 vectors ────────────────────
            % G1 = Y-axis feature, G2 = X-axis feature (meshgrid convention)
            [G2mesh, G1mesh] = meshgrid(data.G2, data.G1);

            % ── 3. Render surface ──────────────────────────────────────────
            hold(ax, 'on');
            surf(ax, G2mesh, G1mesh, data.Z, ...
                'EdgeAlpha', ea, ...
                'FaceAlpha', 0.95);
            colormap(ax, cmap);
            shading(ax, 'interp');
            view(ax, az, el);

            % ── 4. Colorbar ────────────────────────────────────────────────
            if showCB
                cb              = colorbar(ax);
                cb.Color        = C.Text;
                cb.Label.String = zlbl;
                cb.Label.Color  = C.Text;
                cb.Label.FontSize = 9;
            end

            % ── 5. Dark theme axis styling ─────────────────────────────────
            ax.Color     = C.Background;
            ax.GridColor = C.Grid;
            ax.XColor    = C.Text;
            ax.YColor    = C.Text;
            ax.ZColor    = C.Text;
            ax.LineWidth = 0.8;
            grid(ax, 'on');

            % ── 6. Labels ─────────────────────────────────────────────────
            if isfield(options, 'XLabel')
                xlabel(ax, char(options.XLabel), 'Color', C.Text, 'FontSize', 10);
            end
            if isfield(options, 'YLabel')
                ylabel(ax, char(options.YLabel), 'Color', C.Text, 'FontSize', 10);
            end
            zlabel(ax, zlbl, 'Color', C.Text, 'FontSize', 9);
            if isfield(options, 'Title')
                title(ax, char(options.Title), 'Color', C.Text, 'FontSize', 11);
            end

            hold(ax, 'off');
        end
    end

    methods (Access = private)
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
