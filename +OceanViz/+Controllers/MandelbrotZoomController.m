classdef MandelbrotZoomController < handle
    % MANDELBROTZOOMMCONTROLLER  Post-replay interactive zoom for the True
    %   Mandelbrot panel. After the animation loop finishes, attach this
    %   controller to the reference axes. When the user zooms in, it
    %   recomputes the escape-time map at high resolution for the new
    %   bounding box, revealing fractal detail the 150x150 training grid
    %   cannot show.
    %
    %   USAGE (in replay script, after engine.run() returns):
    %     ctrl = OceanViz.Controllers.MandelbrotZoomController();
    %     ctrl.attach(engine.Axes.axSample, 'MaxIter', 200, 'GridRes', 400);
    %
    %   ZOOMING:
    %     Use MATLAB's standard scroll-wheel zoom or the right-click zoom
    %     menu on the left panel. The image auto-updates when you release.

    properties
        MaxIter  (1,1) double = 200  % Iterations for zoomed detail
        GridRes  (1,1) double = 400  % Pixels per axis (400x400 = 160k pts)
    end

    properties (Access = private)
        Ax_              % Managed axes handle
        Fig_             % Parent figure
        ZoomHandle_      % zoom() object
        StatusLabel_     % uicontrol text label
    end

    methods
        function attach(obj, ax, opts)
            % ATTACH  Attach interactive zoom to the given axes.
            %
            %   ctrl.attach(ax)
            %   ctrl.attach(ax, 'MaxIter', 300, 'GridRes', 500)
            arguments
                obj
                ax       matlab.graphics.axis.Axes
                opts.MaxIter (1,1) double = obj.MaxIter
                opts.GridRes (1,1) double = obj.GridRes
            end
            obj.MaxIter = opts.MaxIter;
            obj.GridRes = opts.GridRes;
            obj.Ax_     = ax;
            obj.Fig_    = ancestor(ax, 'figure');

            % Enable scroll-wheel zoom on the figure
            obj.ZoomHandle_ = zoom(obj.Fig_);
            obj.ZoomHandle_.Enable = 'on';

            % Only zoom the reference axes
            setAllowAxesZoom(obj.ZoomHandle_, ax, true);

            % Fire recompute ONCE after each zoom gesture ends
            obj.ZoomHandle_.ActionPostCallback = @(~, ev) obj.onZoom(ev);

            % Status label in the figure corner
            obj.StatusLabel_ = uicontrol(obj.Fig_, ...
                'Style', 'text', 'Units', 'pixels', ...
                'Position', [120 15 280 20], ...
                'String', 'Scroll to zoom  |  Double-click to reset', ...
                'FontSize', 8, 'HorizontalAlignment', 'left', ...
                'BackgroundColor', obj.Fig_.Color, 'ForegroundColor', [0.7 0.7 0.7]);

            fprintf('[MandelbrotZoom] Attached. Scroll to zoom on the left panel.\n');
        end

        function onZoom(obj, ev)
            % ONZOOM  Called once after a zoom gesture completes.
            ax = obj.Ax_;
            if ~isvalid(ax), return; end
            if ~isequal(ev.Axes, ax), return; end  % Only care about our axes

            re_range = xlim(ax);
            im_range = ylim(ax);

            % Guard: reject degenerate ranges
            if diff(re_range) < 1e-10 || diff(im_range) < 1e-10
                return;
            end

            set(obj.StatusLabel_, 'String', sprintf( ...
                'Computing %dx%d grid  [%.4f..%.4f]  [%.4fi..%.4fi] ...', ...
                obj.GridRes, obj.GridRes, re_range, im_range));
            drawnow;

            % Recompute Mandelbrot for zoomed region at higher resolution
            G     = obj.GridRes;
            a_vec = linspace(re_range(1), re_range(2), G);
            b_vec = linspace(im_range(1), im_range(2), G);
            [AA, BB] = meshgrid(a_vec, b_vec);

            C          = complex(AA, BB);
            Z          = zeros(size(C));
            escapeIter = obj.MaxIter * ones(size(C));

            for iter = 1:obj.MaxIter
                m = abs(Z) <= 2;
                Z(m) = Z(m).^2 + C(m);
                escaped = (abs(Z) > 2) & (escapeIter == obj.MaxIter);
                escapeIter(escaped) = iter;
            end

            img = escapeIter / obj.MaxIter;   % [G x G], values in [0,1]

            % Update the existing imagesc in-place — no redraw of axes needed
            h = findobj(ax, 'Type', 'Image');
            if ~isempty(h)
                set(h(end), 'XData', a_vec, 'YData', b_vec, 'CData', img);
            end

            set(obj.StatusLabel_, 'String', sprintf( ...
                'Zoom  [%.4f..%.4f] x [%.4fi..%.4fi]  |  Dbl-click to reset', ...
                re_range(1), re_range(2), im_range(1), im_range(2)));
        end
    end
end
