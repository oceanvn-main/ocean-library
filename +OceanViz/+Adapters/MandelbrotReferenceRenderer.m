classdef MandelbrotReferenceRenderer < OceanViz.Adapters.ISampleRenderer
    % MANDELBROTREFERENCERENDED  Shows the TRUE deterministic Mandelbrot
    %   escape-time map as a static reference panel inside NNReplayEngine.
    %
    %   Once setup() is called, this panel is FIXED — it never changes
    %   per epoch. This is correct because the Mandelbrot set is fully
    %   deterministic. It acts as the "Ground Truth" reference so the user
    %   can compare the NN approximation (right panel) against the true shape.
    %
    %   Additionally, each frame it marks the current demo sample point on
    %   the reference image with a colored dot.
    %
    % See also: OceanViz.Adapters.ISampleRenderer

    properties (Access = private)
        Config_    struct
        Colors_    struct
        DotHandle_         % Animated marker for the current demo point
    end

    methods
        function setup(obj, ax, config, colors)
            obj.Colors_ = colors;
            obj.Config_ = config;

            ck        = config.Checkpoint;
            RE_RANGE  = ck.RE_RANGE;
            IM_RANGE  = ck.IM_RANGE;

            set(ax, 'Color', 'k');

            % Draw true Mandelbrot escape-time map
            imagesc(ax, ck.a_d, ck.b_d, ck.groundTruth);
            colormap(ax, 'hot');
            clim(ax, [0 1]);
            set(ax, 'YDir', 'normal', 'XColor', colors.Grid, 'YColor', colors.Grid);
            xlim(ax, RE_RANGE);
            ylim(ax, IM_RANGE);
            xlabel(ax, 'Re(c)', 'Color', colors.Text, 'FontSize', 8);
            ylabel(ax, 'Im(c)', 'Color', colors.Text, 'FontSize', 8);
            title(ax, 'True Mandelbrot (Ground Truth)', ...
                'Color', colors.Text, 'FontSize', 9);
            hold(ax, 'on');

            % Animated sample dot
            obj.DotHandle_ = scatter(ax, NaN, NaN, 60, 'c', 'filled', ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
        end

        function render(obj, ax, frame, ep)
            ck = obj.Config_.Checkpoint;
            % De-normalize from [-1,1] back to real complex coordinates
            re = (frame.X(1) + 1) / 2 * diff(ck.RE_RANGE) + ck.RE_RANGE(1);
            im = (frame.X(2) + 1) / 2 * diff(ck.IM_RANGE) + ck.IM_RANGE(1);

            % Color the dot by true label (inside=bright, outside=dark)
            brightness = frame.Y;   % already in [0,1] (escape_norm)
            if brightness > 0.95
                col = [0.2, 0.9, 1.0]; % cyan = inside set
            else
                col = [1.0, 0.4, 0.1]; % orange = outside
            end
            set(obj.DotHandle_, 'XData', re, 'YData', im, 'CData', col);
        end

        function t = getTitle(~, frame, ep)
            lbl = 'Outside';
            if frame.Y > 0.95, lbl = 'Inside'; end
            t = sprintf('[%s]  epoch %d', lbl, ep);
        end
    end
end
