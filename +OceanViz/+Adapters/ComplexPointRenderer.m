classdef ComplexPointRenderer < OceanViz.Adapters.ISampleRenderer
    % COMPLEXPOINTRENDERER  Renders a single complex point c = a + b*i on
    %   the complex plane as the sample panel inside NNReplayEngine.
    %
    %   Displays the bounding box of the experiment and highlights the
    %   current demo sample point with its true class (Inside / Outside).
    %
    %   CONFIG fields (passed to setup):
    %     RE_RANGE  - [min max] real axis range     (default [-2.5 1.0])
    %     IM_RANGE  - [min max] imaginary axis range (default [-1.2 1.2])
    %
    % Example:
    %   r = OceanViz.Adapters.ComplexPointRenderer();
    %   r.setup(ax, struct('RE_RANGE', [-2.5 1], 'IM_RANGE', [-1.2 1.2]), colors);
    %
    % See also: OceanViz.Adapters.ISampleRenderer

    properties (Access = private)
        Config_   struct
        Colors_   struct
        DotHandle_        % handle to the animated scatter point
    end

    methods
        function setup(obj, ax, config, colors)
            obj.Config_ = config;
            if ~isfield(obj.Config_, 'RE_RANGE'), obj.Config_.RE_RANGE = [-2.5 1.0]; end
            if ~isfield(obj.Config_, 'IM_RANGE'), obj.Config_.IM_RANGE = [-1.2 1.2]; end
            obj.Colors_ = colors;

            set(ax, 'Color', colors.Background, ...
                'XColor', colors.Grid, 'YColor', colors.Grid);
            xlim(ax, obj.Config_.RE_RANGE);
            ylim(ax, obj.Config_.IM_RANGE);
            xlabel(ax, 'Re(c)', 'Color', colors.Text, 'FontSize', 8);
            ylabel(ax, 'Im(c)', 'Color', colors.Text, 'FontSize', 8);
            title(ax,  'Current Sample (c)', 'Color', colors.Text, 'FontSize', 9);
            grid(ax, 'on');
            hold(ax, 'on');

            % Draw the origin crosshair
            xline(ax, 0, 'Color', colors.Grid, 'LineStyle', ':', 'Alpha', 0.5);
            yline(ax, 0, 'Color', colors.Grid, 'LineStyle', ':', 'Alpha', 0.5);

            % Create the animated dot (initially off-screen)
            obj.DotHandle_ = scatter(ax, NaN, NaN, 80, 'w', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        end

        function render(obj, ax, frame, ep)
            % frame.X is [2 x 1]: [a_norm; b_norm]
            % De-normalize from [-1,1] back to real coords
            re = (frame.X(1) + 1) / 2 * diff(obj.Config_.RE_RANGE) + obj.Config_.RE_RANGE(1);
            im = (frame.X(2) + 1) / 2 * diff(obj.Config_.IM_RANGE) + obj.Config_.IM_RANGE(1);

            % Color: Inside = cyan (Primary), Outside = red (Secondary)
            if frame.Y == 1
                col = obj.Colors_.Primary;
                lbl = 'Inside';
            else
                col = obj.Colors_.Secondary;
                lbl = 'Outside';
            end

            set(obj.DotHandle_, 'XData', re, 'YData', im, 'CData', col);
            title(ax, obj.getTitle(frame, ep), 'Color', obj.Colors_.Text, 'FontSize', 9);
        end

        function t = getTitle(~, frame, ep)
            lbl = 'Outside'; if frame.Y == 1, lbl = 'Inside'; end
            t = sprintf('c = (%.3f, %.3fi)  [%s]  ep %d', ...
                frame.X(1), frame.X(2), lbl, ep);
        end
    end
end
