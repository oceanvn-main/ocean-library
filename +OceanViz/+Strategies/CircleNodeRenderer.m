classdef CircleNodeRenderer < OceanViz.Strategies.INodeRenderer
    % CIRCLENODERENDERER  Default node renderer: filled scatter circles.
    %
    %   This is the standard renderer used by NetworkGraphStrategy.
    %   Nodes appear as filled circles with a slightly darker edge ring.
    %   The color and size are driven fully by the NNGraphData struct.
    %
    %   For custom node shapes (icon, image, text), create a new class that
    %   inherits INodeRenderer and implement draw().
    %
    % See also: OceanViz.Strategies.INodeRenderer
    %           OceanViz.Strategies.NetworkGraphStrategy

    methods
        function draw(~, ax, x, y, sz, color, state, isDead)
            % DRAW  Render one node as a filled circle on ax at (x, y).

            if strcmp(state, 'Highlighted')
                % Bright white ring for highlighted nodes (clearly visible)
                edgeColor = [1.0  1.0  1.0];
                edgeLW    = 2.5;
            elseif strcmp(state, 'Disabled')
                edgeColor = color * 0.5;
                edgeLW    = 0.8;
            else
                edgeColor = min(color * 1.4, 1);  % slightly brighter rim
                edgeLW    = 1.2;
            end

            scatter(ax, x, y, sz, color, 'filled', ...
                'LineWidth',        edgeLW, ...
                'MarkerEdgeColor',  edgeColor);

            % Dead neuron overlay: red × character
            if isDead
                text(ax, x, y, '\times', ...
                    'Color',               [1.0, 0.20, 0.20], ...
                    'FontSize',            10, ...
                    'FontWeight',          'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment',   'middle', ...
                    'HitTest',             'off', ...
                    'PickableParts',       'none');
            end
        end
    end
end
