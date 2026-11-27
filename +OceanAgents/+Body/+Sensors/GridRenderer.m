classdef (Sealed) GridRenderer < OceanAgents.Body.Sensors.IRenderer
    % GRIDRENDERER  Draws a grid world as a one-channel image: 1 at the agent's cell, 0 elsewhere. The view's
    % observation is the cell index in row-major order (as the native grid environments number their cells).
    %
    %   g = OceanAgents.Body.Sensors.GridRenderer(4, 12);    % cliff walking
    %   img = g.render(view);                                % 4 x 12 (x 1)

    properties (SetAccess = immutable)
        Rows double
        Cols double
        Size double
    end

    methods
        function obj = GridRenderer(rows, cols)
            arguments
                rows (1,1) double {mustBeInteger, mustBePositive}
                cols (1,1) double {mustBeInteger, mustBePositive}
            end
            obj.Rows = rows;
            obj.Cols = cols;
            obj.Size = [rows cols 1];
        end

        function img = render(obj, view)
            s = view.Observation;
            if ~(isnumeric(s) && isscalar(s) && s >= 1 && s <= obj.Rows * obj.Cols && s == fix(s))
                error('OceanAgents:Signals:BadValue', 'GridRenderer: %s is not a cell of a %d x %d grid.', ...
                    mat2str(s), obj.Rows, obj.Cols);
            end
            img = zeros(obj.Rows, obj.Cols);
            img(floor((s - 1) / obj.Cols) + 1, mod(s - 1, obj.Cols) + 1) = 1;
        end
    end
end
