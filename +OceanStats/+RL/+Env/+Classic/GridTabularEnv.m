classdef (Abstract) GridTabularEnv < OceanStats.RL.Env.TabularEnvironment
    % GRIDTABULARENV  Shared helpers for the small grid worlds (row 1 = top, index = (row-1)*Cols + col).
    properties (SetAccess = protected)
        Rows (1,1) double = 0
        Cols (1,1) double = 0
    end
    methods
        function [r, c] = rc(obj, s)
            % RC  row and column of state index s
            r = floor((s - 1) / obj.Cols) + 1;
            c = mod(s - 1, obj.Cols) + 1;
        end
        function s = idx(obj, r, c), s = (r - 1) * obj.Cols + c; end
    end
end
