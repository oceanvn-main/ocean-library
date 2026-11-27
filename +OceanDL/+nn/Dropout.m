classdef Dropout < OceanDL.nn.Module
    % DROPOUT  Inverted dropout (Srivastava et al. 2014); identity in eval().

    properties (SetAccess = protected)
        P (1,1) double {mustBeGreaterThanOrEqual(P, 0), mustBeLessThan(P, 1)} = 0.5
    end

    methods
        function obj = Dropout(p)
            if nargin > 0
                obj.P = p;
            end
        end

        function y = forward(obj, x)
            y = OceanDL.Autograd.F.dropout(x, obj.P, obj.Training);
        end
    end
end
