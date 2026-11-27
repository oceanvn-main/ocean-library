classdef ConstantStepSize < OceanStats.RL.TD.StepSize.IStepSize
    % CONSTANTSTEPSIZE  alpha_n = Alpha (tracks non-stationarity; converges in mean, not almost surely).
    properties (SetAccess = private)
        Alpha (1,1) double {mustBePositive} = 0.1
    end
    properties (Constant)
        Latex = "\alpha_n = \alpha"
        Reference = "constant step size (Sutton & Barto sec. 2.5)"
    end
    methods
        function obj = ConstantStepSize(alpha)
            arguments
                alpha (1,1) double {mustBePositive, mustBeLessThanOrEqual(alpha, 1)} = 0.1
            end
            obj.Alpha = alpha;
        end
        function a = alpha(obj, ~), a = obj.Alpha; end
        function s = name(~), s = "constant"; end
    end
end
