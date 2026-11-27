classdef L1Penalty < OceanMath.Regularization.IRobustPenalty
    % L1PENALTY  Total Variation / L1 penalty rho(d) = |d|.
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05

    properties
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function obj = L1Penalty(options)
            arguments
                options.Epsilon (1,1) double = 1e-6
            end
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(~, d)
            val = abs(d);
        end

        function w = weight(obj, d)
            % w(d) = rho'(d)/d = 1/|d|, with |d| floored at Epsilon
            w = 1.0 ./ max(abs(d), obj.Epsilon);
        end

        function fp = derivative(obj, d)
            fp = d ./ max(abs(d), obj.Epsilon);
        end

        function str = getName(~)
            str = "L1 / Total Variation";
        end

        function tf = isConvex(~)
            tf = true;
        end
    end
end
