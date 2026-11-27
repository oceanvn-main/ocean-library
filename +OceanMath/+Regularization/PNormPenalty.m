classdef PNormPenalty < OceanMath.Regularization.IRobustPenalty
    % PNORMPENALTY  General p-norm penalty rho(d) = |d|^p, p in (0, 2].
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05

    properties
        P       (1,1) double {mustBePositive} = 1.0
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function obj = PNormPenalty(options)
            arguments
                options.P       (1,1) double = 1.0
                options.Epsilon (1,1) double = 1e-6
            end
            obj.P       = options.P;
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(obj, d)
            val = abs(d) .^ obj.P;
        end

        function w = weight(obj, d)
            w = max(abs(d), obj.Epsilon) .^ (obj.P - 2);
        end

        function fp = derivative(obj, d)
            fp = obj.P * sign(d) .* (max(abs(d), obj.Epsilon) .^ (obj.P - 1));
        end

        function str = getName(obj)
            str = sprintf('p-Norm (p=%.2f)', obj.P);
        end

        function tf = isConvex(obj)
            tf = (obj.P >= 1.0);
        end
    end
end
