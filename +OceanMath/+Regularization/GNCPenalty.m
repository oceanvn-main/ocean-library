classdef GNCPenalty < OceanMath.Regularization.IRobustPenalty
    % GNCPENALTY  Blake-Zisserman (1987) Graduated Non-Convexity penalty min(d^2, V).
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05

    properties
        V       (1,1) double {mustBePositive} = 1.0
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function obj = GNCPenalty(options)
            arguments
                options.V       (1,1) double = 1.0
                options.Epsilon (1,1) double = 1e-6
            end
            obj.V       = options.V;
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(obj, d)
            val = min(d .^ 2, obj.V);
        end

        function w = weight(obj, d)
            abs_d2 = d .^ 2;
            mask   = (abs_d2 <= obj.V);
            w      = zeros(size(d));
            w(mask)  = 1.0;
            w(~mask) = obj.V ./ max(abs_d2(~mask), obj.Epsilon);
        end

        function fp = derivative(obj, d)
            abs_d2 = d .^ 2;
            mask   = (abs_d2 <= obj.V);
            fp     = zeros(size(d));
            fp(mask) = 2 * d(mask);
        end

        function str = getName(obj)
            str = sprintf('GNC (V=%.2f)', obj.V);
        end

        function tf = isConvex(~)
            tf = false;
        end
    end
end
