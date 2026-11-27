classdef QuadraticPenalty < OceanMath.Regularization.IRobustPenalty
    % QUADRATICPENALTY  Standard L2 quadratic penalty rho(d) = d^2.
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05

    methods
        function val = evaluate(~, d)
            val = d .^ 2;
        end

        function w = weight(~, d)
            % w(d) = rho'(d)/d = 2d/d = 2
            w = 2 * ones(size(d));
        end

        function fp = derivative(~, d)
            fp = 2 * d;
        end

        function str = getName(~)
            str = "Quadratic (L2)";
        end

        function tf = isConvex(~)
            tf = true;
        end
    end
end
