classdef QuadraticPenalty < OceanMath.Regularization.IRobustPenalty
    % QUADRATICPENALTY  Standard L2 quadratic penalty rho(d) = d^2.
    %
    %   This is the baseline case (p=2 norm). The regularization energy becomes
    %   the classical Tikhonov / L2 smoothness term:
    %
    %       E_IR = sum_{i,j} (f(i+1,j) - f(i,j))^2  +  (f(i,j+1) - f(i,j))^2
    %
    %   Properties:
    %     - IRLS weight w(d) = rho'(d)/d = 2  (constant — so IRLS converges in
    %       exactly 1 step)
    %     - Convex
    %     - Produces blurry edges (penalises ALL differences equally)
    %     - Reduces to a single sparse linear system  (I + 2 * lambda * D'D) f = g,
    %       the normal equations of (1/2)||f - g||^2 + lambda * ||D f||^2
    %
    %   Usage:
    %       pen = OceanMath.Regularization.QuadraticPenalty();
    %       val = pen.evaluate([-2, 0, 1.5]);   % [4, 0, 2.25]
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2, Eq. 4.27

    methods
        function val = evaluate(~, d)
            % rho(d) = d^2
            val = d .^ 2;
        end

        function w = weight(~, d)
            % w(d) = rho'(d)/d = 2d/d = 2
            % Constant, so no numerical stabilisation is needed at d = 0.
            w = 2 * ones(size(d));
        end

        function fp = derivative(~, d)
            % rho'(d) = 2d
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
