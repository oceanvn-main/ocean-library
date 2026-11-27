classdef QuadraticPenalty < OceanMath.Image.Regularization.IRobustPenalty
    % QUADRATICPENALTY  Standard L2 quadratic penalty: rho(x) = x^2.
    %
    %   This is the baseline case (p=2 norm). The regularization energy becomes
    %   the classical Tikhonov / L2 smoothness term:
    %
    %       E_IR = sum_{i,j} (f(i+1,j) - f(i,j))^2  +  (f(i,j+1) - f(i,j))^2
    %
    %   Properties:
    %     - IRLS weight w(d) = 1  (constant — so IRLS converges in exactly 1 step)
    %     - Convex
    %     - Produces blurry edges (penalises ALL differences equally)
    %     - Reduces to a single sparse linear system  (I + lambda * D'D) f = g
    %
    %   Usage:
    %       pen = OceanMath.Image.Regularization.QuadraticPenalty();
    %       val = pen.evaluate([-2, 0, 1.5]);   % [4, 0, 2.25]
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2, Eq. 4.27

    methods
        function val = evaluate(~, d)
            % rho(d) = d^2
            val = d .^ 2;
        end

        function w = weight(~, d)
            % w(d) = rho(d)/d^2 = 1  (constant — IRLS is exact in 1 iteration)
            %
            % The limit at d=0 is also 1 (L'Hôpital: (2d)/(2d) = 1), so no
            % numerical stabilisation is needed here.
            w = ones(size(d));
        end

        function g = derivative(~, d)
            % rho'(d) = 2d
            g = 2 .* d;
        end

        function name = getName(~)
            name = 'Quadratic (L2)';
        end

        function tf = isConvex(~)
            tf = true;
        end
    end
end
