classdef L1Penalty < OceanMath.Image.Regularization.IRobustPenalty
    % L1PENALTY  Total Variation penalty: rho(x) = |x|.
    %
    %   This is the p=1 norm, also known as the Total Variation (TV) penalty
    %   (Rudin, Osher & Fatemi 1992; Chan & Shen 2005). Forming the energy:
    %
    %       E_TV = sum_{i,j}  |f(i+1,j) - f(i,j)|  +  |f(i,j+1) - f(i,j)|
    %
    %   Properties:
    %     - Convex (isConvex = true)
    %     - Yields a globally unique solution when combined with an L2 data term
    %     - Preserves sharp discontinuities by not penalising large differences
    %       disproportionately more than small ones
    %     - Produces the "staircase" or piecewise-constant effect on smooth regions
    %
    %   IRLS weight:  w(d) = 1/|d|   (diverges at d=0 — numerically stabilised)
    %   Derivative:   rho'(d) = sign(d)  (subgradient)
    %
    %   Numerical note:
    %       The weight w(d) = |d|^{-1} is numerically problematic at d=0.
    %       We stabilise it as  w(d) = 1 / max(|d|, Epsilon)
    %       where Epsilon = 1e-6 by default.  This tiny epsilon has negligible
    %       effect on the solution while preventing division by zero.
    %
    %   Usage:
    %       pen = OceanMath.Image.Regularization.L1Penalty();
    %       pen = OceanMath.Image.Regularization.L1Penalty('Epsilon', 1e-8);
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2.2, Total Variation

    properties
        % EPSILON  Numerical floor for the IRLS weight denominator.
        %   Prevents division by zero when d -> 0.
        %   Default: 1e-6  (safe for typical image pixel differences in [0,1])
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function obj = L1Penalty(options)
            % L1PENALTY  Constructor.
            %
            %   pen = L1Penalty()
            %   pen = L1Penalty('Epsilon', 1e-8)
            arguments
                options.Epsilon (1,1) double {mustBePositive} = 1e-6
            end
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(~, d)
            % rho(d) = |d|
            val = abs(d);
        end

        function w = weight(obj, d)
            % w(d) = rho(d)/d^2 = |d|/d^2 = 1/|d|
            % Numerically stabilised: w = 1 / max(|d|, epsilon)
            w = 1 ./ max(abs(d), obj.Epsilon);
        end

        function g = derivative(~, d)
            % rho'(d) = sign(d)   [subgradient; 0 at d=0 by MATLAB's sign convention]
            g = sign(d);
        end

        function name = getName(~)
            name = 'L1 / Total Variation';
        end

        function tf = isConvex(~)
            tf = true;
        end
    end
end
