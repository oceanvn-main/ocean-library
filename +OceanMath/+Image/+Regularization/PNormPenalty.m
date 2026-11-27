classdef PNormPenalty < OceanMath.Image.Regularization.IRobustPenalty
    % PNORMPENALTY  General p-norm penalty: rho(x) = |x|^p,  p in (0, 2].
    %
    %   The p-norm family interpolates between all major regularization regimes:
    %
    %       p = 2  →  Quadratic (L2), fully smooth, blurs edges
    %       p = 1  →  Total Variation (L1), convex, sharp edges
    %       p < 1  →  Hyper-Laplacian, non-convex, even sparser than L1
    %                 (natural images: p ≈ 0.5–0.8, Krishnan & Fergus 2009)
    %
    %   The IRLS weight is:
    %       w(d) = rho(d)/d^2 = |d|^p / d^2 = |d|^(p-2)
    %
    %   which is the famous "iterative reweighted" formula.  For p<1, w blows up
    %   near d=0 and is stabilised with Epsilon.
    %
    %   Convexity:
    %       p >= 1  →  convex   (isConvex = true)
    %       p <  1  →  non-convex  (use IRLSSolver with warm restart)
    %
    %   Usage:
    %       pen = OceanMath.Image.Regularization.PNormPenalty();          % p=1.0
    %       pen = OceanMath.Image.Regularization.PNormPenalty('P', 0.8);  % hyper-Laplacian
    %       pen = OceanMath.Image.Regularization.PNormPenalty('P', 1.5);  % intermediate
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2, Eq. 4.29; Krishnan & Fergus (2009)

    properties
        % P  The exponent of the norm.  Must be in (0, 2].
        P       (1,1) double = 1.0

        % EPSILON  Numerical floor for IRLS weights at d=0.
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function obj = PNormPenalty(options)
            % PNORMPENALTY  Constructor.
            %
            %   pen = PNormPenalty()
            %   pen = PNormPenalty('P', 0.8)
            %   pen = PNormPenalty('P', 0.8, 'Epsilon', 1e-8)
            arguments
                options.P       (1,1) double {mustBeInRange(options.P, 0, 2, 'exclusive-inclusive')} = 1.0
                options.Epsilon (1,1) double {mustBePositive} = 1e-6
            end
            obj.P       = options.P;
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(obj, d)
            % rho(d) = |d|^p
            val = abs(d) .^ obj.P;
        end

        function w = weight(obj, d)
            % w(d) = |d|^(p-2)
            % Numerically stabilised: replace |d| with max(|d|, epsilon)
            % before raising to (p-2) < 0 to avoid Inf at d=0.
            d_safe = max(abs(d), obj.Epsilon);
            w = d_safe .^ (obj.P - 2);
        end

        function g = derivative(obj, d)
            % rho'(d) = p * |d|^(p-1) * sign(d)
            %
            % At d=0: the subgradient is 0 (by convention, matches MATLAB sign(0)=0).
            % For p < 1 the derivative diverges at d=0; we clip it via Epsilon.
            d_safe = max(abs(d), obj.Epsilon);
            g = obj.P .* (d_safe .^ (obj.P - 1)) .* sign(d);
        end

        function name = getName(obj)
            name = sprintf('p-Norm  (p=%.2g)', obj.P);
        end

        function tf = isConvex(obj)
            % Convex for p >= 1, non-convex for p < 1.
            tf = (obj.P >= 1.0);
        end
    end
end
