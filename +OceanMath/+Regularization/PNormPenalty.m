classdef PNormPenalty < OceanMath.Regularization.IRobustPenalty
    % PNORMPENALTY  General p-norm penalty rho(d) = |d|^p, p in (0, 2].
    %
    %   The p-norm family interpolates between all major regularization regimes:
    %
    %       p = 2  →  Quadratic (L2), fully smooth, blurs edges
    %       p = 1  →  Total Variation (L1), convex, sharp edges
    %       p < 1  →  Hyper-Laplacian, non-convex, even sparser than L1
    %                 (natural images: p ≈ 0.5–0.8, Krishnan & Fergus 2009)
    %
    %   The IRLS weight is:
    %       w(d) = rho'(d)/d = p * |d|^(p-1) / |d| = p * |d|^(p-2)
    %
    %   which is the "iterative reweighted" formula.  For p<2, w blows up
    %   near d=0 and is stabilised with Epsilon.
    %
    %   Convexity:
    %       p >= 1  →  convex   (isConvex = true)
    %       p <  1  →  non-convex  (use IRLSSolver with warm restart)
    %
    %   Usage:
    %       pen = OceanMath.Regularization.PNormPenalty();          % p=1.0
    %       pen = OceanMath.Regularization.PNormPenalty('P', 0.8);  % hyper-Laplacian
    %       pen = OceanMath.Regularization.PNormPenalty('P', 1.5);  % intermediate
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2, Eq. 4.29; Krishnan & Fergus (2009)

    properties
        % P  The exponent of the norm.  Intended range (0, 2].
        P       (1,1) double {mustBePositive} = 1.0

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
                options.P       (1,1) double = 1.0
                options.Epsilon (1,1) double = 1e-6
            end
            obj.P       = options.P;
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(obj, d)
            % rho(d) = |d|^p
            val = abs(d) .^ obj.P;
        end

        function w = weight(obj, d)
            % w(d) = rho'(d)/d = p*|d|^(p-2), with |d| floored at Epsilon
            % before raising to (p-2) < 0, to avoid Inf at d=0.
            w = obj.P * max(abs(d), obj.Epsilon) .^ (obj.P - 2);
        end

        function fp = derivative(obj, d)
            % rho'(d) = p * |d|^(p-1) * sign(d)
            %
            % At d=0 the subgradient is 0 (MATLAB sign(0)=0). For p < 1 the
            % derivative diverges at d=0; it is clipped via Epsilon.
            fp = obj.P * sign(d) .* (max(abs(d), obj.Epsilon) .^ (obj.P - 1));
        end

        function str = getName(obj)
            str = sprintf('p-Norm (p=%.2f)', obj.P);
        end

        function tf = isConvex(obj)
            % Convex for p >= 1, non-convex for p < 1.
            tf = (obj.P >= 1.0);
        end
    end
end
