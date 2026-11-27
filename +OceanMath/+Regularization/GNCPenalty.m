classdef GNCPenalty < OceanMath.Regularization.IRobustPenalty
    % GNCPENALTY  Blake-Zisserman (1987) Graduated Non-Convexity penalty min(d^2, V).
    %
    %   The classic non-convex penalty for discontinuity-preserving regularization:
    %
    %       rho(d, V) = min(d^2, V)
    %
    %   This function behaves quadratically for small differences d^2 <= V, but
    %   completely ignores differences beyond the clamp threshold V.
    %
    %   Physical interpretation:
    %     - Small differences (noise)   → penalised quadratically (smoothed)
    %     - Large differences (edges)   → penalty clamped at V (hard outlier reject)
    %     - V controls how "sharp" an edge must be before it is left alone
    %
    %   Why "Graduated" Non-Convexity?
    %     The penalty is non-convex (isConvex = false), so direct energy
    %     minimization gets stuck in local minima.  The GNCSolver handles this
    %     via a continuation strategy: start with large V (nearly quadratic, convex),
    %     then gradually reduce V toward the desired non-convex regime.
    %
    %   Asymptotic behaviour:
    %     V → ∞   :  rho(d,V) = d^2  for all finite d  →  QuadraticPenalty
    %     V → 0   :  rho(d,V) = 0    for all d          →  no regularization
    %
    %   IRLS weight  w(d) = rho'(d)/d:
    %       w(d) = 2     if d^2 <= V  (quadratic zone)
    %            = 0     if d^2 >  V  (clamped: this edge is a discontinuity)
    %
    %   Usage:
    %       pen = OceanMath.Regularization.GNCPenalty();
    %       pen = OceanMath.Regularization.GNCPenalty('V', 0.5);
    %
    %   To use with GNCSolver (continuation), GNCSolver will modify the
    %   active V across iterations — do not freeze V externally.
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Blake & Zisserman (1987); Szeliski (2021) §4.2, Eq. 4.30

    properties
        % V  Clamp threshold for the non-convex clamped penalty.
        %   Differences with d^2 > V are treated as discontinuities (weight = 0).
        V       (1,1) double {mustBePositive} = 1.0
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function obj = GNCPenalty(options)
            % GNCPENALTY  Constructor.
            %
            %   pen = GNCPenalty()           % V = 1.0 (default)
            %   pen = GNCPenalty('V', 0.5)
            arguments
                options.V       (1,1) double = 1.0
                options.Epsilon (1,1) double = 1e-6
            end
            obj.V       = options.V;
            obj.Epsilon = options.Epsilon;
        end

        function val = evaluate(obj, d)
            % rho(d) = min(d^2, V)
            val = min(d .^ 2, obj.V);
        end

        function w = weight(obj, d)
            abs_d2 = d .^ 2;
            mask   = (abs_d2 <= obj.V);
            % w(d) = rho'(d)/d:  2 in the quadratic zone, 0 where the
            % penalty is clamped (the edge is treated as a discontinuity).
            w      = zeros(size(d));
            w(mask)  = 2.0;
        end

        function fp = derivative(obj, d)
            % rho'(d) = 2d    if d^2 <= V
            %         = 0     if d^2 >  V  (clamped: constant, so gradient is 0)
            abs_d2 = d .^ 2;
            mask   = (abs_d2 <= obj.V);
            fp     = zeros(size(d));
            fp(mask) = 2 * d(mask);
        end

        function str = getName(obj)
            str = sprintf('GNC (V=%.2f)', obj.V);
        end

        function tf = isConvex(~)
            % GNC is non-convex — the energy landscape has local minima.
            % The GNCSolver handles this via continuation annealing.
            tf = false;
        end
    end
end
