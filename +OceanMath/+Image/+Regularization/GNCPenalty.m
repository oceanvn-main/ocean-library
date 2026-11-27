classdef GNCPenalty < OceanMath.Image.Regularization.IRobustPenalty
    % GNCPENALTY  Blake-Zisserman Graduated Non-Convexity (GNC) clamped penalty.
    %
    %   The classic non-convex penalty for discontinuity-preserving regularization:
    %
    %       rho(d, V) = min(d^2, V)
    %
    %   This function behaves quadratically for small differences |d|^2 < V, but
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
    %   IRLS weight:
    %       w(d) = 1     if d^2 < V   (quadratic zone)
    %            = 0     if d^2 >= V  (clamped: this edge is a discontinuity)
    %
    %   Usage:
    %       pen = OceanMath.Image.Regularization.GNCPenalty();
    %       pen = OceanMath.Image.Regularization.GNCPenalty('V', 0.5);
    %
    %   To use with GNCSolver (continuation), GNCSolver will modify the
    %   active V across iterations — do not freeze V externally.
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Blake & Zisserman (1987); Szeliski (2021) §4.2, Eq. 4.30

    properties
        % V  Clamp threshold for the non-convex clamped penalty.
        %   Differences with d^2 >= V are treated as discontinuities (weight = 0).
        V (1,1) double {mustBePositive} = 1.0
    end

    methods
        function obj = GNCPenalty(options)
            % GNCPENALTY  Constructor.
            %
            %   pen = GNCPenalty()           % V = 1.0 (default)
            %   pen = GNCPenalty('V', 0.5)
            arguments
                options.V (1,1) double {mustBePositive} = 1.0
            end
            obj.V = options.V;
        end

        function val = evaluate(obj, d)
            % rho(d) = min(d^2, V)
            val = min(d .^ 2, obj.V);
        end

        function w = weight(obj, d)
            % IRLS weight  w(d) = rho(d)/d^2:
            %   quadratic zone  d^2 < V  → w = 1
            %   clamped zone    d^2 >= V → w = V/d^2
            %
            % In the clamped zone the gradient of rho w.r.t d is 0, so hard
            % outlier rejection means these edges get zero weight in IRLS.
            % We implement it exactly: w = min(1, V/d^2).
            %
            % At d=0: the limit is 1 (quadratic zone covers d=0 since 0 < V).
            d_sq = d .^ 2;
            % Avoid 0/0: d=0 is always in the quadratic zone (0 < V), so set
            % d_sq_safe = max(d^2, epsilon) only for the ratio, not for the mask.
            quad_mask = (d_sq < obj.V);
            w = zeros(size(d));
            w( quad_mask) = 1;
            denom = max(d_sq(~quad_mask), 1e-30);
            w(~quad_mask) = obj.V ./ denom;
        end

        function g = derivative(obj, d)
            % rho'(d) = 2d    if d^2 < V
            %         = 0     if d^2 >= V  (clamped: constant, so gradient is 0)
            g = zeros(size(d));
            quad_mask = (d .^ 2 < obj.V);
            g(quad_mask) = 2 .* d(quad_mask);
        end

        function name = getName(obj)
            name = sprintf('GNC / Blake-Zisserman  (V=%.2g)', obj.V);
        end

        function tf = isConvex(~)
            % GNC is non-convex — the energy landscape has local minima.
            % The GNCSolver handles this via continuation annealing.
            tf = false;
        end
    end
end
