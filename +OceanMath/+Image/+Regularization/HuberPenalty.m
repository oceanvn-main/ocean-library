classdef HuberPenalty < OceanMath.Image.Regularization.IRobustPenalty
    % HUBERPENALTY  Piecewise Huber penalty for robust image regularization.
    %
    %   This is the CLASSIC piecewise Huber (NOT the pseudo-Huber smooth
    %   approximation used in OceanStats ML boosting).
    %
    %   Formula (Szeliski §4.2):
    %       rho(d) = 0.5 * d^2                          if |d| <= sigma
    %              = sigma * (|d| - 0.5 * sigma)         if |d| >  sigma
    %
    %   The transition point sigma controls the L2 → L1 crossover:
    %       |d| <= sigma  →  quadratic regime (smooth, noise suppression)
    %       |d| >  sigma  →  linear regime    (edge-preserving, robust to outliers)
    %
    %   IRLS weight:
    %       w(d) = 1             if |d| <= sigma
    %            = sigma / |d|   if |d| >  sigma
    %
    %   Properties:
    %     - Convex (isConvex = true)
    %     - Differentiable everywhere (unlike the hard L1 subgradient)
    %     - A smooth compromise between QuadraticPenalty and L1Penalty
    %
    %   Design note:
    %       Distinct from OceanStats.Loss.Regression.HuberLoss which is the
    %       pseudo-Huber (smooth everywhere) for gradient boosting trees.
    %       This class is for image field regularization (Szeliski context).
    %
    %   Usage:
    %       pen = OceanMath.Image.Regularization.HuberPenalty();
    %       pen = OceanMath.Image.Regularization.HuberPenalty('Sigma', 0.5);
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2; Huber (1964)

    properties
        % SIGMA  Transition threshold between quadratic and linear regimes.
        %   Smaller sigma → more robust (L1-like), larger sigma → more quadratic.
        Sigma (1,1) double {mustBePositive} = 1.0
    end

    methods
        function obj = HuberPenalty(options)
            % HUBERPENALTY  Constructor.
            %
            %   pen = HuberPenalty()
            %   pen = HuberPenalty('Sigma', 0.5)
            arguments
                options.Sigma (1,1) double {mustBePositive} = 1.0
            end
            obj.Sigma = options.Sigma;
        end

        function val = evaluate(obj, d)
            % Piecewise Huber:
            %   quadratic zone  |d| <= sigma  → 0.5 * d^2
            %   linear zone     |d| >  sigma  → sigma * (|d| - 0.5*sigma)
            s   = obj.Sigma;
            ad  = abs(d);
            val = zeros(size(d));

            quad_zone = (ad <= s);
            val( quad_zone) = 0.5 .* d(quad_zone).^2;
            val(~quad_zone) = s .* (ad(~quad_zone) - 0.5 .* s);
        end

        function w = weight(obj, d)
            % IRLS weight  w(d) = rho(d)/d^2:
            %   quadratic zone  → w = 1
            %   linear zone     → w = sigma/|d|
            %
            % At d=0: both rho(0)=0 and d^2=0, but the limit is 1 (L'Hôpital),
            % consistent with the quadratic zone.
            s  = obj.Sigma;
            ad = abs(d);
            w  = ones(size(d));

            lin_zone = (ad > s);
            w(lin_zone) = s ./ ad(lin_zone);
        end

        function g = derivative(obj, d)
            % Piecewise derivative (continuous everywhere):
            %   rho'(d) = d           if |d| <= sigma
            %           = sigma*sign(d)  if |d| >  sigma
            s  = obj.Sigma;
            ad = abs(d);
            g  = zeros(size(d));

            quad_zone = (ad <= s);
            g( quad_zone) = d(quad_zone);
            g(~quad_zone) = s .* sign(d(~quad_zone));
        end

        function name = getName(obj)
            name = sprintf('Huber  (\\sigma=%.2g)', obj.Sigma);
        end

        function tf = isConvex(~)
            tf = true;
        end
    end
end
