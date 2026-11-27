classdef HuberPenalty < OceanMath.Regularization.IRobustPenalty
    % HUBERPENALTY  Piecewise Huber penalty rho(d) (smooth L2/L1 compromise).
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
    %   IRLS weight  w(d) = rho'(d)/d:
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
    %       This class is for field regularization (Szeliski context).
    %
    %   Usage:
    %       pen = OceanMath.Regularization.HuberPenalty();
    %       pen = OceanMath.Regularization.HuberPenalty('Sigma', 0.5);
    %
    %   Package: OceanMath.Regularization
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
                options.Sigma (1,1) double = 1.0
            end
            obj.Sigma = options.Sigma;
        end

        function val = evaluate(obj, d)
            % Piecewise Huber:
            %   quadratic zone  |d| <= sigma  → 0.5 * d^2
            %   linear zone     |d| >  sigma  → sigma * (|d| - 0.5*sigma)
            abs_d = abs(d);
            mask  = (abs_d <= obj.Sigma);
            val   = zeros(size(d));
            val(mask)  = 0.5 * (d(mask) .^ 2);
            val(~mask) = obj.Sigma * (abs_d(~mask) - 0.5 * obj.Sigma);
        end

        function w = weight(obj, d)
            % w(d) = rho'(d)/d:  1 for |d| <= sigma,  sigma/|d| beyond it.
            abs_d = abs(d);
            mask  = (abs_d <= obj.Sigma);
            w     = ones(size(d));
            w(~mask) = obj.Sigma ./ max(abs_d(~mask), 1e-12);
        end

        function fp = derivative(obj, d)
            % Piecewise derivative (continuous everywhere):
            %   rho'(d) = d              if |d| <= sigma
            %           = sigma*sign(d)  if |d| >  sigma
            abs_d = abs(d);
            mask  = (abs_d <= obj.Sigma);
            fp    = zeros(size(d));
            fp(mask)  = d(mask);
            fp(~mask) = obj.Sigma * sign(d(~mask));
        end

        function str = getName(obj)
            str = sprintf('Huber (sigma=%.2f)', obj.Sigma);
        end

        function tf = isConvex(~)
            tf = true;
        end
    end
end
