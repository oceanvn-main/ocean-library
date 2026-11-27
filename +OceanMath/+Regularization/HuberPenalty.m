classdef HuberPenalty < OceanMath.Regularization.IRobustPenalty
    % HUBERPENALTY  Piecewise Huber penalty rho(d) (smooth L2/L1 compromise).
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05

    properties
        Sigma (1,1) double {mustBePositive} = 1.0
    end

    methods
        function obj = HuberPenalty(options)
            arguments
                options.Sigma (1,1) double = 1.0
            end
            obj.Sigma = options.Sigma;
        end

        function val = evaluate(obj, d)
            abs_d = abs(d);
            mask  = (abs_d <= obj.Sigma);
            val   = zeros(size(d));
            val(mask)  = 0.5 * (d(mask) .^ 2);
            val(~mask) = obj.Sigma * (abs_d(~mask) - 0.5 * obj.Sigma);
        end

        function w = weight(obj, d)
            % Library convention (see VariationalEnergy): w(d) = rho(d) / d^2.
            %   |d| <= sigma : (d^2/2) / d^2                 = 1/2
            %   |d| >  sigma : sigma*(|d| - sigma/2) / d^2
            % The quadratic zone includes d = 0, where the limit is 1/2.
            abs_d = abs(d);
            mask  = (abs_d <= obj.Sigma);
            w     = 0.5 * ones(size(d));
            w(~mask) = obj.Sigma * (abs_d(~mask) - 0.5 * obj.Sigma) ./ (abs_d(~mask) .^ 2);
        end

        function fp = derivative(obj, d)
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
