classdef PolynomialKernel < OceanStats.Supervised.SVM.Base.ISVMKernel
    % POLYNOMIALKERNEL  dth-Degree Polynomial kernel.
    %
    %   K(x, z) = (Gamma * <x, z> + Coef0)^Degree
    %
    %   Matches ESL Equation 12.22 (where Gamma=1, Coef0=1):
    %   K(x, x') = (1 + <x, x'>)^d
    %
    %   Controls the nonlinearity of the decision boundary. A degree of 1
    %   is equivalent to the LinearKernel if Coef0=0 and Gamma=1.
    %
    %   ESL Reference: §12.3.1 (Eq 12.22)
    %   Author: Ocean Library | 2026-03-12

    properties
        Degree (1,1) double {mustBePositive, mustBeInteger} = 2
        Gamma  (1,1) double = 1.0
        Coef0  (1,1) double = 1.0
    end

    methods
        function obj = PolynomialKernel(options)
            arguments
                options.Degree (1,1) double {mustBePositive, mustBeInteger} = 2
                options.Gamma  (1,1) double = 1.0
                options.Coef0  (1,1) double = 1.0
            end
            obj.Degree = options.Degree;
            obj.Gamma  = options.Gamma;
            obj.Coef0  = options.Coef0;
        end

        function K = compute(obj, X1, X2)
            % COMPUTE  K = (Gamma * (X1 * X2') + Coef0)^Degree
            arguments
                obj
                X1  (:,:) double
                X2  (:,:) double
            end
            K = (obj.Gamma .* (X1 * X2') + obj.Coef0) .^ obj.Degree;
        end

        function s = name(obj)
            s = sprintf('Polynomial(d=%d, g=%.3g, c=%.3g)', ...
                        obj.Degree, obj.Gamma, obj.Coef0);
        end
    end
end
