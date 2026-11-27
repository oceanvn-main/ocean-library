classdef SigmoidKernel < OceanStats.Supervised.SVM.Base.ISVMKernel
    % SIGMOIDKERNEL  Neural Network (Sigmoid) kernel for SVM.
    %
    %   K(x, z) = tanh(Kappa1 * <x, z> + Kappa2)
    %
    %   Matches ESL Equation 12.22:
    %   K(x, x') = tanh(kappa1 * <x, x'> + kappa2)
    %
    %   Note: This kernel is not strictly positive semi-definite (PSD)
    %   for all values of kappa1 and kappa2, but it has been successfully
    %   used as a proxy for a two-layer neural network perceptron.
    %
    %   ESL Reference: §12.3.1 (Eq 12.22)
    %   Author: Ocean Library | 2026-03-12

    properties
        Kappa1 (1,1) double = 1.0    % Slope (often referred to as gamma or alpha)
        Kappa2 (1,1) double = -1.0   % Intercept (often referred to as coef0 or c)
    end

    methods
        function obj = SigmoidKernel(options)
            arguments
                options.Kappa1 (1,1) double = 1.0
                options.Kappa2 (1,1) double = -1.0
            end
            obj.Kappa1 = options.Kappa1;
            obj.Kappa2 = options.Kappa2;
        end

        function K = compute(obj, X1, X2)
            % COMPUTE  K = tanh(Kappa1 * (X1 * X2') + Kappa2)
            arguments
                obj
                X1  (:,:) double
                X2  (:,:) double
            end
            K = tanh(obj.Kappa1 .* (X1 * X2') + obj.Kappa2);
        end

        function s = name(obj)
            s = sprintf('Sigmoid/NN(K1=%.3g, K2=%.3g)', ...
                        obj.Kappa1, obj.Kappa2);
        end
    end
end
