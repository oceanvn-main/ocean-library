classdef ComplexFunctionsGenerator < OceanData.Generators.IDataGenerator
    % COMPLEXFUNCTIONSGENERATOR Creates complex multi-dimensional functional benchmarks.
    % Includes Tensor Products, Neural Networks (ESL 9.4.2), and Friedman's functions.
    
    properties
        Function     (1,1) string {mustBeMember(Function, ["TensorProduct", "NeuralNetwork", "Friedman1"])} = "TensorProduct"
        Features     (1,1) double = -1 % -1 means use default for the function
        Sigma        (1,1) double = -1 % -1 means use DefaultNoise
        Distribution (1,1) string {mustBeMember(Distribution, ["Gaussian", "Uniform"])} = "Gaussian"
    end
    
    methods
        function obj = ComplexFunctionsGenerator(opts)
            arguments
                opts.N            (1,1) double = 100
                opts.Function     (1,1) string = "TensorProduct"
                opts.Features     (1,1) double = -1
                opts.Sigma        (1,1) double = -1
                opts.Distribution (1,1) string = "Gaussian"
            end
            obj.N            = opts.N;
            obj.Function     = opts.Function;
            obj.Features     = opts.Features;
            obj.Sigma        = opts.Sigma;
            obj.Distribution = opts.Distribution;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            sig = obj.Sigma;
            if sig == -1, sig = obj.DefaultNoise; end
            
            p = obj.Features;

            switch obj.Function
                case "TensorProduct"
                    % Scenario 1 & 2 from ESL 9.4.2
                    if p == -1, p = 2; end
                    if obj.Distribution == "Uniform"
                        X = 5 * rand(obj.N, p) - 2.5;
                    else
                        X = randn(obj.N, p);
                    end
                    
                    % Target depends only on X1 and X2 (max operator simulates hinges)
                    f_true = max(X(:,1)-1, 0) + max(X(:,1)-1, 0) .* max(X(:,2)-0.8, 0);
                    Y = f_true + sig * randn(obj.N, 1);
                    
                case "NeuralNetwork"
                    % Scenario 3 from ESL 9.4.2
                    if p == -1, p = 10; end
                    X = randn(obj.N, p);
                    
                    sigma_fn = @(t) 1 ./ (1 + exp(-t));
                    l1 = sum(X(:, 1:min(5, p)), 2);
                    
                    if p >= 10
                        l2 = X(:, 6) - X(:, 7) + X(:, 8) - X(:, 9) + X(:, 10);
                    else
                        l2 = zeros(obj.N, 1);
                    end
                    
                    f_true = sigma_fn(l1) + sigma_fn(l2);
                    Y = f_true + sig * randn(obj.N, 1);
                    
                case "Friedman1"
                    % Famous Friedman dataset from the original MARS paper
                    if p == -1, p = 10; end
                    % Friedman 1 expects uniform [0,1] features
                    X = rand(obj.N, p);
                    
                    % 5 active features, remaining are noise
                    f_true = 10 * sin(pi * X(:,1) .* X(:,2)) + 20 * (X(:,3) - 0.5).^2 + 10 * X(:,4) + 5 * X(:,5);
                    Y = f_true + sig * randn(obj.N, 1);
            end
            
            obj.MetaData.f_true = f_true;
        end
    end
end
