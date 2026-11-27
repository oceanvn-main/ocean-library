classdef RegressionGenerator < OceanData.Generators.IDataGenerator
    % REGRESSIONGENERATOR Creates 1D non-linear regression data.
    % Model: Y = f(X) + Epsilon
    
    properties
        Function        (1,1) string {mustBeMember(Function, ["Sine", "Poly", "Step"])} = "Sine"
        Degree          (1,1) double {mustBePositive, mustBeInteger} = 3
        Sigma           (1,1) double = -1 % -1 means use DefaultNoise
        Domain          (1,2) double = [0, 1]
        CustomFunc      function_handle = function_handle.empty
        Heteroscedastic (1,1) logical = false
    end
    
    methods
        function obj = RegressionGenerator(opts)
            arguments
                opts.N               (1,1) double = 100
                opts.Function        (1,1) string = "Sine"
                opts.Degree          (1,1) double = 3
                opts.Sigma           (1,1) double = -1
                opts.Domain          (1,2) double = [0, 1]
                opts.CustomFunc      function_handle = function_handle.empty
                opts.Heteroscedastic (1,1) logical = false
            end
            obj.N               = opts.N;
            obj.Function        = opts.Function;
            obj.Degree          = opts.Degree;
            obj.Sigma           = opts.Sigma;
            obj.Domain          = opts.Domain;
            obj.CustomFunc      = opts.CustomFunc;
            obj.Heteroscedastic = opts.Heteroscedastic;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            sig = obj.Sigma;
            if sig == -1, sig = obj.DefaultNoise; end

            % 1. Generate Domain (Uniform [a, b])
            X = obj.Domain(1) + (obj.Domain(2)-obj.Domain(1)) * rand(obj.N, 1);
            X = sort(X); % Sorting aids visualization plots

            % 2. Generate True Function f(x)
            if ~isempty(obj.CustomFunc)
                f_true = obj.CustomFunc(X);
            else
                switch obj.Function
                    case "Sine"
                        f_true = sin(2 * pi * X);
                    case "Poly"
                        % Random polynomial coefficients
                        coeffs = randn(obj.Degree + 1, 1);
                        f_true = polyval(coeffs, X);
                        % Standardize to mean=0, std=1 for consistent scale
                        f_true = (f_true - mean(f_true)) / (std(f_true) + eps);
                    case "Step"
                        f_true = double(X > mean(obj.Domain));
                end
            end

            % 3. Generate Noise (Vectorized)
            if obj.Heteroscedastic
                % Scale noise linearly based on X location
                scale = 1 + 4 * (X - min(X)); 
                epsilon = sig .* randn(obj.N, 1) .* scale;
            else
                epsilon = sig .* randn(obj.N, 1);
            end

            Y = f_true + epsilon;
            
            % Save truth for evaluation
            obj.MetaData.f_true = f_true;
        end
    end
end
