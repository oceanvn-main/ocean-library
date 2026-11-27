classdef HighDimGenerator < OceanData.Generators.IDataGenerator
    % HIGHDIMGENERATOR Creates sparse, high-dimensional regression.
    % Model: Y = X * Beta + Epsilon
    % Features:
    %   - Supports p >> N (more features than samples).
    %   - Simulates feature correlation (Multicollinearity).
    %   - Generates 'Sparse' true Beta (few non-zeros).
    
    properties
        Features    (1,1) double {mustBePositive, mustBeInteger} = 50
        Active      (1,1) double {mustBePositive, mustBeInteger} = 5
        Correlation (1,1) double {mustBeGreaterThanOrEqual(Correlation, 0), mustBeLessThan(Correlation, 1)} = 0.0
        Sigma       (1,1) double = -1 % -1 means use DefaultNoise
    end
    
    methods
        function obj = HighDimGenerator(opts)
            arguments
                opts.N           (1,1) double = 100
                opts.Features    (1,1) double = 50
                opts.Active      (1,1) double = 5
                opts.Correlation (1,1) double = 0.0
                opts.Sigma       (1,1) double = -1
            end
            obj.N           = opts.N;
            obj.Features    = opts.Features;
            obj.Active      = opts.Active;
            obj.Correlation = opts.Correlation;
            obj.Sigma       = opts.Sigma;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            sig = obj.Sigma;
            if sig == -1, sig = obj.DefaultNoise; end

            % 1. Generate Correlated Predictors X
            if obj.Correlation > 0
                % Construct Toeplitz Covariance Matrix: Sigma_{ij} = rho^|i-j|
                idx = 1:obj.Features;
                distMat = abs(idx' - idx); 
                SigmaMat = obj.Correlation .^ distMat;
                
                % Use Cholesky decomposition via mvnrnd (Optimized)
                mu = zeros(1, obj.Features);
                X = mvnrnd(mu, SigmaMat, obj.N);
            else
                X = randn(obj.N, obj.Features);
            end

            % 2. Generate Sparse Signal Beta
            Beta = zeros(obj.Features, 1);
            permIdx = randperm(obj.Features, obj.Active);
            % Set active coefficients to ~ +/- (1 to 3) to ensure SNR
            Beta(permIdx) = randn(obj.Active, 1) + sign(randn(obj.Active, 1));

            % 3. Generate Targets
            Y = X * Beta + sig * randn(obj.N, 1);
            
            % Save true coefficients
            obj.MetaData.Beta = Beta;
        end
    end
end
