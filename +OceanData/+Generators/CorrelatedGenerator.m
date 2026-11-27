classdef CorrelatedGenerator < OceanData.Generators.IDataGenerator
    % CORRELATEDGENERATOR Generates synthetic block-correlated regression data.
    %
    % This generator mimics the simulation setting of ESL Section 16.2.3.
    % It generates a dictionary of Gaussian variables strongly correlated 
    % within blocks, but uncorrelated between blocks. 
    % It then assigns non-zero coefficients to a subset of variables (one 
    % drawn from each block) and adds Gaussian noise with a specific
    % noise-to-signal ratio.
    %
    % Note: Number of samples (N) is inherited from IDataGenerator.
    %
    % Properties:
    %   NumFeatures      (p)     - Default 1000
    %   BlockSize        (k)     - Default 20
    %   Rho              (rho)   - Default 0.95
    %   ActivePerBlock           - Default 1
    %   NoiseSignalRatio (NSR)   - Default 0.72
    %
    % The ground truth coefficients (beta) and noiseless signal (f_X) 
    % are stored in the MetaData property after generation.
    
    properties
        NumFeatures      (1,1) double {mustBePositive, mustBeInteger} = 1000
        BlockSize        (1,1) double {mustBePositive, mustBeInteger} = 20
        Rho              (1,1) double {mustBeGreaterThanOrEqual(Rho, 0), mustBeLessThan(Rho, 1)} = 0.95
        ActivePerBlock   (1,1) double {mustBePositive, mustBeInteger} = 1
        NoiseSignalRatio (1,1) double {mustBeNonnegative} = 0.72
    end
    
    methods
        function obj = CorrelatedGenerator(options)
            arguments
                options.N = 60
                options.NumFeatures = 1000
                options.BlockSize = 20
                options.Rho = 0.95
                options.ActivePerBlock = 1
                options.NoiseSignalRatio = 0.72
            end
            
            obj.N = options.N;
            obj.NumFeatures = options.NumFeatures;
            obj.BlockSize = options.BlockSize;
            obj.Rho = options.Rho;
            obj.ActivePerBlock = options.ActivePerBlock;
            obj.NoiseSignalRatio = options.NoiseSignalRatio;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            % GENERATECORE Creates the synthetic dataset.
            % Returns [X, Y] and assigns TrueBeta to MetaData.
            
            p = obj.NumFeatures;
            k = obj.BlockSize;
            rho = obj.Rho;
            
            if mod(p, k) ~= 0
                error('OceanData:Generators:CorrelatedGenerator:InvalidBlockSize', ...
                    'NumFeatures (%d) must be exactly divisible by BlockSize (%d).', p, k);
            end
            
            numBlocks = p / k;
            
            if obj.ActivePerBlock > k
                error('OceanData:Generators:CorrelatedGenerator:InvalidActivePerBlock', ...
                    'ActivePerBlock (%d) cannot exceed BlockSize (%d).', obj.ActivePerBlock, k);
            end
            
            % 1. Generate Correlated Features (X) using LatentVariableSampler
            [A, D] = OceanMath.Random.Samplers.LatentVariableSampler.buildBlockCorrelated(p, k, rho); %Build Latent Matrix and NoiseStd Vector 
            sampler = OceanMath.Random.Samplers.LatentVariableSampler('Loadings', A, 'NoiseStd', D);
            
            mu = zeros(1, p);
            % Dummy distribution, LatentVariableSampler doesn't actually need real covariance
            dummySigma = eye(p); 
            dist = OceanMath.Information.Distributions.MultivariateNormalDistribution(mu, dummySigma);
            
            X = sampler.sample(dist, obj.N);
            
            % 2. Generate True Coefficients (Beta)
            beta = zeros(p, 1);
            
            for m = 1:numBlocks
                % Indices for the current block
                startIdx = (m - 1) * k + 1;
                
                % Randomly pick 'ActivePerBlock' features
                activeIndices = startIdx - 1 + randperm(k, obj.ActivePerBlock);
                
                % Assign standard Gaussian coefficients
                beta(activeIndices) = randn(obj.ActivePerBlock, 1);
            end
            
            % Store true beta in metadata for future analysis (e.g., plotting paths)
            obj.MetaData.TrueBeta = beta;
            
            % 3. Calculate True Signal
            f_X = X * beta;
            
            % 4. Inject Noise
            % Calculate empirical variance of the true signal
            varSignal = var(f_X);
            
            if varSignal > 0
                % Calculate required noise variance based on NSR
                varNoise = obj.NoiseSignalRatio * varSignal;
                noise = sqrt(varNoise) * randn(obj.N, 1);
            else
                % Edge case: all betas are exactly 0 (virtually impossible)
                noise = zeros(obj.N, 1);
            end
            
            Y = f_X + noise;
            
            % Store true noiseless signal in metadata
            obj.MetaData.TrueSignal = f_X;
        end
    end
end
