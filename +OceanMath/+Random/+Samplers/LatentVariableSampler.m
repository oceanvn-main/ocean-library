classdef LatentVariableSampler < OceanMath.Random.Samplers.ISampler
    % LATENTVARIABLESAMPLER Fast multivariate sampling via latent factor construction.
    %
    % This sampler bypasses the O(p^3) Cholesky bottleneck when the target 
    % covariance matrix Sigma can be decomposed into a factor model: 
    %       Sigma = A * A^T + diag(D.^2)
    %
    % It generates features as a linear combination of M shared latent signals
    % and p independent idiosyncratic noise signals:
    %       X = U * A^T + Epsilon * diag(D) + Mu
    %
    % where U ~ N(0, I_M) and Epsilon ~ N(0, I_p).
    %
    % Usage (Generic Factor Analysis):
    %   sampler = LatentVariableSampler('Loadings', A, 'NoiseStd', D);
    %   X = sampler.sample(dist, 1000);
    %
    % Usage (Block-Correlated ESL Example):
    %   [A, D] = OceanMath.Random.Samplers.LatentVariableSampler.buildBlockCorrelated(p, k, rho);
    %   sampler = LatentVariableSampler('Loadings', A, 'NoiseStd', D);
    %
    % See also: OceanMath.Random.Samplers.CholeskySampler
    
    properties
        Loadings  (:,:) double  % (p x M) matrix A
        NoiseStd  (:,1) double  % (p x 1) vector D
    end
    
    methods
        function obj = LatentVariableSampler(options)
            arguments
                options.Loadings (:,:) double
                options.NoiseStd (:,1) double
            end
            obj.Loadings = options.Loadings;
            obj.NoiseStd = options.NoiseStd(:);
            
            if size(obj.Loadings, 1) ~= numel(obj.NoiseStd)
                error('OceanMath:Random:Samplers:DimensionMismatch', ...
                    'Number of rows in Loadings must match length of NoiseStd.');
            end
        end
        
        function samples = sample(obj, dist, N)
            arguments
                obj
                dist OceanMath.Information.Distributions.IDistribution
                N (1,1) double {mustBePositive, mustBeInteger}
            end
            
            mu = dist.getMean();
            p = numel(mu);
            
            if size(obj.Loadings, 1) ~= p
                error('OceanMath:Random:Samplers:DimensionMismatch', ...
                    'Sampler dimension (%d) does not match distribution dimension (%d).', ...
                    size(obj.Loadings, 1), p);
            end
            
            M = size(obj.Loadings, 2);
            
            % Step 1: Generate M Latent Factors
            % U ~ N(0, 1) for i in 1..N, m in 1..M
            U = randn(N, M);
            
            % Step 2: Generate p Idiosyncratic Noise signals
            % Eps ~ N(0, 1) for i in 1..N, j in 1..p
            Eps = randn(N, p);
            
            % Step 3: Construct the Final Features
            % Data Assumption. This is Forward Problem (Simulation/ Theory to reality)
            % X = U * A^T + Eps .* D^T + mu
            samples = U * obj.Loadings' + Eps .* (obj.NoiseStd');
            
            % Add mean vector mu
            samples = samples + repmat(mu, N, 1);
        end
    end
    
    methods (Static)
        function [A, D] = buildBlockCorrelated(p, k, rho, var)
            % BUILDBLOCKCORRELATED Helper to construct factor matrices for 
            % block-diagonal correlation structures (e.g., ESL 16.2.3).
            %
            % INPUTS:
            %   p   - Total number of features
            %   k   - Block size (features per block)
            %   rho - Constant intra-block correlation
            %   var - (Optional) Diagonal variance. Default: 1.0.
            %
            % OUTPUTS:
            %   A   - (p x M) Loadings matrix
            %   D   - (p x 1) Noise standard deviations
            arguments
                p   (1,1) double {mustBePositive, mustBeInteger}
                k   (1,1) double {mustBePositive, mustBeInteger}
                rho (1,1) double {mustBeGreaterThanOrEqual(rho, 0), mustBeLessThan(rho, 1)}
                var (1,1) double = 1.0
            end
            
            if mod(p, k) ~= 0
                error('OceanMath:Random:Samplers:InvalidBlockSize', ...
                    'Total features p (%d) must be divisible by block size k (%d).', p, k);
            end
            
            M = p / k;
            
            % A is a block matrix where each block is a k x 1 vector of sqrt(rho * var)
            % Kronecker product perfectly constructs this p x M matrix
            A = kron(eye(M), ones(k, 1) * sqrt(rho * var));
            
            % D is the remaining idiosyncratic standard deviation
            D = ones(p, 1) * sqrt(var * (1 - rho));
        end
    end
end
