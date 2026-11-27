classdef CholeskySampler < OceanMath.Random.Samplers.ISampler
    % CHOLESKYSAMPLER Multivariate normal sampling via Cholesky decomposition.
    %
    % This is the standard, general-purpose method for sampling from any dense
    % positive-definite covariance matrix. It decomposes Sigma = L * L^T and
    % transforms standard normal noise Z ~ N(0, I) into target samples:
    % X = Z * L^T + Mu.
    %
    % Time Complexity: O(p^3) for decomposition, O(N * p^2) for sampling.
    % This is generic but can become a bottleneck for massive p.
    %
    % See also: OceanMath.Random.Samplers.LatentVariableSampler
    
    methods
        function samples = sample(~, dist, N)
            arguments
                ~
                dist OceanMath.Information.Distributions.IDistribution
                N (1,1) double {mustBePositive, mustBeInteger}
            end
            
            mu = dist.getMean();
            sigma = dist.getCovariance();
            p = numel(mu);
            
            % Decompose Sigma = L * L^T
            try
                L = chol(sigma, 'lower');
            catch
                % Robust fallback for numerically unstable / positive semi-definite matrices
                [V, D] = eig((sigma + sigma') / 2);
                D = diag(max(diag(D), 0)); % Clip negative eigenvalues to 0
                L = V * sqrt(D);
            end
            
            % Generate standard normal noise Z: (N x p)
            Z = randn(N, p);
            
            % Transform Z to X: X = Z * L^T + mu
            % Dimensions: (N x p) * (p x p) + (1 x p) -> (N x p)
            samples = Z * L' + repmat(mu, N, 1);
        end
    end
end
