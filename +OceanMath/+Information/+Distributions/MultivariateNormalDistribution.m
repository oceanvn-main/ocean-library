classdef MultivariateNormalDistribution < OceanMath.Information.Distributions.IDistribution
    % MULTIVARIATENORMALDISTRIBUTION A multivariate Gaussian distribution.
    %
    % This class implements the IDistribution interface for a p-dimensional
    % Gaussian distribution. It provides an exact, closed-form entropy calculation
    % and supports vectorized PDF evaluation.
    %
    % Construction:
    %   dist = MultivariateNormalDistribution(mu, sigma)
    %
    %   mu    - (1 x p) mean vector
    %   sigma - (p x p) covariance matrix
    %
    % See also: OceanMath.Information.Distributions.IDistribution
    
    properties (SetAccess = private)
        Mu_    (1,:) double
        Sigma_ (:,:) double
        p_     (1,1) double
        InvSigma_ (:,:) double
        LogDetSigma_ (1,1) double
    end
    
    methods
        function obj = MultivariateNormalDistribution(mu, sigma)
            arguments
                mu    (1,:) double
                sigma (:,:) double
            end
            
            p = numel(mu);
            if size(sigma, 1) ~= p || size(sigma, 2) ~= p
                error('OceanMath:Information:MultivariateNormalDistribution:SizeMismatch', ...
                    'Covariance matrix must be p x p where p is the length of mu.');
            end
            
            % Check symmetry
            if norm(sigma - sigma') > 1e-8
                error('OceanMath:Information:MultivariateNormalDistribution:NotSymmetric', ...
                    'Covariance matrix must be symmetric.');
            end
            
            obj.Mu_ = mu;
            obj.Sigma_ = sigma;
            obj.p_ = p;
            
            % Precompute inverse and log-determinant for fast PDF evaluation
            % using Cholesky decomposition for stability
            [L, flag] = chol(sigma, 'lower');
            if flag ~= 0
                error('OceanMath:Information:MultivariateNormalDistribution:NotPosDef', ...
                    'Covariance matrix must be strictly positive definite.');
            end
            
            obj.InvSigma_ = (L') \ (L \ eye(p));
            obj.LogDetSigma_ = 2 * sum(log(diag(L)));
        end
        
        function p = evaluate(obj, x)
            % EVALUATE Query the multivariate density at one or more points.
            %
            % INPUTS:
            %   x - (N x p) matrix of query points
            %
            % OUTPUTS:
            %   p - (N x 1) vector of density values
            if size(x, 2) ~= obj.p_
                error('OceanMath:Information:MultivariateNormalDistribution:DimensionMismatch', ...
                    'Input x must be (N x %d).', obj.p_);
            end
            
            N = size(x, 1);
            x_centered = x - repmat(obj.Mu_, N, 1);
            
            % Mahalanobis distance squared: (x - mu) * Sigma^-1 * (x - mu)'
            % Vectorized calculation for N points
            mahal_sq = sum((x_centered * obj.InvSigma_) .* x_centered, 2);
            
            % Multivariate normal PDF
            norm_const = -0.5 * (obj.p_ * log(2 * pi) + obj.LogDetSigma_);
            p = exp(norm_const - 0.5 * mahal_sq);
        end
        
        function tf = isValid(obj)
            % ISVALID  Check if distribution is valid (sigma is positive definite).
            [~, flag] = chol(obj.Sigma_);
            tf = (flag == 0);
        end
        
        function n = getSupportSize(obj)
            n = Inf;
        end
        
        function supp = getSupport(obj)
            % GETSUPPORT  Returns (p x 2) matrix of [-Inf, Inf]
            supp = repmat([-Inf, Inf], obj.p_, 1);
        end
        
        function h = entropy(obj)
            % ENTROPY  Exact closed-form differential entropy in bits.
            % h = 0.5 * log2( det(2 * pi * e * Sigma) )
            % We use the precomputed natural log-determinant.
            
            natural_entropy = 0.5 * obj.p_ * (1 + log(2 * pi)) + 0.5 * obj.LogDetSigma_;
            h = natural_entropy / log(2); % Convert to bits
        end
        
        function mu = getMean(obj)
            mu = obj.Mu_;
        end
        
        function sigma = getCovariance(obj)
            sigma = obj.Sigma_;
        end
    end
end
