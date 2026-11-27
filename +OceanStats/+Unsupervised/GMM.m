classdef GMM < OceanStats.Base.ProbabilisticModel
    % GMM Gaussian Mixture Model (Pure Model).
    %
    % Represents a Mixture of Gaussians distribution.
    % Knows NOTHING about how it is trained (EM, Gradient Descent, etc.).
    %
    % Properties:
    %   Mu: (K x D) Matrix of means.
    %   Sigma: (D x D x K) Array of covariance matrices.
    %   Pi: (1 x K) Vector of mixing coefficients.
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    properties
        K (1,1) double      % Number of components
        Mu (:,:) double     % Means (K x D)
        Sigma (:,:,:) double % Covariances (D x D x K)
        Pi (1,:) double     % Mixing Coefficients (1 x K)
        
        Regularization (1,1) double = 1e-6; 
        CovarianceType (1,1) string = "full";
    end

    methods (Access = public)
        function obj = GMM(K, options)
            arguments
                K (1,1) double = 2;
                options.Regularization = 1e-6;
                options.CovarianceType = "full";
            end
            obj.K = K;
            obj.Regularization = options.Regularization;
            obj.CovarianceType = options.CovarianceType;
        end

        % -----------------------------------------------------------------
        % Likelihood & PDF (The Essence of the Model)
        % -----------------------------------------------------------------

        % -----------------------------------------------------------------
            % DERIVATION: Log-Likelihood with Log-Sum-Exp Trick
            % -----------------------------------------------------------------
            % 1. Objective: Compute ln( P(X | params) )
            %    LL = Sum_n [ ln( Sum_k [ pi_k * N(x_n | mu_k, Sigma_k) ] ) ]
            %
            % 2. The Numerical Problem:
            %    The term inside ln(...) is a sum of probabilities.
            %    If x_n is far from mu_k, N(x_n) can be tiny (e.g., 1e-100).
            %    Summing these small numbers can cause underflow to 0.
            %    Then ln(0) = -Inf, breaking the algorithm.
            %
            % 3. The Solution: Work in the Log-Domain (Log-Sum-Exp)
            %    Let a_nk = ln( pi_k * N(x_n) ) = ln(pi_k) + ln(N(x_n))
            %    We want to compute: ln( Sum_k [ exp(a_nk) ] )
            %
            % 4. The Identity:
            %    Let M_n = max_k( a_nk )  (The max log-prob for data point n)
            %    
            %    Sum_k [ exp(a_nk) ] = Sum_k [ exp(a_nk - M_n + M_n) ]
            %                        = exp(M_n) * Sum_k [ exp(a_nk - M_n) ]
            %
            %    Take ln(...) of both sides:
            %    ln( Sum ) = ln( exp(M_n) ) + ln( Sum_k [ exp(a_nk - M_n) ] )
            %              = M_n + ln( Sum_k [ exp(a_nk - M_n) ] )
            %
            % 5. Why this is stable:
            %    The exponent (a_nk - M_n) is always <= 0.
            %    The largest term is exp(M_n - M_n) = exp(0) = 1.
            %    So the sum is at least 1, and ln(sum) >= 0.
            %    No underflow!
            % -----------------------------------------------------------------
        function ll = logLikelihood(obj, data, params)
            % LOGLIKELIHOOD Log P(X | params)
            
            if nargin > 2
                mu = params.Mu;
                sigma = params.Sigma;
                pi_vals = params.Pi;
            else
                mu = obj.Mu;
                sigma = obj.Sigma;
                pi_vals = obj.Pi;
            end
            
            [N, D] = size(data);
            K_ = size(mu, 1);
            
            log_probs = zeros(N, K_);
            for k = 1:K_
                 log_pdf = obj.log_mvnpdf(data, mu(k,:), sigma(:,:,k));
                 log_probs(:,k) = log(pi_vals(k) + eps) + log_pdf;
            end
            
            max_log = max(log_probs, [], 2);
            ll_per_sample = max_log + log(sum(exp(log_probs - max_log), 2));
            ll = sum(ll_per_sample);
        end
        
        function y = log_mvnpdf(obj, X, Mu, Sigma)
            % LOG_MVNPDF Robust computation of log( N(X | Mu, Sigma) )
            % Using Cholesky decomposition for numerical stability.
            
            [N, D] = size(X);
            try
                % -----------------------------------------------------------------
                % DERIVATION: Log-Likelihood of Multivariate Gaussian
                % -----------------------------------------------------------------
                % 1. Standard PDF Formula:
                %    P(x) = (2pi)^(-D/2) * |Sigma|^(-1/2) * exp( -0.5 * (x-mu)' * inv(Sigma) * (x-mu) )
                %
                % 2. Log-PDF:
                %    ln P(x) = -0.5 * [ D*ln(2pi) + ln|Sigma| + (x-mu)' * inv(Sigma) * (x-mu) ]
                %
                % 3. Optimization: Cholesky Decomposition
                %    Let Sigma = R' * R (where R is upper triangular from chol())
                % -----------------------------------------------------------------
                
                R = chol(Sigma);

                % -----------------------------------------------------------------
                % 4. Term A: Log-Determinant ( ln|Sigma| )
                %    |Sigma| = |R' * R| = |R'| * |R| = |R|^2
                %    ln|Sigma| = ln(|R|^2) = 2 * ln|R|
                %    Since R is triangular, |R| is the product of diagonal elements.
                %    -> log_det = 2 * sum( log(diag(R)) )
                % -----------------------------------------------------------------
                log_det = 2 * sum(log(diag(R)));

                % -----------------------------------------------------------------
                % 5. Term B: Quadratic / Mahalanobis Distance
                %    We need: (x-mu)' * inv(Sigma) * (x-mu)
                %    Substitute Sigma = R' * R:
                %    = (x-mu)' * inv(R' * R) * (x-mu)
                %    = (x-mu)' * inv(R) * inv(R') * (x-mu)
                %
                %    Let z = (x-mu) * inv(R)  <-- Calculated efficiently via linear solve (/)
                %    Then the term becomes: z * z' = sum(z.^2)
                % -----------------------------------------------------------------
                X_centered = X - Mu;
                
                % " / R " solves the system (x-mu) = z * R using forward substitution
                z_scores = X_centered / R; 
                quad_form = sum(z_scores.^2, 2);

                % -----------------------------------------------------------------
                % 6. Final Sum: Constant + LogDet + Quadratic
                % -----------------------------------------------------------------
                y = -0.5 * (D * log(2*pi) + log_det + quad_form);
                
            catch
                % Fallback for ill-conditioned Sigma (Singular matrix)
                y = log(mvnpdf(X, Mu, Sigma) + eps);
            end
        end
        
        % Required Abstract Methods
        function y_hat = predict(obj, X, params), y_hat=[]; end
        function data_new = boot_gen(obj, params, n), data_new=[]; end
        function lp = logPrior(obj, params), lp=0; end
        function p = fit(obj, data), error('Use OceanStats.Optimization.GenericOptimizer with a Recipe to train this model.'); end
        
    end
end
