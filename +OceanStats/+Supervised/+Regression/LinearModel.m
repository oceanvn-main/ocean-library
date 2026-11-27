classdef LinearModel < OceanStats.Base.ProbabilisticModel
    % LINEARMODEL Robust Linear Regression with Bayesian & Bootstrap support.
    %
    % This class is the workhorse for Experiment 1 (Grand Unification).
    % It implements the full ProbabilisticModel interface, providing:
    % 1. MLE Fitting (fit)
    % 2. Log-Likelihood Calculation (logLikelihood)
    % 3. Bayesian Prior & Posterior (logPrior, posterior)
    % 4. Bootstrap Data Generation (boot_gen)
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    properties
        % Model Hyperparameters (for Bayesian Prior)
        PriorPrecision (1,1) double = 0; % Tau (0 = Flat Prior, Inf = Point Mass)
        PriorBeta      (:,1) double      % Mean of Beta Prior (default 0)
    end

    methods
        function obj = LinearModel(options)
            % Constructor
            arguments
                options.PriorPrecision = 0; % Default to non-informative (MLE)
                options.PriorBeta      = [];
            end
            obj.PriorPrecision = options.PriorPrecision;
            obj.PriorBeta      = options.PriorBeta;
        end

        % -----------------------------------------------------------------
        % Core Fitting & Prediction (MLE)
        % -----------------------------------------------------------------
        function params = fit(obj, data, weights)
            % FIT Standard OLS / MLE fit.
            % Returns beta and sigma_mle.
            % Supports optional weights for WLS (required for Mixture Models/HME).
            X = data(:, 1:end-1);
            y = data(:, end);
            
            % Add intercept if not present (assuming X is raw features)
            if ~all(X(:,1)==1)
                 X = [ones(size(X,1), 1), X];
            end

            % Fitting Solve
            if nargin < 3 || isempty(weights) || all(weights == 1)
                % Standard OLS Solution: beta = (X'X)^-1 X'y
                beta = X \ y;
                y_hat = X * beta;
                epsilon = y - y_hat;
                N = length(y);
                sigma_sq = sum(epsilon.^2) / N;
            else
                % Weighted Least Squares (WLS)
                % Formula: beta = (X' W X)^-1 X' W y
                % Efficient implementation using sqrt(W)
                W = max(weights(:), 1e-10); % Numerical safety
                sqrtW = sqrt(W);
                Xw = X .* sqrtW;
                yw = y .* sqrtW;
                
                beta = Xw \ yw;
                
                % Weighted Sigma ML
                y_hat = X * beta;
                epsilon = y - y_hat;
                sigma_sq = sum(W .* (epsilon.^2)) / sum(W);
            end
            
            params = struct('beta', beta, 'sigma', sqrt(sigma_sq));
        end

        function y_hat = predict(obj, X_new, params)
            % PREDICT
            if nargin < 3 || isempty(params)
                error('LinearModel requires params for prediction.');
            end
            
             % Handle intercept
            if ~all(X_new(:,1)==1)
                 X_new = [ones(size(X_new,1), 1), X_new];
            end
            
            y_hat = X_new * params.beta;
        end

        % -----------------------------------------------------------------
        % Probabilistic Interface (Likelihood, Prior, Bootstrap)
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
            % DERIVATION: Log-Likelihood of N i.i.d. Gaussian Observations
            % -----------------------------------------------------------------
            % 1. Model Assumption:
            %    y_i = y_hat_i + epsilon_i, where epsilon_i ~ N(0, sigma^2)
            %    Therefore, P(y_i | beta, sigma) is a univariate Gaussian.
            %
            % 2. PDF for a Single Observation:
            %    P(y_i) = (2 * pi * sigma^2)^(-1/2) * exp( - (y_i - y_hat_i)^2 / (2 * sigma^2) )
            %
            % 3. Joint Likelihood L (assuming N independent observations):
            %    L = Product_{i=1 to N} [ P(y_i) ]
            %
            % 4. Log-Likelihood ll = log(L) = Sum_{i=1 to N} [ log(P(y_i)) ]:
            %    ll = Sum [ log( (2*pi*sigma^2)^(-1/2) ) - (y_i - y_hat_i)^2 / (2*sigma^2) ]
            %       = Sum [ -0.5 * log(2*pi*sigma^2) ]   - Sum [ (y_i - y_hat_i)^2 / (2*sigma^2) ]
            %
            % 5. Simplify Terms for Computation:
            %    A. Normalization Term (Sum of Constants):
            %       The term -0.5 * log(2*pi*sigma^2) is constant for all i.
            %       Summing it N times implies multiplying by N.
            %       -> Term 1 = -0.5 * N * log(2 * pi * sigma^2)
            %
            %    B. Error Term (Sum of Squared Residuals):
            %       Factor out -1 / (2*sigma^2) from the summation.
            %       Sum residuals: sum( (y - y_hat).^2 )
            %       -> Term 2 = -0.5 * sum(residuals.^2) / sigma^2
            % -----------------------------------------------------------------

        function ll = logLikelihood(obj, data, params)
            % LOGLIKELIHOOD Log P(y | X, beta, sigma)
            % Assumes Gaussian noise.
            X = data(:, 1:end-1);
            y = data(:, end);
            if ~all(X(:,1)==1), X = [ones(size(X,1),1), X]; end
            
            y_hat = X * params.beta;
            sigma = params.sigma;
            N = length(y);
            
            % Log-Likelihood of N i.i.d. Gaussians
            % -N/2 * log(2*pi*sigma^2) - (1/(2*sigma^2)) * sum(residuals^2)
            term1 = -0.5 * N * log(2 * pi * sigma^2);
            term2 = -0.5 * sum((y - y_hat).^2) / sigma^2;
            ll = term1 + term2;
        end

        function lp = logPrior(obj, params)
            % LOGPRIOR Log P(beta) ~ N(0, 1/tau * I)
            % If PriorPrecision (tau) is 0, return 0 (Flat Prior).
            if obj.PriorPrecision == 0
                lp = 0;
                return;
            end
            
            beta = params.beta;
            k = length(beta);
            tau = obj.PriorPrecision;

            % -----------------------------------------------------------------
            % DERIVATION: Log PDF of Multivariate Gaussian Prior
            % -----------------------------------------------------------------
            % 1. General PDF for k-dimensional Gaussian N(mu, Sigma):
            %    P(beta) = (2*pi)^(-k/2) * |Sigma|^(-1/2) * exp( -0.5 * (beta - mu)' * Sigma^-1 * (beta - mu) )
            %
            % 2. Assumptions for this model:
            %    - Mean mu = 0
            %    - Precision Matrix Lambda = Sigma^-1 = tau * I (Spherical/Isotropic)
            %    - Dimensionality = k
            %
            % 3. Rewrite using Precision Lambda:
            %    |Sigma|^(-1/2) = |Sigma^-1|^(1/2) = |Lambda|^(1/2)
            %    P(beta) = (2*pi)^(-k/2) * |Lambda|^(1/2) * exp( -0.5 * beta' * Lambda * beta )
            %
            % 4. Take the Natural Logarithm (log P(beta)):
            %    log P = log( (2*pi)^(-k/2) ) + log( |Lambda|^(1/2) ) - 0.5 * beta' * Lambda * beta
            %          = [Constant]           + 0.5 * log(|Lambda|)    - 0.5 * beta' * Lambda * beta
            %
            % 5. Simplify Terms for Computation:
            %    A. Determinant Term (0.5 * log(|Lambda|)):
            %       Since Lambda = tau * I (k x k matrix):
            %       |Lambda| = tau^k
            %       log(|Lambda|) = log(tau^k) = k * log(tau)
            %       -> Term 1 = 0.5 * k * log(tau)
            %
            %    B. Quadratic Term (-0.5 * beta' * Lambda * beta):
            %       beta' * (tau * I) * beta = tau * (beta' * beta) = tau * sum(beta.^2)
            %       -> Term 2 = -0.5 * tau * sum(beta.^2)
            % -----------------------------------------------------------------

            % Log PDF of Gaussian Prior
            % const + 0.5 * log(det(Precision)) - 0.5 * (beta)' * Precision * (beta)
            % Precision = tau * I

            term1 = 0.5 * k * log(tau); % (ignoring 2pi const)
            term2 = -0.5 * tau * sum(beta.^2);
            lp = term1 + term2;
        end

        function data_new = boot_gen(obj, params, n, x_context)
            % BOOT_GEN Parametric Bootstrap Generation
            % Generates y* = X * beta + N(0, sigma)
            if nargin < 4, error('LinearModel requires x_context.'); end
            
            y_mean = obj.predict(x_context, params);
            epsilon = params.sigma * randn(size(x_context, 1), 1);
            
            % If x_context was raw, return [X, y_new]
            % If x_context has intercept, strip it? Usually we return [RawX, Y]
            % Let's rely on x_context being the full feature set provided by user.
            data_new = [x_context, y_mean + epsilon];
        end
        
        % -----------------------------------------------------------------
        % Analytical Posterior (Special for Exp 1)
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
            % DERIVATION: Analytical Posterior for Bayesian Linear Regression
            % -----------------------------------------------------------------
            % 1. Bayes' Theorem:
            %    P(beta | y, X) proportional to P(y | X, beta) * P(beta)
            %    Since both Prior and Likelihood are Gaussian, the Posterior is Gaussian.
            %
            % 2. Log-Posterior Expansion (Completing the Square):
            %    log P(beta | y) = log P(prior) + log P(likelihood) + const
            %
            %    A. Log Prior (Gaussian N(0, Lambda_0^-1)):
            %       -0.5 * beta' * Lambda_0 * beta
            %
            %    B. Log Likelihood (Gaussian N(X*beta, sigma^2*I)):
            %       -0.5 * (y - X*beta)' * (sigma^-2 * I) * (y - X*beta)
            %       Expands to: -0.5 * sigma^-2 * [ y'y - 2*beta'*X'*y + beta'*X'*X*beta ]
            %
            % 3. Grouping Quadratic and Linear Terms in beta:
            %    Combine the beta' * A * beta terms (Precision) and beta' * b terms (Mean).
            %    
            %    log P(beta | y) = -0.5 * [ beta' * (Lambda_0 + sigma^-2 * X'*X) * beta 
            %                             - 2 * beta' * (sigma^-2 * X'*y) ]
            %
            % 4. Matching to Standard Multivariate Gaussian Form:
            %    Standard Form: -0.5 * [ beta' * Lambda_n * beta - 2 * beta' * Lambda_n * mu_n ]
            %
            %    A. Posterior Precision (Lambda_n):
            %       The matrix inside the quadratic term:
            %       Lambda_n = Lambda_0 + (1/sigma^2) * X'*X
            %       (Posterior Certainty = Prior Certainty + Data Certainty)
            %
            %    B. Posterior Mean (mu_n):
            %       Match the linear term: Lambda_n * mu_n = (1/sigma^2) * X'*y
            %       Solve for mu_n:
            %       mu_n = inv(Lambda_n) * ( (1/sigma^2) * X'*y )
            %       (This is a weighted average of Prior Mean (0) and MLE solution)
            % -----------------------------------------------------------------

        function post_params = analytical_posterior(obj, data, known_sigma)
            % ANALYTICAL_POSTERIOR Calculates exact N(mu_n, Sigma_n)
            % for Bayesian Regression with Gaussian Prior N(0, 1/tau).
            % 
            % This is CRITICAL for Experiment 1 convergence check.
            
            X = data(:, 1:end-1);
            y = data(:, end);
            if ~all(X(:,1)==1), X = [ones(size(X,1),1), X]; end
            
            [~, p] = size(X);
            tau = obj.PriorPrecision;
            
            % TEXTBOOK DEFINITION MATCH:
            % Prior Precision = (1/tau) * I
            % This matches the term (1/tau)*Sigma^-1 in Eq 8.28 where Sigma=I
            Lambda_0 = (1 / tau) * eye(p);
            
            % Likelihood Precision (X'X / sigma^2)
            % If known_sigma not provided, estimate from OLS
            if nargin < 3
                params_mle = obj.fit(data);
                sigma_sq = params_mle.sigma^2;
            else
                sigma_sq = known_sigma^2;
            end
            
            % Posterior Precision Lambda_n = Lambda_0 + sigma^-2 * X'X
            XTX = X' * X;
            Lambda_n = Lambda_0 + (1/sigma_sq) * XTX;
            
            % Posterior Covariance Sigma_n = inv(Lambda_n)
            Sigma_n = inv(Lambda_n);
            
            % Posterior Mean mu_n = Sigma_n * (Lambda_0*mu_0 + sigma^-2 * X'y)
            % Assuming prior mean mu_0 = 0
            mu_n = Sigma_n * ((1/sigma_sq) * X' * y);
            
            post_params = struct('mean', mu_n, 'cov', Sigma_n, 'sigma_noise', sqrt(sigma_sq));
        end
    end
end