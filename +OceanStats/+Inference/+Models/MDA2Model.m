classdef MDA2Model < handle
    % MDA2MODEL Pure state container for MDA2 (shared-center mixture model).
    %
    % THEORY (ESL Exercise 12.11):
    %   Joint density: P(G,X) = Sum_r pi_r * P_r(G) * phi(x; mu_r, Sigma)
    %   R Gaussian centers are SHARED across all K classes.
    %   Each center r has a class profile P_r(G=k) describing how strongly
    %   it is associated with each class.
    %
    % PROPERTIES:
    %   R             - Number of shared mixture centers
    %   K             - Number of classes
    %   Mu            - (R x p) shared Gaussian centers
    %   Sigma         - (p x p) shared pooled covariance
    %   Pi            - (R x 1) mixture component priors
    %   ClassProfile  - (R x K) matrix: P_r(G=k)
    %   ClassPriors   - (K x 1) class prior probabilities
    %   Regularization - Ridge term for covariance stability
    %
    % This class holds NO fitting logic. Training is delegated to
    % Recipe_MDA2_EM via the GenericOptimizer.
    %
    % See also: Recipe_MDA2_EM, OceanStats.Optimization.GenericOptimizer
    %
    % Author: Ocean Library System
    % Date:   2026-03-24

    properties
        R    (1,1) double = 10   % Number of shared centers
        K    (1,1) double = 2    % Number of classes
        p    (1,1) double = 2    % Feature dimensionality

        Mu            double = [] % (R x p) centers
        Sigma         double = [] % (p x p) shared covariance
        Pi            double = [] % (R x 1) component priors
        ClassProfile  double = [] % (R x K) class profiles P_r(G=k)
        ClassPriors   double = [] % (K x 1) class priors
        Classes       double = [] % (K x 1) unique class labels

        Regularization (1,1) double = 1e-6
    end

    methods
        function obj = MDA2Model(R, K, p)
            % Constructor: allocate state for R centers, K classes, p dims
            arguments
                R (1,1) double = 10
                K (1,1) double = 2
                p (1,1) double = 2
            end
            obj.R = R;
            obj.K = K;
            obj.p = p;
        end

        function ll = logLikelihood(obj, data)
            % LOGLIKELIHOOD  Observed-data log-likelihood for MM framework.
            %
            % data = [X, y]  where X is (N x p), y is (N x 1)
            %
            % log P(data) = Sum_i log[ Sum_r pi_r * P_r(G=g_i) * phi(x_i; mu_r, Sigma) ]

            X = data(:, 1:end-1);
            y = data(:, end);
            [N, ~] = size(X);

            % Precompute Gaussian log-densities for all (i, r) pairs
            logPhi = obj.computeLogPhi(X);  % (N x R)

            ll = 0;
            for i = 1:N
                k = find(obj.Classes == y(i), 1);
                % log[ Sum_r pi_r * P_r(G=k) * phi(x_i; mu_r, Sigma) ]
                log_terms = log(obj.Pi + eps) + log(obj.ClassProfile(:, k) + eps) + logPhi(i, :)';
                % Log-sum-exp
                maxTerm = max(log_terms);
                ll = ll + maxTerm + log(sum(exp(log_terms - maxTerm)));
            end
        end

        function logPhi = computeLogPhi(obj, X)
            % COMPUTELOGPHI  Log Gaussian densities phi(x_i; mu_r, Sigma) for all (i,r).
            %
            % Returns (N x R) matrix.

            [N, D] = size(X);
            R = obj.R;

            % Cholesky for numerical stability
            L = chol(obj.Sigma, 'lower');  % Sigma = L * L'
            logDetSigma = 2 * sum(log(diag(L)));
            constTerm = -0.5 * D * log(2*pi) - 0.5 * logDetSigma;

            logPhi = zeros(N, R);
            for r = 1:R
                diff = X - obj.Mu(r, :);         % (N x D)
                v = diff / L';                     % Solve: v = diff * inv(L')
                mahal = sum(v.^2, 2);              % Mahalanobis distance squared
                logPhi(:, r) = constTerm - 0.5 * mahal;
            end
        end
    end
end
