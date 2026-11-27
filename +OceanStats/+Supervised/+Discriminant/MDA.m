classdef MDA < OceanStats.Supervised.Discriminant.IDiscriminantModel
    % MDA Mixture Discriminant Analysis Model
    %
    % THEORY OVERVIEW:
    %   Standard LDA assumes each class is a single Gaussian with a shared 
    %   pooled covariance matrix. Mixture Discriminant Analysis (MDA) relaxes 
    %   this by modeling each class as a Mixture of Gaussians (subclasses).
    %   
    %   ESL 12.6 defines MDA with a shared pooled covariance matrix `Sigma` 
    %   across ALL subclasses in ALL classes. This yields linear discriminant 
    %   boundaries between subclasses, while the overall class boundaries 
    %   become highly non-linear.
    %
    % MODES:
    %   1. 'independent': (QDA-like MDA) Fits an independent GMM for each 
    %      class using `fitgmdist`. Covariances are not shared across classes.
    %   2. 'pooled': (Strict ESL 12.6) Custom EM algorithm that shares a 
    %      single pooled covariance matrix across all subclasses.
    %
    % PROPERTIES:
    %   NumSubclasses - Number of Gaussian components per class (default: 3).
    %                   Can be a scalar (same for all) or vector of length K.
    %   Mode          - 'independent' or 'pooled' (default).
    %   MaxIter       - EM maximum iterations (default: 100).
    %   Tol           - EM convergence tolerance (default: 1e-4).
    
    properties
        NumSubclasses = 3
        Mode {mustBeMember(Mode, {'independent', 'pooled'})} = 'pooled'
        MaxIter (1,1) double = 100
        Tol (1,1) double = 1e-4
        Verbose (1,1) logical = false
        
        % For PMDA / Optimal Scoring in M-Step
        BaseModelConstructor 
        
        % State
        
        % For 'independent' mode: K-length cell array of gmdistribution objects
        GMMs cell         
        
        % For 'pooled' mode:
        SubclassWeights   % Cell array of mixing proportions per class
        SubclassMeans     % Cell array of Subclass means per class
        PooledCovariance  % The shared p x p covariance matrix (W)
        BetweenCovariance % Between class covariance (B)
        
        % Fitted regressor models for projection
        SubclassModels    % R total fitted base models
    end
    
    methods
        function obj = MDA(varargin)
            % MDA Constructor
            % Syntax: 
            %   model = MDA('NumSubclasses', 3, 'Mode', 'pooled')
            %   pmda = MDA('BaseModelConstructor', @() PenalizedRegression(...))
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'NumSubclasses', 3);
            addParameter(p, 'Mode', 'pooled');
            addParameter(p, 'MaxIter', 100);
            addParameter(p, 'Tol', 1e-4);
            addParameter(p, 'Verbose', false);
            addParameter(p, 'BaseModelConstructor', []);
            parse(p, varargin{:});
            
            obj.NumSubclasses = p.Results.NumSubclasses;
            obj.Mode = p.Results.Mode;
            obj.MaxIter = p.Results.MaxIter;
            obj.Tol = p.Results.Tol;
            obj.Verbose = p.Results.Verbose;
            obj.BaseModelConstructor = p.Results.BaseModelConstructor;
            
            % If no base model provided, use standard linear regression (equivalent to standard MDA)
            if isempty(obj.BaseModelConstructor)
                obj.BaseModelConstructor = @() OceanStats.Supervised.Regression.LinearModel();
            end
        end
        
        function fit(obj, X, y)
            % FIT Trains the MDA model on X and y
            obj.Classes = unique(y);
            obj.NumClasses = length(obj.Classes);
            K = obj.NumClasses;
            [N, p] = size(X);
            
            % Handle scalar NumSubclasses -> expand to vector
            if isscalar(obj.NumSubclasses)
                R = repmat(obj.NumSubclasses, K, 1);
            else
                R = obj.NumSubclasses(:);
                assert(length(R) == K, 'NumSubclasses must be scalar or length K.');
            end
            
            % Compute Class Priors
            obj.Priors = zeros(K, 1);
            for k = 1:K
                obj.Priors(k) = sum(y == obj.Classes(k)) / N;
            end
            
            if strcmp(obj.Mode, 'independent')
                % --- Independent Mode (Standard GMM per class) ---
                obj.GMMs = cell(K, 1);
                for k = 1:K
                    X_k = X(y == obj.Classes(k), :);
                    % Fit standard GMM. Note: this might fail if covariance 
                    % becomes singular. Regularization might be needed.
                    try
                        obj.GMMs{k} = fitgmdist(X_k, R(k), 'CovarianceType', 'full', ...
                            'Regularize', 1e-5, 'Options', statset('MaxIter', obj.MaxIter, 'TolFun', obj.Tol));
                    catch ME
                        if obj.Verbose
                            fprintf('GMM for class %d failed, falling back to 1 component.\n', k);
                        end
                        obj.GMMs{k} = fitgmdist(X_k, 1, 'Regularize', 1e-5);
                    end
                end
                
            elseif strcmp(obj.Mode, 'pooled')
                % --- Pooled Mode (ESL 12.6 Custom EM) ---
                obj.SubclassWeights = cell(K, 1);
                obj.SubclassMeans = cell(K, 1);
                
                % Step 1: Initialize
                % For each class, run K-means to get initial subclass centers
                % and weights. Compute initial pooled covariance from residuals.
                pooledCov = zeros(p, p);
                for k = 1:K
                    X_k = X(y == obj.Classes(k), :);
                    N_k = size(X_k, 1);
                    if N_k < R(k)
                        error('Class %d has fewer samples (%d) than requested subclasses (%d).', k, N_k, R(k));
                    end
                    
                    % Initial centers via k-means
                    [idx, C] = kmeans(X_k, R(k), 'Replicates', 3, 'EmptyAction', 'singleton');
                    
                    obj.SubclassMeans{k} = C;
                    
                    % Initial mixing weights
                    pi_k = zeros(R(k), 1);
                    for r = 1:R(k)
                        pi_k(r) = sum(idx == r) / N_k;
                    end
                    obj.SubclassWeights{k} = pi_k;
                    
                    % Add to pooled covariance estimate
                    for r = 1:R(k)
                        X_kr = X_k(idx == r, :);
                        if ~isempty(X_kr)
                            diffs = bsxfun(@minus, X_kr, C(r, :)); %How far away a point is from the center of its assigned sub-cluster
                            pooledCov = pooledCov + (diffs' * diffs);
                        end
                    end
                end
                
                % Normalize pooled covariance
                obj.PooledCovariance = pooledCov / N;
                % Add slight ridge to prevent singularity
                obj.PooledCovariance = obj.PooledCovariance + eye(p) * 1e-6;
                
                % Step 2: Custom EM Loop
                for iter = 1:obj.MaxIter
                    
                    % --- E-Step: Calculate Subclass Responsibilities ---
                    % gammas{k}(i, r) = prob that sample i belongs to subclass r of class k
                    % Note: We only compute responsibilities for samples IN class k.
                    gammas = cell(K, 1);
                    logLikelihood = 0;
                    
                    invCov = inv(obj.PooledCovariance);
                    detCov = det(obj.PooledCovariance);
                    constTerm = -0.5 * p * log(2*pi) - 0.5 * log(detCov);
                    
                    for k = 1:K
                        X_k = X(y == obj.Classes(k), :);
                        N_k = size(X_k, 1);
                        pi_k = obj.SubclassWeights{k};
                        mu_k = obj.SubclassMeans{k};
                        
                        % Compute log probabilities. log(pi_kr phi(x_i; mu_kr, Sigma)).  Numerator in Eq 12.62
                        
                        logP = zeros(N_k, R(k));
                        for r = 1:R(k)
                            %How likelikely it is that sub-cluster r generate data point in X_k
                            diffs = bsxfun(@minus, X_k, mu_k(r, :));
                            % Mahalanobis distance squared
                            mahalDist = sum((diffs * invCov) .* diffs, 2);
                            logP(:, r) = constTerm - 0.5 * mahalDist + log(pi_k(r) + 1e-10);
                        end
                        
                        % Log-sum-exp trick for numerical stability
                        maxLogP = max(logP, [], 2);
                        sumP = exp(bsxfun(@minus, logP, maxLogP));
                        normSumP = sum(sumP, 2);
                        
                        % Soft assignments. Eq 12.62. w_ikr
                        gammas{k} = bsxfun(@rdivide, sumP, normSumP);
                        
                        % Accumulate log likelihood. EQ 12.61
                        logLikelihood = logLikelihood + sum(maxLogP + log(normSumP));
                    end
                    
                    if iter > 1 && abs(logLikelihood - prevLL) < obj.Tol
                        if obj.Verbose
                            fprintf('MDA Pooled EM converged at iteration %d. LL=%.4f\n', iter, logLikelihood);
                        end
                        break;
                    end
                    prevLL = logLikelihood;
                    
                    % --- M-Step: Update Means, Weights, and Pooled Covariance ---
                    
                    % 1. Update Subclass Weights
                    for k = 1:K
                        N_kr = sum(gammas{k}, 1)';
                        obj.SubclassWeights{k} = N_kr / sum(N_kr);
                    end
                    
                    % 2. Update Subclass Means via Regularized Update (PMDA)
                    % This elegantly supports both standard MDA and PMDA.
                    
                    newSigma = zeros(p, p);
                    globalMean = mean(X)';
                    newBetween = zeros(p, p);

                    
                    % This is mathematically cleaner:
                    newSigma = zeros(p, p);
                    for k = 1:K
                        X_k = X(y == obj.Classes(k), :);
                        gamma_k = gammas{k}; 
                        N_kr = sum(gamma_k, 1)'; 
                        
                        % Update means 
                        for r = 1:R(k)
                            if N_kr(r) > 1e-10
                                % Generate smooth subclass center
                                % Check if Base constructor is PenalizedRegression
                                testObj = obj.BaseModelConstructor();
                                if isprop(testObj, 'PenaltyMatrix') && ~isempty(testObj.PenaltyMatrix)
                                    % PMDA smooth update: (N_kr * I + lambda * Omega) \ (X_k^T * gamma_k)
                                    Lambda = testObj.Lambda;
                                    Omega = testObj.PenaltyMatrix;
                                    rhs = X_k' * gamma_k(:, r);
                                    % Solve symmetric positive definite system
                                    smooth_mean = (N_kr(r) * eye(p) + Lambda * Omega) \ rhs;
                                    obj.SubclassMeans{k}(r, :) = smooth_mean';
                                else
                                    % Standard MDA raw update
                                    obj.SubclassMeans{k}(r, :) = (gamma_k(:, r)' * X_k) / N_kr(r);
                                end
                            end
                            
                            % Accumulate for new pooled covariance (using the derived centers)
                            diffs = bsxfun(@minus, X_k, obj.SubclassMeans{k}(r, :));
                            weightedDiffs = bsxfun(@times, diffs, gamma_k(:, r));
                            newSigma = newSigma + (weightedDiffs' * diffs);
                            
                            % Accumulate for between class covariance
                            meanDiff = obj.SubclassMeans{k}(r, :)' - globalMean;
                            newBetween = newBetween + N_kr(r) * (meanDiff * meanDiff');
                        end
                    end
                    
                    obj.PooledCovariance = newSigma / N + eye(p) * 1e-6;
                    obj.BetweenCovariance = newBetween / N;
                end
            end
        end
        
        function [y_pred, posterior] = predict(obj, X)
            % PREDICT Calculates class predictions and posterior probabilities
            [N_test, p] = size(X);
            K = length(obj.Classes);
            logPost = zeros(N_test, K);
            
            if strcmp(obj.Mode, 'independent')
                for k = 1:K
                    gmm = obj.GMMs{k};
                    % evaluate pdf
                    % Warning: pdf can underflow to 0. Log is safer if supported.
                    % gmdistribution doesn't explicitly expose logpdf easily.
                    % We'll use posterior probabilities and pdf carefully.
                    
                    % P(X|class k)
                    pk = pdf(gmm, X);
                    logPost(:, k) = log(pk + 1e-100) + log(obj.Priors(k));
                end
                
            elseif strcmp(obj.Mode, 'pooled')
                % ln phi(x; u; sigma) = -p/2 ln(2pi) - 1/2 ln|sigma|   - 1/2 (x-u)' sigma^-1 (x-u)
                invCov = inv(obj.PooledCovariance);
                detCov = det(obj.PooledCovariance);
                constTerm = -0.5 * p * log(2*pi) - 0.5 * log(detCov);  %-p/2 ln(2pi) - 1/2 ln|sigma| 
                
                for k = 1:K
                    mu_k = obj.SubclassMeans{k};
                    pi_k = obj.SubclassWeights{k};
                    R_k = length(pi_k);
                    
                    logP_sub = zeros(N_test, R_k);
                    for r = 1:R_k
                        diffs = bsxfun(@minus, X, mu_k(r, :));
                        mahalDist = sum((diffs * invCov) .* diffs, 2);
                        logP_sub(:, r) = constTerm - 0.5 * mahalDist + log(pi_k(r) + 1e-10);
                    end
                    
                    % Log-sum-exp over subclasses to get P(X | Class k)
                    maxLogP_sub = max(logP_sub, [], 2);
                    sumP_sub = exp(bsxfun(@minus, logP_sub, maxLogP_sub));
                    %Sum inside the log
                    logP_k = maxLogP_sub + log(sum(sumP_sub, 2));
                    
                    logPost(:, k) = logP_k + log(obj.Priors(k));
                end
            end
            
            % Resolve highest posterior
            [~, maxIdx] = max(logPost, [], 2);
            y_pred = obj.Classes(maxIdx);
            
            % Convert logPost to normalized probabilities for `posterior`
            if nargout > 1
                maxLogPost = max(logPost, [], 2);
                expPost = exp(bsxfun(@minus, logPost, maxLogPost));
                posterior = bsxfun(@rdivide, expPost, sum(expPost, 2));
            end
        end
        
        function X_proj = project(obj, X, dims)
            % PROJECT Projects data onto the canonical variates of the MDA model
            %
            % Syntax: X_proj = model.project(X, 2)
            %
            % Description:
            %   Computes the eigen-decomposition of W^{-1} B where W is the 
            %   pooled within-subclass covariance and B is the between-subclass 
            %   covariance. It then projects X onto the leading `dims` eigenvectors.
            
            if nargin < 3
                dims = 2;
            end
            
            if strcmp(obj.Mode, 'independent')
                error('Canonical projection is not supported for independent mode MDA. Use pooled mode.');
            end
            
            p = size(obj.PooledCovariance, 1);
            W = obj.PooledCovariance;
            B = obj.BetweenCovariance;
            
            % Solve generalized eigenvalue problem: B v = lambda W v
            % equivalent to (W^-1 B) v = lambda v
            [V, D] = eig(B, W);
            
            % Extract eigenvalues and sort descending
            eigenvalues = diag(D);
            [~, sortIdx] = sort(eigenvalues, 'descend');
            
            % Select top 'dims' eigenvectors (Canonical Discriminant Vectors)
            % Also normalize them such that V' W V = I
            V_sorted = V(:, sortIdx);
            V_top = V_sorted(:, 1:dims);
            
            % Normalize eigenvectors
            for i = 1:dims
                v = V_top(:, i);
                normFactor = sqrt(v' * W * v);
                V_top(:, i) = v / normFactor;
            end
            
            % Center X based on global training mean before projection
            % Wait, standard canonical variates usually don't strictly require 
            % centering for relative distances, but it's best practice. We'll leave X as is 
            % to match standard LDA behavior which projects raw X.
            X_proj = X * V_top;
        end
    end
end
