classdef Recipe_FixedSigmaGMM_EM < OceanStats.Optimization.IMMRecipe
    % RECIPE_FIXEDSIGMAGMM_EM  EM for GMM with frozen isotropic covariance.
    %
    % ESL Exercise 13.1: All components share Sigma_r = sigma * I where
    % sigma is a fixed hyperparameter (NOT learned).
    %
    % E-step: Compute soft responsibilities gamma_nk using the fixed sigma
    % M-step: Update Mu and Pi, but DO NOT update Sigma
    %
    % As sigma → 0, this recovers K-means (hard assignments + unweighted means).
    %
    % USAGE:
    %   model = OceanStats.Unsupervised.GMM(K);
    %   recipe = OceanStats.Inference.Recipes.Recipe_FixedSigmaGMM_EM(model, sigma);
    %   optimizer = OceanStats.Optimization.GenericOptimizer('MaxIter', 50, 'Verbose', false);
    %   [params, history] = optimizer.run(recipe, X);
    %
    %   gammas = recipe.getResponsibilities(X);   % for visualization
    %
    % Ref: Hastie et al. (2009), ESL Exercise 13.1
    
    properties
        Model           % OceanStats.Unsupervised.GMM
        SigmaScale      % scalar sigma (covariance = sigma * I)
        
        % Optimization state
        LogLikelihood       (1,1) double = -inf
        PrevLogLikelihood   (1,1) double = -inf
        Tol                 (1,1) double = 1e-6
        
        % Trajectory tracking
        CentroidHistory     % cell array of Mu snapshots
    end
    
    methods
        function obj = Recipe_FixedSigmaGMM_EM(model, sigmaScale, opts)
            % Constructor.
            %   recipe = Recipe_FixedSigmaGMM_EM(model, sigmaScale)
            %   recipe = Recipe_FixedSigmaGMM_EM(model, sigmaScale, 'Tol', 1e-8)
            arguments
                model
                sigmaScale  (1,1) double {mustBePositive} = 1.0
                opts.Tol    (1,1) double = 1e-6
            end
            obj.Model = model;
            obj.SigmaScale = sigmaScale;
            obj.Tol = opts.Tol;
            obj.CentroidHistory = {};
        end
        
        function initialize(obj, data, initMu)
            % INITIALIZE  Set model params before running optimizer.
            %   recipe.initialize(X, InitMu)
            %   InitMu: (K x D) initial centroids
            arguments
                obj
                data    double
                initMu  double = []
            end
            [~, D] = size(data);
            K = obj.Model.K;
            
            if isempty(initMu)
                ridx = randperm(size(data, 1), K);
                initMu = data(ridx, :);
            end
            
            obj.Model.Mu = initMu;
            obj.Model.Pi = ones(1, K) / K;
            
            % Freeze covariance: sigma * I for all components
            obj.Model.Sigma = zeros(D, D, K);
            for k = 1:K
                obj.Model.Sigma(:,:,k) = obj.SigmaScale * eye(D);
            end
            
            obj.CentroidHistory = {initMu};
        end
        
        % ---- IOptimizationRecipe interface ----
        
        function val = evaluate(obj, data)
            obj.PrevLogLikelihood = obj.LogLikelihood;
            obj.LogLikelihood = obj.Model.logLikelihood(data);
            val = -obj.LogLikelihood;   % GenericOptimizer minimizes
        end
        
        function [converged, metric] = isConverged(obj)
            diff = abs(obj.LogLikelihood - obj.PrevLogLikelihood);
            converged = diff < obj.Tol;
            metric = sprintf('LL Diff: %.6e', diff);
        end
        
        function params = getParams(obj)
            params = struct('Mu', obj.Model.Mu, 'Sigma', obj.Model.Sigma, 'Pi', obj.Model.Pi);
        end
        
        function name = getName(~)
            name = "Fixed-Sigma GMM via EM (ESL Ex 13.1)";
        end
        
        % ---- Public helpers ----
        
        function gammas = getResponsibilities(obj, data)
            % GETRESPONSIBILITIES  Compute E-step responsibilities.
            %   gammas = recipe.getResponsibilities(X)   % (N x K)
            gammas = obj.minorize(data);
        end
    end
    
    methods (Access = protected)
        
        % E-Step: Compute responsibilities (same as standard GMM)
        function gammas = minorize(obj, data)
            mu = obj.Model.Mu;
            sigma = obj.Model.Sigma;
            pi_vals = obj.Model.Pi;
            
            [N, ~] = size(data);
            K = length(pi_vals);
            log_rho = zeros(N, K);
            
            for k = 1:K
                log_pdf = obj.Model.log_mvnpdf(data, mu(k,:), sigma(:,:,k));
                log_rho(:,k) = log(pi_vals(k) + eps) + log_pdf;
            end
            
            max_log = max(log_rho, [], 2);
            log_sum = max_log + log(sum(exp(log_rho - max_log), 2));
            gammas = exp(log_rho - log_sum);
        end
        
        % M-Step: Update Mu and Pi ONLY — Sigma is FROZEN
        function params = maximize(obj, data, gammas)
            [N, D] = size(data);
            K = size(gammas, 2);
            N_k = sum(gammas, 1);
            
            mu_new = zeros(K, D);
            pi_new = zeros(1, K);
            
            for k = 1:K
                if N_k(k) < eps
                    mu_new(k,:) = obj.Model.Mu(k,:);   % keep old
                    pi_new(k) = 1 / K;
                    continue;
                end
                
                % Responsibility-weighted mean
                mu_new(k,:) = sum(gammas(:,k) .* data, 1) / N_k(k);
                pi_new(k) = N_k(k) / N;
            end
            
            % Sigma stays FROZEN — key difference from standard Recipe_GMM_EM
            params = struct('Mu', mu_new, 'Sigma', obj.Model.Sigma, 'Pi', pi_new);
        end
        
        % Update model (record trajectory)
        function updateModel(obj, params)
            obj.Model.Mu = params.Mu;
            obj.Model.Pi = params.Pi;
            % Sigma NOT updated — stays frozen at sigma * I
            
            % Record trajectory
            obj.CentroidHistory{end+1} = params.Mu;
        end
    end
end
