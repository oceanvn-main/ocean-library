classdef Recipe_GMM_EM < OceanStats.Optimization.IMMRecipe
    % RECIPE_GMM_EM The Recipe for Training GMM via EM (MM Framework).
    %
    % Inherits from IMMRecipe to enforce Minorize-Maximize structure.
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    properties
        Model       % OceanStats.Unsupervised.GMM (Pure Model)
        
        % Optimization State
        LogLikelihood (1,1) double = -inf
        PrevLogLikelihood (1,1) double = -inf
        Tol (1,1) double = 1e-6
    end

    methods
        function obj = Recipe_GMM_EM(model, tol)
            arguments
                model
                tol (1,1) double = 1e-6
            end
            obj.Model = model;
            obj.Tol = tol;
        end
        
        % -----------------------------------------------------------------
        % IOptimizationRecipe Implementation
        % -----------------------------------------------------------------
        
        function val = evaluate(obj, data)
            obj.PrevLogLikelihood = obj.LogLikelihood;
            obj.LogLikelihood = obj.Model.logLikelihood(data);
            val = obj.LogLikelihood;
        end
        
        function [converged, metric] = isConverged(obj)
            diff = abs(obj.LogLikelihood - obj.PrevLogLikelihood);
            converged = diff < obj.Tol;
            metric = sprintf('LL Diff: %.6e', diff);
        end
        
        function params = getParams(obj)
            params = struct('Mu', obj.Model.Mu, 'Sigma', obj.Model.Sigma, 'Pi', obj.Model.Pi);
        end
        
        function name = getName(obj)
            name = "GMM via EM (Minorize-Maximize)";
        end
    end
        
    % -----------------------------------------------------------------
    % IMMRecipe Implementation (The Recipe)
    % -----------------------------------------------------------------
        
    methods (Access = protected)
        
        % E-Step: Construct Minorizer (Responsibilities)
        function gammas = minorize(obj, data)
            mu = obj.Model.Mu;
            sigma = obj.Model.Sigma;
            pi_vals = obj.Model.Pi;
            
            [N, D] = size(data);
            K = length(pi_vals);
            log_rho = zeros(N, K);
            
            for k = 1:K
                try
                     log_pdf = obj.Model.log_mvnpdf(data, mu(k,:), sigma(:,:,k));
                catch
                     log_pdf = log(mvnpdf(data, mu(k,:), sigma(:,:,k)) + eps);
                end
                
                log_rho(:,k) = log(pi_vals(k) + eps) + log_pdf;
            end
            
            max_log = max(log_rho, [], 2);
            log_sum = max_log + log(sum(exp(log_rho - max_log), 2));
            gammas = exp(log_rho - log_sum);
        end
        
        % M-Step: Maximize Minorizer
        function params = maximize(obj, data, gammas)
            [N, D] = size(data);
            K = size(gammas, 2);
            N_k = sum(gammas, 1);
            
            mu_new = zeros(K, D);
            sigma_new = zeros(D, D, K);
            pi_new = zeros(1, K);
            
            for k = 1:K
                if N_k(k) < eps
                    sigma_new(:,:,k) = eye(D);
                    continue; 
                end
                
                weight_sum = sum(gammas(:,k) .* data, 1);
                mu_new(k,:) = weight_sum / N_k(k);
                
                diff = data - mu_new(k,:);
                w_diff = diff .* sqrt(gammas(:,k));
                s = (w_diff' * w_diff) / N_k(k);
                
                % Inject Mathematical Guardrails for p > N architectures
                if obj.Model.CovarianceType == "diagonal"
                    s = diag(diag(s));
                end
                
                s = s + eye(D) * obj.Model.Regularization;
                sigma_new(:,:,k) = s;
                
                pi_new(k) = N_k(k) / N;
            end
            
            params = struct('Mu', mu_new, 'Sigma', sigma_new, 'Pi', pi_new);
        end
        
        % Update Model Reference
        function updateModel(obj, params)
            obj.Model.Mu = params.Mu;
            obj.Model.Sigma = params.Sigma;
            obj.Model.Pi = params.Pi;
        end
    end
    
    % -----------------------------------------------------------------
    % Visualization Support
    % -----------------------------------------------------------------
    methods (Access = public)
        function q = calculate_Q(obj, data, params_new, params_old)
            % Q(theta | theta_old)
            
            % 1. Get Gammas from OLD params (Re-run E-step logic for visualization)
            % We can't reuse internal state easily because visualizer asks for arbitrary params_old.
            % So we extract E-step logic or call minorize with temporary state injection?
            % Since minorize uses obj.Model, we would need to set obj.Model to params_old temporarily.
            % That's messy.
            % Better: Helper function that is stateless.
            
            % Temporarily set model state? No, avoid side effects.
            % Just reimplement the math briefly or use a static helper.
            
            % Manual E-Step for arbitrary params_old:
            mu_old = params_old.Mu;
            sigma_old = params_old.Sigma;
            pi_old = params_old.Pi;
            
            [N, D] = size(data);
            K = length(pi_old);
            log_rho = zeros(N, K);
            
            for k = 1:K
                 try
                     log_pdf = obj.Model.log_mvnpdf(data, mu_old(k,:), sigma_old(:,:,k));
                 catch
                     log_pdf = log(mvnpdf(data, mu_old(k,:), sigma_old(:,:,k)) + eps);
                 end
                 log_rho(:,k) = log(pi_old(k) + eps) + log_pdf;
            end
            max_log = max(log_rho, [], 2);
            log_sum = max_log + log(sum(exp(log_rho - max_log), 2));
            gammas_old = exp(log_rho - log_sum);
            
            % 2. Expected LL
            q = 0;
            for k = 1:K
                 try
                     log_pdf = obj.Model.log_mvnpdf(data, params_new.Mu(k,:), params_new.Sigma(:,:,k));
                 catch
                     log_pdf = log(mvnpdf(data, params_new.Mu(k,:), params_new.Sigma(:,:,k)) + eps);
                 end
                 term = log(params_new.Pi(k) + eps) + log_pdf;
                 q = q + sum(gammas_old(:,k) .* term);
            end
        end
    end
end
