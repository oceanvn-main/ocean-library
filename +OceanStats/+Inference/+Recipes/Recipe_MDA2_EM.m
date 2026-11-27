classdef Recipe_MDA2_EM < OceanStats.Optimization.IMMRecipe
    % RECIPE_MDA2_EM  EM Recipe for MDA2 (Shared-Center Mixture Discriminant).
    %
    % Inherits IMMRecipe (Minorize-Maximize framework).
    % Plugs into GenericOptimizer for the iteration loop.
    %
    % THEORY (ESL Exercise 12.11):
    %   E-Step (minorize): compute responsibilities gamma_ir
    %     gamma_ir = pi_r * P_r(G=g_i) * phi(x_i; mu_r, Sigma) / normalizer
    %
    %   M-Step (maximize): update shared parameters
    %     mu_r         = Sum_i gamma_ir * x_i  /  Sum_i gamma_ir
    %     pi_r         = (1/N) * Sum_i gamma_ir
    %     P_r(G=k)     = Sum_{i: g_i=k} gamma_ir  /  Sum_i gamma_ir
    %     Sigma        = (1/N) * Sum_i Sum_r gamma_ir * (x_i - mu_r)(x_i - mu_r)'
    %
    % DATA FORMAT:
    %   data = [X, y]  where X is (N x p), y is (N x 1) class labels
    %
    % See also: MDA2Model, GenericOptimizer, Recipe_GMM_EM
    %
    % Author: Ocean Library System
    % Date:   2026-03-24

    properties
        Model   % OceanStats.Inference.Models.MDA2Model

        % Convergence tracking
        LogLikelihood     (1,1) double = -inf
        PrevLogLikelihood (1,1) double = -inf
        Tol               (1,1) double = 1e-4
    end

    methods
        function obj = Recipe_MDA2_EM(model, tol)
            arguments
                model
                tol (1,1) double = 1e-4
            end
            obj.Model = model;
            obj.Tol = tol;
        end

        % =================================================================
        %  IOptimizationRecipe Interface
        % =================================================================

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
            params = struct( ...
                'Mu',           obj.Model.Mu, ...
                'Sigma',        obj.Model.Sigma, ...
                'Pi',           obj.Model.Pi, ...
                'ClassProfile', obj.Model.ClassProfile);
        end

        function name = getName(~)
            name = "MDA2 via EM (Shared-Center Mixture Discriminant)";
        end
    end

    % =====================================================================
    %  IMMRecipe Implementation (E-Step / M-Step)
    % =====================================================================
    methods (Access = protected)

        function gammas = minorize(obj, data)
            % E-STEP: Compute responsibilities gamma_ir.
            %
            % gamma_ir = pi_r * P_r(G=g_i) * phi(x_i; mu_r, Sigma)
            %            / Sum_s pi_s * P_s(G=g_i) * phi(x_i; mu_s, Sigma)

            X = data(:, 1:end-1);
            y = data(:, end);
            [N, ~] = size(X);
            R = obj.Model.R;

            % Log Gaussian densities: (N x R)
            logPhi = obj.Model.computeLogPhi(X);

            % Build log-numerator for each (i, r)
            gammas = zeros(N, R);

            for i = 1:N
                k = find(obj.Model.Classes == y(i), 1);
                log_num = log(obj.Model.Pi + eps) ...
                        + log(obj.Model.ClassProfile(:, k) + eps) ...
                        + logPhi(i, :)';    % (R x 1)

                % Log-sum-exp normalization
                maxVal = max(log_num);
                gammas(i, :) = exp(log_num - maxVal)';
                gammas(i, :) = gammas(i, :) / (sum(gammas(i, :)) + eps);
            end
        end

        function params = maximize(obj, data, gammas)
            % M-STEP: Update Mu, Pi, ClassProfile, Sigma.

            X = data(:, 1:end-1);
            y = data(:, end);
            [N, D] = size(X);
            R = obj.Model.R;
            K = obj.Model.K;

            N_r = sum(gammas, 1)';  % (R x 1) effective count per center

            % --- 1. Update centers: mu_r = (Sum_i gamma_ir * x_i) / N_r ---
            Mu_new = zeros(R, D);
            for r = 1:R
                if N_r(r) > eps
                    Mu_new(r, :) = (gammas(:, r)' * X) / N_r(r);
                else
                    Mu_new(r, :) = obj.Model.Mu(r, :);  % keep old
                end
            end

            % --- 2. Update priors: pi_r = N_r / N ---
            Pi_new = N_r / N;

            % --- 3. Update class profiles: P_r(G=k) = Sum_{i:g_i=k} gamma_ir / N_r ---
            CP_new = zeros(R, K);
            for k = 1:K
                classMask = (y == obj.Model.Classes(k));
                CP_new(:, k) = sum(gammas(classMask, :), 1)' ./ (N_r + eps);
            end
            % Normalize rows to sum to 1 (P_r(G) is a probability over classes)
            rowSums = sum(CP_new, 2);
            CP_new = CP_new ./ (rowSums + eps);

            % --- 4. Update covariance: Sigma = (1/N) Sum_i Sum_r gamma_ir (x_i - mu_r)(x_i - mu_r)' ---
            Sigma_new = zeros(D, D);
            for r = 1:R
                diff = X - Mu_new(r, :);                    % (N x D)
                wDiff = diff .* sqrt(gammas(:, r));          % weighted
                Sigma_new = Sigma_new + (wDiff' * wDiff);
            end
            Sigma_new = Sigma_new / N;
            Sigma_new = Sigma_new + eye(D) * obj.Model.Regularization;

            params = struct( ...
                'Mu',           Mu_new, ...
                'Sigma',        Sigma_new, ...
                'Pi',           Pi_new, ...
                'ClassProfile', CP_new);
        end

        function updateModel(obj, params)
            obj.Model.Mu           = params.Mu;
            obj.Model.Sigma        = params.Sigma;
            obj.Model.Pi           = params.Pi;
            obj.Model.ClassProfile = params.ClassProfile;
        end
    end
end
